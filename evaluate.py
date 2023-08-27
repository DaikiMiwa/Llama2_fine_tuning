import os
import sys
import re
import time

import peft
import torch
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
import wandb
import datasets
import numpy as np
import fire

def group_batch(batch):
    return {k: [v] for k, v in batch.items()}

def generate_prompt(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        if example["input"][i] != "":
            output_texts.append(
                f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                    ### Instruction:
                    {example["instruction"][i]}

                    ### Input:
                    {example["input"][i]}

                    ### Response:
                    """
            )
        else:
            output_texts.append(
                f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                    ### Instruction:
                    {example["instruction"][i]}


                    ### Response:
                    """
            )
    return output_texts

def extract_answer_letter(sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r"A|B|C|D|E", sentence_)
    if pred_answers:
        if not pred_answers:
            return ""
        return pred_answers[0]
    else:
        return ""

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float("inf")
    return pred_answer

def evaluate():

# パス
    base_model = "meta-llama/Llama-2-7b-hf"
    adapter = "lora"
    num_train_epochs = 3
    device_map = {"": 0}
    batch_size = 16
    dataset = "SVAMP"
    load_in_8bit = True
    load_in_4bit = False

    if not load_in_8bit ^ load_in_4bit:
        raise ValueError("load_in_8bit and load_in_4bit cannot be True at the same time.")
    checkpoint_path = f"./trained_models/{base_model}_{adapter}_{num_train_epochs}/result"
    data_path = f"./dataset/{dataset}/test.json"

    wandb.init(
            project_name="llm-evaluation",
            config={
                "base_model": base_model,
                "adapter": adapter,
                "checkpoint_path": checkpoint_path,
                "num_train_epochs": num_train_epochs,
                "batch_size": batch_size,
                "dataset": dataset,
                "load_in_8bit": load_in_8bit,
                "load_in_4bit": load_in_4bit,
            },
    )

    print("loading weights from {}".format(checkpoint_path))

# modelの読み込み
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model = peft.PeftModel.from_pretrained(model, checkpoint_path)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token_id = 0

    # generate configの設定
    # 要見直し
    generation_config = transformers.GenerationConfig(max_new_tokens=256)

    # load dataset
    dataset = datasets.load_dataset("json", data_files=data_path)
    num_examples = len(dataset["train"])

    dataset = dataset["train"].map(group_batch, batched=True, batch_size=4)

    number_of_correct_answers = 0
    number_of_examples = 0

    for batch in dataset:
        # どうやってバッチ化する?
        batched_input = generate_prompt(batch)
        batched_answer = list(map(float, batch["answer"]))
        batched_encoded_input = tokenizer(
            batched_input, padding=True, return_tensors="pt"
        ).to("cuda")

        start_time = time.time()
        batched_output = model.generate(
            **batched_encoded_input,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
        )
        end_time = time.time()
        print(f"Generate response takes {end_time - start_time}s ...")
        batched_output_sequences = batched_output.sequences
        batched_decoded_output = tokenizer.batch_decode(
            batched_output.sequences, skip_special_tokens=True
        )
        batched_generated_answer = [
            extract_answer_number(decoded_output.split("### Response:")[1])
            for decoded_output in batched_decoded_output
        ]

        number_of_correct_answers += np.sum(
            np.array(batched_answer) == np.array(batched_generated_answer)
        )
        number_of_examples += len(batched_answer)

        accuracy = number_of_correct_answers / number_of_examples
        wandb.log({"accuracy": accuracy})

if __name__ == "__main__":
    fire.Fire(evaluate)

