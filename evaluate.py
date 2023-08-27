import os
import sys
import re
import time

import peft
import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import datasets
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# パス
model_path = "output_dir"
base_model =  "meta-llama/Llama-2-7b-hf"
checkpoint_dirs = os.listdir(model_path)
checkpoint_path = os.path.join(model_path,sorted(checkpoint_dirs)[-1])

print("loading weights from {}".format(checkpoint_path))

if torch.cuda.is_available():
    # 一つ目のgpuだけ使う
    device = {"":0}
else:
    device = torch.device("cpu")

# modelの読み込み
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,device_map={"":0}
)
tokenizer.pad_token_id = 0

# ベースモデルを読み込んでからpeftの重みを読み込む
model = AutoModelForCausalLM.from_pretrained(base_model, load_in_8bit=True, device_map={"": 0})
model = peft.PeftModel.from_pretrained(model,checkpoint_path)

# generate configの設定
generation_config = transformers.GenerationConfig(max_new_tokens=256)

# load dataset
data_path = "./dataset/SVAMP/test.json"
dataset = datasets.load_dataset("json", data_files=data_path)
num_examples = len(dataset["train"])

def group_batch(batch):
    return {k: [v] for k, v in batch.items()}

def generate_prompt(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        if example["input"][i] != "":
            output_texts.append(f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                    ### Instruction:
                    {example["instruction"][i]}

                    ### Input:
                    {example["input"][i]}

                    ### Response:
                    """)
        else:
            output_texts.append(f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                    ### Instruction:
                    {example["instruction"][i]}


                    ### Response:
                    """)
    return  output_texts

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


dataset = dataset['train'].map(group_batch, batched=True, batch_size=4)

number_of_correct_answers = 0
number_of_examples = 0

for batch in dataset:
    # どうやってバッチ化する?
    batched_input = generate_prompt(batch)
    batched_answer = list(map(float,batch["answer"]))
    batched_encoded_input = tokenizer(batched_input, padding=True, return_tensors="pt").to("cuda")

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
    batched_decoded_output = tokenizer.batch_decode(batched_output.sequences,skip_special_tokens=True)
    batched_generated_answer = [
        extract_answer_number(decoded_output.split("### Response:")[1])
        for decoded_output in batched_decoded_output
    ]

    for decoded_output in batched_decoded_output:
        print("------------")
        print(decoded_output)
        print("------------")

    number_of_correct_answers += np.sum(
        np.array(batched_answer) == np.array(batched_generated_answer)
    )
    number_of_examples += len(batched_answer)