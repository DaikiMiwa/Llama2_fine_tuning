import os

import peft
import torch
import transformers
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
import wandb
import fire
from trl import SFTTrainer

def formatting_func(example: dict):
    # packing=Falseの時に必要
    """データセットのフォーマットを設定する関数

    Args :
        example (dict): データセットのdict
    Returns
        (str): フォーマットされたデータセット
    """
    output_texts = []
    for i in range(len(example["instruction"])):
        if example["input"][i] != "":
            output_texts.append(f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {example["instruction"][i]}
                
                ### Input:
                {example["input"][i]}
                
                ### Response:
                {example["output"][i]}""")
        else:
            output_texts.append(f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {example["instruction"][i]}
                
                ### Response:
                {example["output"][i]}""")
        
    return output_texts

def generate_peft_config(adapter: str):

    if "lora" == adapter:
        # とりまデフォルトで・・・
        lora_config = peft.LoraConfig(
            task_type="CASUAL_LM",
        )
        return lora_config
    elif "adalora" == adapter:
        # とりまデフォルトで・・・
        return peft.AdaLoraConfig(
            task_type="CASUAL_LM",
        )
    else:
        raise ValueError(f"adapter {adapter} is not supported.")

def train(
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
        base_model: str = "",
        adapter: str = "",
        data_path : str = "math_data.json",
        resume_from_checkpoint: bool = False,
        ):

    # 量子化読み込みの設定
    if not load_in_8bit ^ load_in_4bit:
        raise ValueError("load_in_8bit and load_in_4bit cannot be True at the same time.")

    # 学習に関する設定
    ## 大事だから設定して欲しいやつ
    if base_model == "":
        raise ValueError("base_model is not set.")
    if adapter not in ["lora","adalora"]:
        raise ValueError(f"{adapter} is not supported.")

    # 大事じゃない
    batch_size = 16
    micro_batch_size = 4
    use_gradient_checkpointing = True
    num_train_epochs = 1
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = {"":0} # single gpuでの学習を想定
    torch_dtype = torch.float16 # or torch.bfloat16
    logging_steps = 10
    learining_rate = 3e-4
    max_steps = -1 # データセットとepoch数から最大ステップ数を計算
    eval_steps = 200
    save_steps = 200
    save_total_limit = 3
    max_seq_length = 256
    group_by_length = False # ?

    output_dir = f"trained_models/{base_model}_{adapter}_{num_train_epochs}"

    # wandbの設定
    wandb.init(
            project="llm-fine-tuning"
            config = {
                "base_model": base_model,
                "adapter": adapter,
                "load_in_8bit": load_in_8bit,
                "load_in_4bit": load_in_4bit,
                "batch_size": batch_size,
                "micro_batch_size": micro_batch_size,
                "use_gradient_checkpointing": use_gradient_checkpointing,
                "num_train_epochs": num_train_epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "device_map": device_map,
                "torch_dtype": torch_dtype,
                "logging_steps": logging_steps,
                "learining_rate": learining_rate,
                "max_steps": max_steps,
                "eval_steps": eval_steps,
                "save_steps": save_steps,
                "save_total_limit": save_total_limit,
                "max_seq_length": max_seq_length,
                "group_by_length": group_by_length,
                "output_dir": output_dir,
                "data_path": data_path,
                "resume_from_checkpoint": resume_from_checkpoint,
            }
    )

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

    tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            add_eos_token=True
            ) 
    tokenizer.pad_token_id = 0

    # peftの設定
    peft_config = generate_peft_config(adapter)
    model = peft.prepare_model_for_kbit_training(model,use_gradient_checkpointing=use_gradient_checkpointing)
    model = peft.get_peft_model(model,peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learining_rate,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        optim="adamw_torch",
        max_steps=max_steps,
        output_dir=output_dir,
        remove_unused_columns=False, # 必要
        report_to="wandb"
    )

    dataset = load_dataset("json", data_files=data_path)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        max_seq_length=max_seq_length,
        formatting_func=_formatting_func,
        packing=False,
        # peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=os.path.join(output_dir))
    trainer.save_model(os.path.join(os.path.join(output_dir, f"result")))

if __name__ == "__main__":
    fire.Fire(train)
