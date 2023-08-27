import os

import peft
import torch
import transformers
from datasets import load_dataset
# todo : CasualLM と CasualLLM の違いを調べる
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # 設定
use_gradient_checkpointing = True

output_dir = "./llama2-7b-lora"

# # 量子化読み込みの設定

# 8bit量子化を行う
load_in_8bit = True
load_in_4bit = False

quantization_config = BitsAndBytesConfig(
    load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
)

# GPUの0番目を使う
device_map = {"":0}
# bfloat16を使う
torch_dtype = torch.float16

# # モデルの読み込み

# +
# hugging faceのモデルを指定
# use llama2
base_model = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
)
# -
tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True)
tokenizer.pad_token_id = 0

# peftの設定
lora_r: int = 8
lora_alpha: int = 16
lora_dropout: float = 0.05
lora_target_modules: list[str] = None

lora_config = peft.LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    bias="none",
    task_type="CASUAL_LM",
)

peft_config = lora_config
# この関数でモオリジナルのモデルをフリーズさせる
model = peft.prepare_model_for_kbit_training(model,use_gradient_checkpointing=use_gradient_checkpointing)
model = peft.get_peft_model(model,peft_config)

# checkpointのフォルダからの読み込み
# checkpoint_dir = None
# if os.path.exists(output_dir):
#     files = os.listdir(output_dir)
#     checkpoint_files = list(filter(
#             lambda x:os.path.isfile(os.path.join(output_dir,x)),
#             os.listdir(output_dir),
#             ))
#     checkpoint_dirs = list(filter(
#             lambda x:os.path.isdir(os.path.join(output_dir,x)),
#             os.listdir(output_dir),
#            ))
#     if "adapter_config.json" in checkpoint_files:
#         peft_weight = peft.load_peft_weights(output_dir,device=device_map)
#         model = peft.set_peft_model_state_dict(model,peft_weight)
#         checkpoint_dir = None
#         print("-------------------------------------")
#         print(f"find adapter_config.json in {output_dir}")
#         print("-------------------------------------")
#     elif len(checkpoint_dirs) > 0 :
#         print(checkpoint_dirs)
#         for checkpoint_dir in sorted(checkpoint_dirs,reverse=True):
#             print(os.listdir(os.path.join(output_dir,checkpoint_dir)))
#             if "adapter_config.json" in os.listdir(os.path.join(output_dir,checkpoint_dir)):
#                 peft_weight = torch.load(os.path.join(output_dir,checkpoint_dir,"adapter_model.bin"))
#                 model = peft.set_peft_model_state_dict(model,peft_weight)
#                 checkpoint_dir = os.path.join(output_dir,checkpoint_dir)
#                 print("-------------------------------------")
#                 print(f"find adapter_config.json in {checkpoint_dir}")
#                 print("-------------------------------------")
#                 lora_config=None
#                 break
#     else :
#         print("-------------------------------------")
#         print("can't find adapter_config.json")
#         print("-------------------------------------")
# else :
#     print("-------------------------------------")
#     print("can't find adapter_config.json")
#     print("-------------------------------------")

model.print_trainable_parameters()

# # 学習
# ## 基本の学習パラメータの設定
batch_size = 16
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size
learining_rate = 3e-4
num_train_epochs = 1
logging_steps = 10
# データセットとepoch数から最大ステップ数を計算する
max_steps = -1
eval_steps = 200
save_steps = 200
save_total_limit = 3
val_set_size = 120
group_by_length = False
max_seq_length = 256


def formatting_func(example: dict):
    """データセットのフォーマットを設定する関数

    Args :
        example (dict): データセットのdict
    Returns
        (str): フォーマットされたデータセット
    """
    if example["input"] != "":
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

            ### Instruction:
            {example["instruction"]}
            
            ### Input:
            {example["input"]}
            
            ### Response:
            {example["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

            ### Instruction:
            {example["instruction"]}
            
            ### Response:
            {example["output"]}"""

def _formatting_func(example: dict):
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

# def formatting_func(example):
#     text = f"###Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n ### Instruction: {example['instruction']}\n ### Input: {example['input']}\n ### Response: {example['output']}"
#     return text

# print("------------------------------------")
# print()
# if checkpoint_dir is None:
#     resume_from_checkpoint = None
#     print(f"not resume_from_checkpoint")
# else:
#     print(f"resume_from_checkpoint : {checkpoint_dir}")
# print("------------------------------------")
# print()

training_args = TrainingArguments(
    per_device_train_batch_size=micro_batch_size,
    num_train_epochs=3,
    learning_rate=learining_rate,
    logging_steps=logging_steps,
    # eval_steps, save_stepsごとに評価と保存を行う
    eval_steps=eval_steps,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    # optimizerは自然に任せる
    optim="adamw_torch",
    max_steps=max_steps,
    output_dir=output_dir,
    remove_unused_columns=False,
    # resume_from_checkpoint=checkpoint_dir,
)

dataset = load_dataset("json", data_files="math_data.json")

# peftの重みの読み込み

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    max_seq_length=max_seq_length,
    formatting_func=_formatting_func,
    packing=False,
    # peft_config=peft_config,
)

trainer.train(resume_from_checkpoint=os.path.join(output_dir,"checkpoint"))
trainer.save_model(os.path.join(os.path.join(output_dir, f"checkpoint")))
