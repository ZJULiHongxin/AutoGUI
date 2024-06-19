import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
import ast
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)

index = 1
server = ['L20', '118', 'A6000'][index]
qwen_path = [
    "/mnt/nvme0n1p1/hongxin_li/hf_home/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8",
    "Qwen/Qwen-VL-Chat",
    "/data2/hongxin_li/hf_home/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8"
][index]

ROOT = [
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/images/seeclick_web_imgs"
]
screenspot_imgs = [
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/ScreenSpot/screenspot_imgs",
    "/data0/jingran/workspace/UI_training_data/screen_spot/images",
    "/data2/hongxin_li/UI_training_data/ScreenSpot/screenspot_imgs"
][index]

parser = argparse.ArgumentParser()
parser.add_argument('--lora_path', type=str, default="/data2/hongxin_li/seeclick_exp/checkpoints/reproduce_seeclick_aitw/checkpoint-400")

args, unknown = parser.parse_known_args()

tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)

# use lora
print("load Lora")
lora_path = args.lora_path
model = AutoPeftModelForCausalLM.from_pretrained(lora_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()

for name, parameter in model.named_parameters():
    if parameter.dtype != torch.bfloat16:
        parameter.data = parameter.data.to(device=model.device, dtype=torch.bfloat16)


print("Load Success")
model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)

model.save_pretrained(args.lora_path+"-merged")
model.base_model.save_pretrained(args.lora_path+"-merged")