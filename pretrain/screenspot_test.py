import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
import random
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
from process_utils import pred_2_point, extract_bbox
from datetime import datetime
from prompt_lib import web_loca_all_point_prompt

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)

index = 1
server = ['L20', '118', 'A6000'][index]
qwen_path = [
    "/mnt/nvme0n1p1/hongxin_li/hf_home/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8",
    "Qwen/Qwen-VL-Chat",
    "/data2/hongxin_li/hf_home/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8"
][index]

screenspot_imgs = [
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/ScreenSpot/screenspot_imgs",
    "/data0/jingran/workspace/UI_training_data/screen_spot/images",
    "/data2/hongxin_li/UI_training_data/ScreenSpot/screenspot_imgs"
][index]

parser = argparse.ArgumentParser()
parser.add_argument('--our_model_path', type=str, default="/data2/hongxin_li/SeeClick/checkpoints/reproduce_seeclick_full")
parser.add_argument('--lora_path', type=str, default="/mnt/nvme0n1p1/hongxin_li/seeclick_exp/checkpoints/funcpred625k_wicrico330k_llava150k_cauldron197k_refGnd_iconResample_v2")
parser.add_argument('--screenspot_imgs', type=str, default=screenspot_imgs)
parser.add_argument('--screenspot_test', type=str, default=os.path.dirname(screenspot_imgs))
parser.add_argument('--task', type=str, default="all")
args, unknown = parser.parse_known_args()

tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)

if args.lora_path.lower():
    # use lora
    print("load Lora")
    lora_path = args.lora_path
    model = AutoPeftModelForCausalLM.from_pretrained(lora_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()
    
    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(device=model.device, dtype=torch.bfloat16)
else:
    # use Qwen-VL-Chat
    model_path = qwen_path if not args.our_model_path else args.our_model_path
    print("Loading ", model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()

print("Load Success")
model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)

eval_result_dir = os.path.join('/'.join(__file__.split('/')[:-2]), "eval_results/screenspot")

if args.lora_path:
    if "snapshots" in args.lora_path:
        postfix = args.lora_path[args.lora_path.find("models--") + 8: args.lora_path.find("snapshots") - 1]
    else:
        run_name, ckpt = args.lora_path.split('/')[-2:]
        postfix = os.path.join(run_name, ckpt)
else:
    if "snapshots" in qwen_path:
        postfix = qwen_path[qwen_path.find("models--") + 8: qwen_path.find("snapshots") - 1]
    elif len(qwen_path.split('/')) == 2:
        postfix = qwen_path.replace('/', '--')
    else:
        postfix = qwen_path.split('/')[-1]
        
os.makedirs(os.path.join(eval_result_dir, postfix), exist_ok=True)

if args.task == "all":
    tasks = ["mobile", "desktop", "web"]
else:
    tasks = [args.task]
tasks_result = {}
result = []
all_correct = {"text": [0,0], "icon": [0,0]}

for task in tasks:
    dataset = "screenspot_" + task + ".json"
    screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
    print(f"[{task}] Num of sample: {len(screenspot_data)}")
    prompt_origin = "In the UI, where should I click if I want to complete instruction \"{}\" (with point)?"
    prompt_origin_qwen = "Generate the bounding box of {}"
    num_action = 0
    corr_action = 0
    text_correct = []
    icon_correct = []
    num_wrong_format = 0
    for j, item in tqdm(enumerate(screenspot_data), total=len(screenspot_data), desc=f"{postfix} | {task}"):
        num_action += 1
        filename = item["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        if not os.path.exists(img_path):
            print("img not found")
            input()
        image = Image.open(img_path)
        instruction = item["instruction"]
        bbox = item["bbox"]
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        img_size = image.size
        bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]

        # prompt = prompt_origin.format(instruction)
        prompt = random.choice(web_loca_all_point_prompt) + ' ' + instruction
        query = tokenizer.from_list_format([{'image': img_path},  # Either a local path or an url
                                            {'text': prompt}, ])
        # print(query)
        response, history = model.chat(tokenizer, query=query, history=None)
        print(response)
        
        temp = {"img_path": img_path, "text": instruction, "gt": bbox,
                           "type": item["data_type"], "source": item["data_source"], "response": response}
        try:
            if 'box' in response:
                pred_bbox = extract_bbox(response)
                click_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2, (pred_bbox[0][1] + pred_bbox[1][1]) / 2]
                click_point = [item / 1000 for item in click_point]
            else:
                click_point = pred_2_point(response, scale=100)
            
            temp["pred"] = click_point
            
            if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                corr_action += 1
                if item["data_type"] == 'text':
                    text_correct.append(1)
                else:
                    icon_correct.append(1)
                temp["status"] = "correct"
                logging.info("match " + str(corr_action / num_action))
            else:
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                temp["status"] = "wrong"
                logging.info("unmatch " + str(corr_action / num_action))
            
        except:
            num_wrong_format += 1
            if item["data_type"] == 'text':
                text_correct.append(0)
            else:
                icon_correct.append(0)
            
            temp["status"] = "invalid"
            logging.info("Step: " + str(j) + " wrong format")

        result.append(temp)

    logging.info("Action Acc: " + str(corr_action / num_action))
    logging.info("Total num: " + str(num_action))
    logging.info("Wrong format num: " + str(num_wrong_format))
    logging.info("Text Acc: " + str(sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0))
    logging.info("Icon Acc: " + str(sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0))

    text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0
    icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0
    
    all_correct["text"][0] += sum(text_correct); all_correct["text"][1] += len(text_correct)
    all_correct["icon"][0] += sum(icon_correct); all_correct["icon"][1] += len(icon_correct)
    
    tasks_result[task] = {"text": [text_acc, len(text_correct)], "icon": [icon_acc, len(icon_correct)]}

logging.info(args.lora_path)
logging.info(tasks_result)



tasks_result["all"] = {
                        "text": [all_correct["text"][0] / all_correct["text"][1], all_correct["text"]],
                       "icon": [all_correct["icon"][0] / all_correct["icon"][1], all_correct["icon"]],
                       "overall": [(all_correct["text"][0] + all_correct["icon"][0]) / (all_correct["text"][1] + all_correct["icon"][1]), all_correct["text"][1] + all_correct["icon"][1]]
                    }

print(tasks_result["all"])
time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
save_to = os.path.join(eval_result_dir, postfix, datetime.now().strftime("%m-%d-%H-%M-%S") + ".json")

print(f"Finished at {time_str}. Save eval results to", save_to)
with open(save_to, "w") as f:
    json.dump({"logs": result, "eval_result": tasks_result, "time": time_str}, f, indent =2)
