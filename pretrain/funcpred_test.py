import torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
import numpy as np
import json
import re
import argparse
import os
from PIL import Image, ImageDraw
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

funcpred_images = [
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/ScreenSpot/funcpred_images",
    "/data0/jingran/workspace/UI_training_data/func_pred/imgs",
    "/data2/hongxin_li/UI_training_data/func_pred/imgs"
][index]

parser = argparse.ArgumentParser()
parser.add_argument('--our_model_path', type=str, default="")
parser.add_argument('--lora_path', type=str, default="/data0/jingran/workspace/hongxin_li/seeclick_exp/checkpoints/L20_funcpred25k_llava150k_cauldron197k_refGnd_resample_0605")
parser.add_argument('--funcpred_images', type=str, default=funcpred_images)
parser.add_argument('--funcpred_test', type=str, default=os.path.dirname(funcpred_images))

args, unknown = parser.parse_known_args()

tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)

DEBUG = False
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

eval_result_dir = os.path.join('/'.join(__file__.split('/')[:-2]), "eval_results/funcpred")

if args.lora_path:
    if "snapshots" in args.lora_path:
        postfix = args.lora_path[args.lora_path.find("models--") + 8: args.lora_path.find("snapshots") - 1]
    else:
        run_name, ckpt = args.lora_path.split('/')[-2:]
        postfix = os.path.join(run_name, ckpt)
else:
    if "snapshots" in model_path:
        postfix = model_path[model_path.find("models--") + 8: qwen_path.find("snapshots") - 1]
    elif len(model_path.split('/')) == 2:
        postfix = model_path.replace('/', '--')
    else:
        postfix = model_path.replace('/', '-')
        
os.makedirs(os.path.join(eval_result_dir, postfix), exist_ok=True)


tasks_result = {}
result = []

all_correct = {"mobile": {"text": [], "icon": []}, "web": {"text": [], "icon": []}, "overall": {"num_samples": 0, "all": 0}}

funcpred_data = json.load(open(os.path.join((os.path.dirname(funcpred_images)), "func_pred.json"), 'r'))

print(f"Num of sample: {len(funcpred_data)}")
prompt_origin = "In the UI, where should I click if I want to complete instruction \"{}\" (with point)?"
prompt_origin_qwen = "Generate the bounding box of {}"

num_wrong_format = 0
random.shuffle(funcpred_data)
for j, item in tqdm(enumerate(funcpred_data), total=len(funcpred_data), desc=f"FuncPred {postfix}"):
    if DEBUG and j == 16: break
    filename = item["image"]
    img_path = os.path.join(args.funcpred_images, filename)
    if not os.path.exists(img_path):
        print("img not found", img_path)
        input()
    image = Image.open(img_path)
    W, H = image.size
    
    instruction = item["func"]
    bbox = item["unnormalized_box"]

    # Debug
    if False:
        draw = ImageDraw.Draw(image)
        draw.rectangle([(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))], outline=(0,0,255))
        image.save("test.png")
        continue
    bbox = [bbox[0] / W, bbox[1] / H, bbox[2] / W, bbox[3] / H]

    # prompt = apply_vlm_template(instruction, model_name=postfix, output_box=True)
    prompt = random.choice(web_loca_all_point_prompt) + ' ' + instruction
    query = tokenizer.from_list_format([{'image': img_path},  # Either a local path or an url
                                        {'text': prompt}, ])
    # print(query)
    response, history = model.chat(tokenizer, query=query, history=None)
    
    temp = {"img_path": img_path, "instruc": instruction, "gt_point": item["point"], "gt_box": bbox, "type": item["elem_type"], "device": item["device"], "elem_text": item["elem_text"], "response": response}
    
    all_correct["overall"]["num_samples"] += 1
    
    try:
        if 'box' in response:
            pred_bbox = extract_bbox(response)
            click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
            click_point = [item / 1000 for item in click_point]
        else:
            click_point = pred_2_point(response, scale=100)
        
        temp["pred"] = click_point
        
        if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
            all_correct["overall"]["all"] += 1
            all_correct[item["device"]][item["elem_type"]].append(1)
            temp["status"] = "correct"
        else:
            all_correct[item["device"]][item["elem_type"]].append(0)
            temp["status"] = "wrong"
        
    except Exception as e:
        num_wrong_format += 1
        all_correct[item["device"]][item["elem_type"]].append(0)
        click_point = [-1,-1]
        temp["status"] = "invalid"
        logging.info("Step: " + str(j) + " wrong format")
    
    box_str = ', '.join(f'{x:.2f}' for x in bbox)
    print(f"Response: {response} | Extracted point: {click_point} | GT Box: {box_str}")

    logging.info(f'Acc(%): {all_correct["overall"]["all"] / all_correct["overall"]["num_samples"]*100:.1f}')

    result.append(temp)

all_correct["overall"]["all"] = all_correct["overall"]["all"] / all_correct["overall"]["num_samples"]

logging.info(f"Overall Acc (%): {all_correct['overall']['all'] * 100:.2f}")
logging.info(f"Total num: {all_correct['overall']['num_samples']}")
logging.info("Wrong format num: " + str(num_wrong_format))

all_correct["overall"]["text"] = np.mean(all_correct["mobile"]["text"] + all_correct["web"]["text"]).item()
all_correct["overall"]["icon"] = np.mean(all_correct["mobile"]["icon"] + all_correct["web"]["icon"]).item()
all_correct["overall"]["all"] = all_correct["overall"]["all"]

logging.info("Text Acc: {:.1f}%".format(all_correct["overall"]["text"] * 100))
logging.info("Icon Acc: {:.1f}%".format(all_correct["overall"]["icon"] * 100))


logging.info(args.lora_path)


time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"Finished at {time_str}")
with open(os.path.join(eval_result_dir, postfix, datetime.now().strftime("%m-%d-%H-%M-%S") + ".json"), "w") as f:
    json.dump({"logs": result, "eval_result": all_correct, "time": time_str}, f, indent =2)
