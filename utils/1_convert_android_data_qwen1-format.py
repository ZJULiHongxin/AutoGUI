# 将安卓模拟器样本转成Qwen-VL格式
import os, json, random
from PIL import Image
from tqdm import tqdm

android_file = "/data0/jingran/workspace/UI_training_data/Ours-Pretrain/scaling_exp/android_system/android_system_samples_test.json"

with open(android_file, 'r') as f:
    data = json.load(f)

new_samples = []

for x in tqdm(data, total=len(data)):
    new_messages = []
    for i, message in enumerate(x['conversations']):
        if message['from'] == 'human':
            this_message = message["value"].replace("<image>","").strip()
            next_message = x['conversations'][i+1]['value']
            
            if not next_message.startswith('(') and not next_message.endswith(')') or len(next_message) >= 8:
                left_b = this_message.find('(')
                this_message = this_message[:left_b] + '<ref>' + this_message[left_b:]

            new_messages.append({"from": "user", "value": this_message})
        else:
            new_messages.append({"from": "assistant", "value": message["value"].replace("<image>","").strip()})
    
    image_path = os.path.join("/data0/jingran/workspace/UI_training_data/Ours-Pretrain/scaling_exp", x['image'])
    assert os.path.exists(image_path), image_path

    new_messages[0]['value'] = f"Picture 1: <img>{image_path}</img>\n{new_messages[0]['value']}"
    new_samples.append({
        "id": x['id'],
        "conversations": new_messages,
        "wxh": x['wxh'],
        "device": "mobile"
    })

autogui625k = "/data0/jingran/workspace/UI_training_data/Ours-Pretrain/raw/funcpred/scale_exp/funcpred_refGnd_625k.json"

with open(autogui625k, 'r') as f:
    autogui625k_data = json.load(f)

merged = new_samples + autogui625k_data
random.shuffle(merged)
num_QAs = sum(len(x['conversations']) // 2 for x in merged)
save_to = f"/data0/jingran/workspace/UI_training_data/Ours-Pretrain/raw/funcpred/scale_exp/funcpred_refGnd_{num_QAs}.json"

with open(save_to.replace(".json", "_sample.json"), 'w') as f:
    json.dump(merged[:128], f, indent=2)


with open(save_to, 'w') as f:
    json.dump(merged, f, indent=2)