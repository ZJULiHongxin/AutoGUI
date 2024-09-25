# 将安卓模拟器样本转成Qwen2-VL格式

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
            
            new_messages.append({"role": "user", "content": this_message})
        else:
            new_messages.append({"role": "assistant", "content": message["value"].replace("<image>","").strip()})
    
    image_path = os.path.join("/data0/jingran/workspace/UI_training_data/Ours-Pretrain/scaling_exp", x['image'])
    assert os.path.exists(image_path), image_path

    new_samples.append({
        "id": x['id'],
        "images": [image_path],
        "messages": new_messages,
        "wxh": x['wxh']
    })

autogui625k = "/data0/jingran/workspace/UI_training_data/Ours-Pretrain/raw/funcpred/scale_exp/funcpred_refGnd_625k.json"

with open(autogui625k, 'r') as f:
    data = json.load(f)

for x in tqdm(data, total=len(data)):
    new_messages = x['conversations']
    image_path, first_message = x['conversations'][0]['value'].split("</img>")
    image_path = image_path[len("Picture 1: <img>"):].strip()

    new_messages[0]['value'] = first_message.strip()
    
    img = Image.open(image_path)
    
    for message in new_messages:
        if message['from'] == 'user':
            message['role'] = message.pop('from')
            message['content'] = message.pop('value')
        elif message['from'] == 'assistant':
            message['role'] = message.pop('from')
            message['content'] = message.pop('value')

    new_samples.append({
        "id": x['id'],
        "images": [image_path],
        "messages": new_messages,
        "wxh": f"{img.width}x{img.height}"
    })

# add <image> indicator
for x in new_samples:
    x['messages'][0]['content'] = '<image>' + x['messages'][0]['content']

num_QAs = sum(len(x['messages']) // 2 for x in new_samples)
save_to = f"/data0/jingran/workspace/UI_training_data/Ours-Pretrain/raw/funcpred/scale_exp/funcpred_refGnd_{num_QAs}.json"

random.shuffle(new_samples)

with open(save_to.replace(".json", "_sample.json"), 'w') as f:
    json.dump(new_samples[:128], f, indent=2)


with open(save_to, 'w') as f:
    json.dump(new_samples, f, indent=2)