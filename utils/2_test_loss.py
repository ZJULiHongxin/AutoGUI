import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json


#
data_files = {
    "autogui": {
        "data_files": ["/mnt/nvme0n1p1/hongxin_li/UI_training_data/raw/funcpred/20241127_diverse/funcgnd_textloc_widgetlisting_560k_dedup_1536samples.json"],
        "format": "qwen"}
}

# Load pre-trained model and tokenizer
model_name = "/mnt/vdb1/hongxin.li/uipro_ckpt/Qwen2-VL-7B_UIPro-Dedup16M/lora/merged/checkpoint-7952"  # You can choose any other model
# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Calculate loss for each sample
for dataset_name, info in data_files.items():
    data = []
    for data_file in info["data_files"]:
        with open(data_file, "r") as f:
            data.extend(json.load(f))
    
    losses = []
    with torch.no_grad():
        for i, x in tqdm(enumerate(data), total=len(data), desc=dataset_name):
            if info["format"] == 'qwen':
                img, query = x['messages'][0]['content'].split('</img>')
                img = img[16:]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
            # Get model output
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            
            # Extract loss
            loss = outputs.loss.item()
            losses.append(loss)

# Plot the loss histogram
plt.figure(figsize=(10, 6))
plt.hist(losses, bins=10, color='skyblue', edgecolor='black')
plt.title('Loss Histogram')
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
