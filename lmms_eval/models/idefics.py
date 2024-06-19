import requests
from PIL import Image
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

url_1 = "/data0/jingfan/code/ui_detection/datasets/images/rico/57800.jpg"
#url_2 = "http://images.cocodataset.org/val2017/27545.jpg"

image_1 = Image.open(url_1)
#image_2 = Image.open(requests.get(url_2, stream=True).raw)
images = [image_1, 
          # image_2
          ]

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "Detect the icon targeted by the task 'click to move to trash'?"},
        {"type": "image"},
        #@{"type": "image"},
    ],
}]

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b")
model.to(device)

# at inference time, one needs to pass `add_generation_prompt=True` in order to make sure the model completes the prompt
text = processor.apply_chat_template(messages, add_generation_prompt=True)
print(text)
# 'User: What’s the difference between these two images?<image><image><end_of_utterance>\nAssistant:'

inputs = processor(images=images, text=text, return_tensors="pt").to(device)

generated_text = model.generate(**inputs, max_new_tokens=500)
generated_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
print("Generated text:", generated_text)