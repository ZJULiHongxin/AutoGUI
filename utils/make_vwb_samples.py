import os
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
vwb_eg_raw = load_dataset("sujr/vwb_eg_extend", split='train')
vwb_ag_raw = load_dataset("sujr/vwb_ag_extend", split='train')


new_samples = []

for sample in tqdm(vwb_eg_raw):
    img = Image.open(os.path.join("/data0/jingran/workspace/hongxin_li/WebEval", sample['raw_image'].replace("img_box", "img")))
    
    W, H = img.size
    unnorm_box = sample['bbox']
    norm_box = [unnorm_box[0] / W, unnorm_box[1] / H, unnorm_box[2] / W, unnorm_box[3] / H]
    new_samples.append({'image': img, 'task': 'elem-gnd', 'detailed_elem_desc': sample['elem_desc'], 'elem_desc': sample['old_elem_desc'], 'unnormalized_box': unnorm_box, 'box': norm_box, 'image_size': f'{W}x{H}'})


for sample in tqdm(vwb_ag_raw):
    img = Image.open(os.path.join("/data0/jingran/workspace/hongxin_li/WebEval", sample['raw_image'].replace("img_box", "img")))
    
    W, H = img.size
    unnorm_box = sample['bbox']
    norm_box = [unnorm_box[0] / W, unnorm_box[1] / H, unnorm_box[2] / W, unnorm_box[3] / H]
    new_samples.append({'image': img, 'task': 'action-gnd', 'detailed_elem_desc': sample['elem_desc'], 'elem_desc': sample['old_elem_desc'], 'unnormalized_box': unnorm_box, 'box': norm_box, 'image_size': f'{W}x{H}'})


test_dataset = Dataset.from_list(new_samples)

# check
# for x in test_dataset:
#     img = x['image']
#     draw = ImageDraw.Draw(img)
#     x1,y1,x2,y2 = x['unnormalized_box']
#     draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=5)
#     img.save("test.png")
#     1+1

dataset_dict = DatasetDict({
    "test": test_dataset,
    #"test": test_dataset,
})

# Save the dataset locally before pushing (optional)

# Push to Hugging Face
# dataset_dict.push_to_hub("my_username/my_dataset")

# Make a HF dataset
dataset_dict.push_to_hub("HongxinLi/VWB-EVAL", private=False, token='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')