import os
import json, glob
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import pickle

import plotly.graph_objects as go
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm_llm_exp.utils import parse_action, test_inside_box
from pretrain.prompt_lib import apply_vlm_template

from autogui_model import AutoGUI, SLIME, SeeClick
from .utils import get_target_center_screenshot

MAX_W, MAX_H = 1280, 720
SCALE = 1

box_size_ranges = np.linspace(0.02, 0.8, 40)

DEBUG = False

"""
Load dataset
"""
splits = ['test_website', 'test_task', 'test_domain'][:1]
dataset_all = load_dataset("osunlp/Multimodal-Mind2Web")

# extract each traj
temp_split_info = os.path.join(os.path.dirname(__file__), "temp_split_info.pkl")

if not os.path.exists(temp_split_info):
    samples = {}
else:
    with open(temp_split_info, "rb") as f:
        samples = pickle.load(f)

for split in splits:
    dataset = dataset_all[split]
    for i, sample in tqdm(enumerate(dataset)):
        if DEBUG and i == 100: break
        sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
        if sample_id in samples: continue
        samples[sample_id] = sample

# with open(temp_split_info, "wb") as f:
#     pickle.dump(samples, f)

webfun_dir = "/data0/jingran/workspace/hongxin_li/WebFun/"

"""
Load subtask annotation results
"""
# subtask_anno_dir
subtask_anno_name = "mind2web_test_website_0721"
subtask_anno_result_dir = f"{webfun_dir}/exp_results/mind2web_test_website_0721_llama3-70b/results"

processed_idxs = [int(os.path.basename(x).split('_')[0]) for x in glob.glob(os.path.join(subtask_anno_result_dir, "*subtask.txt"))]

meta_info_file = os.path.join(webfun_dir, "test_samples", f"{subtask_anno_name}.json")
with open(meta_info_file, "r") as f:
    meta_info = json.load(f)

invalid_cnt = 0

"""
Load a VLM
"""
vlm = [
    # SLIME(pretrained="/home/jingran_su/.cache/huggingface/hub/SliME-Llama-3-8B-AutoGUI_3epochs/checkpoint-14619"),
    # AutoGUI(pretrained="/data0/jingran/workspace/hongxin_li/seeclick_exp/checkpoints/seeclick_scaling_funcpred625k_llava150k_cauldron197k_refGnd_resample_v2/final"),
    SeeClick("cckevinn/SeeClick")
][-1]

results = []

for sample in tqdm(meta_info):
    sample_id, sample_idx = f"{sample['task_id']}_{sample['step']}", sample["sample_id"]
    if sample_idx in processed_idxs:
        1+1
    if sample_id not in samples or sample_idx not in processed_idxs:
        invalid_cnt += 1
        continue

    pos_cand_box = sample['unnormalized_box'] # x1, y1, x2, y2

    # crop the image
    cropped_image, left, upper, right, lower = get_target_center_screenshot(samples[sample_id]["screenshot"], MAX_H, MAX_W, pos_cand_box)
    
    # Adjust the boxes of pos/neg candidates
    pos_cand_box = [pos_cand_box[0] - left, pos_cand_box[1] - upper, pos_cand_box[2] - left, pos_cand_box[3] - upper]
    
    img_path = os.path.join(os.path.dirname(__file__), f"temp.png")
    cropped_image.save(img_path)

    with open(os.path.join(subtask_anno_result_dir, f"{sample['sample_id']}_subtask.txt"), "r") as f:
        subtask = f.read().split("Subtask:")[-1].strip(' .')

    subtask = apply_vlm_template(task_instruction=subtask.replace("The subtask is to", "This element is used to"), model_name=vlm.model_name)
    
    gnd_result = vlm.generate(subtask, os.path.abspath(img_path))

    action_type, action_target = parse_action(gnd_result)
    
    pred_point = [action_target[0] / SCALE, action_target[1] / SCALE]
    pos_point = (pos_cand_box[0] + pos_cand_box[2]) / 2 / MAX_W, (pos_cand_box[1] + pos_cand_box[3]) / 2 / MAX_H
    inside_boxes = test_inside_box(pred_point, pos_point, box_size_ranges)
    
    results.append(inside_boxes)

# plot the center-acc vs. box size curve using Plotly
aggr_results = np.array(results).mean(axis=0) * 100

# plot a line curve
fig = go.Figure(data=go.Scatter(x=box_size_ranges, y=aggr_results, mode='lines+markers+text', text=[f'{x:.1f}' for x in aggr_results.tolist()],
                textposition='top center'))

# Update layout
fig.update_layout(
    title='Gnd acc vs. Threshold',
    xaxis_title='Threshold [0-1]',
    yaxis_title='%',
        xaxis=dict(
        tickmode='array',
        tickvals=box_size_ranges,
        ticktext=[f'{val:.2f}' for val in box_size_ranges],
        range=[0, 1]
    ),
)

# Show the plot
fig.show()

    
print(invalid_cnt)