# 该脚本用于初步尝试VLM预测元素功能，再用LLM校验，以验证这种规划方案的可行性
import os
import json, glob
from datasets import load_dataset, load_from_disk
import numpy as np
from tqdm import tqdm
import pickle, copy
from PIL import Image, ImageDraw
import plotly.graph_objects as go
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm_llm_exp.utils import parse_action, test_inside_box, get_target_center_screenshot, compute_target_overlap_ratio, draw_dashed_rectangle, format_input_multichoice, postprocess_action_llm, calculate_f1, postprocess_action
from pretrain.prompt_lib import apply_vlm_template, FUNC_CAP_PROMPT

from autogui_model import AutoGUI, SLIME, SeeClick

import sys
sys.path.append("/data0/jingran/workspace/hongxin_li")
from WebFun.utils.tools import LLM

MAX_W, MAX_H = 1280, 720
SCALE = 100
TH = 0.25
box_size_ranges = np.linspace(0.02, 0.8, 40)

SHIFT = 1131/1216

DEBUG = False

"""
Load dataset
"""
splits = ['test_website', 'test_task', 'test_domain'][:1]
dataset_all = load_dataset("osunlp/Multimodal-Mind2Web")

# processed_data = load_from_disk(os.path.join("/data0/jingran/workspace/hongxin_li/Mind2Web/Mind2Web_data", f"processed_{splits[0]}"))
# processed_samples = [f"{x['annotation_id']}_{x['action_uid']}" for x in processed_data]

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
        if DEBUG and len(samples) == 40: break
        sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
        if sample_id in samples: continue
        samples[sample_id] = sample

axtree_dir = "/data0/jingran/workspace/hongxin_li/globus/raw_dump/task"
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
Load VLMs
"""
vlm = [
    # SLIME(pretrained="/home/jingran_su/.cache/huggingface/hub/SliME-Llama-3-8B-AutoGUI_3epochs/checkpoint-14619"),
    AutoGUI(pretrained="/data0/jingran/workspace/hongxin_li/seeclick_exp/checkpoints/seeclick_scaling_funcpred625k_llava150k_cauldron197k_refGnd_resample_v2/final"),
    #SeeClick("cckevinn/SeeClick")
][-1]

"""
Load LLMs
"""
llm = LLM("llama3-70b-fireworks")

with open(os.path.join(os.path.dirname(__file__), "prompt.json")) as f:
    PROMPT = json.load(f)

for sample in tqdm(meta_info):
    sample_id, sample_idx = f"{sample['task_id']}_{sample['step']}", sample["sample_id"]
    if sample_idx in processed_idxs:
        1+1
    if sample_id not in samples or sample_idx not in processed_idxs:
        invalid_cnt += 1
        continue

    pos_cand_box = list(map(round, sample['unnormalized_box'])) # x1, y1, x2, y2
    pos_id = json.loads(json.loads(samples[sample_id]['pos_candidates'][0])['attributes'])['backend_node_id']
    # crop the image
    cropped_image, left, upper, right, lower = get_target_center_screenshot(samples[sample_id]["screenshot"], MAX_H, MAX_W, pos_cand_box)

    img_path = os.path.join(os.path.dirname(__file__), f"temp.png")
    cropped_image.save(img_path)

    pos_cand_box = [pos_cand_box[0] - left, pos_cand_box[1] - upper, pos_cand_box[2] - left, pos_cand_box[3] - upper]

    # Load subtask
    with open(os.path.join(subtask_anno_result_dir, f"{sample['sample_id']}_subtask.txt"), "r") as f:
        subtask = f.read().split("Subtask:")[-1].strip(' .')

    subtask = apply_vlm_template(task_instruction=subtask.replace("The subtask is to", "This element is used to"), model_name=vlm.model_name)

    # grounding
    gnd_result = vlm.generate(subtask, os.path.abspath(img_path))

    action_type, action_target = parse_action(gnd_result)
    
    """
    Find elements near the gnd point
    """
    pred_point = [action_target[0] / SCALE, action_target[1] / SCALE]

    pred_region = list(map(round, [
        max(0, (pred_point[0] - TH) * MAX_W),
        max(0, (pred_point[1] - TH) * MAX_H),
        min(MAX_W, (pred_point[0] + TH) * MAX_W),
        min(MAX_H, (pred_point[1] + TH) * MAX_H),
    ]))

    # 方法一：read the raw axtree （axtree的bbox有问题）
    with open(os.path.join(axtree_dir, sample['task_id'], "axtree", f"{sample['step']}_before_clean.json"), "r") as f:
        axtree = json.load(f)
        original_axtree = axtree['content']
        nodes = axtree["obs_nodes_info"]

    # node_inside = []
    # sub_axtree = [nodes["1"]["text"]]
    # for node in nodes.values():
    #     node_union_bound = node["union_bound"]
    #     if "'Deals'" in node['text']:
    #         1+1
    #     x1, y1 = (node_union_bound[0] - left) * SHIFT, node_union_bound[1] - upper
    #     x2, y2 = x1 + node_union_bound[2], y1 + node_union_bound[3]

    #     node_box = [round(x1), round(y1), round(x2), round(y2)]

    #     if compute_target_overlap_ratio(node_box, pred_region) > 0.8:
    #         node_inside.append([node, node_box])
    #         sub_axtree.append(node['text'])
    
    # sub_axtree='\n'.join(sub_axtree)


    # 方法二：use multi-modal Mind2Web
    # 问题：有些neg是子树结点，覆盖了很多子节点，需要剔除掉
    candidate_ids, node_inside, node_boxes_inside = [], [], []
    for neg in samples[sample_id]["neg_candidates"]:
        attrs = json.loads(json.loads(neg)["attributes"])
        neg_box = attrs["bounding_box_rect"]
        x1, y1, W, H = list(map(lambda x: float(x), neg_box.split(',')))
        
        x1, y1 = x1 - left, y1 - upper
        x2, y2 = x1+W, y1+H

        node_box = [round(x1), round(y1), round(x2), round(y2)]
        ratio = compute_target_overlap_ratio(node_box, pred_region)

        if ratio > 0.8:
            candidate_ids.append(attrs['backend_node_id'])
            node_inside.append(neg)
            node_boxes_inside.append(node_box)

    samples[sample_id]['previous_actions'] = samples[sample_id]['action_reprs'][:int(samples[sample_id]['target_action_index'])]
    seq_context, seq_in, target_out, choices = format_input_multichoice(
        samples[sample_id], candidate_ids, pos_id, keep_html_brackets=True
    )
    _, target_action = postprocess_action(target_out)

    # Adjust the boxes of pos/neg candidates
    if True:
        test_img = cropped_image.copy()
        draw = ImageDraw.Draw(test_img)
        from PIL import Image, ImageDraw

        # Draw a rectangle for the GT box
        rectangle_outline_color = "red"
        draw.rectangle(pos_cand_box, # [left, top, right, bottom]
                       outline=rectangle_outline_color)

        # Draw the pred_region
        draw_dashed_rectangle(draw, pred_region, outline_color="red")

        # Draw a circle
        circle_center = [round(pred_point[0] * MAX_W), round(pred_point[1] * MAX_H)]  # (x, y)
        circle_radius = 6
        circle_outline_color = "green"
        draw.ellipse(
            (circle_center[0] - circle_radius, circle_center[1] - circle_radius, 
            circle_center[0] + circle_radius, circle_center[1] + circle_radius),
            outline=circle_outline_color
        )

        # Draw recs for all nodes
        for node_box in node_boxes_inside:
            draw_dashed_rectangle(draw, node_box, outline_color="purple") 

        # Save or show the image
        test_img.save("output_image.png")  # This will save the image to a file
    
    1+1

    for idx, choice in enumerate(choices):
        # convert to ascii A, B, C, D, ...
        seq_in += f"{chr(66 + idx)}. {choice[1]}\n"

    prompt = copy.deepcopy(PROMPT)
    prompt[-1]['content'] = f"'''\n{seq_context}\n'''\n\n{seq_in}"

    resp = llm.query_LLM(prompt)
    answer = resp[0]

    print("Verif...", end='')
    retry = 0
    while True:
        # LLM verif
        selected, pred_action = postprocess_action_llm(answer)
        if selected[0] != "A":
            # convert B, C, D to 0, 1, 2
            selected = ord(selected[0]) - ord("B")

        # VLM cap
        selected_node_id = choices[selected][0]
        selected_node_box = node_boxes_inside[candidate_ids.index(selected_node_id)]
        selected_center = f"({int((selected_node_box[0] + selected_node_box[2]) / 2 / MAX_W * SCALE)},{int((selected_node_box[1] + selected_node_box[3]) / 2 / MAX_W * SCALE)})"
        vlm_cap_prompt = FUNC_CAP_PROMPT.format(selected_center)
        cap = vlm.generate(vlm_cap_prompt, os.path.abspath(img_path))
        
        retry += 1
        print(f" {retry} ", end='')
        prompt.extend(
            [
                {"role": 'assistant', "content": answer},
                {"role": 'user', "content": f'The funcitonality of the element you just selected is: {cap}. Please verify you selection step-by-step to ensure the selected element can help to advance towards task completion. If so, answer "Yes"; otherwise,predict another action. Please follow this output format: Reason: ... Response: ...'},
            ]
        )

        resp = llm.query_LLM(prompt)[0]

        extracted_verif_answer = resp[resp.rfind(':')+1:].strip()

        if extracted_verif_answer.lower() == 'yes':
            print(); break
        else:
            answer = extracted_verif_answer
    
    pred_element, pred_action = postprocess_action(answer, choices)
            
    action_f1 = calculate_f1(pred_action, target_action)

    elem_acc = (pos_cand_box[0] <= (selected_node_box[0] + selected_node_box[2]) / 2 <= pos_cand_box[2]) and (pos_cand_box[1] <= (selected_node_box[1] + selected_node_box[3]) / 2 <= pos_cand_box[3])
