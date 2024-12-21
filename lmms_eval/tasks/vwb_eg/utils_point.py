import re, random
import json
import logging
from PIL import Image
import pandas as pd
from enum import Enum
from pretrain.prompt_lib import web_loca_all_point_prompt, apply_vlm_template
from pretrain.process_utils import pred_2_point
import os 

eval_logger = logging.getLogger("lmms-eval")

class UIObjectType(Enum):
  """Types of the different UI objects."""
  UNKNOWN = 0
  BUTTON = 1
  CHECKBOX = 2
  CHECKEDTEXTVIEW = 3
  EDITTEXT = 4
  IMAGEBUTTON = 5
  IMAGEVIEW = 6
  RADIOBUTTON = 7
  SLIDINGDRAWER = 8
  SPINNER = 9
  SWITCH = 10
  TABWIDGET = 11
  TEXTVIEW = 12
  TOGGLEBUTTON = 13
  VIDEOVIEW = 14

simplified_action_space_dict = {'type': 3, 'click': 4, 'swipe': 0} # id borrowed from seeclick
simplified_action_space = repr([k for k in simplified_action_space_dict.keys()])

ui_object_type_dict = {member.value: member.name for member in UIObjectType}

"""
NOTES:
    1. Task Formulation: 
        - screenshot + task instruct + step_instruct[Option] + prev_action + action_space --> bbox + action 
        - SoM screenshot + task instruct + step_instruct[Option] + prev_action + gt_action --> bbox
        - prev_action number is set to PREV_ACTION_LENGTH
    2. Positive element Extraction:
    3. Action Space to Text Space:
        - The above simplification is happend in action2str().
    4. General Response Format: We unify the output format as a json, includes "action type" and "bbox"
""".format(action_space = simplified_action_space)

def vwb_doc_to_visual(doc):
    # load from bytes
    img = doc['image'].convert("RGB")
    return [img]

def vwb_doc_to_target(doc):
    return [doc['box']]

def vwb_doc_to_target_wa(doc):
    return doc['pos_bbox']

def vwb_doc_to_text(doc, model_name='', model_specific_prompt_kwargs=None):
    '''
    Args:
        doc: dict
            A dictionary of data instance.
        model_specific_prompt_kwargs: dict
            A dictionary containing the following
                pre_prompt: str
                    A string to be prepended to the prompt
                post_prompt: str
                    A string to be appended to the prompt
    Returns:    
        text: str
            prompt
    '''

    instruc = doc["elem_desc"] if doc['task'] == 'elem-gnd' else doc["detailed_elem_desc"]
    pre_prompt = ""
    post_prompt = ""

    if model_specific_prompt_kwargs is None:
        prompt = apply_vlm_template(instruc, model_name)
    else:
        # Use random prompt templates
        if model_specific_prompt_kwargs['format'] == 'random':
            prompt = random.choice(web_loca_all_point_prompt) + f' {instruc}'
        else: # Use model-specific prompt tempalte
            if "pre_prompt" in model_specific_prompt_kwargs:
                pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
            if "post_prompt" in model_specific_prompt_kwargs:
                # if model_name == 'uipro' and doc['task'] == 'elem-gnd':
                #     model_specific_prompt_kwargs["post_prompt"] = "Locate the text \"{goal_info}\". (with bbox)"

                post_prompt = model_specific_prompt_kwargs["post_prompt"].format(goal_info=instruc)
            
            prompt = f"{pre_prompt}{post_prompt}"
    
    # if the we require a box-format output, the prompt should be modified accordingly
    return prompt



def vwb_point_process_results(doc, result, model_specific_process_kwargs=None):
    '''
    Args:
        doc: dict
            A list of data instance.
        result: list
            A list of model outputs.
    '''
    
    # process the preds and computes squad f1 score before passing to metrics
    pred = result[0]['response'] if len(result) > 0 else ""

    scale = model_specific_process_kwargs.get("scale", 1) if model_specific_process_kwargs is not None else 1

    try:
        pred = pred_2_point(pred, keep_box=False, scale=scale)
    except:
        pred = [-1,-1,-1,-1]

    # normalize the bbox
    bbox = doc['box']
            
    point = None
    correct = 0
    if len(pred) == 2:
        center_x, center_y = pred
    elif len(pred) == 4:
        center_x = (pred[0] + pred[2]) / 2
        center_y = (pred[1] + pred[3]) / 2
    else:
        center_x, center_y = -1, -1

    if (bbox[0] <= center_x <= bbox[2]) and (bbox[1] <= center_y <= bbox[3]):
        correct = 1
    else:
        correct = 0
    
    data_dict = {"prompt": result[0]["prompt"], "response": result[0]["response"],"task": doc["task"], "pred": pred, "acc": correct, 'bbox': doc['box'], 'unnormalized_box': doc['unnormalized_box']}

    return {
        f"vwb_elem-gnd_result":  data_dict,
    }

def vwb_gnd_result(results):
    # STEP2 calculate grounding score, by grouping trace_id
    # df = pd.DataFrame(results)
    # grouped = df.groupby('trace_id')['acc']
    # results = grouped.agg(mean_acc='sum')
    # partial acc
    acc = []
    for result in results:
        acc.append(result['acc'])
        # if result['task'] == 'elem-gnd': eg_acc.append(result['acc'])
        # if result['task'] == 'action-gnd': ag_acc.append(result['acc'])
    score = sum(acc)/len(acc)
    # eg_acc, ag_acc = sum(eg_acc) / len(eg_acc) if len(eg_acc) else 0, sum(ag_acc) / len(ag_acc) if len(ag_acc) else 0
    print(f'VWB overall acc (0.0-1.0): {score} = {sum(acc)} / {len(acc)}')
    ## print(f'VWB overall acc (0.0-1.0): {score} | EG: {eg_acc} | AG: {ag_acc}')
    return score

def vwb_complete_aggregation_result(results):
    # STEP2 calculate grounding score, by grouping trace_id
    df = pd.DataFrame(results)
    grouped = df.groupby('trace_id')['acc']
    results = grouped.agg(success_rate=lambda x: (x == 1).all().astype(int))
    # trajectory acc
    complete_score = results['success_rate'].mean()
    return complete_score

def vwb_partial_aggregation_result(results):
    # STEP2 calculate grounding score, by grouping trace_id
    df = pd.DataFrame(results)
    grouped = df.groupby('trace_id')['acc']
    results = grouped.agg(mean_acc='mean')
    # partial acc
    partial_score = results['mean_acc'].mean()
    print('partial_score', partial_score)
    return partial_score

def vwb_nogroup_aggregation_result(results):
    # STEP2 calculate grounding score, by grouping trace_id
    df = pd.DataFrame(results)
    partial_score = df['acc'].mean()

    return partial_score

# is instruction English
def is_english_simple(text):
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

# bbox -> point (str)
def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str)
def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str