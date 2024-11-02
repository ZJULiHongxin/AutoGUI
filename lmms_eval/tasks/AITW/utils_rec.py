import re, random, ast, os
import logging
from PIL import Image
from datasets import Dataset
from collections import defaultdict
from pretrain.prompt_lib import web_loca_all_point_prompt, apply_vlm_template
from pretrain.process_utils import pred_2_point
from lmms_eval.tasks.AITW.action_matching import *

eval_logger = logging.getLogger("lmms-eval")

STEP_SR = "action_acc(step_sr)"
ACTIONTYPE_ACC = "action_type_acc"
CLICK_ACC = "click_acc"
TEXT_ACC = "text_acc"
SWIPE_ACC = "swipe_acc"
WRONG_FORMAT_RATIO = "wrong_format"
AITW_METRICS = [STEP_SR, ACTIONTYPE_ACC, CLICK_ACC, TEXT_ACC, SWIPE_ACC, WRONG_FORMAT_RATIO]



def aitw_doc_to_visual(doc):
    # Image is presented as is
    image_dir = doc.get('image_dir', '')
    if image_dir:
        image = Image.open(os.path.join(image_dir, doc["image"]))
    else:
        image = doc["image"]
    return [image.convert("RGB")]


def aitw_doc_to_text(doc, model_name='', model_specific_prompt_kwargs=None):
    pre_prompt = ""
    post_prompt = ""
    
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"].format(task=doc['step']["goal"], history=doc["history"])
    
    prompt = f"{pre_prompt}{post_prompt}"
    
    # if the we require a box-format output, the prompt should be modified accordingly
    return prompt


def parse_float_sequence_within(input_str):
    """
    Extract the first sequence of four floating-point numbers within square brackets from a string.

    Args:
    input_str (str): A string that may contain a sequence of four floats within square brackets.

    Returns:
    list: A list of four floats if the pattern is found, or a list of four zeros if the pattern is not found.
    """
    # Define the regex pattern to find the first instance of four floats within square brackets
    pattern = r'\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]'
    
    # Use re.search to find the first match of the pattern in the input string
    match = re.search(pattern, input_str)
    
    # If a match is found, convert the captured groups into a list of floats
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    
    # If the input does not contain the pattern, return the null float sequence
    return [0, 0, 0, 0]


# convert action to prediction format
def aitw_action2step(step_data, scale=1):
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':  # for click action, we calculate midpoint of touch and lift as the click point
            touch_point = step_data["touch"]
            lift_point = step_data["lift"]
            action_type_new = 4
            click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]
            
            if scale == 1:
                click_point = [f"{item:.2f}" for item in click_point]
            else:
                click_point = [f"{int(item*scale):d}" for item in click_point]
            click_point = "({},{})".format(click_point[0], click_point[1])
            action = "{{\"action_type\": {}, \"target\": {}}}".format(action_type_new, click_point)
        else:  # for scroll action, we assign an action_type_id for each scroll
            if step_data["action_type_text"] == 'scroll down':
                action_type_new = 0
            elif step_data["action_type_text"] == 'scroll up':
                action_type_new = 1
            elif step_data["action_type_text"] == 'scroll left':
                action_type_new = 8
            elif step_data["action_type_text"] == 'scroll right':
                action_type_new = 9
            action = "{{\"action_type\": {}}}".format(action_type_new)
    elif action_type == 3:
        typed_text = step_data["type_text"]
        action_type_new = action_type
        action = "{{\"action_type\": {}, \"typed_text\": \"{}\"}}".format(action_type_new, typed_text)
    else:
        action_type_new = action_type
        action = "{{\"action_type\": {}}}".format(action_type_new)

    return action

# aitw 585条测试轨迹
def aitw_process_result(doc, result, model_specific_process_kwargs=None):
    """
    Args:
        doc: a instance of the eval dataset
        results: [{"prompt": prompt, "response": response}]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    step = doc['step']
    action_ref = action_2_format(step, scale=scale)
    
    try:
        raw_action_pred = ast.literal_eval(result[0]["response"])
        ref_action_type = raw_action_pred.pop('action_type')
        ref_action_attr = raw_action_pred
        wrong_format = False
    except:
        wrong_format = True

    action_acc = click_acc = swipe_acc = text_acc = False

    if not wrong_format:
        model_name = model_specific_process_kwargs.get('model', '').lower()

        if model_name == 'seeclick':
            pred_action_type, pred_action_attr = pred_2_format_seeclick(raw_action_pred)
        elif model_name in ['autogui', 'uipro']:
            pred_action_type, pred_action_attr  = pred_2_format_autogui(raw_action_pred)

        scale = model_specific_process_kwargs.get("scale", 1) if model_specific_process_kwargs is not None else 1
        
        action_matching_result = check_actions_match(
            ref_action_type=ref_action_type,
            ref_action_attr=ref_action_attr,
            pred_action_type=pred_action_type,
            pred_action_attr=pred_action_attr,
            annotation_positions=step['annotations']
            )
        
        if action_matching_result[0] == 'click':
            action_acc = click_acc = action_matching_result[2]
        elif action_matching_result[0] == 'swipe':
            action_acc = swipe_acc = action_matching_result[2]
        elif action_matching_result[0] == 'input_text':
            action_acc = text_acc = action_matching_result[2]

    data_dict = {"prompt": result[0]["prompt"], "response": result[0]["response"], STEP_SR: action_acc, ACTIONTYPE_ACC: action_matching_result[1], CLICK_ACC: click_acc, SWIPE_ACC: swipe_acc, TEXT_ACC: text_acc, WRONG_FORMAT_RATIO: wrong_format}
    return {metric: data_dict for metric in AITW_METRICS}

def aggr_aitw_performance(results, metric):
    """
    Aggregate the results of the screenspot evaluation task using the specified metric.

    Args:
    - results (list of dict): List of result dictionaries.
    - metric (str): Metric to use for aggregation.

    Returns:
    - dict: Dictionary containing the aggregated results for the specified metric.
    """
    num_samples = len(results)
    
    result_str = []
    for k in AITW_METRICS:
        corr_num = sum([x[k] for x in results])
        result_str.append(f'{k}:\t{corr_num / num_samples:.4f}\t({corr_num} / {num_samples})\n')

    return '\n'.join(result_str)
