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
ENTER_ACC = "enter_acc"
HOME_ACC = "home_acc"
BACK_ACC = "back_acc"
COMPLETE_ACC = "complete_acc"
INFEASIBLE_ACC = "infeasible_acc"
WRONG_FORMAT = "wrong_format"
AITW_METRICS = [CLICK_ACC, TEXT_ACC, SWIPE_ACC, ENTER_ACC, HOME_ACC, BACK_ACC, COMPLETE_ACC, INFEASIBLE_ACC]
AITW_METRIC = "AITW_METRIC"


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
    
    scale = model_specific_process_kwargs.get("scale", 1) if model_specific_process_kwargs is not None else 1
    ref_action_type, ref_action_attr = action_2_format(step)
    
    try:
        response = result[0]["response"]
        response = response[response.rfind('{"'):]
        raw_action_pred = ast.literal_eval(response)
        wrong_format = False
    except:
        wrong_format = True
    
    action_matching_result = [ref_action_type, False, False]
    action_acc = click_acc = swipe_acc = text_acc = home_acc = back_acc = enter_acc = complete_acc = infeasible_acc = False
    click_num = swipe_num = input_text_num = home_num = back_num = enter_num = complete_num = infeasible_num = 0

    ann_positions = step.pop("annot_position")
    if not wrong_format:
        model_name = model_specific_process_kwargs.get('model', '').lower()

        if model_name == 'seeclick':
            pred_action_type, pred_action_attr = pred_2_format_seeclick(raw_action_pred)
        elif model_name in ['autogui', 'uipro']:
            pred_action_type, pred_action_attr  = pred_2_format_autogui(raw_action_pred, scale=scale)

        action_matching_result = check_actions_match(
            ref_action_type=ref_action_type,
            ref_action_attr=ref_action_attr,
            pred_action_type=pred_action_type,
            pred_action_attr=pred_action_attr,
            annotation_positions=np.array(
                    [ann_positions[i:i + 4] for i in range(0, len(ann_positions), 4)])
            )
        
        action_acc = action_matching_result[2]
        if action_matching_result[0] == 'click':
            click_num, click_acc = click_num + 1, action_acc
        elif action_matching_result[0] == 'swipe':
            swipe_num, swipe_acc = swipe_num + 1, action_acc
        elif action_matching_result[0] == 'input_text':
            input_text_num, text_acc = input_text_num + 1, action_acc
        elif action_matching_result[0] == 'navigate_home':
            home_num, home_acc = home_num + 1, action_acc
        elif action_matching_result[0] == 'navigate_back':
            back_num, back_acc = back_num + 1, action_acc
        elif action_matching_result[0] == 'status':
            if ref_action_attr['goal_status'] == 'successful':
                complete_num, complete_acc = complete_num + 1, action_acc
            else:
                infeasible_num, infeasible_acc = infeasible_num + 1, action_acc
        elif action_matching_result[0] == 'enter':
            enter_num, enter_acc = enter_num + 1, action_acc
    else: pred_action_type, pred_action_attr = None, None
        
    data_dict = {"step_info": step, "prompt": result[0]["prompt"], "response": result[0]["response"], 'scenario': doc['image'].split('/')[0], 'pred_action': {'action_type':pred_action_type, 'attr': pred_action_attr}, 'gt_action': {'action_type': ref_action_type, 'attr': ref_action_attr}, STEP_SR: action_acc, ACTIONTYPE_ACC: action_matching_result[1], "action_match_details": {CLICK_ACC: click_acc, SWIPE_ACC: swipe_acc, TEXT_ACC: text_acc, ENTER_ACC: enter_acc, HOME_ACC: home_acc, BACK_ACC: back_acc, COMPLETE_ACC: complete_acc, INFEASIBLE_ACC: infeasible_acc, WRONG_FORMAT: wrong_format}, "action_counts": {CLICK_ACC: click_num, SWIPE_ACC: swipe_num, TEXT_ACC: input_text_num, ENTER_ACC: enter_num, HOME_ACC: home_num, BACK_ACC: back_num, COMPLETE_ACC: complete_num, INFEASIBLE_ACC: infeasible_num}}
    return {AITW_METRIC: data_dict}

def aggr_aitw_performance(results):
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
    
    # global metric
    scenario_metrics = {}
    for scenario in ['average', 'general', 'install', 'googleapps', 'single', 'webshopping']:
        scenario_metrics[f'{scenario}-stepsr'] = [0,0]
        scenario_metrics[f'{scenario}-action_type'] = [0,0]
    
    for result in results:
        is_action_corr, is_actiontype_corr = result[STEP_SR], result[ACTIONTYPE_ACC]
        scenario_metrics['average-stepsr'][0] += is_action_corr; scenario_metrics['average-stepsr'][1] += 1
        scenario_metrics['average-action_type'][0] += is_actiontype_corr; scenario_metrics['average-action_type'][1] += 1
        
        scenario = result['scenario']
        scenario_metrics[f'{scenario}-stepsr'][0] += is_action_corr; scenario_metrics[f'{scenario}-stepsr'][1] += 1
        scenario_metrics[f'{scenario}-action_type'][0] += is_actiontype_corr; scenario_metrics[f'{scenario}-action_type'][1] += 1

    for k, metric in scenario_metrics.items():
        ratio = metric[0] / metric[1] if metric[1] > 0 else 0.0
        result_str.append(f'{k}: {ratio:.4f} ({metric[0]} / {metric[1]})')
    
    # detailed metrics
    for k in AITW_METRICS:
        corr_num = sum([x['action_match_details'][k] for x in results])
        num_actions = sum([x['action_counts'][k] for x in results])
        
        ratio = corr_num / num_actions if num_actions > 0 else 0.0
        result_str.append(f'{k}: {ratio:.4f} ({corr_num} / {num_actions})')

    return '\n'.join(result_str)
