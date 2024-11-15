import re, random, ast, os
import logging
from PIL import Image
from datasets import Dataset
from collections import defaultdict
from pretrain.prompt_lib import web_loca_all_point_prompt, apply_vlm_template
from pretrain.process_utils import pred_2_point
from lmms_eval.tasks.AITW.action_matching import *

eval_logger = logging.getLogger("lmms-eval")

Op_match = "Op_match"
Ele_match = "Ele_match"
Op_F1 = "Op_F1"
Step_SR = "Step_SR"
Macro_Ele_match = "Macro_Ele_match"
Macro_Step_SR = "Macro_Step_SR"
Macro_Op_F1 = "Op_F1"
CLICK_ACC = "click_acc"
TEXT_ACC = "text_acc"
SELECT_ACC = "select_acc"
WRONG_FORMAT = "wrong_format"

MIND2WEB_METRICS = [Op_match, Ele_match, Op_F1, Step_SR, Macro_Ele_match, Macro_Step_SR, Macro_Op_F1, CLICK_ACC, TEXT_ACC, SELECT_ACC]
MIND2WEB_METRIC = "MIND2WEB_METRIC"


def mind2web_doc_to_visual(doc):
    # Image is presented as is
    image_dir = doc.get('image_dir', '')
    if image_dir:
        image = Image.open(os.path.join(image_dir, doc["image"]))
    else:
        image = doc["image"]
    return [image.convert("RGB")]


def mind2web_doc_to_text(doc, model_name='', model_specific_prompt_kwargs=None):
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
def mind2web_action2step(step_data, scale=1, return_bbox=False):
    action_type = step_data["operation"]["original_op"]
    
    norm_x1, norm_y1, norm_x2, norm_y2 = step_data["normalized_bbox"]
    
    click_point = [(norm_x1 + norm_x2) / 2, (norm_y1 + norm_y2) / 2]

    if scale == 1:
        click_point = [round(item, 3) for item in click_point]
        click_point = [f"{item:.2f}" for item in click_point]
    else:
        click_point = [f"{int(scale * item):d}" for item in click_point]

    click_point = "({},{})".format(click_point[0], click_point[1])

    if return_bbox:
        bbox = [round(item, 3) for item in step_data["normalized_bbox"]]

    if action_type in ['CLICK', 'HOVER', 'ENTER']:
        action_step = "{{\"action_type\": {}, \"target\": {}}}".format(4, click_point)
    elif action_type == 'SELECT':
        select_value = step_data["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"target\": {}, \"value\": \"{}\"}}".format(2, click_point,
                                                                                               select_value)
    elif action_type == 'TYPE':
        typed_text = step_data["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"target\": {}, \"value\": \"{}\"}}".format(3, click_point,
                                                                                               typed_text)

    if return_bbox:
        return action_step, bbox
    else:
        return action_step

# calculate action f1 following mind2web
def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# aitw 585条测试轨迹
def mind2web_process_result(doc, result, model_specific_process_kwargs=None):
    """
    Args:
        doc: a instance of the eval dataset
        results: [{"prompt": prompt, "response": response}]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    step = doc['step']
    
    scale = model_specific_process_kwargs.get("scale", 1) if model_specific_process_kwargs is not None else 1

    action_ref, bbox_ref = mind2web_action2step(step)
    action_ref = ast.literal_eval(action_ref)

    try:
        response = result[0]["response"]
        response = response[response.rfind('{"'):]
        action_pred = ast.literal_eval(response)
        wrong_format = False
    except:
        wrong_format = True

    action_match = elem_match = step_acc = click_acc = inputtext_acc = select_acc = False
    Op_F1 = 0.0

    action_type_ref = action_ref.pop("action_type")
    action_type_pred = action_pred.pop("action_type")
    if not wrong_format:
        if action_type_pred == action_type_ref:
            action_match = True
        
        click_point = action_pred["target"]

        if (bbox_ref[0] <= click_point[0] / scale <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] / scale <= bbox_ref[3]):
            elem_match = True

        pred_str = str(action_type_pred)
        if action_type_pred in [3, "TYPE"] or action_type_pred in [2, "SELECT"]:
            pred_str += ' '
            pred_str += action_pred["value"].lower()
        ref_str = str(action_type_ref)
        if action_type_ref in [3, "TYPE"] or action_type_ref in [2, "SELECT"]:
            ref_str += ' '
            ref_str += action_ref["value"].lower()

        Op_F1 = calculate_f1(pred_str, ref_str)

        step_acc = Op_F1 == 1.0 and elem_match

        if step_acc:
            if action_type_ref == 'CLICK': click_acc = True
            elif action_type_ref == 'SELECT': select_acc = True
            elif action_type_ref == 'TYPE': inputtext_acc = True

    data_dict = {"step_info": step, "prompt": result[0]["prompt"], "response": result[0]["response"], 'scenario': doc['image'].split('/')[0], 'pred_action': {'action_type':action_type_pred, 'attr': action_pred}, 'gt_action': {'action_type': action_type_ref, 'attr': action_ref}, Step_SR: step_acc, Op_match: action_match, "action_match_details": {CLICK_ACC: click_acc, SELECT_ACC: select_acc, TEXT_ACC: inputtext_acc, , WRONG_FORMAT: wrong_format}, "action_counts": {CLICK_ACC: click_num, SWIPE_ACC: swipe_num, TEXT_ACC: input_text_num, ENTER_ACC: enter_num, HOME_ACC: home_num, BACK_ACC: back_num, COMPLETE_ACC: complete_num, INFEASIBLE_ACC: infeasible_num}}
    return {AITW_METRIC: data_dict}

def aggr_mind2web_performance(results):
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
