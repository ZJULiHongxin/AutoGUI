import re, random
import logging
from datasets import Dataset
from collections import defaultdict
from pretrain.prompt_lib import web_loca_all_point_prompt, apply_vlm_template
from pretrain.process_utils import pred_2_point

eval_logger = logging.getLogger("lmms-eval")

REC_METRICS = ["Center_ACC"]



def func_gnd_doc_to_visual(doc):
    # Image is presented as is
    image = doc["image"].convert("RGB")
    return [image.convert("RGB")]


def func_gnd_doc_to_text(doc, model_name='', model_specific_prompt_kwargs=None):
    instruc = doc["func"]
    pre_prompt = ""
    post_prompt = ""

    # Use random prompt templates
    if model_specific_prompt_kwargs is None:
        prompt = apply_vlm_template(instruc, model_name)
    else:
        if model_specific_prompt_kwargs['format'] == 'random':
            prompt = random.choice(web_loca_all_point_prompt) + f" {instruc}"
        else: # Use model-specific prompt tempalte
            if "pre_prompt" in model_specific_prompt_kwargs:
                pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
            if "post_prompt" in model_specific_prompt_kwargs:
                post_prompt = model_specific_prompt_kwargs["post_prompt"].format(goal_info=instruc)
            
            prompt = f"{pre_prompt}{post_prompt}"
    
    # if the we require a box-format output, the prompt should be modified accordingly
    return prompt

# "Bounding box coordinates are specified in the format (top-left x, top-left y, bottom-right x, bottom-right y). All values are floating point numbers bounded between 0 and 1 with two decimal places of precision (e.g., 0.15). Please provide the bounding box coordinates of the region that corresponds to the command: " + doc["instruction"]

# f'In this UI screenshot, what are the bounding box coordinates of the element corresponding to the command "{doc["instruction"]}"? Output the normalized X and Y coordinates, ranging from 0.0 to 1.0. Note that the X-axis runs horizontally from left (0.0) to right (1.0), and the Y-axis runs vertically from top (0.0) to bottom (1.0). Your should carefully view the image before finally predicting the required bounding box coordinates in the format [X_min, Y_min, X_max, Y_max].' # 


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

def func_gnd_process_result(doc, result, model_specific_process_kwargs=None):
    """
    Args:
        doc: a instance of the eval dataset
        results: [{"prompt": prompt, "response": response}]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0]["response"] if len(result) > 0 else ""
    
    scale = model_specific_process_kwargs.get("scale", 1) if model_specific_process_kwargs is not None else 1
        
    try:
        pred = pred_2_point(pred, keep_box=False, scale=scale)
    except:
        pred = [0,0,0,0]

    w, h = list(map(int, doc['image_size'].split('x')))
    box = [doc['unnormalized_box'][0] / w, doc['unnormalized_box'][1] / h, doc['unnormalized_box'][2] / w, doc['unnormalized_box'][3] / h]

    data_dict = {"prompt": result[0]["prompt"], "response": result[0]["response"], "func": doc["func"], "pred": pred, 'bbox': box, 'elem_text': doc['elem_text'], 'elem_role': doc['elem_role'], 'image_size': doc['image_size'], 'device': doc['device']}
    return {metric: data_dict for metric in REC_METRICS}


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    
    if len(box1) != 4 or len(box2) != 4:
        return 0.0

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the Intersection over Union
    iou = intersection_area / union_area

    return iou


def compute_accuracy(box1, box2, threshold=0.5):
    """
    Compute the accuracy of two bounding boxes based on a specified threshold.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - threshold (float): Threshold for the IoU to consider the prediction correct.

    Returns:
    - float: Accuracy of the prediction based on the IoU threshold.
    """
    iou = compute_iou(box1, box2)
    return iou >= threshold


def compute_center_accuracy(box1, box2):
    """
    Compute if the center point of box 2 is within box 1.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - bool: True if the center point of box 2 is within box 1, False otherwise.
    """
    if isinstance(box1, str):
        box1 = list(map(int, box1.strip('[]()').split(',')))
    if isinstance(box2, str):
        box2 = list(map(int, box2.strip('[]()').split(',')))
    # Compute the center point of box 2
    if len(box2) == 2:
        center_x, center_y = box2
    elif len(box2) == 4:
        center_x = (box2[0] + box2[2]) / 2
        center_y = (box2[1] + box2[3]) / 2
    else:
        center_x, center_y = -1, -1

    # Check if the center point is within box 1
    return box1[0] <= center_x <= box1[2] and box1[1] <= center_y <= box1[3]


def func_gnd_aggregation_result(results, metric):
    """
    Aggregate the results of the screenspot evaluation task using the specified metric.

    Args:
    - results (list of dict): List of result dictionaries.
    - metric (str): Metric to use for aggregation.

    Returns:
    - dict: Dictionary containing the aggregated results for the specified metric.
    """
    scorers = {
        'Center_ACC': compute_center_accuracy
    }
    results_dict = defaultdict(list)
    results_dict[metric] = []

    for result in results:
        # Extract the ground truth and predicted bounding boxes
        gt = result['bbox']
        pred = result['pred']

        # Compute the specified metric between the ground truth and predicted bounding boxes
        score = scorers[metric](gt, pred)

        results_dict[metric].append(score)
        
        results_dict[f"{metric}-{result['device']}"].append(score)

    for key in results_dict:
        if len(results_dict[key]) == 0:
            results_dict[key] = 0
        else:
            results_dict[key] = sum(results_dict[key]) / len(results_dict[key])

        print(f"{key}: {results_dict[key]:0.4f}")
    return results_dict[metric]


def func_gnd_center_acc(results):
    return func_gnd_aggregation_result(results, "Center_ACC")
