import re , random
import logging
from datasets import Dataset
from pretrain.prompt_lib import web_loca_all_point_prompt, apply_vlm_template
from pretrain.process_utils import pred_2_point

eval_logger = logging.getLogger("lmms-eval")

REC_METRICS = ["Center_ACC"]



def screenspot_rec_doc_to_visual(doc):
    # Image is presented as is
    image = doc["image"].convert("RGB")
    return [image.convert("RGB")]


def screenspot_rec_doc_to_text(doc, model_name='', model_specific_prompt_kwargs=None):
    instruc = doc["instruction"]
    pre_prompt = ""
    post_prompt = ""

    if model_specific_prompt_kwargs is None:
        prompt = apply_vlm_template(instruc, model_name)
    else:
        # Use random prompt templates
        if model_specific_prompt_kwargs['format'] == 'random':
            prompt = random.choice(web_loca_all_point_prompt) + f" This element is used for \"{instruc}\""
        else: # Use model-specific prompt tempalte
            if "pre_prompt" in model_specific_prompt_kwargs:
                pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
            if "post_prompt" in model_specific_prompt_kwargs:
                post_prompt = model_specific_prompt_kwargs["post_prompt"].format(goal_info=instruc)
            
            prompt = f"{pre_prompt}{post_prompt}"
    
    # if the we require a box-format output, the prompt should be modified accordingly
    return prompt

# f'What are the bounding box coordinates of the element corresponding to the command "{doc["instruction"]}" in this UI screenshot?(with grounding)'
# "Bounding box coordinates are specified in the format (top-left x, top-left y, bottom-right x, bottom-right y). All values are floating point numbers bounded between 0 and 1 with two decimal places of precision (e.g., 0.15). Please provide the bounding box coordinates of the region that corresponds to the command: " + doc["instruction"]

# f'In this UI screenshot, what are the bounding box coordinates of the element corresponding to the command "{doc["instruction"]}"? Output the normalized X and Y coordinates, ranging from 0.0 to 1.0. Note that the X-axis runs horizontally from left (0.0) to right (1.0), and the Y-axis runs vertically from top (0.0) to bottom (1.0). Your should carefully view the image before finally predicting the required bounding box coordinates in the format [X_min, Y_min, X_max, Y_max].' # 

def screenspot_rec_process_result(doc, result, model_specific_process_kwargs=None):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0]['response'] if len(result) > 0 else ""

    scale = model_specific_process_kwargs.get("scale", 1) if model_specific_process_kwargs is not None else 1

    try:
        pred = pred_2_point(pred, keep_box=False, scale=scale)
    except:
        pred = [0,0,0,0]
    ann_id = doc["file_name"]
    data_dict = {"prompt": result[0]["prompt"], "response": result[0]["response"],"instruction": doc["instruction"], "pred": pred, "ann_id": ann_id, 'bbox': doc['bbox'], 'data_type': doc['data_type'], 'data_source': doc['data_source']}
    return {f"screenspot_{metric}": data_dict for metric in REC_METRICS}


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


def screenspot_rec_aggregation_result(results, metric):
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
    results_dict = {
        metric: [], 
        metric + '-mobile_text': [], 
        metric + '-mobile_icon': [],
        metric + '-web_text': [], 
        metric + '-web_icon': [],
        metric + '-desktop_text': [], 
        metric + '-desktop_icon': [],
    }
    for result in results:
        # Extract the ground truth and predicted bounding boxes
        gt = result['bbox']
        pred = result['pred']

        # Compute the specified metric between the ground truth and predicted bounding boxes
        score = scorers[metric](gt, pred)

        results_dict[metric].append(score)
        if result['data_type'] == 'text':
            if 'ios' in result['data_source'] or 'android' in result['data_source']:
                results_dict[metric + '-mobile_text'].append(score)
            elif 'macos' in result['data_source'] or 'windows' in result['data_source']:
                results_dict[metric + '-desktop_text'].append(score)
            else:
                results_dict[metric + '-web_text'].append(score)
        else:
            if 'ios' in result['data_source'] or 'android' in result['data_source']:
                results_dict[metric + '-mobile_icon'].append(score)
            elif 'macos' in result['data_source'] or 'windows' in result['data_source']:
                results_dict[metric + '-desktop_icon'].append(score)
            else:
                results_dict[metric + '-web_icon'].append(score)

    for key in results_dict:
        if len(results_dict[key]) == 0:
            results_dict[key] = 0
        else:
            results_dict[key] = sum(results_dict[key]) / len(results_dict[key])

        print(f"{key}: {results_dict[key]:0.4f}")
    return results_dict[metric]


def screenspot_rec_iou(results):
    return screenspot_rec_aggregation_result(results, "IoU")


def screenspot_rec_acc01(results):
    return screenspot_rec_aggregation_result(results, "ACC@0.1")

def screenspot_rec_acc03(results):
    return screenspot_rec_aggregation_result(results, "ACC@0.3")


def screenspot_rec_acc05(results):
    return screenspot_rec_aggregation_result(results, "ACC@0.5")


def screenspot_rec_acc07(results):
    return screenspot_rec_aggregation_result(results, "ACC@0.7")


def screenspot_rec_acc09(results):
    return screenspot_rec_aggregation_result(results, "ACC@0.9")


def screenspot_rec_center_acc(results):
    return screenspot_rec_aggregation_result(results, "Center_ACC")

if __name__ == '__main__':
    print(pred_2_point("123,123"))