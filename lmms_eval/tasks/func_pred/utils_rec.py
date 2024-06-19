import re 
import logging
from datasets import Dataset

eval_logger = logging.getLogger("lmms-eval")

REC_METRICS = ["Center_ACC"]



def screenspot_rec_doc_to_visual(doc):
    # Image is presented as is
    image = doc["image"].convert("RGB")
    return [image.convert("RGB")]


def screenspot_rec_doc_to_text(doc):
    return doc["instruction"]
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

# bbox (qwen str) -> bbox
SEECLICK_BOX_PATTERN = re.compile(r"\((\d+,\d+)\),\((\d+,\d+)\)")
def extract_bbox(pred):
    # Regular expression to find the content inside <box> and </box>
    matches = SEECLICK_BOX_PATTERN.findall(pred)
    # Convert the tuples of strings into tuples of integers
    
    try:
        points = []
        
        for point in matches[-1]:
            x, y = point.split(',')
            points.extend([int(x) / 1000, int(y) / 1000])
    except:
        points = None

    return points

# point (str) -> point
BRACKET_COORD_PATTERN = re.compile(r'\[(.*?)\]')
GENERAL_COORD_PATTERN = re.compile(r'-?\d+\.?\d*')


def pred_2_point(pred, keep_box=True):
    click_point = None
    if '[[' in pred: # For CogAgent
        coords_start = pred.find('[[')
        if coords_start != -1:
            coords_end = pred.find(']]')
            if coords_end != -1:
                coords_str = pred[coords_start+2:coords_end]
                try:
                    # The bounding box coordinates in the CogAgent's output use the format [[x1, y1, x2, y2]], with the origin at the top left corner, the x-axis to the right, and the y-axis downward. (x1, y1) and (x2, y2) are the top-left and bottom-right corners, respectively, with values as relative coordinates multiplied by 1000 (prefixed with zeros to three digits).
                    click_point = [x / 1000 for x in map(float, coords_str.split(','))]
                except:
                    raise ValueError("Cannot extract click point from {}".format(pred))
    elif '[' in pred:
        matches = [(match.group(), (match.start(), match.end())) for match in BRACKET_COORD_PATTERN.finditer(pred)]

        if matches:
            # We take the last one
            last_valid_match_id = len(matches) - 1
            while last_valid_match_id >=0:
                click_point_str, start_end = matches[last_valid_match_id]
                try:
                    click_point = list(map(float, click_point_str[1:-1].split(',')))
                    break
                except: pass
                last_valid_match_id -= 1
            else:
                raise ValueError("Cannot extract click point from {}".format(pred))

            # If there are two coordinates enclosed with brackets and they are different and their appearances in the response are not far away, they may be represent the top-left and bottom-right corners, respectively.
            if len(click_point) == 2 and last_valid_match_id > 0 and (start_end[0] - matches[last_valid_match_id-1][1][1]) < 30:
                try:
                    another_point = list(map(float, matches[last_valid_match_id-1][0][1:-1].split(', ')))
                    if len(another_point) == 2:
                        click_point = [(another_point[0] + click_point[0]) / 2, (another_point[1] + click_point[1]) / 2]
                except: pass

    if click_point is None: # For SeeClick
        if '<box>' in pred: # For QWen-VL-Chat
            click_point = extract_bbox(pred)
        else:
            floats = GENERAL_COORD_PATTERN.findall(pred)
            
            if floats:
                click_point = []
                for num in floats:
                    try:
                        num = float(num)
                        if 0.0 <= num <=1.0: click_point.append(num)
                    except: pass
        
    assert click_point is not None, "Cannot extract click point from {}".format(pred)
    assert len(click_point) in [2,4], "Invalid click point {} found in {}".format(click_point, pred)
    
    if not keep_box and len(click_point) == 4:
        click_point = [(click_point[0]+click_point[2])/2, (click_point[1]+click_point[3])/2]

    return click_point


def screenspot_rec_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    # pred = parse_float_sequence_within(pred)
    try:
        pred = pred_2_point(pred)
    except:
        pred = [0,0,0,0]
    ann_id = doc["file_name"]
    data_dict = {"instruction": doc["instruction"], "pred": pred, "ann_id": ann_id, 'bbox': doc['bbox'], 'data_type': doc['data_type'], 'data_source': doc['data_source']}
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
        metric + '-downsize1': [], 
        metric + '-downsize2': [],
        metric + '-downsize4': []
    }
    for result in results:
        # Extract the ground truth and predicted bounding boxes
        gt = result['bbox']
        pred = result['pred']

        # Compute the specified metric between the ground truth and predicted bounding boxes
        score = scorers[metric](gt, pred)

        results_dict[metric].append(score)
        
        results_dict[f"{metric}-{result['data_type']}"].append(score)

    for key in results_dict:
        if len(results_dict[key]) == 0:
            results_dict[key] = 0
        else:
            results_dict[key] = sum(results_dict[key]) / len(results_dict[key])

        print(f"{key}: {results_dict[key]:0.4f}")
    return results_dict[metric]


def screenspot_rec_center_acc(results):
    return screenspot_rec_aggregation_result(results, "Center_ACC")
