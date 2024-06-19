import re
import io
import json
import logging
import datasets
from PIL import Image
from torchvision.ops import box_iou
import torch
import pandas as pd
from enum import Enum
from tqdm import tqdm
import ast
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
# MC_NUM = 8
# PREV_ACTION_LENGTH = 3

# def vwb_preprocess_dataset(dataset: datasets.Dataset):
#     ds_pd = dataset.to_pandas()
#     def get_pos_element(example):
#         '''
#         Extract target/positive element type and desc in text space, which would be friendly for prompt formulation.
#         '''
#         width = Image.open(io.BytesIO(example["image"]['bytes'])).width
#         height = Image.open(io.BytesIO(example["image"]['bytes'])).height

#         # step1: collect the positive object, xyxy --> normalized xyxy
#         if example['ui_pos_obj_screen_bbox'] is not None:
#             pos_bbox = [example['ui_pos_obj_screen_bbox'][0] / width,
#                                                     example['ui_pos_obj_screen_bbox'][1] / height,
#                                                     example['ui_pos_obj_screen_bbox'][2] / width,
#                                                     example['ui_pos_obj_screen_bbox'][3] / height]
#         else:
#             pos_bbox = []
#         pos_element_type = ui_object_type_dict[example['ui_pos_obj_type']]

#         return [pos_element_type, example['ui_pos_obj_desc_str'], pos_bbox, width, height]

#     # add key
#     tqdm.pandas()
#     ds_pd[['pos_element_type', 'pos_element_desc', 'pos_bbox', 'image_width', 'image_height']] = ds_pd.progress_apply(get_pos_element, axis=1, result_type='expand')

#     # Action Type: click, type, swipe 
#     #       ```unique_ds = self.ds_polars.unique(subset = ['action'], maintain_order=True)```
    
#     def agg_history_action(episode: list[dict]):
#         # in each epsisode, we add previous actions to current step_data, 
#         # episode = episode.sort_values(by='step_id')
#         new_colums = ['action_id', 'action_type','typed_text', 'element_type', 'element_description', 'action_str', 'action_str_seeclick'] # vwb have no swipe direction.
#         history = {k: [[]] for k in new_colums}
#         for index, step_data in episode.iterrows():
#             # original dataset contains element type, element desc, and bbox, here we merge them to help vlm understand.
#             action = action2str_vwb(step_data)
#             for k in history.keys():
#                 last = history[k][-1].copy()
#                 last += [action[k]]
#                 history[k].append(last)
#         for k, v in history.items():
#             # new_col = pl.Series(k, v[:-1])
#             episode.insert(0, "prev_"+k, v[:-1])
#             episode.insert(0, k, v[-1])
#         episode.insert(0, 'step_id', range(len(episode)))
        
#         return episode

#     # STEP1 prepare: group episode based on ep_id, and add previous actions for each step_data
#     ds_new = ds_pd.groupby('trace_id').progress_apply(agg_history_action)
    
#     new_dataset = datasets.Dataset.from_pandas(ds_new)

#     return new_dataset


def vwb_doc_to_visual(doc):
    # load from bytes
    root_path = '/data0/jingran/workspace/hongxin_li/WebEval/'
    img_path = os.path.join(root_path, doc['raw_image'].replace('img_box', 'img'))
    img = Image.open(img_path).convert("RGB")
    return [img]

def vwb_doc_to_target(doc):
    return [doc['bbox']]

def vwb_doc_to_target_wa(doc):
    return doc['pos_bbox']

def vwb_doc_to_text(doc, model_specific_prompt_kwargs=None):
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

    # if 'format' in model_specific_prompt_kwargs and model_specific_prompt_kwargs['format'] == 'seeclick':
    #     prev_actions_str = ""
    #     for prev_a_str in doc['prev_action_str_seeclick'][-PREV_ACTION_LENGTH:]:
    #         prev_actions_str += prev_a_str + "\n"
    # else:
    #     prev_actions_str = ""
    #     for prev_a_str in doc['prev_action_str'][-PREV_ACTION_LENGTH:]:
    #         prev_actions_str += prev_a_str + "\n"
    
    global_instr = doc['elem_desc']
    # step_instr = doc['elem_desc']

    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"].format(goal_info=global_instr)
    text = f"{pre_prompt}{post_prompt}"
    return text




def vwb_point_process_results(doc, result, model_specific_process_kwargs=None):
    '''
    Args:
        doc: dict
            A list of data instance.
        result: list
            A list of model outputs.
    '''
    
    # process the preds and computes squad f1 score before passing to metrics
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = result[0]

    # normalize the bbox
    root_path = '/data0/jingran/workspace/hongxin_li/WebEval/'
    img_path = os.path.join(root_path, doc['raw_image'].replace('img_box', 'img'))
    img = Image.open(img_path)
    width = img.width
    height = img.height
    # to xyxy n
    bbox = [round(doc['bbox'][0] / width, 1), round(doc['bbox'][1] / height, 1), round(doc['bbox'][2] / width, 1), round(doc['bbox'][3] / height, 1)]
            
    point = None
    correct = 0
    try:
        point = pred_2_point(resAns)
        if model_specific_process_kwargs == 'qwen_vl_chat':
            point = [round(point[0] / 1000, 1), round(point[1] / 1000,1)]
        elif model_specific_process_kwargs == 'seeclick':
            point = [point[0]/100, point[1]/100]
            point = [round(point[0] ,1), round(point[1] ,1)]
        elif model_specific_process_kwargs == 'cogagent_chat_hf':
            # point = [point[0]/1000, point[1]/1000]
            point = [round(point[0] ,1), round(point[1] ,1)]
        else:
            point = [round(point[0] , 1), round(point[1] , 1)]
        
        if (bbox[0] <= point[0] <= bbox[2]) and (bbox[1] <= point[1] <= bbox[3]):
            correct =1
    except Exception as e:
        print(e)
        correct = 0

    return {
        "vwb_gnd_result":  {'acc': correct},
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
    score = sum(acc)/len(acc)
    print('score', score)
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

# point (str) -> point
BRACKET_COORD_PATTERN = re.compile(r'\[(.*?)\]')
GENERAL_COORD_PATTERN = re.compile(r'-?\d+\.?\d*')

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
            points.extend([int(x), int(y)])
    except:
        points = None

    return points

def pred_2_point(pred, keep_box=False):
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
                        click_point.append(num)
                    except: pass
        
    assert click_point is not None, "Cannot extract click point from {}".format(pred)
    assert len(click_point) in [2,4], "Invalid click point {} found in {}".format(click_point, pred)
    
    if not keep_box and len(click_point) == 4:
        click_point = [(click_point[0]+click_point[2])/2, (click_point[1]+click_point[3])/2]

    return click_point

def pred_2_format_vwb(resAns, format='default'):
    """
    Extract action from response of VLM. 
    
    Args:
        step_data (dict): The step data containing the action details.
        
        NOTE:
        
        We believe the VLM has the point localiation ability, if action type is **click or type**, we directly extract the xy position as touch and lift.

    Return
        extracted_dicts: List[Dict]
    """
    ANSWER_PATTERN = r'\{.*?\}'

    extract_res = {}

    # Try to parse the matched string as a JSON object and append to the list
    try:
        if format == 'default':
            match = re.findall(ANSWER_PATTERN, resAns, re.DOTALL)
            action = json.loads(match)
            assert 'action_type' in action and isinstance(action['action_type'], str), f"action_type should be a string, but got {action['action_type']}"
            if 'click' in action["action_type"]: # dual point
                extract_res['action_type'] = 'click'
                extract_res['typed_text'] = ''
                extract_res['bbox'] = action['bbox']
            elif 'type' in action["action_type"]:
                extract_res['action_type'] = 'type'
                extract_res['typed_text'] = action['typed_text']
                extract_res['bbox'] = action['bbox']
            elif 'swipe' in action["action_type"]:
                extract_res['action_type'] = 'swipe'
                extract_res['typed_text'] = ''
                extract_res['bbox'] = [0.5,0.5,0.5,0.5]
                # swipe do not have target element
        elif format == 'seeclick':
            action = ast.literal_eval(resAns)
            assert 'action_type' in action and isinstance(action['action_type'], int), f"action_type should be a int, but got {action['action_type']}"
            if action['action_type'] == simplified_action_space_dict['click']:
                extract_res['action_type'] = 'click'
                if len(action['click_point']) == 2:
                    extract_res['click_point'] = action['click_point']
                elif len(action['click_point']) == 4:
                    extract_res['bbox'] = action['click_point']
            elif action['action_type'] == simplified_action_space_dict['type']:
                extract_res['action_type'] = 'type'
                extract_res['typed_text'] = action['typed_text']
                if 'click_point' in action:
                    extract_res['click_point'] = action['click_point']
                else:
                    extract_res['click_point'] = [0.5,0.5]
            elif action['action_type'] == [0,1,8,9]:
                extract_res['action_type'] = 'swipe'
                extract_res['click_point'] = [0.5,0.5]
            else:
                raise ValueError(f"Unknown action type in vwb: {action['action_type']}")
        elif format =='cogagent_chat_hf': # pred format: 
            resAns = resAns.split("Grounded Operation:")[-1]
            if 'tap' in resAns:
                # get point
                extract_res['action_type'] = 'click'
                extract_res['typed_text'] = ''
                pattern_point = r'\[\[(\d+),(\d+)\]\]'
                pattern_bbox = r'\[\[(\d+),(\d+),(\d+),(\d+)\]\]'
                matches_point = re.findall(pattern_point, resAns)
                matches_bbox = re.findall(pattern_bbox, resAns)
                if matches_bbox:
                    bbox = [int(num)/1000 for num in matches_bbox[0]]
                    extract_res['bbox'] = bbox
                elif matches_point:
                    center = [int(num)/1000 for num in matches_point[0]]
                    extract_res['click_point'] = center
                else:
                    raise ValueError(f"Unknown grounding patter: {resAns}")
            elif 'type' in resAns:
                extract_res['action_type'] = 'type'
                extract_res['typed_text'] = resAns.split("typed_text:")[1].split(",")[0]
                extract_res['bbox'] = [0.5,0.5,0.5,0.5]
            elif 'swipe' in resAns:
                extract_res['action_type'] = 'swipe'
                extract_res['typed_text'] = ''
                extract_res['bbox'] = [0.5,0.5,0.5,0.5]
            else:
                raise ValueError(f"Unknown action type in vwb: {resAns}")
    except Exception as e:
        extract_res = {"action_type": "swipe", 'typed_text': '', 'bbox': [0.5,0.5,0.5,0.5]}
    return extract_res

