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
import numpy as np
from lmms_eval.tasks.screenspot.utils_rec import pred_2_point

eval_logger = logging.getLogger("lmms-eval")

simplified_action_space_dict = {'type': 3, 'click': 4, 'swipe': 0} # id borrowed from seeclick
simplified_action_space = repr([k for k in simplified_action_space_dict.keys()])


icon_noun = [
    'icon','widget','ui element','element'
]

input_function_prompt_list = [
    "Where is the {icon} that has the functionality: {desc}?",
    "Can you point me to the {icon} that does {desc}?",
    "Which location features the {icon} associated with the functionality of {desc}?",
    "Could you help me locate the {icon} that handles {desc}?",
    "I'm trying to find the {icon} for {desc}. Any ideas where it might be?",
    "Where can I find the {icon} that performs the function: {desc}?",
    "Can you show me the {icon} that is responsible for {desc}?",
    "Identify the {icon} that enables {desc}.",
    "Where is the {icon} for {desc}?",
    "Location of the {icon} for {desc}?"
]

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
MC_NUM = 8
PREV_ACTION_LENGTH = 3

def wic_preprocess_dataset(dataset: datasets.Dataset):
    ds_pd = dataset.to_pandas()
    def get_pos_element(example):
        '''
        Extract target/positive element type and desc in text space, which would be friendly for prompt formulation.
        '''

        # step1: collect the positive object, xyxy --> normalized xyxy*1000
        bbox = example['bbox']
        bbox = [int(x*1000) for x in bbox] # xyxy

        desc_list = []
        for desc in example['captions']:
            input_prompt = np.random.choice(input_function_prompt_list)
            icon = np.random.choice(icon_noun)
            input_prompt = input_prompt.format(icon = icon, desc = desc)
            desc_list.append(input_prompt)

        return [desc_list]

    # add key
    tqdm.pandas()
    ds_pd[['input_prompt_list']] = ds_pd.progress_apply(get_pos_element, axis=1, result_type='expand')
    
    new_dataset = datasets.Dataset.from_pandas(ds_pd)

    return new_dataset

def wic_doc_to_visual(doc):
    # load from bytes
    image_path = doc['file_name']
    data_root = '/data3/workspace/hongxin_li/UI_training_data/images'
    try:
        img = Image.open(f"{data_root}/{image_path}")
    except:
        print("Invalid img path", f"{data_root}/{image_path}")
    return [img]

def wic_doc_to_target(doc):
    bbox = doc['bbox']
    bbox = [int(x*1000) for x in bbox] # xyxy
    return [bbox]

def wic_doc_to_text(doc, model_specific_prompt_kwargs=None):
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
    
    global_instr = np.random.choice(doc['captions'])
    prompt_instr = np.random.choice(doc['input_prompt_list'])

    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"].format(question=prompt_instr)
    text = f"{pre_prompt}{post_prompt}"
    return text

def wic_process_results(doc, result, model_specific_process_kwargs=None):
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
    # point_str = re.search(r'<point>\((\d+),(\d+)\)</point>', resAns)
    try:
        point = pred_2_point(resAns) # [x, y] in [0,1]
    except:
        print("invalid_point: ", resAns)
        point=[0,0]
    # if point_str:
    #     point = [int(point_str.group(1)), int(point_str.group(2))]
    # else:
    #     point = [0, 0]
    gt_bbox = doc['bbox']
    # gt_bbox = [int(x*1000) for x in bbox] # xyxy
    x, y = point
    x1, y1, x2, y2 = gt_bbox
    if x >= x1 and x <= x2 and y >= y1 and y <= y2:
        correct = 1
    else:
        correct = 0

    return {
        "point_acc":  {'acc': correct, 'screenId': doc['screenId']},
    }

def wic_aggregation_result(results):
    # STEP2 calculate grounding score, by grouping trace_id
    df = pd.DataFrame(results)
    # trajectory acc
    complete_score = df['acc'].mean()
    return complete_score



def action2str_motif(step_data):
    """
    Determines the action detail type and converts the action to a string.

    Args:
        step_data (dict): The step data containing the action details.

    Returns:
        str: The action in string format.
        step_data with additional action_detail_type field.
    """
    # https://huggingface.co/datasets/cjfcsjt/MoTIF-automation/viewer/default/test_au_tu
    # type, click, swipe
    action_type = step_data["action"] 

    bbox = step_data["pos_bbox"] # normalized xyxy
    if len(bbox) == 0:
        center_point = [0.5, 0.5]
    else:
        center_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    pos_element_type = step_data["pos_element_type"] if step_data["pos_element_type"] is not None else ""
    pos_element_desp = step_data["pos_element_desc"] if step_data["pos_element_desc"] is not None else ""
    typed_text = step_data["ui_pos_obj_input_str"] if step_data["ui_pos_obj_input_str"] is not None else ""

    if action_type == 'click':
        # unnoralized bbox xyxy, normalize to [0,1]
        # bbox = [0,0,0,0]
        # bbox[0] = step_data["ui_pos_obj_screen_bbox"][0] / step_data["screen_w"] 
        # bbox[1] = step_data["ui_pos_obj_screen_bbox"][1] / step_data["screen_h"] 
        # bbox[2] = step_data["ui_pos_obj_screen_bbox"][0] / step_data["screen_w"]
        # bbox[3] = step_data["ui_pos_obj_screen_bbox"][1] / step_data["screen_h"] 
        # bbox_str = [f"{b:.2f}" for b in bbox]
        # bbox_str = "[{}, {}, {}, {}]".format(bbox_str[0], bbox_str[1], bbox_str[2], bbox_str[3])
        action_str = "{{\"action_type\": \"click\", element_type: \"{}\"}}".format(pos_element_type)  # formated here for easy json load
        action_str_seeclick = "{{\"action_type\": {}, \"click_point\": \"{}\"}}".format(simplified_action_space_dict['click'], center_point)
        action = {'action_id': simplified_action_space_dict['click'],
                  'action_type': action_type, 
                  'typed_text': '',
                  'element_type': pos_element_type,
                  'element_description': pos_element_desp,
                  'action_str': action_str,
                  'action_str_seeclick': action_str_seeclick}   
    elif action_type == 'type':
        # unnoralized bbox xyxy, normalize to [0,1]
        # bbox = [0,0,0,0]
        # bbox[0] = step_data["ui_pos_obj_screen_bbox"][0] / step_data["screen_w"] 
        # bbox[1] = step_data["ui_pos_obj_screen_bbox"][1] / step_data["screen_h"] 
        # bbox[2] = step_data["ui_pos_obj_screen_bbox"][0] / step_data["screen_w"]
        # bbox[3] = step_data["ui_pos_obj_screen_bbox"][1] / step_data["screen_h"] 
        # bbox_str = [f"{b:.2f}" for b in bbox]
        # bbox_str = "[{}, {}, {}, {}]".format(bbox_str[0], bbox_str[1], bbox_str[2], bbox_str[3])
        action_str = "{{\"action_type\": \"type\", \"typed_text\": \"{}\", element_type: \"{}\"}}".format(typed_text, pos_element_type) # motif has both typed_text and positive element_type
        action_str_seeclick = "{{\"action_type\": {}, \"typed_text\": \"{}\", element_type: \"{}\"}}".format(simplified_action_space_dict['click'], typed_text, pos_element_type)
        action = {'action_id': simplified_action_space_dict['type'],
                  'action_type': action_type, 
                  'typed_text': typed_text,
                  'element_type': pos_element_type,
                  'element_description': pos_element_desp,
                  'action_str': action_str,
                  'action_str_seeclick': action_str_seeclick}   
    elif action_type == 'swipe':
        action_str = "{{\"action_type\": \"swipe\"}}"  # motif has no swipe direction
        action_str_seeclick = "{{\"action_type\": {} }}".format(simplified_action_space_dict['swipe'])  # motif has no swipe direction
        action = {'action_id': simplified_action_space_dict['swipe'],
                  'action_type': action_type, 
                  'typed_text':typed_text,
                  'element_type': pos_element_type,
                  'element_description': pos_element_desp,
                  'action_str': action_str,
                  'action_str_seeclick': action_str_seeclick}   
    else:
        raise ValueError(f"Unknown action type: {action_str}")
    
    return action


def pred_2_format_motif(resAns, format='default'):
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
                raise ValueError(f"Unknown action type in MOTIF: {action['action_type']}")
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
                raise ValueError(f"Unknown action type in MOTIF: {resAns}")
    except Exception as e:
        extract_res = {"action_type": "swipe", 'typed_text': '', 'bbox': [0.5,0.5,0.5,0.5]}
    return extract_res

