import re, random
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
from pretrain.prompt_lib import web_loca_all_point_prompt, apply_vlm_template
from pretrain.process_utils import pred_2_point

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
MC_NUM = 8
PREV_ACTION_LENGTH = 3

def motif_preprocess_dataset(dataset: datasets.Dataset):
    ds_pd = dataset.to_pandas()
    def get_pos_element(example):
        '''
        Extract target/positive element type and desc in text space, which would be friendly for prompt formulation.
        '''
        width = Image.open(io.BytesIO(example["image"]['bytes'])).width
        height = Image.open(io.BytesIO(example["image"]['bytes'])).height

        # step1: collect the positive object, xyxy --> normalized xyxy
        if example['ui_pos_obj_screen_bbox'] is not None:
            pos_bbox = [example['ui_pos_obj_screen_bbox'][0] / width,
                                                    example['ui_pos_obj_screen_bbox'][1] / height,
                                                    example['ui_pos_obj_screen_bbox'][2] / width,
                                                    example['ui_pos_obj_screen_bbox'][3] / height]
        else:
            pos_bbox = []
        pos_element_type = ui_object_type_dict[example['ui_pos_obj_type']]

        return [pos_element_type, example['ui_pos_obj_desc_str'], pos_bbox, width, height]

    # add key
    tqdm.pandas()
    ds_pd[['pos_element_type', 'pos_element_desc', 'pos_bbox', 'image_width', 'image_height']] = ds_pd.progress_apply(get_pos_element, axis=1, result_type='expand')

    # Action Type: click, type, swipe 
    #       ```unique_ds = self.ds_polars.unique(subset = ['action'], maintain_order=True)```
    
    def agg_history_action(episode: list[dict]):
        # in each epsisode, we add previous actions to current step_data, 
        # episode = episode.sort_values(by='step_id')
        new_colums = ['action_id', 'action_type','typed_text', 'element_type', 'element_description', 'action_str', 'action_str_seeclick'] # motif have no swipe direction.
        history = {k: [[]] for k in new_colums}
        for index, step_data in episode.iterrows():
            # original dataset contains element type, element desc, and bbox, here we merge them to help vlm understand.
            action = action2str_motif(step_data)
            for k in history.keys():
                last = history[k][-1].copy()
                last += [action[k]]
                history[k].append(last)
        for k, v in history.items():
            # new_col = pl.Series(k, v[:-1])
            episode.insert(0, "prev_"+k, v[:-1])
            episode.insert(0, k, v[-1])
        episode.insert(0, 'step_id', range(len(episode)))
        
        return episode

    # STEP1 prepare: group episode based on ep_id, and add previous actions for each step_data
    ds_new = ds_pd.groupby('trace_id').progress_apply(agg_history_action)
    
    new_dataset = datasets.Dataset.from_pandas(ds_new)

    return new_dataset


def motif_doc_to_visual(doc):
    # load from bytes
    img = Image.open(io.BytesIO(doc['image']['bytes']))
    return [img]

def motif_doc_to_target(doc):
    return [doc['pos_bbox'], doc['action']]

def motif_doc_to_target_wa(doc):
    return doc['pos_bbox']

def motif_doc_to_text(doc, model_name='', model_specific_prompt_kwargs=None):
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
    global_instr = doc['goal']
    step_instr = doc['instr']
    pre_prompt = ""
    post_prompt = ""

    if model_specific_prompt_kwargs is None:
        prompt = apply_vlm_template(global_instr, model_name)
    else:
        # Use random prompt templates
        if model_specific_prompt_kwargs is not None:
            if model_specific_prompt_kwargs['format'] == 'random':
                post_prompt = random.choice(web_loca_all_point_prompt) + f" This element is used for \"{global_instr}\""
                
            elif model_specific_prompt_kwargs['format'] == 'seeclick':
                prev_actions_str = ""
                for prev_a_str in doc['prev_action_str_seeclick'][-PREV_ACTION_LENGTH:]:
                    prev_actions_str += prev_a_str + "\n"
            else:
                prev_actions_str = ""
                for prev_a_str in doc['prev_action_str'][-PREV_ACTION_LENGTH:]:
                    prev_actions_str += prev_a_str + "\n"

            if "pre_prompt" in model_specific_prompt_kwargs:
                pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
            if "post_prompt" in model_specific_prompt_kwargs:
                post_prompt = model_specific_prompt_kwargs["post_prompt"].format(goal_info=global_instr, step_instr=step_instr, previous_actions=prev_actions_str, action_space= simplified_action_space)
        
        else:
            post_prompt = random.choice(web_loca_all_point_prompt) + f" This element is used for \"{global_instr}\""

        prompt = f"{pre_prompt}{post_prompt}"
    return prompt

def motif_doc_to_text_wa(doc, model_specific_prompt_kwargs=None):
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

    prev_actions_str = ""
    for prev_a_str in doc['prev_action_str'][-PREV_ACTION_LENGTH:]:
        prev_actions_str += prev_a_str + "\n"
    
    global_instr = doc['goal']
    action_type = doc['action_type_str']

    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"].format(goal_info=global_instr, previous_actions=prev_actions_str, action_type=action_type)
    text = f"{pre_prompt}{post_prompt}"
    return text

def motif_process_results(doc, result, model_specific_process_kwargs=None):
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
    # The response from vlm is high-level semantics. 
    # Here we extract back to the ```action type: int```` and made easy for the evaluation funciton ```def check_actions_match()``` 
    action_pred = pred_2_format_motif(resAns, model_specific_process_kwargs)
    
    pred_action_type = action_pred["action_type"]
    gt_bbox = doc['pos_bbox'] # normalized xyxy
    gt_action_type = doc['action'] # str, type, click, swipe
    iou_threshold = 0.1
    '''
    Eval on Acc@IoU=0.1, model required to predict one correct bbox, without using candidates bbox. (ScreenAI) 
    '''

    if not pred_action_type == gt_action_type:
        correct = 0
        correct_action = 0
    else:
        correct_action = 1
        correct = 0
        if pred_action_type == 'swipe':
            correct = 1
        else:
            # type and click
            if 'bbox' in action_pred:
                p_bbox = action_pred["bbox"]
                try:
                    gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32).view(-1, 4)
                    p_bbox = torch.tensor(p_bbox, dtype=torch.float32).view(-1, 4)
                    iou = box_iou(p_bbox, gt_bbox)
                    iou = iou.item()
                    if iou >= iou_threshold:
                        correct = 1
                    else:
                        correct = 0
                except:
                    correct = 0
            elif 'click_point' in action_pred:
                try:
                    # click_point in bbox?
                    click_point = action_pred["click_point"]
                    xmin, ymin, xmax, ymax = gt_bbox  # Unpack the bounding box coordinates
                    x, y = click_point  # Unpack the click point coordinates
                    # Check conditions
                    inside_x = (x >= xmin) and (x <= xmax)
                    inside_y = (y >= ymin) and (y <= ymax)
                    inside_bbox = inside_x and inside_y
                    if inside_bbox:
                        correct = 1
                    else:
                        correct = 0
                except:
                    correct = 0
    return {
        "partial_action_bbox_acc":  {'acc': correct, 'trace_id': doc['trace_id'], 'step_id': doc['step_id']},
        "complete_action_bbox_acc": {'acc': correct, 'trace_id': doc['trace_id'], 'step_id': doc['step_id']},
        "nogroup_action_bbox_acc": {'acc': correct, 'trace_id': doc['trace_id'], 'step_id': doc['step_id']},
        "partial_action_acc":  {'acc': correct_action, 'trace_id': doc['trace_id'], 'step_id': doc['step_id']},
        "complete_action_acc": {'acc': correct_action, 'trace_id': doc['trace_id'], 'step_id': doc['step_id']},
    }

def motif_point_process_results(doc, result, model_specific_process_kwargs=None):
    '''
    Args:
        doc: dict
            A list of data instance.
        result: list
            A list of model outputs.
    '''
    
    # process the preds and computes squad f1 score before passing to metrics
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    
    pred = result[0]['response'] if len(result) > 0 else ""
    scale = model_specific_process_kwargs.get("scale", 1) if model_specific_process_kwargs is not None else 1

    try:
        pred = pred_2_point(pred, keep_box=False, scale=scale)
    except:
        pred = [0,0,0,0]
    # action_pred = pred_2_format_motif(resAns, model_specific_process_kwargs)
    
    # pred_action_type = action_pred["action_type"]
    gt_bbox = doc['pos_bbox'] # normalized xyxy
    gt_action_type = doc['action'] # str, type, click, swipe

    '''
    Eval on Acc@IoU=0.1, model required to predict one correct bbox, without using candidates bbox. (ScreenAI) 
    '''    
    correct = 0
    if gt_action_type == 'swipe':
        correct = 1
    else:
        # type and click
        try:
            # click_point in bbox?
            if len(pred) == 2:
                center_x, center_y = pred
            elif len(pred) == 4:
                center_x = (pred[0] + pred[2]) / 2
                center_y = (pred[1] + pred[3]) / 2
            else:
                center_x, center_y = -1, -1

            if gt_bbox[0] <= center_x <= gt_bbox[2] and gt_bbox[1] <= center_y <= gt_bbox[3]:
                correct = 1
            else:
                correct = 0
        except:
            correct = 0
    
    # pop image
    for k in ["image", "image_w_bbox", "ui_neg_objs_type_id", "ui_neg_objs_clickable", "ui_neg_objs_bbox", "ui_neg_objs_str"]:
        if k in doc: doc.pop(k)
        
    return {
        "motif_gnd_result":  {'goal': doc['goal'], 'step_instruc': doc['instr'], 'prompt': result[0]['prompt'], 'response': result[0]['response'], 'pred': pred, 'gt_box': gt_bbox, 'acc': correct, 'trace_id': doc['trace_id'], 'step_id': doc['step_id'], 'action': gt_action_type},
    }

def motif_gnd_result(results):
    # STEP2 calculate grounding score, by grouping trace_id
    # df = pd.DataFrame(results)
    # grouped = df.groupby('trace_id')['acc']
    # results = grouped.agg(mean_acc='sum')
    # partial acc
    acc = []
    acc_ = []
    for result in results:
        if result['action'] != 'swipe':
            acc.append(result['acc'])
        acc_.append(result['acc'])
    score = sum(acc)/len(acc)
    score_ = sum(acc_) / len(acc_)
    print('score', score)
    print('score_', score_)
    return score

def motif_complete_aggregation_result(results):
    # STEP2 calculate grounding score, by grouping trace_id
    df = pd.DataFrame(results)
    grouped = df.groupby('trace_id')['acc']
    results = grouped.agg(success_rate=lambda x: (x == 1).all().astype(int))
    # trajectory acc
    complete_score = results['success_rate'].mean()
    return complete_score

def motif_partial_aggregation_result(results):
    # STEP2 calculate grounding score, by grouping trace_id
    df = pd.DataFrame(results)
    grouped = df.groupby('trace_id')['acc']
    results = grouped.agg(mean_acc='mean')
    # partial acc
    partial_score = results['mean_acc'].mean()
    print('partial_score', partial_score)
    return partial_score

def motif_nogroup_aggregation_result(results):
    # STEP2 calculate grounding score, by grouping trace_id
    df = pd.DataFrame(results)
    partial_score = df['acc'].mean()

    return partial_score


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

