
web_loca_all_point_prompt = [
    "In the screenshot of this web page, please give me the coordinates of the element I want to click on according to my instructions (with point).",
    "Based on the screenshot of the page, I give a text description and you give its corresponding location (with point).",
    "In the image above, I will give a series of descriptions of the element to be clicked. Please predict where you want to click (with point).",
    "I will give textual descriptions of a certain element in the screenshot. Please predict the location of the corresponding element (with point).",
    "Please identify the coordinates of the webpage element I describe based on the provided screenshot (with point).",
    "Given a screenshot, I will describe a specific element; your task is to predict their locations (with point).",
    "Using the image of this webpage, can you determine the coordinates of the element I describe (with point)?",
    "In this webpage capture, I will describe a certain element. Please locate it for me (with point).",
    "I'll provide textual descriptions of the element in this webpage screenshot. Can you find their coordinates (with point)?",
    "From the given webpage screenshot, I need you to identify the locations of the described element (with point).",
    "Based on this screenshot, I'll describe an element. Please pinpoint their exact locations (with point).",
    "For the element I describe in this page capture, can you predict their positions (with point)?",
    "I will describe an element from a webpage screenshot; your role is to locate it (with point).",
    "Using the attached screenshot of a webpage, please find the coordinates of the described element (with point).",
    "From the image of this webpage, I will describe an element for you to locate (with point).",
    "I'll give descriptions of a certain webpage element; please identify where they are in this screenshot (with point).",
    "On this webpage screenshot, I will point out an element; please predict their exact coordinates (with point).",
    "In this web page image, please locate the element as I describe it (with point).",
    "Given this screenshot of a webpage, I'll describe an element; locate it for me (with point).",
    "Please use the provided webpage screenshot to locate the element I describe (with point).",
    "In the provided web page image, I'll describe a specific element. Identify their locations, please (with point).",
    "With this screenshot of a webpage, can you locate the element I describe (with point)?",
    "I will describe features on this webpage screenshot; please predict their positions (with point).",
    "Using the screenshot of this webpage, identify the coordinates of the element I describe (with point).",
    "On this webpage capture, I'll point out a specific element for you to locate (with point).",
    "Please determine the location of the element I describe in this webpage screenshot (with point).",
    "I'll describe certain an element on this webpage image; your task is to find their locations (with point).",
    "Using this webpage screenshot, I'll describe an element. Please locate it (with point).",
    "Based on my descriptions, find the locations of the mentioned element in this webpage screenshot (with point).",
    "In this web page capture, please predict the positions of the element I describe (with point).",
    "I'll give textual clues about an element in this webpage screenshot; identify their coordinates (with point).",
    "Using the provided screenshot, I'll describe a webpage element for you to locate (with point).",
    "From this webpage image, I will describe a specific element. Please predict their exact locations (with point)."
]

import random
random.seed(1234)

# ScreenSpot
POSITION_PROMPT = """In this UI screenshot, what is the position of the element corresponding to the command "{}"? Output the normalized X and Y coordinates, ranging from 0.00 to 1.00. Note that the X-axis runs horizontally from left (0.00) to right (1.00), and the Y-axis runs vertically from top (0.00) to bottom (1.00). Your should carefully view the image before finally predicting the required position in the format [X, Y] (two decimal places)."""

BOX_PROMPT = """In this UI screenshot, what are the bounding box coordinates of the element corresponding to the command "{}"? Output the normalized X and Y coordinates, ranging from 0.00 to 1.00. Note that the X-axis runs horizontally from left (0.00) to right (1.00), and the Y-axis runs vertically from top (0.00) to bottom (1.00). Your should carefully view the image before finally predicting the required bounding box coordinates in the format [X_min, Y_min, X_max, Y_max]."""

LLAVA16_START, LLAVA16_END = "[INST] ", " [/INST]"
LLAVA15_START, LLAVA15_END = "USER: ", "\nASSISTANT:"
LLAVA_IMAGE_PLACEHOLDER = "<image>"

# Llava-1.6 Example: "[INST] <image>\nIn this UI screenshot, what is the position of the element corresponding to the command "{}"? [/INST]"
def get_llava_prompt(is_v16, add_special_tokens=True, output_box=False):
    prompt = BOX_PROMPT if output_box else POSITION_PROMPT
    
    if add_special_tokens:
        prompt = f'{LLAVA16_START}{LLAVA_IMAGE_PLACEHOLDER}{prompt}{LLAVA16_END}' if is_v16 else f'{LLAVA15_START}{LLAVA_IMAGE_PLACEHOLDER}{prompt}{LLAVA15_END}'
    
    return prompt

# DeepSeek-VL: https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat#simple-inference-example
DEEPSEEK_IMAGE_PLACEHOLDER = '<image_placeholder>'

# DeepSeek-vl Example: '<image_placeholder>In this UI screenshot, what is the position of the element corresponding to the command "{question}"?'
def get_deepseek_prompt(output_box):
    prompt = BOX_PROMPT if output_box else POSITION_PROMPT
    
    prompt = f'{DEEPSEEK_IMAGE_PLACEHOLDER}{prompt}'
    
    return prompt

# CogAgent-chat: https://huggingface.co/THUDM/cogagent-chat-hf

# GUI Agent Task: Use the Agent template and replace <TASK> with the task instruction enclosed in double quotes. This query can make CogAgent infer Plan and Next Action. If adding (with grounding) at the end of the query, the model will return a formalized action representation with coordinates. (https://github.com/THUDM/CogVLM/tree/main?tab=readme-ov-file#task-prompts)
cogagent_templates = [
    "Can you advise me on how to {}?",
    "I'm looking for guidance on how to {}.",
    "What steps do I need to take to {}?",
    "Could you provide instructions for {}?",
    "I'm wondering what the process is for {}.",
    "How can I go about {}?",
    "I need assistance with planning to {}.",
    "Do you have any recommendations for {}?",
    "Please share some tips for {}.",
    "I'd like to know the best way to {}.",
    "What's the most effective way to {}?",
    "I'm seeking advice on accomplishing {}.",
    "Could you guide me through the steps to {}?",
    "I'm unsure how to start with {}.",
    "Is there a strategy for successfully {}?",
    "What's the proper procedure for {}?",
    "How should I prepare for {}?",
    "I'm not sure where to begin with {}.",
    "I need some insights on {}.",
    "Can you explain how to tackle {}?",
    "I'm interested in the process of {}.",
    "Could you enlighten me on {}?",
    "What are the recommended steps for {}?",
    "Is there a preferred method for {}?",
    "I'd appreciate your advice on {}.",
    "Can you shed light on {}?",
    "What would be the best approach to {}?",
    "How do I get started with {}?",
    "I'm inquiring about the procedure for {}.",
    "Could you share your expertise on {}?",
    "I'd like some guidance on {}.",
    "What's your recommendation for {}?",
    "I'm seeking your input on how to {}.",
    "Can you provide some insights into {}?",
    "How can I successfully accomplish {}?",
    "What steps are involved in {}?",
    "I'm curious about the best way to {}.",
    "Could you show me the ropes for {}?",
    "I need to know how to go about {}.",
    "What are the essential steps for {}?",
    "Is there a specific method for {}?",
    "I'd like to get some advice on {}.",
    "Can you explain the process of {}?",
    "I'm looking for guidance on how to approach {}.",
    "What's the proper way to handle {}?",
    "How should I proceed with {}?",
    "I'm interested in your expertise on {}.",
    "Could you walk me through the steps for {}?",
    "I'm not sure where to begin when it comes to {}.",
    "What should I prioritize when doing {}?",
    "How can I ensure success with {}?",
    "I'd appreciate some tips on {}.",
    "Can you provide a roadmap for {}?",
    "What's the recommended course of action for {}?",
    "I'm seeking your guidance on {}.",
    "Could you offer some suggestions for {}?",
    "I'd like to know the steps to take for {}.",
    "What's the most effective way to achieve {}?",
    "How can I make the most of {}?",
    "I'm wondering about the best approach to {}.",
    "Can you share your insights on {}?",
    "What steps should I follow to complete {}?",
    "I'm looking for advice on {}.",
    "What's the strategy for successfully completing {}?",
    "How should I prepare myself for {}?",
    "I'm not sure where to start with {}.",
    "What's the procedure for {}?",
    "Could you provide some guidance on {}?",
    "I'd like to get some tips on how to {}.",
    "Can you explain how to tackle {} step by step?",
    "I'm interested in understanding the process of {}.",
    "What are the key steps to {}?",
    "Is there a specific method that works for {}?",
    "I'd appreciate your advice on successfully completing {}.",
    "Can you shed light on the best way to {}?",
    "What would you recommend as the first step to {}?",
    "How do I initiate {}?",
    "I'm inquiring about the recommended steps for {}.",
    "Could you share some insights into {}?",
    "I'm seeking your expertise on {}.",
    "What's your recommended approach for {}?",
    "I'd like some guidance on where to start with {}.",
    "Can you provide recommendations for {}?",
    "What's your advice for someone looking to {}?",
    "I'm seeking your input on the process of {}.",
    "How can I achieve success with {}?",
    "What's the best way to navigate {}?",
    "I'm curious about the steps required for {}.",
    "Could you show me the proper way to {}?",
    "I need to know the necessary steps for {}.",
    "What's the most efficient method for {}?",
    "I'd appreciate your guidance on {}.",
    "Can you explain the steps involved in {}?",
    "I'm looking for recommendations on how to approach {}.",
    "What's the right way to handle {}?",
    "How should I manage {}?",
    "I'm interested in your insights on {}.",
    "Could you provide a step-by-step guide for {}?",
    "I'm not sure how to start when it comes to {}.",
    "What are the key factors to consider for {}?",
    "How can I ensure a successful outcome with {}?",
    "I'd like some tips and tricks for {}.",
    "Can you offer a roadmap for accomplishing {}?",
    "What's the preferred course of action for {}?",
    "I'm seeking your expert advice on {}.",
    "Could you suggest some best practices for {}?",
    "I'd like to understand the necessary steps to complete {}.",
    "What's the most effective strategy for {}?",
]

COGAGENT_BOX_PROMPT = 'Can you point out the element specified by the command {} in the UI screenshot and provide the bounding box of its location?'
# 'Can you point out the element specified by the command {} in the UI screenshot and provide the bounding box of its location?' # 在ScreenSpot上少量[[x0,y0,x1,y1]]，其他都是不带坐标的plan; 在RefExp上都是不带坐标的plan
# "Please provide the bounding box coordinate of the region described by the sentence: {}.(with grounding)" # 在ScreenSpot上CogAgent只输出[[X,Y]]
# 'Can you point out the element specified by the command {} in the UI screenshot and provide the bounding box of its location?(with grounding)' # 在ScreenSpot上只输出[[X,Y]]； 在RefExp上只输出[[X,Y]] 
# 'Where is {}? answer in [[x0,y0,x1,y1]] format.(with grounding)' # 在RefExp上只输出[[X,Y]] 
# 'Where is {}? answer in [[x0,y0,x1,y1]] format.'  # 在RefExp上只输出极少的[[x0,y0,x1,y1]]，其他都是不带坐标的plan
def get_cogagent_prompt(output_box):
    prompt = COGAGENT_BOX_PROMPT if output_box else random.choice(cogagent_templates) + "(with grounding)"
    
    return prompt

# QWen-VL: https://huggingface.co/Qwen/Qwen-VL-Chat

QWENVL_BOX_PROMPT = """What are the bounding box coordinates of the element corresponding to the functionality "{}" in this UI screenshot?"""
def get_qwenvl_prompt(output_box):
    prompt = QWENVL_BOX_PROMPT if output_box else POSITION_PROMPT
        
    return prompt

# SeeClick: https://github.com/njucckevin/SeeClick
SEECLICK_POINT_PROMPT = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with point)?"

SEECLICK_BOX_PROMPT = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with bbox)?"

def get_seeclick_prompt(output_box):
    prompt = SEECLICK_BOX_PROMPT if output_box else SEECLICK_POINT_PROMPT
        
    return prompt

def get_paligemma_prompt(output_box):
    prompt = "detect the region targeted by the command \"{}\""
        
    return prompt

def get_default_prompt(output_box):
    prompt = random.choice(web_loca_all_point_prompt)
    if output_box: prompt = prompt.replace("with point", "with bbox")
    
    return prompt

def apply_vlm_template(task_instruction, model_name, output_box=False):
    model_name = model_name.lower()
    if 'llava' in model_name:
        # Llava-1.6 Example: "[INST] <image>\nIn this UI screenshot, what is the position of the element corresponding to the command "{}"? [/INST]"
        prompt = get_llava_prompt(is_v16='1.6' in model_name, output_box=output_box)
    elif 'autogui_plus' in model_name:
        prompt = get_default_prompt(output_box)
        elem_desc = ' This element is used for "{}"' if not task_instruction.startswith("This element") else ' {}'
        prompt = prompt + elem_desc
    elif 'deepseek' in model_name:
        # DeepSeek-vl Example: '<image_placeholder>In this UI screenshot, what is the position of the element corresponding to the command "{question}"?'
        prompt = get_deepseek_prompt(output_box=output_box)
    elif 'cogagent' in model_name:
        # CogAgent-chat Example: "Can you advise me on how to {}?"
        prompt = get_cogagent_prompt(output_box=output_box)
    elif 'qwen' in model_name or 'slime' in model_name:
        # Qwen-VL Example: "What are the bounding box coordinates of the element corresponding to the command \"{}\" in this UI screenshot?"
        prompt = get_qwenvl_prompt(output_box=output_box)
    elif 'monkey' in model_name:
        # Qwen-VL Example: "What are the bounding box coordinates of the element corresponding to the command \"{}\" in this UI screenshot?"
        prompt = get_qwenvl_prompt(output_box=True)
    elif 'seeclick' in model_name:
        # SeeClick Example: "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with bbox)?"
        prompt = get_seeclick_prompt(output_box=output_box)
    elif 'pali' in model_name:
        # SeeClick Example: "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with bbox)?"
        prompt = get_paligemma_prompt(output_box=output_box)
    else:
        prompt = get_default_prompt(output_box)
        elem_desc = ' This element is used for "{}"' if not task_instruction.startswith("This element") else ' {}'
        prompt = prompt + elem_desc

    return prompt.format(task_instruction)

FUNC_CAP_PROMPT = "Describe the function of the element at {} on the screen."