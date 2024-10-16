import os, traceback

AVAILABLE_MODELS = {
    "autogui": "AutoGUI",
    'uipro': "UIPro",
    "llava_model": "Llava",
    "llava_hf": "LlavaHf",
    "llava_sglang": "LlavaSglang",
    "qwen_vl": "Qwen_VL",
    "qwen_vl_chat": "Qwen_VL_Chat",
    "qwen2_vl": "Qwen2_VL",
    "qwen2_vl_alicloud": "Qwen2_VL_Alicoud",
    "fuyu": "Fuyu",
    "gpt4v": "GPT4V",
    "instructblip": "InstructBLIP",
    "minicpm_v": "MiniCPM_V",
    "cogagent_chat_hf": "CogAgentChatHf",
    "seeclick": "SeeClick",
    "internvl_chat": "InternVLChat",
    "slime": "SLIME",
    "monkey": "Monkey"
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except ImportError:
        traceback.print_exc()
        print('Invalid VLM model:', model_name, model_class)


import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
