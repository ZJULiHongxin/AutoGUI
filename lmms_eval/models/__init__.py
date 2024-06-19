import os

AVAILABLE_MODELS = {
    "ui_llava": "UI_Llava",
    "ui_llava_lora": "UI_Llava_Lora",
    "llava": "Llava",
    "llava_hf": "LlavaHf",
    "llava_sglang": "LlavaSglang",
    "qwen_vl": "Qwen_VL",
    "qwen_vl_chat": "Qwen_VL_Chat",
    "fuyu": "Fuyu",
    "gpt4v": "GPT4V",
    "instructblip": "InstructBLIP",
    "minicpm_v": "MiniCPM_V",
    "cogagent_chat_hf": "CogAgentChatHf",
    "seeclick": "SeeClick",
    "paligemma": "PALIGemma",
    "internvl_chat": "InternVLChat"
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except ImportError:
        print('Invalid VLM model:', model_name, model_class)


import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
