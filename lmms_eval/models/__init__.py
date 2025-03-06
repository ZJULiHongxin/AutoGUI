import os, traceback

AVAILABLE_MODELS = {
    "uipro_qwen2_vl": "UIPRO_Qwen2_VL",
    "uipro_qwen2_vl_planning": "UIPRO_Qwen2_VL_Planning",
    "autogui": "AutoGUI",
    "autogui_qwen2_vl": "Autogui_Qwen2_VL",
    'uipro': "UIPro",
    "uipro_llavaov": "UIProLlavaOneVision",
    "llava_model": "Llava",
    "llava_hf": "LlavaHf",
    "llava_sglang": "LlavaSglang",
    "llava_onevision": "LlavaOneVision",
    "qwen_vl": "Qwen_VL",
    "qwen_vl_chat": "Qwen_VL_Chat",
    "qwen2_vl": "Qwen2_VL",
    "qwen2_vl_cloud": "Qwen2_VL_Cloud",
    "fuyu": "Fuyu",
    "gpt4v": "GPT4V",
    "instructblip": "InstructBLIP",
    "minicpm_v": "MiniCPM_V",
    "cogagent_chat_hf": "CogAgentChatHf",
    "seeclick": "SeeClick",
    "internvl_chat": "InternVLChat",
    "slime": "SLIME",
    "monkey": "Monkey",
    "claude": "Claude",
    "tinyclick": "TinyClick",
    "ferretui": "FerretUI",
    "llama_vision": "LlamaVision",
    "uipro_florence2": "UIPro_Florence2",
    "uground": "UGround",
    "osatlas": "OSAtlas",
    "llava_ov": "LLaVAOV",
    "internvl2": "InternVL2",
    "florence2": "Florence2"
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except ImportError:
        traceback.print_exc()
        print('Invalid VLM model:', model_name, model_class)


import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
