import torch
import logging
from tqdm import tqdm
from accelerate import Accelerator, DistributedType
from typing import List, Optional, Union, Tuple
from peft import AutoPeftModelForCausalLM, PeftModel
import os
from colorama import Fore, Style

import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import AutoPeftModelForCausalLM

import sys
sys.path.append("/data0/jingran/workspace/hongxin_li")
from seeclick_exp.pretrain.prompt_lib import apply_vlm_template
from seeclick_exp.autogui_model.modeling_autogui import AutoGUILMHeadModel
from seeclick_exp.autogui_model.configuration_autogui import AutoGUIConfig

class AutoGUI:
    """
    AutoGUI Model
    https://huggingface.co/WebAgent/AutoGUI-Qwen-v0.1-LoRA/blob/main/adapter_config.json
    """

    def __init__(
        self,
        pretrained: str = "WebAgent/AutoGUI-Qwen-v0.1-LoRA",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        self.model_name = 'autogui'
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device
        
        try:
            if not os.path.exists(os.path.join(pretrained, "adapter_config.json")): raise Exception()
            
            lora_cfg_pretrained = AutoGUIConfig.from_pretrained(pretrained)
            self._model = AutoGUILMHeadModel.from_pretrained(pretrained, low_cpu_mem_usage=True, config=lora_cfg_pretrained, device_map=self._device, trust_remote_code=trust_remote_code).eval()
            
            non_lora_module_path = os.path.join(pretrained, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_module_path):
                non_lora_trainables = torch.load(non_lora_module_path, map_location='cpu')
                
                # Correct param names of the inserted mdules
                new_non_lora_trainables = {}
                for name, param in non_lora_trainables.items():
                    new_name = name.replace('base_model.model.', '')
                    new_name = new_name.replace('modules_to_save.default', 'original_module')
                    new_non_lora_trainables[new_name] = param
                self._model.load_state_dict(new_non_lora_trainables, strict=False)
            
            self._model = PeftModel.from_pretrained(model=self._model, model_id=pretrained).eval()
            
            print(Fore.YELLOW + f"Loading from the LoRA adapter directory: {pretrained}" + Style.RESET_ALL)
        except:
            print(Fore.YELLOW + f"Loading a full model from {pretrained}" + Style.RESET_ALL)
            self._model = AutoModelForCausalLM.from_pretrained(pretrained, device_map=self._device, trust_remote_code=trust_remote_code).eval()
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        except:
            self._tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-VL-Chat', trust_remote_code=trust_remote_code)

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        self.prompt = "<img>{}</img>{}"
        self._config = self._model.config
        self.model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eod_id

    @property
    def max_length(self):
        return self._max_length

    # should be deleted since max_new_tokens is decided by gen_kwargs not a model property
    # @property
    # def max_new_tokens(self) -> int:
    #     return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate(self, prompt, img_path) -> List[str]:
        # prompt = apply_vlm_template(instruction, model_name='autogui', output_box=False)
        query = self.tokenizer.from_list_format([{'image': img_path},  # Either a local path or an url
                                            {'text': prompt}, ])
        
        response, history = self.model.chat(self.tokenizer, query=query, history=None, max_new_tokens=32)
        print(prompt)
        print(response)
        return response


import torch, os
import logging
from tqdm import tqdm
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
from transformers import LlavaForConditionalGeneration, AutoProcessor

import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from lmms_eval.models.slime_utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from lmms_eval.models.slime_utils.conversation import conv_templates, SeparatorStyle
from lmms_eval.models.slime_utils.model.builder import load_pretrained_model
from lmms_eval.models.slime_utils.utils import disable_torch_init
from lmms_eval.models.slime_utils.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

class SLIME:
    """
    Slime Model for Hugging Face Transformers: https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava

    Adapted from the InstructBLIP model in lmms_eval/models/instructblip.py

    Example usage:

    accelerate launch --num_processes=8 -m lmms_eval \
        --model llava_hf \
        --model_args pretrained=llava-hf/llava-1.5-7b-hf \
        --tasks mme \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples
    """

    def __init__(
        self,
        pretrained: str = 'yifanzhang114/SliME-Llama3-8B',
        model_base: Optional[str] = None,
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        device_map: str = "",
        chat_template: Optional[str] = None,
        use_cache: bool = True,
        topp: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        self.model_name = 'slime'
        model_name = get_model_name_from_path(pretrained)

        self._tokenizer, self._model, self._image_processor, context_len = load_pretrained_model(pretrained, model_base, model_name, use_flash_attn=True, topp=topp)

        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.chat_template = chat_template
        self.use_cache = use_cache
        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                print("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                print(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            print(f"Using {accelerator.num_processes} devices with pipeline parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            print(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

        if 'llama' in pretrained.lower():
            self.conv_mode = 'llama3'
        elif 'vicuna' in pretrained.lower():
            self.conv_mode = 'vicuna_v1'

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate(self, instruction, img_path) -> List[str]:
        # Some benchmarks like MME do not contain image tokens, so we prepend them to the prompt.
        if DEFAULT_IMAGE_TOKEN not in instruction:
            instruction = f"{DEFAULT_IMAGE_TOKEN}\n{instruction}"
        # Apply chat template

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # the n_dims of img_tensor must be 5 (including the bs dimention); otherwise, the text-guided sampler will not work.
        visuals = [Image.open(img_path)]
        img_sizes = [x.size for x in visuals]
        img_tensor = process_images(visuals, self._image_processor, self._model.config).to(dtype=self.model.dtype, device=self.model.device)
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device=self.model.device)

        
        cont = self.model.generate(
            input_ids,
            images=img_tensor,
            image_sizes=img_sizes,
            max_new_tokens=8,
            use_cache=self.use_cache,
            do_sample=False,
            temperature=0.0
        )

        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

        return text_outputs

import os, torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

class SeeClick:
    def __init__(self, model_path) -> None:
        self.model_name = 'seeclick'
        
        limit = "80GB"
        option = 0
        max_memory = [{0: limit}, {4: limit, 5: limit, 6: limit, 7: limit}][option] # , 2: limit, 3: limit
        
        self.torch_type = torch.bfloat16

        # tokenizer_path: "Qwen/Qwen-VL-Chat"
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            bf16=True,
            trust_remote_code=True,
            device_map="cuda",
            # load_in_4bit=True,
            max_memory=max_memory).eval()

        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    
    def generate(self, instruction, img_path, top_p=1.0, temperature=0.0, max_tokens=13):
        # max_length ( int , optional, defaults to 20) — The maximum length the generated tokens can have. Corresponds to the length of the input prompt + max_new_tokens

        # "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
        # "USER: <image>\nWhat is this?\nASSISTANT: Two cats lying on a bed!\nUSER: <image>\nAnd this?\nASSISTANT:",
                
        query = self.tokenizer.from_list_format([{'image': img_path},  # Either a local path or an url
                                        {'text': instruction}, ])
        with torch.no_grad():
            response, history = self.model.chat(self.tokenizer, query=query, history=None, do_sample=False, temperature=0.0)

        return response
