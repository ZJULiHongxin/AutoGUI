from multiprocessing import context
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from datetime import timedelta
import logging

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState

from typing import Optional, Sequence, List, Tuple, Union
import re
from tqdm import tqdm

eval_logger = logging.getLogger("lmms-eval")

@register_model("internvl-chat")
class InternVLChat(lmms):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL-Chat-V1-5",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        device_map="",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        need_bos: bool = True,
        padding: bool = False,
        half: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.need_bos = need_bos
        self.padding = padding
        self._model = AutoModel.from_pretrained(self.pretrained, device_map=self.device_map, torch_dtype=dtype, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained, trust_remote_code=True)
        self.model.tokenizer = self.tokenizer
        self.batch_size_per_gpu = batch_size

        if accelerator.num_processes > 1:
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
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
    
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

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")
            context = contexts[0]
            assert len(contexts) == 1, "Only one context is supported for InternVL-Chat"
            
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 512
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "repetition_penalty" not in gen_kwargs:
                gen_kwargs["repetition_penalty"] = 1.0
            
            outputs = self.model.chat(
                self.tokenizer, 
                visuals, 
                context, 
                gen_kwargs, 
                history=None, 
                return_history=True
            )
            output_token = outputs[0]
            if output_token[0] == 0 or output_token[0] == 1:
                output_token = output_token[1:]
            output_text = self.model.tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split("[UNUSED_TOKEN_145]")[0].strip()
            output_text = output_text.split("<|im_end|>")[0].strip()
            # if DATASET_TYPE(task) == "multi-choice":
            #     output_text = pattern.findall(output_text)
            #     if len(output_text) == 0:
            #         print("Error:", output_text)
            #         output_text = "Z"
            #     if type(output_text) == list:
            #         output_text = output_text[0]
            res.append(output_text)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        return super().loglikelihood(requests)