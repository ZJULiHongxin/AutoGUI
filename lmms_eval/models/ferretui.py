import torch
import logging
from tqdm import tqdm
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen.qwen_generate_utils import make_context
from accelerate import Accelerator, DistributedType
from typing import List, Optional, Union, Tuple
import uuid
import os
from colorama import Fore, Style

import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import AutoPeftModelForCausalLM
from functools import partial

from ferretui_utils.inference import inference_and_run
from ferretui_utils.conversation import conv_templates, SeparatorStyle
from ferretui_utils.mm_utils import tokenizer_image_token, process_images
from ferretui_utils.builder import load_pretrained_model

def get_model_name_from_path(model_path):
    if 'gemma' in model_path:
        return 'ferret_gemma'
    elif 'llama' or 'vicuna' in model_path:
        return 'ferret_llama'
    else:
        raise ValueError(f"No model matched for {model_path}")

@register_model("ferretui")
class FerretUI(lmms):
    def __init__(
        self,
        pretrained: str = "jadechoghari/Ferret-UI-Gemma2b",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        model_name = get_model_name_from_path(pretrained.lower())
        self.conv_mode = 'ferret_gemma_instruct' if 'gemma' in model_name else 'ferret_llama_3'
        self._tokenizer, self._model, self.image_processor, context_len = load_pretrained_model(pretrained, None, model_name)
        
        print(Fore.YELLOW + f"Loading a full model from {pretrained}" + Style.RESET_ALL)

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
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        pass

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            query = contexts[0]
            if "<image>" in query:
                query = query.split('\n')[1]
            query = "<image>\n" + query
            
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()
            
            img = visuals[0]
            image_size = img.size
            
            if self.model.config.image_aspect_ratio == "square_nocrop":
                image_tensor = self.image_processor.preprocess(img, return_tensors='pt', do_resize=True, 
                                                    do_center_crop=False, size=[336, 336])['pixel_values'][0]
            elif self.model.config.image_aspect_ratio == "anyres":
                image_process_func = partial(self.image_processor.preprocess, return_tensors='pt', do_resize=True, do_center_crop=False, size=[336, 336])
                image_tensor = process_images([img], self.image_processor, self.model.config, image_process_func=image_process_func)[0]
            else:
                image_tensor = process_images([img], self.image_processor, self.model.config)[0]

            image_tensor = image_tensor.to(device=self.device, dtype=self.model.dtype)
            # preconfigure gen_kwargs with defaults
            if "image_sizes" not in gen_kwargs:
                try:
                    gen_kwargs["image_sizes"] = [visuals[0].size]
                except:
                    gen_kwargs["image_sizes"] = None
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            with torch.inference_mode():
                self.model.orig_forward = self.model.forward
                self.model.forward = partial(
                    self.model.orig_forward,
                    region_masks=None
                )
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    region_masks=None,
                    image_sizes=[image_size],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=512,#gen_kwargs["max_new_tokens"],
                    use_cache=True)
                self.model.forward = self.model.orig_forward

                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                text_outputs = outputs.strip()

            if (hasattr(self, "accelerator") and self.accelerator.is_main_process or not hasattr(self, "accelerator") is None) and doc_id[0] % 2 == 0:
                print(f"Generated text for doc ID {doc_id[0]}:")
                print(Fore.CYAN + f"prompt: {query}")
                print(Fore.YELLOW + f"response:{text_outputs}\n" + Style.RESET_ALL)
                
            res.append({"prompt": query, "response": text_outputs})

            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
