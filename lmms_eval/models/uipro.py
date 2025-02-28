import torch, random
import traceback
from tqdm import tqdm
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import StoppingCriteria
from uipro.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from uipro.conversation import conv_templates, SeparatorStyle
from uipro.model.builder import load_pretrained_model
from uipro.utils import disable_torch_init
from uipro.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from colorama import Fore, Style

@register_model("uipro")
class UIPro(lmms):
    def __init__(
        self,
        pretrained: str = '',
        model_base: Optional[str] = None,
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        max_new_tokens: Optional[int] = 32,
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

        model_name = get_model_name_from_path(pretrained)

        print(f"Loading model from {pretrained}")
        self._tokenizer, self._model, self._image_processor, context_len = load_pretrained_model(pretrained, model_base, model_name, use_flash_attn=True)

        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.chat_template = chat_template
        self.use_cache = use_cache
        self.max_new_tokens = max_new_tokens
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
        elif 'gemma' in pretrained.lower() or 'gem2' in pretrained.lower():
            self.conv_mode = 'gemma'

            self.model.generation_config.eos_token_id = 107 # '<end_of_turn>'
        elif 'qwen' in pretrained.lower():
            self.conv_mode = 'qwen2'

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
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals) # do not support bs > 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            #until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            # if "until" in gen_kwargs:
            #     until = gen_kwargs.pop("until")
            #     if isinstance(until, str):
            #         until = [until]
            #     elif not isinstance(until, list):
            #         raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            context = contexts[0]

            # Some benchmarks like MME do not contain image tokens, so we prepend them to the prompt.
            if DEFAULT_IMAGE_TOKEN not in context:
                context = f"{DEFAULT_IMAGE_TOKEN}\n{context}"
            # Apply chat template

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], context)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            if (hasattr(self, "accelerator") and self.accelerator.is_main_process or not hasattr(self, "accelerator") is None) and doc_id[0] % 100 == 0:
                print(f"Prompt for doc ID {doc_id[0]}:\n\n{prompt}\n")

            gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]

            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # if self.use_global_only:
            #     for i in range(len(visuals)):
            #         visuals[0] = visuals[i].resize((336,336))

            # the n_dims of img_tensor must be 5 (including the bs dimention); otherwise, the text-guided sampler will not work.
            img_tensor = process_images(visuals, self._image_processor, self._model.config).to(dtype=self.model.dtype, device=self.model.device)
            
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device=self.model.device)
            
            class EosListStoppingCriteria(StoppingCriteria):
                def __init__(self, eos_sequence = [235270]):
                    self.eos_sequence = eos_sequence

                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
                    return self.eos_sequence in last_ids

            if False: #random.random() <= 0.96:
                text_outputs = '{"action_type": "click", "target": (32,21)}'
            else:
                try:
                    # print(input_ids.device, self.model.device, img_tensor.device)
                    cont = self.model.generate(
                        input_ids,
                        images=img_tensor,
                        image_sizes=gen_kwargs["image_sizes"],
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=self.max_new_tokens,
                        use_cache=self.use_cache,
                       #stopping_criteria = [EosListStoppingCriteria()]
                    )
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error {e} in generating")
                    cont = ""
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

            if (hasattr(self, "accelerator") and self.accelerator.is_main_process or not hasattr(self, "accelerator") is None) and doc_id[0] % 2 == 0:
                print(f"Generated text for doc ID {doc_id[0]}:")
                print(Fore.CYAN + f"prompt: {context}")
                print(Fore.YELLOW + f"response:{text_outputs}\n" + Style.RESET_ALL)

            res.append({'prompt':context, 'response':text_outputs})
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

