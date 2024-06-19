import torch
import os
torch.backends.cuda.matmul.allow_tf32 = True
import re
import logging
import copy
from tqdm import tqdm
from datetime import timedelta

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
import warnings

warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from .model_utils.slice_logic import process_image

if torch.__version__ > "2.1.2":
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"

import sys
module_dir = os.path.expanduser('~/.cache/huggingface/modules')
sys.path.append(module_dir)
warnings.filterwarnings("ignore")

DEFAULT_IMAGE_TOKEN = "<image>"
POINT_START_TOKEN = "<point>"
POINT_TOKEN = "<point_token>"

IMAGE_TOKEN_INDEX = -200
POINT_TOKEN_INDEX = -300
SPLIT_TOKEN_INDEX = -400
TOKEN_INDEX_MAP = {DEFAULT_IMAGE_TOKEN: IMAGE_TOKEN_INDEX, POINT_TOKEN: POINT_TOKEN_INDEX}

# Default chat template: https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/llama-3-chat.jinja



import os
import transformers
import sys
sys.path.append("/data0/jingran/workspace/hongxin_li/ui_llava")
from peft import PeftConfig, get_peft_model
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    
    # Our custom args
    add_point_token: Optional[bool] = field(default=False)
    context_enhanced: Optional[str] = field(default="")
    use_slice_seps: Optional[bool] = field(default=True)



@dataclass
class DataArguments:
    data_path: str = field(default=None,
                        metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    use_oracle_neighbors: Optional[bool] = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="none")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    vit_lora: Optional[bool] = field(default=False)
    llm_lora: Optional[bool] = field(default=False)

def find_and_order_tokens(text, tokens):
    # Create a regular expression pattern to match all tokens
    token_pattern = '|'.join(re.escape(token) for token in tokens)
    
    # Find all matches using re.finditer which returns an iterator yielding Match objects
    matches = re.finditer(token_pattern, text)
    
    # Extract positions and matched tokens from Match objects
    positions = [(match.start(), match.group(0)) for match in matches]
    
    # Since matches are found in order, they are already sorted by position
    return positions

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, point_token_index=POINT_TOKEN_INDEX, split_token_index= SPLIT_TOKEN_INDEX, return_tensors=None):
    target_tokens = ["<point_token>", "<image>"]
    sorted_tokens = find_and_order_tokens(prompt, target_tokens) # [(position, token)]
    for target_token in target_tokens:
        prompt = prompt.replace(target_token, '<split>')
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<split>')]
    

    # def insert_separator(X, sep):
    #     return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    def insert_separator(X, sep):
        ele_l = []
        for index, sublist in enumerate(zip(X, [sep]*len(X))):
            for ele in sublist[:-1]:
                ele_l.append(ele)
            if index < len(sorted_tokens):
                token_name = sorted_tokens[index][1]
                ele_l.append( list(map(lambda x: TOKEN_INDEX_MAP[token_name], sublist[-1])) ) 
        return ele_l

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

@register_model("ui_llava_lora")
class UI_Llava_Lora(lmms):
    """
    Llava Model
    """
    # TODO:
    
    def __init__(
        self,
        pretrained: str = "llava-hf/llava-1.5-7b-hf",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        attn_implementation: Optional[str] = None,
        device_map: str = "",
        chat_template: Optional[str] = "",
        use_cache: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if getattr(accelerator, "num_processes", 0) > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        model_args = ModelArguments()
        data_args = DataArguments()
        training_args = TrainingArguments()

        ckpt_folder = "/data0/jingran/workspace/hongxin_li/ui_llava/checkpoints/use_uidata2ft_xtuner_v11-lora/checkpoint-300"
        model_args.pretrain_mm_mlp_adapter = None #"/data0/jingran/workspace/hongxin_li/ui_llava/checkpoints/use_uidata2ft_xtuner_v11-lora/mm_projector/checkpoint-300.bin"

        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=ckpt_folder,
            trust_remote_code=True
        )

        # tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     pretrained_model_name_or_path=ckpt_folder,
        #     trust_remote_code=True,
        #     padding_side="right",
        #     use_fast=True,
        # )
        self._model.get_model().initialize_vision_modules(
                    model_args=model_args,
                    fsdp=training_args.fsdp
                ) # vision tower + projector
        vision_tower = self._model.get_vision_tower()
        # vision_tower.vision_tower.load_adapter(ckpt_folder + "/adapter")


        # self._model = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=dtype, device_map=self.device_map, trust_remote_code=trust_remote_code)
        
        self._tokenizer = AutoTokenizer.from_pretrained(ckpt_folder, trust_remote_code=trust_remote_code)
        with open(os.path.join(os.path.dirname(__file__), "chat_templates/llama-3-chat.jinja"), "r") as f:
            chat_template = f.read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        self._tokenizer.chat_template = chat_template

        # self._image_processor = AutoProcessor.from_pretrained(pretrained, revision=revision, trust_remote_code=trust_remote_code)
        # Pad from left for batched generation: https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava#usage-tips
        # self._image_processor.tokenizer.padding_side = "left"
        # self._tokenizer = self._image_processor.tokenizer
        self._config = self._model.config
        
        batch_size = int(batch_size)
        assert batch_size == 1, f"Only batch_size=1 is supported, but {batch_size} is given!"
        self.batch_size_per_gpu = int(batch_size)
        self.chat_template = chat_template
        self.use_cache = use_cache
        if getattr(accelerator, "num_processes", 0) > 1 and device_map == "":
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
        elif getattr(accelerator, "num_processes", 0) == 1 and device_map == "auto":
            eval_logger.info(f"Using {getattr(accelerator, 'num_processes', 0)} devices with pipeline parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

        self.resp_start_token = self.tokenizer.encode("<|end_header_id|>", add_special_tokens=False)[0]

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for context, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
            image_tokens = " ".join(image_tokens)
            context = f"{image_tokens}\n{context}"
            # Apply chat template
            messages = [{"role": "user", "content": context}, {"role": "assistant", "content": continuation}]

            prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            prompt_and_continuation = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            formatted_contexts = [prompt]
            formatted_continuation = [prompt_and_continuation]
            model_inputs = self._image_processor(text=formatted_continuation, images=visuals).to(self._device, self.model.dtype)
            labels = model_inputs["input_ids"].clone()
            contxt_id = self._image_processor(text=formatted_contexts, return_tensors="pt")["input_ids"]
            labels[: len(contxt_id)] = -100

            if self.accelerator.is_main_process and doc_id % 100 == 0:
                eval_logger.info(f"Prompt for doc ID {doc_id}:\n\n{formatted_contexts[0]}\n")
                eval_logger.info(f"Prompt and continuation for doc ID {doc_id}:\n\n{formatted_continuation[0]}\n")

            with torch.inference_mode():
                outputs = self.model(**model_inputs, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = model_inputs["input_ids"][:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : model_inputs["input_ids"].shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def preprocess(self, images):
        image_batch_tensors = []
        best_ws = []
        for image in images:
            slices_and_image, best_w, _ = process_image(image) # A list of image slice tensors plus the original image, all resized to 3 x 336 x 336
            image_tensor = torch.cat(slices_and_image, dim = 0)
            image_batch_tensors.append(image_tensor)
            best_ws.append(best_w)

        # Pad each sample
        if all(x is not None and x.shape == image_batch_tensors[0].shape for x in image_batch_tensors):
            image_batch_tensors = torch.stack(image_batch_tensors)
        else:
            max_of_x = 24
            padded_x_tensors = []
            for x in image_batch_tensors:
                padding = torch.zeros(max_of_x - x.size(0), x.size(1), x.size(2), dtype=x.dtype)
                # 在第一个维度上堆叠填充
                padded_x_tensor = torch.cat((padding, x), dim=0)
                padded_x_tensors.append(padded_x_tensor)

            image_batch_tensors = torch.stack(padded_x_tensors)
        
        return image_batch_tensors, best_ws


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
            visuals = self.flatten(visuals) # Already PIL.Image

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            context = contexts[0]

            # Some benchmarks like MME do not contain image tokens, so we prepend them to the prompt.
            if DEFAULT_IMAGE_TOKEN not in context:
                context = f"{DEFAULT_IMAGE_TOKEN}\n{context}"
            # Apply chat template
            messages = [{"role": "user", "content": context}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if hasattr(self, "accelerator") and self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.info(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            # text_splits = text.split(DEFAULT_IMAGE_TOKEN)
            # text_input_ids = []
            # text_attn_masks = []
            # for text_split in text_splits:
            #     if len(text_input_ids) > 0:
            #         text_input_ids.append(torch.full((1,1), IMAGE_TOKEN_INDEX, dtype=torch.int64))
            #         text_attn_masks.append(torch.ones((1,1), dtype=torch.int64))

            #     text_split_ids, text_split_attn_mask = self.tokenizer(text_split, return_tensors="pt").values()
            #     text_input_ids.append(text_split_ids)
            #     text_attn_masks.append(text_split_attn_mask)
            # # Debug: print(self.tokenizer.decode(text_input_ids.where((text_input_ids != IMAGE_TOKEN_INDEX) & (text_input_ids != POINT_TOKEN_INDEX), torch.tensor(10))))
            # text_input_ids = torch.cat(text_input_ids, dim=1)
            # text_attn_masks = torch.cat(text_attn_masks, dim=1)
            #inputs = self.tokenizer(text, return_tensors="pt")#.to(self._device, self.model.dtype)
            points = torch.stack([torch.tensor([[int(x), int(y)] for x, y in re.findall(r"<point>\((\d+),(\d+)\)</point>", text)])], dim=0)
            
            if points.shape[1] > 0:
                text = text.replace("</point>", f"</point>{POINT_TOKEN}")
            text_input_ids = tokenizer_image_token(text, self.tokenizer, image_token_index=IMAGE_TOKEN_INDEX, point_token_index=POINT_TOKEN_INDEX, split_token_index=SPLIT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            # print(self.tokenizer.decode(text_input_ids[0].where((text_input_ids[0] != IMAGE_TOKEN_INDEX) & (text_input_ids[0] != POINT_TOKEN_INDEX), torch.tensor(10))))
            text_attn_masks = torch.ones((1, text_input_ids.shape[1]))

            origin_image_widths, origin_image_heights, gen_kwargs["image_sizes"] = [], [], []
            for image in visuals:
                width, height = image.size
                gen_kwargs["image_sizes"].append(image.size)
                origin_image_widths.append(width)
                origin_image_heights.append(height)

            processed_imags, best_ws = self.preprocess(visuals)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 20
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            resps_ids_batch = self.model.generate(
                input_ids=text_input_ids.to(self.device),
                attention_mask=text_attn_masks.to(self.device, self.model.dtype),
                images = processed_imags,
                points=points,
                origin_image_widths=origin_image_widths,
                origin_image_heights=origin_image_heights,
                best_w=best_ws,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            # except Exception as e:
            #     eval_logger.error(f"Error {e} in generating")
            #     cont = ""
            for resps_ids in resps_ids_batch:
                assistant_resp_start = len(resps_ids) - 1
                while assistant_resp_start >= 0:
                    if resps_ids[assistant_resp_start] == self.resp_start_token: break
                    assistant_resp_start -= 1

            # text_outputs = self.tokenizer.batch_decode(resps_ids, skip_special_tokens=True)[0]
            text_outputs = self.tokenizer.decode(resps_ids[assistant_resp_start + 1:], skip_special_tokens=True)

            if hasattr(self, "accelerator") and self.accelerator.is_main_process and doc_id[0] % 1 == 0:
                eval_logger.info(f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n")

            print(text_outputs)
            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
