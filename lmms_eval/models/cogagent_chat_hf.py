import torch
import logging
from tqdm import tqdm
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from PIL import Image
from lmms_eval.api.registry import register_model
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
from transformers import AutoModelForCausalLM, LlamaTokenizer
from colorama import Fore, Style
import warnings

warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")

@register_model("cogagent_chat_hf")
class CogAgentChatHf(lmms):
    """
    Cogagent Chat Model from Hugging Face # from https://huggingface.co/THUDM/cogagent-chat-hf
    
    Example usage:
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval \
        --model cogagent_chat_hf 
        --model_args pretrained=THUDM/cogagent-chat-hf,device_map='' 
        --tasks motif_bbox_test 
        --batch_size 1 
        --log_samples 
        --log_samples_suffix cogagent-chat-hf_motif_bbox_test 
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str = "THUDM/cogagent-chat-hf",
        revision: str = "main",
        device: str = "cuda:0",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        device_map: str = "",
        chat_template: Optional[str] = None,
        use_cache: bool = True,
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
        self._model = AutoModelForCausalLM.from_pretrained(pretrained, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True, device_map=self.device_map).eval()
        self._tokenizer =  LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
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
            eval_logger.info(f"Using {accelerator.num_processes} devices with pipeline parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
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
    def dtype(self):
        return self._model.dtype
    
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
        pass



    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    
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
        
        def collate_fn(features, tokenizer) -> dict:
            images = [feature.pop('images') for feature in features]
            tokenizer.padding_side = 'left'
            padded_features = tokenizer.pad(features)
            inputs = {**padded_features, 'images': images}
            return inputs
        

        def recur_move_to(item, tgt, criterion_func):
            if criterion_func(item):
                device_copy = item.to(tgt)
                return device_copy
            elif isinstance(item, list):
                return [recur_move_to(v, tgt, criterion_func) for v in item]
            elif isinstance(item, tuple):
                return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
            elif isinstance(item, dict):
                return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
            else:
                return item

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
            visuals = self.flatten(visuals)
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
                
            # batch infernece: https://github.com/THUDM/CogVLM/issues/143
            samples = []
            for visual, context in zip(visuals, contexts):
                sample = self.model.build_conversation_input_ids(self.tokenizer, query=context, history=[], images=[visual])
                samples.append(sample)
            inputs = collate_fn(samples, self.tokenizer)
            inputs = recur_move_to(inputs, 'cuda', lambda x: isinstance(x, torch.Tensor))
            inputs = recur_move_to(inputs, self.dtype, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))

            

            gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            try:
                cont = self.model.generate(
                    **inputs,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature = gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""
            
            outputs = cont[:, inputs['input_ids'].shape[1]:]
            for i in range(len(samples)):
                text_outputs = self.tokenizer.decode(outputs[i]).split("</s>")[0].strip()
                print(f"Generated text for doc ID {doc_id[i]}:")
                print(Fore.CYAN + f"prompt: {contexts[i]}")
                print(Fore.YELLOW + f"response:{text_outputs}\n" + Style.RESET_ALL)
                res.append({'prompt': contexts[i], 'response': text_outputs})
                self.cache_hook.add_partial("generate_until", (contexts[i], gen_kwargs), text_outputs)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res