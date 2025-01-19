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
from lmms_eval.utils import resize_image
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import AutoPeftModelForCausalLM

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

@register_model("uipro_qwen2_vl_planning")
class UIPRO_Qwen2_VL_Planning(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        resize: int = -1,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.pretrained = pretrained
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        print(Fore.YELLOW + f"Loading a full model from {pretrained}" + Style.RESET_ALL)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained, device_map=self._device
        )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        except:
            self._tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', trust_remote_code=trust_remote_code)

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=256*28*28, max_pixels=1344*28*28)
        self.resize = resize
        self._config = self._model.config
        # self.model.tie_weights()
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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            query = []
            visual_paths = []
            for visual in visuals:
                name = uuid.uuid4().hex.upper()[0:6]
                visual.save(f"/tmp/{name}.png")
                visual_paths.append(f"/tmp/{name}.png")
                query.append({"image": f"/tmp/{name}.png"})

            # Make a copy for query to save context (text that needs to be masked)
            context_query = [_ for _ in query]
            context_query.append({"text": contexts})
            query.append({"text": contexts + continuation})

            context_query = self.tokenizer.from_list_format(context_query)
            query = self.tokenizer.from_list_format(query)

            raw_contxt_text, context_tokens = make_context(
                self.tokenizer, context_query, history=None, system="You are a helpful assistant", max_window_size=self.model.generation_config.max_window_size, chat_format=self.model.generation_config.chat_format
            )
            context_tokens = torch.tensor([context_tokens])

            raw_continuation_text, continuation_tokens = make_context(
                self.tokenizer, query, history=None, system="You are a helpful assistant", max_window_size=self.model.generation_config.max_window_size, chat_format=self.model.generation_config.chat_format
            )
            continuation_tokens = torch.tensor([continuation_tokens]).to(self.model.device)
            attn_mask = torch.ones_like(continuation_tokens).to(self.model.device)
            labels = continuation_tokens.clone().to(self.model.device)
            labels[:, : context_tokens.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=continuation_tokens, labels=labels, attention_mask=attn_mask)
            loss = outputs.loss
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = continuation_tokens[:, context_tokens.shape[1] :]
            greedy_tokens = greedy_tokens[:, context_tokens.shape[1] : continuation_tokens.shape[1]]  # [1, seq]
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

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc=f"uipro_qwen2_vl Responding (resize: {self.resize})")
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
            visual_paths = []
            # save images to /tmp, name generated by hash function
            # qwen accept image path. Have to do it here....
            for visual in visuals:
                name = uuid.uuid4().hex.upper()[0:6]
                
                if self.resize > 28:
                    #visual = resize_image(visual, max_size=self.resize)
                    visual = visual.resize((self.resize,self.resize))

                visual.save(f"/tmp/{name}.png")
                visual_paths.append(f"/tmp/{name}.png")

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            # until = [self.tokenizer.decode(self.eot_token_id)]

            # # Update values from gen_kwargs if present
            # if "until" in gen_kwargs:
            #     until = gen_kwargs.pop("until")
            #     if isinstance(until, str):
            #         until = [until]
            #     elif not isinstance(until, list):
            #         raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            content = []
            for visual_path, context in zip(visual_paths, contexts):
                content.append({"type": "image", "image": visual_path})
                content.append({"type": "text", "text": context})
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                # videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 32
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen_kwargs["max_new_tokens"],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"]
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            if (hasattr(self, "accelerator") and self.accelerator.is_main_process or not hasattr(self, "accelerator") is None) and doc_id[0] % 5 == 0:
                print(f"Generated text for doc ID {doc_id[0]}:")
                print(Fore.CYAN + f"prompt: {context}")
                print(Fore.YELLOW + f"response:{output_text}\n" + Style.RESET_ALL)

            res.append({"prompt": context, "response": output_text.strip()})

            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), output_text)
            # remove visuals from tmp
            for visual_path in visual_paths:
                try:
                    os.remove(visual_path)
                except:
                    pass
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
