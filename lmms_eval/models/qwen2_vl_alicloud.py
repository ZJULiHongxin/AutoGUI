import  dashscope, time
import logging
from tqdm import tqdm
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen.qwen_generate_utils import make_context
from typing import List, Optional, Union, Tuple
from colorama import Fore, Style
from http import HTTPStatus
from lmms_eval.utils import resize_image, encode_image
from copy import deepcopy
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import AutoPeftModelForCausalLM

from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

@register_model("qwen2_vl_alicloud")
class Qwen2_VL_Alicoud(lmms):
    """
    Qwen2_VL Model
    https://github.com/QwenLM/Qwen2-VL
    """

    def __init__(
        self,
        model: str = "qwen-vl-max-0809",
        api_key: Optional[str] = "sk-",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        print(f"[Qwen2_VL_Alicoud] Unexpected kwargs: {kwargs}")

        self.model = model
        dashscope.api_key = api_key
        print(Fore.YELLOW + f"Querying {model} on AliCloud" + Style.RESET_ALL)

    def get_model_response(self, prompt: str, image_urls: List[str]) -> (bool, str):
        content = [{
            "text": prompt
        }]
        for image_url in image_urls:
            content.append({
                "image_url": image_url
            })
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        response = dashscope.MultiModalConversation.call(model=self.model, messages=messages, top_p=0.15)
        if response.status_code == HTTPStatus.OK:
            return True, response.output.choices[0].message.content[0]["text"]
        else:
            return False, response.message

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for prompt, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            
            # When there is no image token in the context, append the image to the text
            
            img_urls = []
            for visual in visuals:
                img = encode_image(resize_image(visual, max_size=1280))
                img_urls.append(f"data:image/jpeg;base64,{img}")

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 32
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # payload["max_tokens"] = gen_kwargs["max_new_tokens"]
            # payload["temperature"] = gen_kwargs["temperature"]

            for attempt in range(5):
                try:
                    status, content = self.get_model_response(prompt=prompt, image_urls=img_urls)

                    content = content.strip()
                    
                    if not status: raise Exception("too many requests!")
                    break  # If successful, break out of the loop

                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    time.sleep((attempt+1) * 4)
                    if attempt >= 4:  # If we have retries left, sleep and then continue to next attempt
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        content = "[0,0]"
            
            if doc_id % 2 == 0:
                print(f"Generated text for doc ID {doc_id}:")
                print(Fore.CYAN + f"prompt: {prompt}")
                print(Fore.YELLOW + f"response:{content}\n" + Style.RESET_ALL)

            res.append({'prompt': prompt, 'response': content})
            pbar.update(1)
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Qwen2_VL not support"