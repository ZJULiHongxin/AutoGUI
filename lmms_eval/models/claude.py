from copy import deepcopy
import os
from typing import List, Tuple
from tqdm import tqdm
import requests as url_requests
import time
import logging
import http.client
import json
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils

from PIL import Image
from colorama import Fore, Style

API_TYPE = os.getenv("API_TYPE", "openai")
NUM_SECONDS_TO_SLEEP = 5
eval_logger = logging.getLogger("lmms-eval")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://xiaoai.plus/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }


@register_model("claude")
class Claude(lmms):
    def __init__(
        self,
        model_version: str = "claude-3-5-sonnet-20240620",
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.image_token = "<image>"

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
            prompt = contexts
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            imgs = []
            for visual in visuals:
                img = utils.encode_image(utils.resize_image(visual, max_size=1280))
                imgs.append(img)

            payload = {"model": self.model_version, "messages": []}
            response_json = {"role": "user", "content": []}
            # When there is no image token in the context, append the image to the text
            if self.image_token not in contexts:
                payload["messages"].append(deepcopy(response_json))
                payload["messages"][0]["content"].append({"type": "text", "text": contexts})
                for img in imgs:
                    payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
            else:
                contexts = contexts.split(self.image_token)
                for idx, img in enumerate(imgs):
                    payload["messages"].append(deepcopy(response_json))
                    payload["messages"][idx]["content"].append({"type": "text", "text": contexts[idx]})
                    payload["messages"][idx]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

                # If n image tokens are in the contexts
                # contexts will be splitted into n+1 chunks
                # Manually add it into the payload
                payload["messages"].append(deepcopy(response_json))
                payload["messages"][-1]["content"].append({"type": "text", "text": contexts[-1]})

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
                    conn = http.client.HTTPSConnection("xiaoai.plus")
                    headers = {
                    'Authorization': f'Bearer {API_KEY}',
                    'Content-Type': 'application/json'
                    }
                    conn.request("POST", "/v1/chat/completions", json.dumps(payload), headers)
                    response = conn.getresponse()
                    response_raw = response.read()
                    response_data = eval(response_raw.decode("utf-8"))

                    content = response_data["choices"][0]["message"]["content"].strip()
                    assert 'any' not in content
                    break  # If successful, break out of the loop

                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        content = ""
            
            if doc_id % 2 == 0:
                print(f"Generated text for doc ID {doc_id}:")
                print(Fore.CYAN + f"prompt: {prompt}")
                print(Fore.YELLOW + f"response:{content}\n" + Style.RESET_ALL)

            res.append({'prompt': prompt, 'response': content})
            pbar.update(1)
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"
