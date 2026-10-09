"""Describe + navigation-prediction stage, ported from WebpageFunctionality/describe.py.

Spec deviation #5: the top-level ``import torch`` / ``torch.set_default_device("cuda")``
and the ``predict_with_webpage_descriptions`` transformers/local-model import path from
the source are DROPPED. The navigation prediction that the source performed via a local
transformers model is reimplemented inline using ``llm.query`` with ``prompts.PREDICT_WITH_DESCS``
-- the same OpenAI-compatible client, no local transformers, no torch, no CUDA.

``llm.query_LLM(...)`` from the source becomes ``llm.query(...)`` throughout.
"""
from __future__ import annotations

import os
import re

from ..axtree.web import NUMBERING_PATTERN
from ..axtree.prune import prune_static_text
from .. import prompts


def describe_webpage(
    llm,
    *,
    saved_file: str = '',
    content=None,
    content_file: str = '',
    remove_hidden: bool = False,
    repeat: int = 1,
    desc_limit: int = 150,
    add_tab_title: bool = True,
    resume: bool = True,
) -> list:
    """Describe a webpage hierarchically. Ported from describe.py:80-122."""
    if resume and saved_file and os.path.exists(saved_file):
        with open(saved_file, 'r') as f:
            desc = f.read()

        descriptions = desc[desc.find(':') + 1:desc.rfind("Webpage:\n")].strip().split('Prediction:\n')
    else:
        if content is None:
            with open(content_file, 'r') as f:
                content = f.read()

            content = re.sub(NUMBERING_PATTERN, '', content).split('\n')

            # Prune lengthy StaticText
            prune_static_text(content, line_limit=desc_limit, remove_hidden=remove_hidden)
        else:
            content = content[:desc_limit]

        webpage = '\n'.join(content)
        prompt = [{'role': 'user', 'content': prompts.DESCRIBING_PROMPT.format(content=webpage, exemplar='')}]

        n = 0
        while True:
            resps = llm.query(prompt, do_sample=True, temperature=0.4, repeat=repeat)
            n += 1
            valid = True
            for resp in resps:
                if resp.lower().count("overall functionality:") < 1:
                    valid = False
                    break

            if valid:
                break

        descriptions = ["{}\n{}".format(
            "Tab title:" + content[0][12:] if add_tab_title else '',
            resp
        ) for resp in resps]

        if saved_file:
            with open(saved_file, "w") as f:
                f.write('\n\n'.join(f"Prediction:\n{description}" for description in descriptions) + f'\n\nWebpage:\n{webpage}')

    return descriptions


def _predict_with_webpage_descriptions(before: list, after: list, action_str: str, llm, save_path: str) -> list:
    """Navigation prediction reimplemented inline (spec deviation #5).

    Replaces the source's ``predict_with_webpage_descriptions`` (which used a local
    transformers model) with an inline ``llm.query`` call against
    ``prompts.PREDICT_WITH_DESCS`` -- same OpenAI-compatible client, no local model.
    """
    predictions = []

    for before_desc, after_desc in zip(before, after):
        prompt = [{'role': 'user', 'content': prompts.PREDICT_WITH_DESCS.format(
            before=before_desc, after=after_desc, action_str=action_str, exemplar='')}]

        while True:
            resps = llm.query(prompt, temperature=1.0, repeat=1)

            if f"{prompts.SUMMARY_MARK}:" not in resps[0]:
                continue
            else:
                break

        predictions.append(resps[0])

    if save_path:
        with open(save_path, "w") as f:
            f.write(f"Action: {action_str}\n\n" + '\n\n'.join(f"Prediction:\n{pred}" for pred in predictions))

    return predictions


def describe_predict(
    task_id: int,
    *,
    before_file: str = '',
    action_str: str = '',
    llm=None,
    exp_dir: str = '',
    remove_hidden: bool = False,
    content_dict=None,
    repeat: int = 1,
    desc_limit: int = 150,
    add_tab_title: bool = True,
    resume: bool = True,
) -> list:
    """Describe before/after pages and predict the navigation functionality.

    Ported from describe.py:125-170. The source's local-model
    ``predict_with_webpage_descriptions`` is replaced by the inline
    ``_predict_with_webpage_descriptions`` reimplementation.
    """
    desc = {}

    func_pred_file = os.path.join(exp_dir, f"{task_id}_func.txt")

    predictions = None
    if resume and os.path.exists(func_pred_file):
        with open(func_pred_file, "r") as f:
            pred_raw = f.read()

        if "Prediction:" in pred_raw:
            print(f"Load and skip {func_pred_file}")
            predictions = pred_raw.split("Prediction:\n")[1:]

    if predictions is None:
        for i, key in enumerate(["before", "after"]):
            if content_dict is not None:
                content = content_dict[key][:desc_limit]
            else:
                content_file = before_file if i == 0 else before_file.replace("before", "after")
                with open(content_file, 'r') as f:
                    content = f.read()

                content = re.sub(NUMBERING_PATTERN, '', content).split('\n')

                # Prune lengthy StaticText
                prune_static_text(content, line_limit=desc_limit, remove_hidden=remove_hidden)

            descriptions = describe_webpage(
                llm=llm,
                saved_file=os.path.join(exp_dir, f"{task_id}_{key}_Desc.txt"),
                remove_hidden=remove_hidden,
                content=content,
                repeat=repeat,
                desc_limit=desc_limit,
                add_tab_title=add_tab_title,
                resume=resume,
            )

            desc[key] = descriptions

        predictions = _predict_with_webpage_descriptions(desc['before'], desc['after'], action_str, llm, func_pred_file)

    return predictions
