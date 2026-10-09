"""Annotate stage, ported from WebpageFunctionality/one_stage_difflib.py.

Ports ``predict_with_diff`` (one_stage_difflib.py:32-60) and ``predict_func``
(one_stage_difflib.py:62-124, renamed ``predict_functionality``).

Spec deviations applied here:
- #3 (bounded retry): the source's ``while True: ... continue`` retry-until-
  ``SUMMARY_MARK`` loop in ``predict_with_diff`` becomes a bounded
  ``for _ in range(max_retries)`` loop that raises ``RuntimeError`` if the
  summary mark never appears.
- ``llm.query_LLM(...)`` from the source becomes ``llm.query(...)``.
- The module globals ``LAYOUT`` / ``PROMPT`` selection becomes the ``layout``
  parameter (default ``'axtree'`` -> ``PREDICT_PROMPT_AXTREE``).

Config field mapping (faithful to this source -- see task-11 CRITICAL note):
This file imports ``DIFF_LIMIT = 250`` from utils.tools and does NOT rebind it
(unlike reject.py which locally rebinds it to 150). Therefore:
- ``config.diff_limit`` (250) is used BOTH for ``format_diff(diff_limit=...)``
  AND for the deleted/added-line navigation clauses (``num_*_lines > 250``).
- ``config.nav_token_threshold`` (5500) is used for the token navigation clause
  (``num_tokens_from_string(variation) > 5500``).
- ``config.desc_prediction_threshold`` (0.8) is used for the ratio clause.
- ``config.desc_page_limit`` (150) is passed to ``describe_predict`` as its desc limit.
- ``config.diff_context`` (4) and ``config.diff_token_limit`` are threaded into
  ``difflib.unified_diff`` / ``format_diff``.
``config.nav_diff_line_limit`` (150) is NOT used here -- it is reject-stage-only.
"""
from __future__ import annotations

import difflib
import json
import os

from ..axtree.diff import format_diff, num_tokens_from_string
from .. import prompts
from .describe import describe_predict


def predict_with_diff(
    *,
    task_id,
    variation: str,
    action_str: str,
    llm,
    exp_dir: str,
    resume: bool = True,
    layout: str = 'axtree',
    max_retries: int = 10,
) -> list:
    """Query the LLM for element functionality from a formatted diff.

    Ported from one_stage_difflib.py:32-60. The unbounded retry-until-SUMMARY_MARK
    loop is replaced by a bounded loop that raises ``RuntimeError`` on exhaustion
    (spec deviation #3).
    """
    prompt_tmpl = prompts.PREDICT_PROMPT_SCHEMA if layout == 'schema' else prompts.PREDICT_PROMPT_AXTREE

    func_pred_file = os.path.join(exp_dir, f"{task_id}_func.txt")

    predictions = None
    if resume and os.path.exists(func_pred_file):
        with open(func_pred_file, "r") as f:
            pred_raw = f.read()

        if "Prediction:" in pred_raw:
            print(f"Load and skip {func_pred_file}")
            predictions = pred_raw.split("Prediction:\n")[1:]

    if predictions is None:
        prompt = [{'role': 'user', 'content': prompt_tmpl.format(exemplar='', action_str=action_str, variation=variation)}]

        predictions = None
        for _ in range(max_retries):
            resps = llm.query(prompt, do_sample=True, temperature=1.0, repeat=1)

            if f"{prompts.SUMMARY_MARK}:" not in resps[0]:
                continue
            predictions = resps
            break

        if predictions is None:
            raise RuntimeError(
                f"predict_with_diff: LLM did not produce a '{prompts.SUMMARY_MARK}:' "
                f"response for task {task_id} after {max_retries} attempts."
            )

        with open(func_pred_file, "w") as f:
            f.write(f"Action: {action_str}\n\n" + '\n\n'.join(f"Prediction:\n{pred}" for pred in predictions))

    return predictions


def predict_functionality(
    *,
    task_id,
    content_before: list,
    content_after: list,
    diff_info: dict = None,
    action_str: str = '',
    llm=None,
    exp_res_dir: str = '',
    config=None,
    repeat: int = 1,
    resume: bool = True,
    layout: str = 'axtree',
) -> tuple:
    """Predict element functionality from before/after webpage content.

    Ported from one_stage_difflib.py:62-124 (``predict_func``). Returns
    ``(predictions, is_valid, no_change, is_nav)``. The navigation branch
    delegates to ``describe.describe_predict``; the manipulation branch to
    ``predict_with_diff``.
    """
    # Read / compute the diff
    diff_file = os.path.join(exp_res_dir, f"{task_id}_diff.txt")

    if diff_info is not None:
        formated_diff = diff_info["diff"]
        num_diff_lines = diff_info["num_diff_lines"]
        num_added_lines = diff_info["num_added_lines"]
        num_deleted_lines = diff_info["num_deleted_lines"]
    elif os.path.exists(diff_file):
        with open(diff_file, "r") as f:
            saved = json.load(f)
        formated_diff = saved["diff"]
        num_diff_lines = saved["num_diff_lines"]
        num_added_lines = saved["num_added_lines"]
        num_deleted_lines = saved["num_deleted_lines"]
    else:
        # Generate the unified diff
        diff = difflib.unified_diff(
            content_before, content_after,
            fromfile='original_webpage', tofile='new_webpage',
            n=config.diff_context,
        )

        formated_diff, num_diff_lines, num_added_lines, num_deleted_lines = format_diff(
            diff,
            diff_limit=config.diff_limit,
            diff_token_limit=config.diff_token_limit,
            use_additional_prefixes=layout == 'axtree',
        )

        with open(diff_file, "w") as f:
            json.dump({
                "diff": formated_diff,
                "num_diff_lines": num_diff_lines,
                "num_added_lines": num_added_lines,
                "num_deleted_lines": num_deleted_lines,
            }, f)

    is_valid = True
    no_change = False
    is_nav = False

    if num_diff_lines > 0:
        variation = '\n'.join(formated_diff)
        if num_deleted_lines / len(content_before) >= config.desc_prediction_threshold \
            and num_added_lines / len(content_after) >= config.desc_prediction_threshold \
                or num_deleted_lines > config.diff_limit or num_added_lines > config.diff_limit \
                or num_tokens_from_string(variation) > config.nav_token_threshold \
                or content_before[0] != content_after[0]:
            predictions = describe_predict(
                task_id,
                before_file='',
                action_str=action_str,
                llm=llm,
                exp_dir=exp_res_dir,
                remove_hidden=config.remove_hidden,
                content_dict={"before": content_before, "after": content_after},
                desc_limit=config.desc_page_limit,
                add_tab_title=layout == 'axtree',
                repeat=repeat,
                resume=resume,
            )
            is_nav = True
        else:
            # Provide a hint about whether the interaction caused webpage navigation
            if content_before[0] == content_after[0]:
                if layout == 'axtree' and 'Unchanged RootWebArea' not in formated_diff[0]:
                    variation = f"Unchanged {content_before[0]}\n" + variation
                elif layout == 'schema' and 'Unchanged Tab title' not in formated_diff[0]:
                    variation = f"Unchanged {content_before[0]}\n" + variation

                variation = "After the interaction, we are still on the same webpage but its content has been altered.\n\n" + variation
            else:
                variation = "After the interaction, we probably jumped to a new webpage.\n\n" + variation

            predictions = predict_with_diff(
                task_id=task_id,
                variation=variation,
                action_str=action_str,
                llm=llm,
                exp_dir=exp_res_dir,
                resume=resume,
                layout=layout,
            )
    else:
        print(f"{task_id}: The webpage has no changes.")
        predictions = None
        is_valid = False
        no_change = True

    return predictions, is_valid, no_change, is_nav
