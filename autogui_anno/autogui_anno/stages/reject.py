"""Reject (scoring) stage, ported from WebpageFunctionality/reject.py.

The source's module-level thresholds (DESC_PREDICTION_THRESHOLD, DIFF_LIMIT,
token count, REJECT_REPEAT, MAX_SCORE, ...) are threaded through ``PipelineConfig``.
The source's inline score-parse loop is replaced by ``stages.scoring.parse_scores``.
``llm.query_LLM(...)`` becomes ``llm.query(...)``. The navigation-change predicate
from reject.py:75-78 is extracted into ``is_navigation_change``.
"""
from __future__ import annotations

import os
import json

from ..config import PipelineConfig
from .scoring import parse_scores
from . import describe
from .. import prompts


def is_navigation_change(
    *,
    num_deleted_lines: int,
    num_added_lines: int,
    len_before: int,
    len_after: int,
    variation: str,
    first_line_before,
    first_line_after,
    config: PipelineConfig,
    llm,
) -> bool:
    """Whether the webpage variation should be treated as a navigation (page-jump).

    Extracted from reject.py:75-78. When True, the before/after pages are described
    and compared at the description level; otherwise the raw diff is scored directly.
    """
    return (
        num_deleted_lines / len_before >= config.desc_prediction_threshold
        and num_added_lines / len_after >= config.desc_prediction_threshold
        or num_deleted_lines > config.diff_limit
        or num_added_lines > config.diff_limit
        or llm.num_tokens(variation) > config.diff_token_limit
        or first_line_before != first_line_after
    )


def reject_sample(
    *,
    task_id: int,
    action_str: str,
    llm,
    exp_dir: str,
    content_before=None,
    content_after=None,
    diff_info=None,
    config: PipelineConfig,
    resume: bool = False,
) -> list | None:
    """Score whether a webpage variation is sufficient to predict functionality.

    Ported from reject.py:29-141. Returns the parsed score list, or ``None`` when the
    page is unchanged (zero diff lines). Writes ``{task_id}_reject.txt`` /
    ``{task_id}_diff.txt`` side-effect files.
    """
    rejection_result_file = os.path.join(exp_dir, f"{task_id}_reject.txt")
    # Load the rejection result if it exists
    if resume and os.path.exists(rejection_result_file):
        try:
            with open(rejection_result_file, "r") as f:
                rejection_result = f.read()
                scores = []
                equal_idx = 0
                while True:
                    equal_idx = rejection_result.find("= ", equal_idx + 1)
                    if equal_idx == -1:
                        break
                    scores.append(int(rejection_result[equal_idx + 1:rejection_result.find('\n', equal_idx)].strip('<>= ')))

                return scores
        except Exception:
            pass

    diff_file = os.path.join(exp_dir, f"{task_id}_diff.txt")

    if diff_info is not None:
        formated_diff = diff_info["diff"]
        num_diff_lines = diff_info["num_diff_lines"]
        num_added_lines = diff_info["num_added_lines"]
        num_deleted_lines = diff_info["num_deleted_lines"]
    elif os.path.exists(diff_file):
        with open(diff_file, "r") as f:
            diff_info = json.load(f)
            formated_diff = diff_info["diff"]
            num_diff_lines = diff_info["num_diff_lines"]
            num_added_lines = diff_info["num_added_lines"]
            num_deleted_lines = diff_info["num_deleted_lines"]
    else:
        raise ValueError(
            "reject_sample requires either diff_info or a pre-computed "
            f"{task_id}_diff.txt file in exp_dir"
        )

    variation = '\n'.join(formated_diff)

    if num_diff_lines > 0:
        if is_navigation_change(
            num_deleted_lines=num_deleted_lines,
            num_added_lines=num_added_lines,
            len_before=len(content_before),
            len_after=len(content_after),
            variation=variation,
            first_line_before=content_before[0],
            first_line_after=content_after[0],
            config=config,
            llm=llm,
        ):
            before_descriptions = describe.describe_webpage(
                llm=llm,
                saved_file=os.path.join(exp_dir, f"{task_id}_before_Desc.txt"),
                content=content_before,
                remove_hidden=True,
                desc_limit=config.desc_page_limit,
                add_tab_title=True,
                resume=resume,
            )

            after_descriptions = describe.describe_webpage(
                llm=llm,
                saved_file=os.path.join(exp_dir, f"{task_id}_after_Desc.txt"),
                content=content_after,
                remove_hidden=True,
                desc_limit=config.desc_page_limit,
                add_tab_title=True,
                resume=resume,
            )
            outcome = (
                f"Before the interaction, the current webpage description is:\n{before_descriptions[0]}"
                f"\n\nAfter {action_str}, we navigate to a new webpage whose description is:\n{after_descriptions[0]}"
            )
        else:
            outcome = f"After {action_str}, the webpage exhibits the following variations:\n{variation}"

        # Score the sample if the webpage variation is valid
        reject_prompt = prompts.make_reject_prompt(config.reject_max_score)
        prompt = [{'role': 'user', 'content': reject_prompt.format(
            target_element=action_str[action_str.find(' ') + 1:],
            outcome=outcome)}]

        resps = llm.query(prompt, do_sample=True, temperature=config.reject_temp, repeat=config.reject_repeat, stop=["</", "</score>"])

        scores = parse_scores(resps, repeat=config.reject_repeat, max_score=config.reject_max_score)

        if len(scores) == 0:
            print("Empty score from: ", resps[0])

        with open(diff_file, "w") as f:
            f.write(variation)

        with open(rejection_result_file, "w") as f:
            f.write('\n\n'.join(f"Prediction:\n{resp}" for resp in resps) + '\n\nScore list: {}'.format(','.join(map(str, scores))))
    else:
        scores = None

    return scores
