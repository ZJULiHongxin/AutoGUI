"""Verify stage (cycle-consistency checking).

Ported from ``WebpageFunctionality/cycle_consis_checking.py``:
- ``find_target_line``            <- source L20-28 (verbatim logic).
- ``verify_consistency``          <- ``check_cycle_consistency_score`` (source L133-264).
- ``verify_consistency_multi``    <- new majority-vote wrapper over several verifiers.

The dead grounding variant ``check_cycle_consistency`` (source L30-131) is NOT
ported (spec section 4).

Spec deviations applied here:
- #1 (MAX_SCORE NameError): the source's error-feedback branch references a bare
  module-level ``MAX_SCORE`` that only exists under ``__main__`` -- a latent
  ``NameError`` at import-as-library time. The inline score-parse loop and that
  feedback string are both replaced by ``stages.scoring.parse_verification_scores``,
  which takes ``verify_max_score=config.verify_max_score`` and returns the score
  histogram together with a properly-formatted error-feedback string.
- #2 (score scale): ``is_consistent = final_score == config.verify_max_score``
  (not the literal ``3``).
- #3 (resume short-circuit scale): the resume path's ``sum(scores)/len(scores) == 3``
  becomes ``== config.verify_max_score``.
- Bounded retry: the source's unbounded ``while True`` scoring-retry becomes a
  bounded ``for _ in range(config.cycle_check_repeat)`` loop that raises
  ``RuntimeError`` on exhaustion (consistent with annotate/reject/client).
- ``llm.query_LLM(...)`` becomes ``llm.query(...)``.
- ``config.cycle_check_line_limit`` provides the context window
  (source module global ``CYCLE_CHECK_LINE_LIMIT``) and ``config.cycle_check_repeat``
  the per-query sample count (source module global ``CYCLE_CHECK_REPEAT``).

``get_clean_func`` lazily loads spacy's ``en_core_web_sm`` model, which may be
absent. When it is, we fall back to the raw summary text so the stage stays usable
(and testable) without the model; with the model present behaviour is unchanged.
"""
from __future__ import annotations

import os

from ..axtree.prune import count_tab, CLICKABLE
from .scoring import parse_verification_scores
from .. import prompts


def find_target_line(before_content, elem_text):
    """Locate the first clickable line whose displayed text matches ``elem_text``.

    Ported from cycle_consis_checking.py:20-28.
    """
    for elem_line_id, line in enumerate(before_content):
        # Convert the doubled backslashes to their intended single-character representation
        line = line.encode().decode('unicode-escape')
        node_type = line[:line.find("'")].strip()
        if f"'{elem_text}'" in line and node_type in CLICKABLE:
            return elem_line_id
    else:
        return None


def _clean_func(func_pred_content):
    """Run ``prompts.get_clean_func``; fall back to the raw summary when spacy's
    model is unavailable (so the stage works without ``en_core_web_sm``)."""
    stripped = func_pred_content.strip()
    try:
        return prompts.get_clean_func(stripped)
    except Exception:
        # spacy model missing (or any cleaning failure): fall back to the summary text
        func_start = stripped.find(f"{prompts.SUMMARY_MARK}:")
        if func_start == -1:
            return stripped
        return stripped[stripped.find(':', func_start) + 1:].strip()


def verify_consistency(
    *,
    before_content: list,
    func_pred_content: str,
    elem_text: str,
    llm,
    result_file: str,
    elem_line_id: int = None,
    config,
    resume: bool = True,
) -> tuple:
    """Check whether a predicted functionality is cycle-consistent with the element.

    Ported from ``check_cycle_consistency_score`` (cycle_consis_checking.py:133-264),
    with spec deviations #1/#2/#3 and bounded retry applied. Returns
    ``(resps, scores, elem_line_id, candidate, is_consistent)``.
    """
    max_score = config.verify_max_score

    # Resume: reuse a previously-written checking result (spec deviation #3: scale by max_score)
    if resume and os.path.exists(result_file):
        try:
            with open(result_file, "r") as f:
                checking_result = f.read()
                scores = []
                equal_idx = 0
                while True:
                    equal_idx = checking_result.find("<score>", equal_idx + 1)
                    if equal_idx == -1:
                        break
                    scores.append(int(checking_result[equal_idx + 7:checking_result.find('\n', equal_idx)].strip('<>= ')))

                return 'success', scores, None, None, sum(scores) / len(scores) == max_score
        except Exception:
            pass

    # Locate the element in the AXTree
    if elem_line_id is None:
        elem_line_id = find_target_line(before_content, elem_text)

    if elem_line_id is None:
        return None, None, None, None, None

    # Extract the content window around the interacted element
    half_range = config.cycle_check_line_limit // 2
    start = max(0, elem_line_id - half_range)
    end = min(start + config.cycle_check_line_limit + 1, len(before_content))

    # Add the parent nodes (lines above the window with a shallower indentation)
    current_tab_cnt = count_tab(before_content[elem_line_id])
    i = elem_line_id - 1
    parents_lines = []

    while i >= 0:
        this_line_tab_cnt = count_tab(before_content[i])
        if this_line_tab_cnt < current_tab_cnt:
            current_tab_cnt = this_line_tab_cnt

            if i < start:
                parents_lines.insert(0, before_content[i])

        i -= 1

    minimum_tab_cnt = 0

    grounding_lines = parents_lines + before_content[start:end]
    grouned_content = [f'[{i}] {line[minimum_tab_cnt:]}' for i, line in enumerate(grounding_lines)]

    grounding_idx = elem_line_id - start + len(parents_lines)
    candidate = grouned_content[grounding_idx]

    # If the predicted functionality is not provided, load the saved one from a local path
    result_dir, sample_idx = os.path.dirname(result_file), os.path.basename(result_file).split("_")[0]
    if len(func_pred_content) == 0:
        with open(os.path.join(result_dir, f"{sample_idx}_func.txt"), "r") as f:
            func_pred_content = f.read()  # comprises the reasoning process and the final prediction

    elem_func = _clean_func(func_pred_content)

    # Reload the description of the webpage after interaction.
    # If the sample uses diff to predict functionalities, extract the reasoning process
    # to provide more information for checking cycle consistency.
    diff_file = os.path.join(result_dir, f"{sample_idx}_diff.txt")
    after_desc_file = os.path.join(result_dir, f"{sample_idx}_after_Desc.txt")
    if os.path.exists(diff_file) or not os.path.exists(after_desc_file):
        # Diff-based prediction: func_pred_content carries the reasoning process.
        # (Also the fallback when no after-description file was produced.)
        outcome_info = "After interacting with the candidate element, the webpage exhibits these changes:\n" + func_pred_content[func_pred_content.find(prompts.REASONING_MARK) + len(prompts.REASONING_MARK) + 1:func_pred_content.find(prompts.SUMMARY_MARK)].strip()
    else:
        with open(after_desc_file, "r") as f:
            desc = f.read()
            outcome_info = "After interacting with the candidate element, we navigate to a new webpage that contains these contents:\n" + desc[desc.find(":") + 1:desc.rfind('\n', 50, desc.rfind(prompts.DESCRIPTION_MARK))].strip()  # Do not include the final webpage description as it provides no more information gain

    # Start checking cycle-consistency
    verif_prompt = prompts.make_verif_prompt(max_score)
    prompt = [{'role': 'user', 'content': verif_prompt.format(content='\n'.join(grouned_content), candidate=candidate, functionality=elem_func, outcome_info=outcome_info)}]

    # Bounded scoring-retry (spec deviation #3 pattern): raise on exhaustion.
    scores = None
    query_cnt = 0
    for _ in range(config.cycle_check_repeat):
        resps = llm.query(prompt, do_sample=True, temperature=0.6, repeat=config.cycle_check_repeat, stop=["</score>"])
        query_cnt += 1

        # Spec deviation #1: parse scores + build error feedback via the shared helper,
        # routing the former bare MAX_SCORE through config.verify_max_score.
        scores, error_feedback = parse_verification_scores(resps, verify_max_score=max_score)

        if error_feedback:
            print(f"Invalid node label -> {resps[-1]}")
            if len(prompt) == 1:
                prompt.append({'role': 'assistant', 'content': resps[-1]})
                prompt.append({'role': 'user', 'content': error_feedback})
            else:
                prompt[-1]['content'] = error_feedback
            scores = None
            continue

        break

    if scores is None:
        raise RuntimeError(
            f"verify_consistency: LLM did not produce valid scores after "
            f"{config.cycle_check_repeat} attempts."
        )

    # Scoring Method 2: Strict mode (all scores must be the full score).
    # final_score = sum(i*scores[i]) / sum(scores)  (source strict form).
    final_score = sum(i * scores[i] for i in range(len(scores))) / sum(scores)

    grouned_content[grounding_idx] = "=> " + grouned_content[grounding_idx]

    with open(result_file, 'w') as f:
        f.write(f'Query:\nThis element {elem_func}\n\n' + '\n\n'.join(f"Predictions:\n{x}" for x in resps) + '\n\nFinal score: {}\n{}\n\nQuery count: {}'.format(final_score, '\n'.join(grouned_content), query_cnt))

    # Spec deviation #2: scale by config.verify_max_score, not the literal 3.
    is_consistent = final_score == max_score

    return resps, scores, elem_line_id, candidate, is_consistent


def verify_consistency_multi(*, verifiers: list, **kwargs) -> tuple:
    """Run ``verify_consistency`` once per verifier and MAJORITY-VOTE ``is_consistent``.

    Web passes a 1-element ``verifiers`` list; mobile passes 3. Returns the
    aggregated ``(resps, scores, elem_line_id, candidate, is_consistent)`` tuple:
    the non-voted fields (``resps``, ``scores``, ``elem_line_id``, ``candidate``)
    come from the FIRST verifier's result, and ``is_consistent`` is the majority
    vote across all verifiers.
    """
    results = [verify_consistency(llm=verifier, **kwargs) for verifier in verifiers]

    # If the first verifier short-circuits (element absent), propagate that result.
    first = results[0]
    if first[0] is None:
        return first

    votes = [bool(r[4]) for r in results if r[4] is not None]
    is_consistent = sum(votes) > len(votes) / 2 if votes else False

    resps, scores, elem_line_id, candidate, _ = first
    return resps, scores, elem_line_id, candidate, is_consistent
