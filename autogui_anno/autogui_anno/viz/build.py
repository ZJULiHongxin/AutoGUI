"""Visualizer record builder: pure record assembly + an impure driver.

The pipeline's stages (``reject.reject_sample``, ``annotate.predict_functionality``,
``verify.verify_consistency``) return raw shapes (score lists, prediction tuples,
``(resps, scores, line_id, candidate, is_consistent)`` tuples). The viewer, however,
wants one flat, self-describing record per sample. ``run_builder`` runs the real
stages and reshapes their raw returns into the viewer-schema dicts documented below;
``build_sample_record`` then assembles those dicts into the final record. Keeping the
assembly pure (no LLM, no I/O) is what makes the three early-exit cases trivially
testable and guarantees that an early-exit record is COMPLETE and null-filled rather
than half-written (Review Focus #6).
"""
from __future__ import annotations

import difflib
import glob
import json
import os

from ..axtree.diff import format_diff
from ..config import PipelineConfig
from .. import prompts

# Plain-English, one-paragraph blurb per stage, baked into every record so the
# viewer (Task 20) needs no knowledge of the pipeline internals. Keys are exactly
# the six stage keys the record carries.
STAGE_EXPLAINERS: dict[str, str] = {
    "input": (
        "The pipeline starts from two accessibility trees: a structured text "
        "snapshot of the page before the interaction and another of the page "
        "after it. Each line describes one on-screen element — its role (button, "
        "link, menu, and so on) and its visible text — so the whole screen is "
        "captured as readable text rather than pixels."
    ),
    "diff": (
        "The before and after trees are compared line by line to find what the "
        "interaction actually changed. Added lines are elements that appeared, "
        "deleted lines are elements that went away, and the counts summarize how "
        "big the change was. If nothing changed, the interaction told us nothing "
        "about the element and the sample stops here."
    ),
    "reject": (
        "A language model scores whether the observed change is informative enough "
        "to describe the element's purpose. Trivial or noisy changes score low and "
        "the sample is rejected; a clear, meaningful change is kept and moves on to "
        "annotation. This filter keeps low-quality samples out of the dataset."
    ),
    "annotate": (
        "A language model reads the change and writes a short, high-level "
        "description of what the interacted element does — for example, 'opens the "
        "options menu'. It works either from the line-by-line diff (manipulation "
        "mode) or from a description of the whole page when the interaction "
        "navigated somewhere new (describe mode)."
    ),
    "verify": (
        "To guard against hallucinated descriptions, a second language-model pass "
        "checks consistency: given the proposed functionality, could it point back "
        "to the element that was actually clicked? Each check is scored, and only a "
        "full-confidence agreement marks the description as consistent. Anything "
        "less is flagged as inconsistent."
    ),
    "result": (
        "The final outcome for this sample: the functionality description the "
        "pipeline settled on (when it produced one), alongside the dataset's "
        "ground-truth label when one exists, so a reader can compare the two. The "
        "top-level verdict records where the sample ended up — kept, rejected, "
        "unchanged, uncheckable, or inconsistent."
    ),
}


def build_sample_record(
    *,
    meta: dict,
    before: list,
    after: list,
    diff_result,
    reject_result,
    annotate_result,
    verify_result,
    config: PipelineConfig,
) -> dict:
    """Assemble one viewer record from already-computed stage results.

    PURE: no LLM calls, no file I/O. Every stage key is always present; a stage
    whose result argument is ``None`` (because the pipeline exited early before
    reaching it) has its record value set to ``None``. The top-level ``verdict``
    reflects where the pipeline stopped.

    The ``*_result`` arguments carry the viewer-schema shapes (see the module
    docstring); ``run_builder`` reshapes the raw stage returns into these before
    calling this function.
    """
    sample_id = meta.get("sample_id")
    elem_type = meta.get("elem_type", "")

    # --- Derive the top-level verdict (order matters: earliest exit wins) ---
    diff_empty = diff_result is not None and not diff_result.get("lines")
    if diff_empty:
        verdict = "no_change"
    elif meta.get("target_line") is None:
        verdict = "uncheckable"
    elif reject_result is not None and not reject_result.get("kept"):
        verdict = "rejected"
    elif verify_result is not None and not verify_result.get("consistent"):
        verdict = "inconsistent"
    else:
        verdict = "kept"

    # --- meta block (viewer reads a fixed set of keys) ---
    meta_block = {
        "action_str": meta.get("action_str", ""),
        "elem_type": elem_type,
        "elem_text": meta.get("elem_text", ""),
        "elem_id": meta.get("elem_id"),
        "target_line": meta.get("target_line"),
        "target_line_idx": meta.get("target_line_idx"),
    }

    # --- input block (always present) ---
    input_block = {
        "before": list(before),
        "after": list(after),
        "explainer": STAGE_EXPLAINERS["input"],
    }

    # --- diff block (None only if its arg is None) ---
    if diff_result is None:
        diff_block = None
    else:
        diff_block = {
            "lines": diff_result.get("lines", []),
            "num_added": diff_result.get("num_added", 0),
            "num_deleted": diff_result.get("num_deleted", 0),
            "explainer": STAGE_EXPLAINERS["diff"],
        }

    # --- reject block ---
    if reject_result is None:
        reject_block = None
    else:
        reject_block = {
            "scores": reject_result.get("scores", []),
            "max_score": reject_result.get("max_score"),
            "kept": reject_result.get("kept"),
            "reasoning": reject_result.get("reasoning", ""),
            "explainer": STAGE_EXPLAINERS["reject"],
        }

    # --- annotate block ---
    if annotate_result is None:
        annotate_block = None
        functionality = None
    else:
        functionality = annotate_result.get("functionality")
        annotate_block = {
            "functionality": functionality,
            "mode": annotate_result.get("mode"),
            "reasoning": annotate_result.get("reasoning", ""),
            "explainer": STAGE_EXPLAINERS["annotate"],
        }

    # --- verify block ---
    if verify_result is None:
        verify_block = None
    else:
        verify_block = {
            "scores": verify_result.get("scores", []),
            "final_score": verify_result.get("final_score"),
            "max_score": verify_result.get("max_score"),
            "consistent": verify_result.get("consistent"),
            "candidate": verify_result.get("candidate"),
            "reasoning": verify_result.get("reasoning", ""),
            "explainer": STAGE_EXPLAINERS["verify"],
        }

    # --- result block (always present) ---
    ground_truth = meta.get("gt")
    result_block = {
        "functionality": functionality,
        "ground_truth": ground_truth,
        "has_ground_truth": bool(ground_truth),
        "explainer": STAGE_EXPLAINERS["result"],
    }

    return {
        "schema_version": 1,
        "dataset": meta.get("dataset", ""),
        "sample_id": sample_id,
        "label": f"#{sample_id} {elem_type}",
        "meta": meta_block,
        "input": input_block,
        "diff": diff_block,
        "reject": reject_block,
        "annotate": annotate_block,
        "verify": verify_block,
        "result": result_block,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Impure driver
# ---------------------------------------------------------------------------

def _parse_elem_type(line_type: str, action_str: str) -> str:
    """Best-effort element-type label from the metadata ``type`` or action string."""
    if line_type:
        return str(line_type)
    # Fall back to the <role> embedded in action strings like
    # 'clicking a <button> element named "x"'.
    start = action_str.find("<")
    end = action_str.find(">")
    if 0 <= start < end:
        return action_str[start + 1:end]
    return ""


def _summary_from_prediction(pred: str) -> str:
    """Extract the clean functionality summary from an annotate prediction string.

    Falls back to the text after the ``Summary:`` mark when spacy-based cleaning
    (``prompts.get_clean_func``) is unavailable, mirroring verify._clean_func.
    """
    stripped = (pred or "").strip()
    try:
        return prompts.get_clean_func(stripped)
    except Exception:
        func_start = stripped.find(f"{prompts.SUMMARY_MARK}:")
        if func_start == -1:
            return stripped
        return stripped[stripped.find(":", func_start) + 1:].strip()


def _reasoning_from_prediction(pred: str) -> str:
    """Extract the reasoning section of an annotate prediction, if present."""
    stripped = (pred or "").strip()
    r = stripped.find(f"{prompts.REASONING_MARK}:")
    s = stripped.find(f"{prompts.SUMMARY_MARK}:")
    if r == -1:
        return ""
    if s != -1 and s > r:
        return stripped[r + len(prompts.REASONING_MARK) + 1:s].strip()
    return stripped[r + len(prompts.REASONING_MARK) + 1:].strip()


def _locate_target_line(before: list, elem_text: str, find_target_line) -> int | None:
    """Locate the interacted element line, tolerating ``[n]`` label prefixes.

    ``verify.find_target_line`` matches only when the node role (the text before
    the first quote) is a bare clickable role, so a leading ``[9] `` aria-id label
    hides the match. We retry on label-stripped lines so the locator works for both
    labeled (web, with markers) and unlabeled AXTree files. Returns the index into
    the ORIGINAL ``before`` list.
    """
    if not elem_text:
        return None
    idx = find_target_line(before, elem_text)
    if idx is not None:
        return idx
    from ..axtree.prune import remove_labels
    stripped = [remove_labels(ln).strip() for ln in before]
    return find_target_line(stripped, elem_text)


def run_builder(
    *,
    data_dir: str,
    out_dir: str,
    config: PipelineConfig,
    llm,
    verifiers: list,
    dataset_filter=None,
    resume: bool = True,
) -> dict:
    """Drive the pipeline over sample dirs and write per-sample viewer records.

    IMPURE. Discovers ``data_dir/Mind2Web_*/`` sample directories and their sibling
    ``<name>.json`` metadata list, joins each ``<n>_before.txt`` / ``<n>_after.txt``
    pair to the metadata entry with ``sample_id == n``, runs reject -> annotate ->
    verify honoring early exits, reshapes each raw stage return into the viewer
    schema, calls ``build_sample_record``, and writes:

    - ``out_dir/<dataset>_<sample_id>.json`` per sample, and
    - ``out_dir/index.json`` = ``[{dataset, sample_id, label, verdict, file}]``.

    Resumable: a sample whose output file already exists is skipped (``resume``).

    The ``llm`` and ``verifiers`` are passed in by the caller (built from the model
    registry); this driver never constructs a client with a hardcoded endpoint/key.

    Returns ``{"built": int, "skipped": int, "by_verdict": {verdict: count}}``.
    """
    # Imported here (not at module top) so the pure ``build_sample_record`` surface
    # carries no stage/LLM imports.
    from ..stages.reject import reject_sample
    from ..stages.annotate import predict_functionality
    from ..stages.verify import find_target_line, verify_consistency_multi

    os.makedirs(out_dir, exist_ok=True)

    built = 0
    skipped = 0
    by_verdict: dict[str, int] = {}
    index: list[dict] = []

    sample_dirs = sorted(glob.glob(os.path.join(data_dir, "Mind2Web_*")))
    sample_dirs = [d for d in sample_dirs if os.path.isdir(d)]

    # Normalize the dataset filter to a set of allowed names: callers may pass a
    # single dataset name (str) or a collection of names (the CLI's --datasets
    # yields a list). ``None`` means no filtering.
    if dataset_filter is None:
        allowed_datasets = None
    elif isinstance(dataset_filter, str):
        allowed_datasets = {dataset_filter}
    else:
        allowed_datasets = set(dataset_filter)

    for sample_dir in sample_dirs:
        dataset = os.path.basename(sample_dir)
        if allowed_datasets is not None and dataset not in allowed_datasets:
            continue

        meta_json = sample_dir + ".json"
        if not os.path.exists(meta_json):
            continue
        with open(meta_json, "r") as f:
            records = json.load(f)
        by_sample_id = {r["sample_id"]: r for r in records if "sample_id" in r}

        before_files = sorted(glob.glob(os.path.join(sample_dir, "*_before.txt")))
        for before_file in before_files:
            base = os.path.basename(before_file)
            n_str = base[: base.rfind("_before.txt")]
            try:
                sample_id = int(n_str)
            except ValueError:
                continue

            after_file = os.path.join(sample_dir, f"{n_str}_after.txt")
            if sample_id not in by_sample_id or not os.path.exists(after_file):
                continue

            out_file = os.path.join(out_dir, f"{dataset}_{sample_id}.json")
            rel_file = f"{dataset}_{sample_id}.json"

            entry = by_sample_id[sample_id]
            action_str = entry.get("action_str", "") or ""
            elem_type = _parse_elem_type(entry.get("type", ""), action_str)
            elem_text = entry.get("text", "") or ""

            if resume and os.path.exists(out_file):
                skipped += 1
                # Keep the index coherent on resume: reuse the saved verdict/label.
                try:
                    with open(out_file, "r") as f:
                        saved = json.load(f)
                    index.append({
                        "dataset": dataset, "sample_id": sample_id,
                        "label": saved.get("label", f"#{sample_id} {elem_type}"),
                        "verdict": saved.get("verdict"), "file": rel_file,
                    })
                except Exception:
                    pass
                continue

            with open(before_file, "r") as f:
                before = [ln.rstrip("\n") for ln in f]
            with open(after_file, "r") as f:
                after = [ln.rstrip("\n") for ln in f]

            exp_dir = os.path.join(out_dir, f"{dataset}_{sample_id}_work")
            os.makedirs(exp_dir, exist_ok=True)

            meta = {
                "dataset": dataset,
                "sample_id": sample_id,
                "action_str": action_str,
                "elem_type": elem_type,
                "elem_text": elem_text,
                "elem_id": entry.get("id"),
                "gt": entry.get("gt"),
                "target_line": None,
                "target_line_idx": None,
            }

            # --- diff ---
            diff = difflib.unified_diff(
                before, after,
                fromfile="original_webpage", tofile="new_webpage", n=config.diff_context,
            )
            formated_diff, num_diff_lines, num_added_lines, num_deleted_lines = format_diff(
                diff,
                diff_limit=config.diff_limit,
                diff_token_limit=config.diff_token_limit,
                use_additional_prefixes=True,
            )
            diff_info = {
                "diff": formated_diff,
                "num_diff_lines": num_diff_lines,
                "num_added_lines": num_added_lines,
                "num_deleted_lines": num_deleted_lines,
            }
            diff_result = {
                "lines": formated_diff,
                "num_added": num_added_lines,
                "num_deleted": num_deleted_lines,
            }

            reject_result = None
            annotate_result = None
            verify_result = None

            # Locate the interacted element line (lets verify run and makes the
            # verdict 'uncheckable' when the element is absent from the tree).
            elem_line_id = _locate_target_line(before, elem_text, find_target_line)
            if elem_line_id is not None:
                meta["target_line"] = before[elem_line_id]
                meta["target_line_idx"] = elem_line_id

            # --- reject (early exit: empty diff -> reject_sample returns None) ---
            if num_diff_lines > 0:
                scores = reject_sample(
                    task_id=sample_id,
                    action_str=action_str,
                    llm=llm,
                    exp_dir=exp_dir,
                    content_before=before,
                    content_after=after,
                    diff_info=diff_info,
                    config=config,
                    resume=resume,
                )
                if scores is not None:
                    # reject.py keeps a sample whose mean score clears half the max.
                    max_score = config.reject_max_score
                    kept = bool(scores) and (sum(scores) / len(scores)) >= (max_score / 2)
                    reject_result = {
                        "scores": scores,
                        "max_score": max_score,
                        "kept": kept,
                        "reasoning": "",
                    }

                    # --- annotate (only if kept) ---
                    if kept:
                        predictions, is_valid, no_change, is_nav = predict_functionality(
                            task_id=sample_id,
                            content_before=before,
                            content_after=after,
                            diff_info=diff_info,
                            action_str=action_str,
                            llm=llm,
                            exp_res_dir=exp_dir,
                            config=config,
                            repeat=1,
                            resume=resume,
                            layout="axtree",
                        )
                        if predictions is not None:
                            pred0 = predictions[0]
                            functionality = _summary_from_prediction(pred0)
                            annotate_result = {
                                "functionality": functionality,
                                "mode": "describe" if is_nav else "diff",
                                "reasoning": _reasoning_from_prediction(pred0),
                            }

                            # --- verify (only if annotated and locatable) ---
                            if elem_line_id is not None and verifiers:
                                resps, v_scores, _, candidate, is_consistent = verify_consistency_multi(
                                    verifiers=verifiers,
                                    before_content=before,
                                    func_pred_content=pred0,
                                    elem_text=elem_text,
                                    elem_line_id=elem_line_id,
                                    result_file=os.path.join(exp_dir, f"{sample_id}_cycle.txt"),
                                    config=config,
                                    resume=resume,
                                )
                                if resps is not None:
                                    v_scores = v_scores or []
                                    total = sum(v_scores)
                                    final_score = (
                                        sum(i * v_scores[i] for i in range(len(v_scores))) / total
                                        if total else 0.0
                                    )
                                    verify_result = {
                                        "scores": v_scores,
                                        "final_score": final_score,
                                        "max_score": config.verify_max_score,
                                        "consistent": bool(is_consistent),
                                        "candidate": candidate,
                                        "reasoning": "",
                                    }

            record = build_sample_record(
                meta=meta,
                before=before,
                after=after,
                diff_result=diff_result,
                reject_result=reject_result,
                annotate_result=annotate_result,
                verify_result=verify_result,
                config=config,
            )

            with open(out_file, "w") as f:
                json.dump(record, f, indent=2)

            built += 1
            verdict = record["verdict"]
            by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
            index.append({
                "dataset": dataset, "sample_id": sample_id,
                "label": record["label"], "verdict": verdict, "file": rel_file,
            })

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    return {"built": built, "skipped": skipped, "by_verdict": by_verdict}
