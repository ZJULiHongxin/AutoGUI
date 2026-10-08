"""Web trajectory orchestrator + checkpoints + stats aggregation.

Ported from ``WebpageFunctionality/src/annotate_func.py``:
- ``run_web``          <- ``annotate_func`` (source L43-402): drives one dataset
  of trajectories through the reject -> annotate -> verify stages, locating the
  interacted element via tabbability markers and recording ``uncheckable`` reasons.
- ``aggregate_stats``  <- the overall summary math (source L309-397), MINUS the
  matplotlib plotting block (source L374-382, which moves to Task 16).
- ``_load_checkpoint`` / ``_save_checkpoint`` <- the inline resume try/except
  (source L64-85, L302-307), centralized.

Spec deviations / mappings applied here:
- The source's module globals become parameters / ``config`` fields:
    * output dir ``FUNC_PRED_DIR``          -> ``out_dir``
    * data dir ``TRAJ_DIR`` (``DATA_DIR``)  -> ``data_dir`` (``<data_dir>/traj``)
    * ``DO_REJECTING`` / ``DO_CYCLE_CHECKING`` -> the ``do_rejecting`` /
      ``do_cycle_checking`` parameters
    * ``RESUME``                            -> the ``resume`` parameter
    * ``REMOVE_HIDDEN`` / ``WITH_MARKERS``  -> ``config.remove_hidden`` /
      ``config.with_markers``
    * ``DIFF_CONTEXT`` / ``DIFF_LIMIT`` / ``DESC_PAGE_LIMIT`` -> ``config.*``
    * ``REJECT_TEMP`` / ``REJECT_REPEAT``   -> ``config.reject_temp`` /
      ``config.reject_repeat``
- Token/query counters: the source's ``llm.token_num[0]`` / ``llm.token_num[1]``
  / ``llm.query_cnt`` become ``llm.prompt_tokens`` / ``llm.completion_tokens`` /
  ``llm.query_count`` (the ported ``LLMClient`` property names).
- The source stat fields ``dataset_name`` / ``model_name`` (module globals) are
  dropped from the per-traj stats; the pipeline receives ``llm`` as a parameter
  and does not read model-registry globals.
- The matplotlib plotting block (source L374-382) and ``import matplotlib`` are
  NOT ported (moves to Task 16); ``aggregate_stats`` returns the plain stats.
- Chinese comments in the source are translated to English inline.

The stage calls use the ported keyword surfaces:
``reject.reject_sample``, ``annotate.predict_functionality``,
``verify.verify_consistency_multi`` (web passes a 1-element ``verifiers`` list).
"""
from __future__ import annotations

import difflib
import glob
import json
import os
import time
import traceback
from datetime import datetime

import numpy as np

from .axtree.diff import format_diff
from .axtree.prune import prune_static_text
from .axtree.web import process_axtree
from .config import PipelineConfig
from .llm.client import LLMClient
from .stages.annotate import predict_functionality
from .stages.reject import reject_sample
from .stages.verify import verify_consistency_multi


def _load_checkpoint(result_dir) -> dict | None:
    """Load a trajectory's saved checkpoint, or ``None`` if it is absent/broken.

    Centralizes the inline resume try/except from annotate_func.py:64-85 +
    the ``cycle_check_result.json`` read. Returns a dict with keys ``stats`` and
    ``cycle_checking_results`` on success, or ``None`` when the trajectory should
    be processed afresh (no checkpoint, incomplete directory, or a broken file).
    """
    stats_file = os.path.join(result_dir, "stats.json")
    if not (os.path.exists(stats_file) and os.path.isdir(result_dir)
            and len(os.listdir(result_dir)) > 2):
        return None
    try:
        with open(stats_file, "r") as f:
            stats = json.load(f)
        with open(os.path.join(result_dir, "cycle_check_result.json"), "r") as f:
            cycle_checking_results = json.load(f)
        return {"stats": stats, "cycle_checking_results": cycle_checking_results}
    except Exception as e:
        print(
            f"The stats file of this trajectory ({result_dir}) is broken because: "
            f"{' | '.join(str(a) for a in e.args)}\nDo it again!"
        )
        return None


def _save_checkpoint(result_dir, stats, cycle_results) -> None:
    """Persist a trajectory's checkpoint (annotate_func.py:302-307)."""
    with open(os.path.join(result_dir, "cycle_check_result.json"), "w") as f:
        json.dump(cycle_results, f, indent=2)
    with open(os.path.join(result_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def aggregate_stats(
    stats_list: dict,
    consis_dict: dict,
    *,
    do_rejecting: bool,
    do_cycle_checking: bool,
    start_time: float,
) -> dict:
    """Aggregate per-trajectory stats into the overall ``basic_stats`` dict.

    Ported from annotate_func.py:309-397 MINUS the matplotlib plotting block
    (source L374-382, moved to Task 16). Mutates ``consis_dict['summary']`` with
    the overall consistency rate (when cycle-checking is enabled) and returns the
    ``basic_stats`` dict. The ``all_rejection_scores`` / ``all_cycle_checking_results``
    inputs to the (dropped) plot are no longer needed, so the rejection ranking is
    reduced to the ordered sample list the plot consumed (still written as
    ``rejection_order`` by ``run_web``).
    """
    # Overall cycle-consistency rate
    if do_cycle_checking:
        consis_dict["summary"] = {
            "num_consis": sum(len(x["consis"]) for x in consis_dict["details"].values()),
            "num_inconsis": sum(len(x["inconsis"]) for x in consis_dict["details"].values()),
        }
        denom = consis_dict["summary"]["num_consis"] + consis_dict["summary"]["num_inconsis"]
        consis_dict["summary"]["consis_rate"] = (
            consis_dict["summary"]["num_consis"] / denom if denom else 0.0
        )

    # Percentage of valid / checkable tasks
    valid_tasks_each_traj = {k: x["valid_tasks"] for k, x in stats_list.items()}
    valid_nav_tasks_each_traj = {
        k: set(x["valid_tasks"]).intersection(set(x["nav"])) for k, x in stats_list.items()
    }
    valid_manip_tasks_each_traj = {
        k: set(x["valid_tasks"]) - set(x["nav"]) for k, x in stats_list.items()
    }
    checkable_tasks_each_traj = {k: x["checkable_tasks"] for k, x in stats_list.items()}
    num_all_steps = sum(x["num_samples"] for x in stats_list.values())

    num_all_nav_steps = sum(len(x["nav"]) for x in stats_list.values())
    num_all_manip_steps = num_all_steps - num_all_nav_steps

    # Statistics of each failure type.
    # NOTE: Logical relation: a task being uncheckable implies its functionality
    # is not predictable.
    uncheckable_stats = {}
    if do_cycle_checking:
        for v in stats_list.values():
            for failure_type in v["invalid"]:
                if failure_type not in uncheckable_stats:
                    uncheckable_stats[failure_type] = 0
                uncheckable_stats[failure_type] += len(v["invalid"][failure_type])

    basic_stats = {
        "num_valid_steps": sum(len(x) for x in valid_tasks_each_traj.values()),
        "num_valid_nav_steps": sum(len(x) for x in valid_nav_tasks_each_traj.values()),
        "num_valid_manip_steps": sum(len(x) for x in valid_manip_tasks_each_traj.values()),
        "num_all_steps": num_all_steps,
        "num_all_nav_steps": num_all_nav_steps,
        "num_all_manip_steps": num_all_manip_steps,
        # If a valid task cannot go through cycle checking without raising
        # exceptions, it is uncheckable.
        "num_checkable_steps": sum(len(x) for x in checkable_tasks_each_traj.values()),
        "uncheckable_stats": uncheckable_stats,
        "time elapse": time.time() - start_time,
        "prompt_tokens": sum(x["prompt_tokens"] for x in stats_list.values()),
        "completion_tokens": sum(x["completion_tokens"] for x in stats_list.values()),
        "query_count": sum(x["query_count"] for x in stats_list.values()),
    }

    return basic_stats


def _rank_rejection_samples(all_rejection_scores: dict) -> list:
    """Rank all samples by their mean rejection score (ascending).

    Ported from annotate_func.py:338-345 (the ``samples_to_sort`` build). The
    plot that consumed the ranks (source L347-382) moves to Task 16.
    """
    samples_to_sort = []
    for traj_name, traj_steps_scores in all_rejection_scores.items():
        for step, score in traj_steps_scores.items():
            samples_to_sort.append([f"{traj_name}_{step}", np.mean(score).item()])
    samples_to_sort.sort(key=lambda x: x[1])
    return samples_to_sort


def run_web(
    config: PipelineConfig,
    llm: LLMClient,
    verifiers: list,
    *,
    data_dir: str,
    out_dir: str,
    resume: bool = False,
    do_rejecting: bool = True,
    do_cycle_checking: bool = True,
) -> dict:
    """Annotate the functionality of web elements across a set of trajectories.

    Ported from ``annotate_func`` (annotate_func.py:43-402). Scans
    ``<data_dir>/traj`` for trajectory directories (names starting with ``24-``),
    processes each step pair through reject -> annotate -> verify, records the six
    ``uncheckable`` reasons, writes per-trajectory ``stats.json`` /
    ``cycle_check_result.json`` files, and finally writes ``overall_stats.json``.
    Returns the overall stats dict ``{'basic_stats', 'cycle_consis_info',
    'rejection_order'}``.
    """
    traj_dir = os.path.join(data_dir, "traj")
    os.makedirs(out_dir, exist_ok=True)

    traj_names = [x for x in sorted(os.listdir(traj_dir)) if x.startswith("24-")]

    stats_list = {}
    consis_dict = {"details": {}, "summary": {}}
    all_rejection_scores = {}
    all_cycle_checking_results = {}

    solved, last_token_usage, last_query_cnt = 0, [0, 0], 0
    start = time.time()
    for traj_idx, traj_name in enumerate(traj_names):
        if "result" in traj_name:
            continue

        traj_path = os.path.join(traj_dir, traj_name)
        result_dir = os.path.join(out_dir, traj_name)

        # Load checkpoints
        if resume:
            ckpt = _load_checkpoint(result_dir)
            if ckpt is not None:
                stats = ckpt["stats"]
                cycle_checking_results = ckpt["cycle_checking_results"]

                stats_list[traj_name] = stats
                all_rejection_scores[traj_name] = stats["rejection_scores"]
                all_cycle_checking_results[traj_name] = cycle_checking_results
                consis_dict["details"][traj_name] = {
                    "consis": cycle_checking_results["result"]["consistent"],
                    "inconsis": cycle_checking_results["result"]["inconsistent"],
                }

                print("Skip trajectory:", traj_name)
                continue

        os.makedirs(result_dir, exist_ok=True)

        # Load meta files and axtree files
        meta_files = sorted(glob.glob(os.path.join(traj_path, "*_meta.json")))
        raw_axtree_files = sorted(glob.glob(os.path.join(traj_path, "*_axtree.txt")))
        num_states = len(meta_files)

        cycle_checking_results = {
            "traj_name": traj_name,
            "result": {"consistent": [], "inconsistent": [], "uncheckable": []},
        }

        scores_record = {}

        # Store annotating status
        checkable, valid_tasks, no_change_list, nav = [], [], [], []
        target_not_in_tree, blank_page, broken_marker_list, invalid_axtree = [], [], [], []

        axtree_lines_list, markers_list, nodes_list, steps = [], [], [], []
        error_info_list = []

        # Start processing axtrees
        for i, (meta_file, raw_axtree_file) in enumerate(zip(meta_files, raw_axtree_files)):
            step_id = int(
                raw_axtree_file[raw_axtree_file.rfind("step") + 4:raw_axtree_file.rfind("_act")]
            )
            steps.append(step_id)

            try:  # This try block catches errors caused by any invalid AXTree file
                with open(meta_file, "r") as f:
                    node_list = json.load(f)

                    axtree_lines, invalid_markers = process_axtree(
                        os.path.abspath(raw_axtree_file), resume=resume, node_list=node_list
                    )

                nodes_list.append(node_list)
                axtree_lines_list.append(axtree_lines)

                markers = prune_static_text(
                    axtree_lines,
                    remove_hidden=config.remove_hidden,
                    with_markers=config.with_markers,
                    menuitem_limit=config.menuitem_limit,
                    tab_limit=config.tab_limit,
                    text_max_len=config.text_max_len,
                )

                markers_list.append(
                    {"step": 0, "cnt": len(markers), "markers": markers,
                     "invalid_markers": invalid_markers}
                )
            except Exception:
                error_info = f"The AXtree of {raw_axtree_file} is not valid:\n{traceback.format_exc()}\n"
                print(error_info)
                error_info_list.append(error_info)

                nodes_list.append(None)
                axtree_lines_list.append(None)
                markers_list.append(None)

                invalid_axtree.append([step_id, error_info])
                continue

            if len(axtree_lines) == 0:
                blank_page.append(step_id)

        # Start predicting functionalities
        for i, step_id in enumerate(steps):
            if i == 0:
                continue
            solved += 1

            if axtree_lines_list[i - 1] is None or axtree_lines_list[i] is None or markers_list[i - 1] is None:
                cycle_checking_results["result"]["uncheckable"].append([step_id, "Invalid AXTree"])
                continue

            if steps[i] - steps[i - 1] != 1:
                error_info = f"step_id {step_id}: skipped due to the gap between two steps"
                print(error_info)
                error_info_list.append(error_info)
                cycle_checking_results["result"]["uncheckable"].append([step_id, "Incontinuous steps"])
                continue

            if steps[i] in blank_page or steps[i - 1] in blank_page:
                error_info = f"step_id {step_id}: skipped due to the blank page"
                print(error_info)
                error_info_list.append(error_info)
                cycle_checking_results["result"]["uncheckable"].append([step_id, "Blank page"])
                continue

            meta_file = meta_files[i]
            action_marker = meta_file[meta_file.rfind("_action") + 7:meta_file.rfind("_d")]

            if action_marker not in markers_list[i - 1]["markers"]:
                target_not_in_tree.append(step_id)
                error_info = (
                    f"Step id: {step_id}. The interacted element does not appear in the AXTree "
                    "string possibly because its ancesters are outside of the viewport and thus "
                    "excluded from the AXtree"
                )
                print(error_info)
                error_info_list.append(error_info)
                cycle_checking_results["result"]["uncheckable"].append([step_id, "Target not in AXTree"])
                continue

            for node_info in nodes_list[i - 1]:
                if node_info["hint_marker_text"] == action_marker:
                    break
            else:
                broken_marker_list.append(step_id)
                error_info = (
                    f"Step id: {step_id}. Cannot find the action marker {action_marker} in the "
                    f"meta file {meta_files[i - 1]}"
                )
                print(error_info)
                error_info_list.append(error_info)
                cycle_checking_results["result"]["uncheckable"].append([step_id, "Broken marker"])
                continue

            axtree_before, axtree_after = axtree_lines_list[i - 1], axtree_lines_list[i]

            elem_role = node_info["axtree_node"]["role"]["value"]
            elem_text = node_info["axtree_node"]["name"]["value"]

            action_str = f'clicking a <{elem_role}> element named "{elem_text}"'

            # Generate the unified diff
            diff = difflib.unified_diff(
                axtree_before, axtree_after,
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

            # Reject invalid samples
            if do_rejecting:
                scores = reject_sample(
                    task_id=step_id,
                    action_str=action_str,
                    llm=llm,
                    exp_dir=result_dir,
                    content_before=axtree_before,
                    content_after=axtree_after,
                    diff_info=diff_info,
                    config=config,
                    resume=resume,
                )
                if scores is None:
                    no_change_list.append(step_id)
                    cycle_checking_results["result"]["uncheckable"].append([step_id, "Page not changed"])
                    continue

                scores_record[step_id] = scores

            # Predict the function
            predictions, is_valid, no_change, is_nav = predict_functionality(
                task_id=step_id,
                content_before=axtree_before,
                content_after=axtree_after,
                diff_info=diff_info,
                action_str=action_str,
                llm=llm,
                exp_res_dir=result_dir,
                config=config,
                repeat=1,
                resume=resume,
                layout="axtree",
            )

            if predictions is None:
                no_change_list.append(step_id)
                cycle_checking_results["result"]["uncheckable"].append([step_id, "Page not changed"])
            else:
                func_pred_content = predictions[0]
                valid_tasks.append(step_id)

                if is_nav:
                    nav.append(step_id)

                # Cycle consistency checking
                if do_cycle_checking:
                    resps, scores, _, _, is_consistent = verify_consistency_multi(
                        verifiers=verifiers,
                        before_content=axtree_before,
                        func_pred_content=func_pred_content,
                        elem_text=elem_text,
                        # Supply the element line id when known to improve the
                        # accuracy of consistency checking.
                        elem_line_id=markers_list[i - 1]["markers"][action_marker]
                        if config.with_markers else None,
                        result_file=os.path.join(result_dir, f"{step_id}_cycle.txt"),
                        config=config,
                        resume=resume,
                    )

                    if resps is None:
                        target_not_in_tree.append(step_id)
                    else:
                        print(f"step_id {step_id}: cycle consis succeeds")

                        if is_consistent:
                            cycle_checking_results["result"]["consistent"].append(step_id)
                        else:
                            cycle_checking_results["result"]["inconsistent"].append(step_id)

                        checkable.append(step_id)

        num_consistent = len(cycle_checking_results["result"]["consistent"])

        if len(checkable) > 0:
            cycle_checking_results["result"]["consis_rate"] = (
                f"{num_consistent}/{len(checkable)}={num_consistent / len(checkable):.3f}"
            )

        consis_dict["details"][traj_name] = {
            "consis": cycle_checking_results["result"]["consistent"],
            "inconsis": cycle_checking_results["result"]["inconsistent"],
        }

        stats = {
            "traj_name": traj_name,
            "num_samples": num_states - 1,
            # A task is valid if we predict its functionality successfully.
            "valid_tasks": valid_tasks,
            # A task is checkable if it can go through the cycle consistency check.
            "checkable_tasks": checkable,
            "invalid": {
                "no_change": no_change_list,
                "blank_page": blank_page,
                "target_not_in_tree": target_not_in_tree,
                "broken_marker": broken_marker_list,
                "invalid_axtree": invalid_axtree,
            },
            "nav": nav,
            # The max number of lines to describe a page.
            "desc_page_limit": config.desc_page_limit,
            # The max number of lines in the diff.
            "diff_limit": config.diff_limit,
            # Should be greater than the #menuitem limit.
            "diff_context": config.diff_context,
            "markers": markers_list,
            "error_info_list": error_info_list,
            "rejection_scores": scores_record,
            "prompt_tokens": llm.prompt_tokens - last_token_usage[0],
            "completion_tokens": llm.completion_tokens - last_token_usage[1],
            "query_count": llm.query_count - last_query_cnt,
        }

        last_token_usage[0], last_token_usage[1], last_query_cnt = (
            llm.prompt_tokens, llm.completion_tokens, llm.query_count
        )
        all_rejection_scores[traj_name] = scores_record
        all_cycle_checking_results[traj_name] = cycle_checking_results

        stats_list[traj_name] = stats

        # Save the checking result + stats
        _save_checkpoint(result_dir, stats, cycle_checking_results)

    # Rank samples by rejection score (the ranking the Task-16 plot consumes)
    samples_to_sort = _rank_rejection_samples(all_rejection_scores) if do_rejecting else []

    basic_stats = aggregate_stats(
        stats_list,
        consis_dict,
        do_rejecting=do_rejecting,
        do_cycle_checking=do_cycle_checking,
        start_time=start,
    )

    overall_stats = {
        "basic_stats": basic_stats,
        "cycle_consis_info": consis_dict,
        "rejection_order": samples_to_sort,
    }

    with open(os.path.join(out_dir, "overall_stats.json"), "w") as f:
        json.dump(overall_stats, f, indent=2)

    print("All {} steps have been processed. Time: {}".format(solved, datetime.now()))

    return overall_stats
