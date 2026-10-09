# autogui_anno/tests/test_viz_build.py
import json
import os

from autogui_anno.config import PipelineConfig
from autogui_anno.viz.build import build_sample_record, STAGE_EXPLAINERS
from autogui_anno.viz import build as build_mod

CFG = PipelineConfig()
META = {"action_str": "clicking a <svg> element", "elem_type": "svg",
        "elem_text": "", "elem_id": 9999, "target_line": "[9] button 'x'",
        "target_line_idx": 5}
BEFORE = ["[1] RootWebArea 'x'", "[9] button 'x'"]
AFTER = ["[1] RootWebArea 'x'", "[9] button 'x'", "[30] menu 'opts'"]

def test_explainers_cover_all_stages():
    assert set(STAGE_EXPLAINERS) == {"input", "diff", "reject", "annotate", "verify", "result"}
    assert all(isinstance(v, str) and v for v in STAGE_EXPLAINERS.values())

def test_record_full_pipeline_kept():
    rec = build_sample_record(
        meta=META, before=BEFORE, after=AFTER,
        diff_result={"lines": ["Added [30] menu 'opts'"], "num_added": 1, "num_deleted": 0},
        reject_result={"scores": [1], "max_score": 1, "kept": True, "reasoning": "r"},
        annotate_result={"functionality": "Opens the options menu.", "mode": "diff", "reasoning": "a"},
        verify_result={"scores": [0, 0, 0, 3], "final_score": 3.0, "max_score": 3,
                       "consistent": True, "candidate": "[9] button 'x'", "reasoning": "v"},
        config=CFG,
    )
    assert rec["schema_version"] == 1
    assert rec["verdict"] == "kept"
    assert rec["result"]["functionality"] == "Opens the options menu."
    assert rec["input"]["explainer"] == STAGE_EXPLAINERS["input"]
    for k in ["input", "diff", "reject", "annotate", "verify", "result"]:
        assert k in rec

def test_record_early_exit_no_change_has_null_later_stages():
    rec = build_sample_record(
        meta=META, before=BEFORE, after=BEFORE,
        diff_result={"lines": [], "num_added": 0, "num_deleted": 0},
        reject_result=None, annotate_result=None, verify_result=None, config=CFG,
    )
    assert rec["verdict"] == "no_change"
    assert rec["reject"] is None and rec["annotate"] is None and rec["verify"] is None
    assert rec["result"]["functionality"] is None
    assert rec["result"]["has_ground_truth"] is False

def test_record_rejected_stops_before_annotate():
    rec = build_sample_record(
        meta=META, before=BEFORE, after=AFTER,
        diff_result={"lines": ["Added [30] menu 'opts'"], "num_added": 1, "num_deleted": 0},
        reject_result={"scores": [0], "max_score": 1, "kept": False, "reasoning": "r"},
        annotate_result=None, verify_result=None, config=CFG,
    )
    assert rec["verdict"] == "rejected"
    assert rec["reject"]["kept"] is False
    assert rec["annotate"] is None


def _write_no_change_dataset(root, dataset):
    """Create a `<dataset>/` dir + sibling `<dataset>.json` holding one sample
    whose before == after, so run_builder takes the network-free no_change path."""
    d = os.path.join(root, dataset)
    os.makedirs(d, exist_ok=True)
    tree = "[1] RootWebArea 'x'\n[9] button 'x'\n"
    with open(os.path.join(d, "0_before.txt"), "w") as f:
        f.write(tree)
    with open(os.path.join(d, "0_after.txt"), "w") as f:
        f.write(tree)
    with open(d + ".json", "w") as f:
        json.dump([{"sample_id": 0, "type": "button", "text": "x",
                    "action_str": "clicking", "gt": None}], f)


def test_run_builder_dataset_filter_accepts_list(tmp_path):
    # Two datasets on disk; a LIST filter selects only one. Regression test for
    # the --datasets (nargs='+') -> dataset_filter=list case: a scalar `!=`
    # comparison silently excluded every dataset. before==after keeps it offline.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_no_change_dataset(str(data_dir), "Mind2Web_v1")
    _write_no_change_dataset(str(data_dir), "Mind2Web_v2")
    out_dir = tmp_path / "out"

    summary = build_mod.run_builder(
        data_dir=str(data_dir), out_dir=str(out_dir), config=CFG,
        llm=None, verifiers=[], dataset_filter=["Mind2Web_v1"], resume=False,
    )

    assert summary["built"] == 1  # only the included dataset's sample was built
    idx = json.load(open(os.path.join(str(out_dir), "index.json")))
    datasets = {e["dataset"] for e in idx}
    assert datasets == {"Mind2Web_v1"}  # Mind2Web_v2 excluded by the list filter


def test_run_builder_dataset_filter_accepts_str(tmp_path):
    # The scalar-string form must keep working (back-compat with any caller
    # that passes a single dataset name rather than a list).
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_no_change_dataset(str(data_dir), "Mind2Web_v1")
    _write_no_change_dataset(str(data_dir), "Mind2Web_v2")
    out_dir = tmp_path / "out"

    summary = build_mod.run_builder(
        data_dir=str(data_dir), out_dir=str(out_dir), config=CFG,
        llm=None, verifiers=[], dataset_filter="Mind2Web_v2", resume=False,
    )

    assert summary["built"] == 1
    idx = json.load(open(os.path.join(str(out_dir), "index.json")))
    assert {e["dataset"] for e in idx} == {"Mind2Web_v2"}
