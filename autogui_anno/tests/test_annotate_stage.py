from autogui_anno.config import PipelineConfig
from autogui_anno.stages.annotate import predict_functionality

class FakeLLM:
    def __init__(self, resps): self._resps = resps; self.query_count = 0
    def query(self, messages, **kw): self.query_count += 1; return self._resps
    def num_tokens(self, s): return len(s.split())

def test_predict_functionality_no_change(tmp_path):
    cfg = PipelineConfig()
    preds, is_valid, no_change, is_nav = predict_functionality(
        task_id=1, content_before=["RootWebArea 'x'"],
        content_after=["RootWebArea 'x'"],
        diff_info={"diff": [], "num_diff_lines": 0, "num_added_lines": 0, "num_deleted_lines": 0},
        action_str="clicking a <button> named \"X\"",
        llm=FakeLLM(["Reasoning: ... Summary: This element ..."]),
        exp_res_dir=str(tmp_path), config=cfg, resume=False,
    )
    assert preds is None and no_change is True and is_valid is False

def test_predict_functionality_manipulation(tmp_path):
    cfg = PipelineConfig()
    diff_info = {"diff": ["Added link 'New'"], "num_diff_lines": 5,
                 "num_added_lines": 1, "num_deleted_lines": 1}
    preds, is_valid, no_change, is_nav = predict_functionality(
        task_id=2, content_before=["RootWebArea 'x'", "a", "b"],
        content_after=["RootWebArea 'x'", "a", "c"], diff_info=diff_info,
        action_str="clicking a <button> named \"X\"",
        llm=FakeLLM(["Reasoning: foo Summary: This element adds a link."]),
        exp_res_dir=str(tmp_path), config=cfg, resume=False,
    )
    assert is_valid and not no_change and not is_nav
    assert preds and "Summary:" in preds[0]
