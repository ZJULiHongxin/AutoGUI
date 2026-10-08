# autogui_anno/tests/test_verify_stage.py
from autogui_anno.config import PipelineConfig
from autogui_anno.stages.verify import find_target_line, verify_consistency

def test_find_target_line_locates_clickable():
    before = ["RootWebArea 'x'", "button 'Save'", "StaticText 'hi'"]
    assert find_target_line(before, "Save") == 1

def test_find_target_line_absent_returns_none():
    assert find_target_line(["RootWebArea 'x'"], "Nope") is None

class FakeLLM:
    def __init__(self, resps): self._resps = resps; self.query_count = 0
    def query(self, messages, **kw): self.query_count += 1; return self._resps

def test_verify_consistency_full_score_is_consistent(tmp_path):
    cfg = PipelineConfig()  # verify_max_score=3, cycle_check_repeat=3
    before = ["RootWebArea 'x'", "button 'Save'", "StaticText 'hi'"]
    resps, scores, line_id, cand, is_consistent = verify_consistency(
        before_content=before,
        func_pred_content="Reasoning: foo\n\nSummary: This element saves the file.",
        elem_text="Save", llm=FakeLLM(["<score>3", "<score>3", "<score>3"]),
        result_file=str(tmp_path / "1_cycle.txt"), elem_line_id=1,
        config=cfg, resume=False,
    )
    assert is_consistent is True

def test_verify_consistency_mixed_score_not_consistent(tmp_path):
    cfg = PipelineConfig()
    before = ["RootWebArea 'x'", "button 'Save'", "StaticText 'hi'"]
    resps, scores, line_id, cand, is_consistent = verify_consistency(
        before_content=before,
        func_pred_content="Reasoning: foo\n\nSummary: This element saves the file.",
        elem_text="Save", llm=FakeLLM(["<score>3", "<score>1", "<score>2"]),
        result_file=str(tmp_path / "2_cycle.txt"), elem_line_id=1,
        config=cfg, resume=False,
    )
    assert is_consistent is False
