from autogui_anno.config import PipelineConfig
from autogui_anno.stages.reject import reject_sample


class FakeLLM:
    def __init__(self, resps): self._resps = resps; self.query_count = 0
    def query(self, messages, **kw): self.query_count += 1; return self._resps
    def num_tokens(self, s): return len(s.split())


def test_reject_sample_zero_diff_returns_none(tmp_path):
    cfg = PipelineConfig()
    out = reject_sample(
        task_id=1, action_str="clicking a <button> named \"X\"",
        llm=FakeLLM(["<score>1"]), exp_dir=str(tmp_path),
        content_before=["RootWebArea 'x'"], content_after=["RootWebArea 'x'"],
        diff_info={"diff": [], "num_diff_lines": 0, "num_added_lines": 0, "num_deleted_lines": 0},
        config=cfg, resume=False,
    )
    assert out is None


def test_reject_sample_scores_valid_diff(tmp_path):
    cfg = PipelineConfig()
    diff_info = {"diff": ["Added link 'New'"], "num_diff_lines": 5,
                 "num_added_lines": 1, "num_deleted_lines": 1}
    out = reject_sample(
        task_id=2, action_str="clicking a <button> named \"X\"",
        llm=FakeLLM(["reasoning <score>1"]), exp_dir=str(tmp_path),
        content_before=["RootWebArea 'x'", "a", "b"],
        content_after=["RootWebArea 'x'", "a", "c"],
        diff_info=diff_info, config=cfg, resume=False,
    )
    assert out == [1]
