import os, json
from autogui_anno.config import PipelineConfig
from autogui_anno.pipeline import run_web


class FakeLLM:
    """Canned reject/predict/verify responses; counts queries."""
    def __init__(self):
        self._prompt_tokens = 0; self._completion_tokens = 0; self._query_count = 0
    @property
    def prompt_tokens(self): return self._prompt_tokens
    @property
    def completion_tokens(self): return self._completion_tokens
    @property
    def query_count(self): return self._query_count
    def num_tokens(self, s): return len(s.split())
    def query(self, messages, **kw):
        self._query_count += 1
        text = messages[0]["content"].lower()
        if "score" in text and "candidate" in text:  # verify
            return ["<score>3"] * kw.get("repeat", 1)
        if "sufficient" in text:  # reject (first-line "sufficient for predicting")
            return ["<score>1"] * kw.get("repeat", 1)
        return ["Reasoning: foo\n\nSummary: This element adds a link."] * kw.get("repeat", 1)


def test_run_web_dryrun_produces_stats(tmp_path):
    cfg = PipelineConfig()
    data_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    out_dir = str(tmp_path / "out")
    llm = FakeLLM()
    stats = run_web(cfg, llm, [llm], data_dir=data_dir, out_dir=out_dir, resume=False)
    assert "basic_stats" in stats
    assert os.path.exists(os.path.join(out_dir, "overall_stats.json"))
    assert llm.query_count > 0
