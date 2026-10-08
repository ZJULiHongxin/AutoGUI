# autogui_anno/tests/test_llm_client.py
import types, pytest
from autogui_anno.config import ModelSpec
from autogui_anno.llm.client import LLMClient

SPEC = ModelSpec(name="m", provider="openai", base_url="https://x/v1",
                 api_key_env="FAKE_KEY", model="gpt-5-mini")

def _fake_response(texts, p=10, c=5):
    choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=t)) for t in texts]
    usage = types.SimpleNamespace(prompt_tokens=p, completion_tokens=c)
    return types.SimpleNamespace(choices=choices, usage=usage)

class _FakeOpenAI:
    def __init__(self, scripted):
        self._scripted = scripted
        self.calls = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))
    def _create(self, **kw):
        r = self._scripted[min(self.calls, len(self._scripted)-1)]
        self.calls += 1
        if isinstance(r, Exception):
            raise r
        return r

def test_query_returns_texts_and_counts(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "k")
    fake = _FakeOpenAI([_fake_response(["a", "b"], p=10, c=5)])
    cli = LLMClient(SPEC, client_factory=lambda: fake)
    out = cli.query([{"role": "user", "content": "hi"}], repeat=2)
    assert out == ["a", "b"]
    assert cli.prompt_tokens == 10
    assert cli.completion_tokens == 5
    assert cli.query_count == 1

def test_query_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "k")
    fake = _FakeOpenAI([RuntimeError("transient"), _fake_response(["ok"])])
    cli = LLMClient(SPEC, client_factory=lambda: fake, max_retries=3)
    assert cli.query([{"role": "user", "content": "hi"}]) == ["ok"]

def test_query_raises_after_exhaustion(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "k")
    fake = _FakeOpenAI([RuntimeError("boom")])
    cli = LLMClient(SPEC, client_factory=lambda: fake, max_retries=2)
    with pytest.raises(RuntimeError):
        cli.query([{"role": "user", "content": "hi"}])
