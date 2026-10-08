import pytest
from autogui_anno.config import ModelSpec
from autogui_anno.llm.registry import resolve_api_key

def test_resolve_api_key_present(monkeypatch):
    monkeypatch.setenv("AUTOGUI_LLM_API_KEY", "secret-123")
    spec = ModelSpec(name="m", provider="openai",
                     base_url="https://x/v1", api_key_env="AUTOGUI_LLM_API_KEY",
                     model="gpt-5-mini")
    assert resolve_api_key(spec) == "secret-123"

def test_resolve_api_key_missing_raises_named(monkeypatch):
    monkeypatch.delenv("AUTOGUI_LLM_API_KEY", raising=False)
    spec = ModelSpec(name="m", provider="openai",
                     base_url="https://x/v1", api_key_env="AUTOGUI_LLM_API_KEY",
                     model="gpt-5-mini")
    with pytest.raises(EnvironmentError) as exc:
        resolve_api_key(spec)
    assert "AUTOGUI_LLM_API_KEY" in str(exc.value)
