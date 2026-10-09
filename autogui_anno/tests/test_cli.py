# autogui_anno/tests/test_cli.py
from autogui_anno import cli


def test_main_web_parses_and_dispatches(tmp_path, monkeypatch):
    # minimal config + models files
    pipe = tmp_path / "pipe.yaml"; pipe.write_text("diff_limit: 250\n")
    models = tmp_path / "models.yaml"
    models.write_text(
        "default: default-fast\nmodels:\n  default-fast:\n"
        "    provider: openai\n    base_url: https://x/v1\n"
        "    api_key_env: FAKE_KEY\n    model: gpt-5-mini\n"
    )
    monkeypatch.setenv("FAKE_KEY", "k")
    called = {}
    def fake_run_web(config, llm, verifiers, **kw):
        called.update(kw); called["ok"] = True; return {"basic_stats": {}}
    monkeypatch.setattr(cli, "run_web", fake_run_web)
    # inject a dummy client builder to avoid real OpenAI construction
    rc = cli.main_web([
        "--config", str(pipe), "--models", str(models),
        "--data", str(tmp_path), "--out", str(tmp_path / "o"),
        "--build-client", "dummy",
    ])
    assert rc == 0 and called.get("ok")
