from autogui_anno import cli

def test_main_visualize_build_dispatches(tmp_path, monkeypatch):
    pipe = tmp_path / "pipe.yaml"; pipe.write_text("diff_limit: 250\n")
    models = tmp_path / "models.yaml"
    models.write_text(
        "default: default-fast\nmodels:\n  default-fast:\n"
        "    provider: openai\n    base_url: https://x/v1\n"
        "    api_key_env: FAKE_KEY\n    model: gpt-5-mini\n"
    )
    monkeypatch.setenv("FAKE_KEY", "k")
    called = {}
    def fake_run_builder(**kw):
        called.update(kw); called["ok"] = True
        return {"built": 1, "skipped": 0, "by_verdict": {"kept": 1}}
    monkeypatch.setattr(cli, "run_builder", fake_run_builder)
    rc = cli.main_visualize_build([
        "--config", str(pipe), "--models", str(models),
        "--data", str(tmp_path), "--out", str(tmp_path / "viz"),
        "--build-client", "dummy",
    ])
    assert rc == 0 and called.get("ok")

def test_main_visualize_serve_check_only(tmp_path):
    rc = cli.main_visualize_serve(["--dir", str(tmp_path), "--check-only"])
    assert rc == 0
