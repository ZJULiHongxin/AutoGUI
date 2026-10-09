# autogui_anno/tests/test_config.py
import pytest
from autogui_anno.config import PipelineConfig, RegistryConfig

def test_pipeline_defaults_match_source():
    c = PipelineConfig()
    assert c.diff_limit == 250
    assert c.diff_context == 4
    assert c.reject_max_score == 1
    assert c.verify_max_score == 3
    assert c.desc_prediction_threshold == 0.8

def test_pipeline_from_yaml_overrides(tmp_path):
    p = tmp_path / "pipe.yaml"
    p.write_text("diff_limit: 100\nreject_repeat: 5\n")
    c = PipelineConfig.from_yaml(str(p))
    assert c.diff_limit == 100
    assert c.reject_repeat == 5
    assert c.verify_max_score == 3  # untouched default

def test_pipeline_from_yaml_unknown_key_raises(tmp_path):
    p = tmp_path / "pipe.yaml"
    p.write_text("nonsense_key: 1\n")
    with pytest.raises(ValueError):
        PipelineConfig.from_yaml(str(p))

def test_registry_from_yaml(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(
        "default: default-fast\n"
        "models:\n"
        "  default-fast:\n"
        "    provider: openai\n"
        "    base_url: https://x.example.com/v1\n"
        "    api_key_env: AUTOGUI_LLM_API_KEY\n"
        "    model: gpt-5-mini\n"
    )
    reg = RegistryConfig.from_yaml(str(p))
    assert reg.default == "default-fast"
    assert reg.models["default-fast"].model == "gpt-5-mini"
    assert reg.models["default-fast"].api_key_env == "AUTOGUI_LLM_API_KEY"
