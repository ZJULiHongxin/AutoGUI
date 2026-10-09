# autogui_anno/tests/test_readme.py
import os
def test_readme_mentions_stages_and_env_var():
    p = os.path.join(os.path.dirname(__file__), "..", "README.md")
    txt = open(p).read()
    for stage in ["reject", "annotate", "verify", "generate"]:
        assert stage in txt.lower()
    assert "AUTOGUI_LLM_API_KEY" in txt
    assert "models.example.yaml" in txt
