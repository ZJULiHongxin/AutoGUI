from autogui_anno import prompts


def test_marks_present():
    assert prompts.SUMMARY_MARK == "Summary"
    assert prompts.DESCRIPTION_MARK == "Overall Functionality"


def test_make_reject_prompt_formats_max_score():
    p = prompts.make_reject_prompt(1)
    assert "{target_element}" in p and "{outcome}" in p


def test_make_verif_prompt_builds():
    p = prompts.make_verif_prompt(3)
    assert "{content}" in p and "{functionality}" in p and "{candidate}" in p


def test_get_clean_func_extracts_from_first_verb():
    import pytest
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except Exception:
        pytest.skip("en_core_web_sm model not installed", allow_module_level=False)

    raw = "Reasoning: ...\n\nSummary: This element opens the settings menu."
    out = prompts.get_clean_func(raw)
    assert "settings" in out.lower()
