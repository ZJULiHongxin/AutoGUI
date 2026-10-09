import json, os

WEB = os.path.join(os.path.dirname(__file__), "..", "autogui_anno", "viz", "web")

def test_viewer_assets_exist():
    for f in ["index.html", "app.js", "styles.css", "viz_data/index.json",
              "viz_data/example_record.json"]:
        assert os.path.exists(os.path.join(WEB, f)), f

def test_app_js_renders_all_stages():
    src = open(os.path.join(WEB, "app.js")).read()
    assert "renderRecord" in src and "renderIndex" in src
    for stage in ["input", "diff", "reject", "annotate", "verify", "result"]:
        assert stage in src

def test_example_record_matches_schema():
    rec = json.load(open(os.path.join(WEB, "viz_data", "example_record.json")))
    assert rec["schema_version"] == 1
    for k in ["dataset", "sample_id", "label", "meta", "input", "result", "verdict"]:
        assert k in rec
    assert {"before", "after", "explainer"} <= set(rec["input"])

def test_index_lists_example():
    idx = json.load(open(os.path.join(WEB, "viz_data", "index.json")))
    assert isinstance(idx, list) and len(idx) >= 1
    assert {"dataset", "sample_id", "label", "verdict", "file"} <= set(idx[0])
