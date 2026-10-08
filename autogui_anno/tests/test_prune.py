import json, os
from autogui_anno.axtree.prune import (
    count_tab, remove_labels, extract_labels, get_markers,
    prune_static_text, MARKER_STR,
)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "axtree_sample.json")

def test_count_tab():
    assert count_tab("\t\tbutton 'x'") == 2
    assert count_tab("button 'x'") == 0
    assert count_tab("") == 0

def test_remove_and_extract_labels():
    assert remove_labels("[12] link 'Home'") == " link 'Home'"
    assert extract_labels("[12] link [34]") == ["12", "34"]

def test_get_markers_strips_markers_and_maps():
    lines = ["RootWebArea 'x'", f"link 'Home'{MARKER_STR}abc]"]
    markers = get_markers(lines)
    assert markers["abc"] == 1
    assert MARKER_STR not in lines[1]  # stripped in place

def test_prune_static_text_removes_hidden_and_returns_markers():
    with open(FIX) as f:
        lines = json.load(f)
    before = len(lines)
    markers = prune_static_text(lines, remove_hidden=True, with_markers=True)
    assert isinstance(markers, dict)
    assert len(lines) <= before
    assert not any("hidden: True" in ln for ln in lines)

def test_prune_static_text_caps_menuitems():
    lines = ["RootWebArea 'x'"] + [f"\tmenuitem 'm{i}'" for i in range(5)]
    prune_static_text(lines, menuitem_limit=3)
    assert sum("menuitem" in ln for ln in lines) <= 3
