import os
import shutil
import xml.etree.ElementTree as ET

from autogui_anno.axtree.android import process_xml, find_all_elem_texts_boxes

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "android_sample.xml")


def test_process_xml_produces_text_tree(tmp_path):
    # process_xml's real signature is process_xml(xml_file, target_box=None,
    # resume=True, skip_statusbar=True, skip_lang=False). When given a path
    # ending in .xml it reads the file and writes a sibling *_axtree.json cache,
    # so point it at a temp copy of the fixture. It returns (tree_lines, all_boxes).
    xml_path = tmp_path / "android_sample.xml"
    shutil.copy(FIX, xml_path)
    tree_lines, all_boxes = process_xml(str(xml_path), resume=False)
    text = "\n".join(tree_lines)
    assert isinstance(text, str) and len(text) > 0


def test_find_all_elem_texts_boxes_parses_bounds():
    with open(FIX) as f:
        root = ET.fromstring(f.read())
    items = find_all_elem_texts_boxes(root)
    assert any(it["box"] is not None for it in items)
    assert all({"tag", "text", "box", "is_leaf", "is_interactable"} <= set(it) for it in items)
