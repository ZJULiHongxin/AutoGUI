from autogui_anno.axtree.web import NUMBERING_PATTERN, clean_accessibility_tree


def test_numbering_pattern_strips_labels():
    assert NUMBERING_PATTERN.sub("", "[12] link 'Home'") == " link 'Home'"


def test_clean_accessibility_tree_callable():
    # smoke: the function exists under the corrected spelling and runs on a
    # minimal tree-line list without raising. clean_accessibility_tree takes
    # the list[str] of indented AXTree lines produced by
    # parse_accessibility_tree and returns a de-duplicated list[str].
    assert callable(clean_accessibility_tree)
    lines = [
        "link 'Home'",
        "\tStaticText 'Home'",   # duplicate of parent -> dropped
        "\tStaticText 'Welcome'",  # new text -> kept
    ]
    out = clean_accessibility_tree(lines)
    assert isinstance(out, list)
    assert "link 'Home'" in out
    assert "\tStaticText 'Welcome'" in out
