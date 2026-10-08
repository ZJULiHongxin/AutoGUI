from autogui_anno.axtree.diff import format_diff

def test_format_diff_counts_and_prefixes():
    before = ["RootWebArea 'x'", "link 'Home'", "link 'Old'"]
    after = ["RootWebArea 'x'", "link 'Home'", "link 'New'"]
    import difflib
    diff = difflib.unified_diff(before, after, fromfile="a", tofile="b", n=4)
    formated, cnt, added, deleted = format_diff(list(diff), diff_limit=250)
    assert added >= 1 and deleted >= 1
    assert any(ln.startswith(("Added", "Deleted", "Unchanged")) for ln in formated)
