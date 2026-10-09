from autogui_anno.tasks.generate import normalize_coord, build_grounding_sample

def test_normalize_coord_zero_padded():
    assert normalize_coord(1280, 2560) == "500"
    assert normalize_coord(0, 2560) == "000"

def test_build_grounding_sample_shape():
    rect = {"top": 100, "left": 200, "width": 50, "height": 40}
    s = build_grounding_sample(func="opens the menu", rect=rect, scale=1,
                               full_w=2560, full_h=1440,
                               image_rel_path="ui/img.png", step_id=5)
    assert s["conversations"][0]["value"].startswith("Please locate the element supporting this functionality:")
    assert s["conversations"][1]["value"].startswith("<point>[")
    assert s["bbox"].count(",") == 3
    assert s["image"] == "ui/img.png"
