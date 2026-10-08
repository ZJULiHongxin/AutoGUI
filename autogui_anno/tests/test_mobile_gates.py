import numpy as np
from autogui_anno.stages.mobile_gates import is_pure_color, is_element_too_large, is_tap_action

def test_is_pure_color_true_on_flat_patch():
    img = np.full((20, 20, 3), 128, dtype=np.uint8)
    assert is_pure_color(img, (0, 0, 10, 10)) is True

def test_is_pure_color_false_on_noisy_patch():
    img = (np.random.rand(20, 20, 3) * 255).astype(np.uint8)
    assert is_pure_color(img, (0, 0, 19, 19)) is False

def test_is_element_too_large():
    assert is_element_too_large([0, 0, 100, 100], 100, 100, ratio=0.65) is True
    assert is_element_too_large([0, 0, 10, 10], 100, 100, ratio=0.65) is False

def test_is_tap_action():
    assert is_tap_action("DualPoint") is True
    assert is_tap_action("Type") is False
