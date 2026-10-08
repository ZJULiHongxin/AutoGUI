"""Pure invalidity gates for the mobile orchestrator.

Ported from:
- ``is_pure_color`` <- ``WebpageFunctionality/utils/tools.py:683-711`` (the numpy-std,
  ``roi=(x1,y1,x2,y2)`` version — ported VERBATIM; the DUPLICATE at tools.py:971 is NOT
  ported).
- ``is_element_too_large`` <- the inline element-area ratio gate at
  ``annotate_func_android.py:220``. The source gate is
  ``AREA > 0 and (box area) / AREA >= 0.65`` where ``AREA`` is a module global
  (annotate_func_android.py:42) left at ``0`` — i.e. effectively disabled until set.
  This parameterized version takes real screen dims so the gate actually works, while
  guarding against zero area (``screen_w*screen_h <= 0`` -> ``False``) to preserve the
  source's ``AREA > 0`` short-circuit faithfully.
- ``is_tap_action`` <- the inline check at ``annotate_func_android.py:202``
  (``action_type != 'DualPoint'`` -> uncheckable), re-expressed positively.
"""
from __future__ import annotations

import numpy as np


def is_pure_color(image, roi, *, threshold: int = 5) -> bool:
    """Determine if a region of an image is of pure color, supporting both grayscale and color images.

    Parameters:
        image (numpy.ndarray): The image in which to check the color purity.
        roi (tuple): A tuple of (x1, y1, x2, y2) specifying the region of interest.
        threshold (int or float): The threshold for the standard deviation to consider the region as pure color.

    Returns:
        bool: True if the region is of pure color, False otherwise.
    """
    # Extract the region of interest from the image
    x1, y1, x2, y2 = roi
    region = image[y1:y2, x1:x2]

    # Check if the image is grayscale or color
    if len(region.shape) == 2:
        # Grayscale image (2D array)
        std_dev = np.std(region)
    else:
        # Color image (3D array)
        std_dev = np.std(region, axis=(0, 1))

    # Check if the standard deviation is below the threshold for all color channels (or the single channel in grayscale)
    if isinstance(std_dev, np.ndarray):
        return bool(np.all(std_dev < threshold))
    else:
        return bool(std_dev < threshold)


def is_element_too_large(box, screen_w, screen_h, *, ratio: float = 0.65) -> bool:
    """Return True if the interacted element's area is at least ``ratio`` of the screen.

    Mirrors annotate_func_android.py:220 (``AREA > 0 and (box area)/AREA >= 0.65``).
    ``screen_w*screen_h <= 0`` returns False to preserve the source's ``AREA > 0``
    short-circuit (the gate is disabled when the screen area is unknown/zero).
    """
    screen_area = screen_w * screen_h
    if screen_area <= 0:
        return False
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    return box_area / screen_area >= ratio


def is_tap_action(action_type: str) -> bool:
    """Return True if the action is a tap (``action_type == 'DualPoint'``).

    Mirrors annotate_func_android.py:202 (which treats ``!= 'DualPoint'`` as
    not-a-tap -> uncheckable).
    """
    return action_type == "DualPoint"
