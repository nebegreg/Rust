"""
Color Utilities for Ultimate Rotoscopy
=======================================
"""

import numpy as np


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to LAB color space."""
    import cv2

    if rgb.dtype == np.float32 or rgb.dtype == np.float64:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

    # OpenCV expects BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    return lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB to RGB color space."""
    import cv2

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb


def apply_color_correction(
    image: np.ndarray,
    gamma: float = 1.0,
    saturation: float = 1.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """
    Apply color correction to an image.

    Args:
        image: Input RGB image
        gamma: Gamma correction (1.0 = no change)
        saturation: Saturation adjustment (1.0 = no change)
        brightness: Brightness offset (-1 to 1)
        contrast: Contrast multiplier (1.0 = no change)

    Returns:
        Color-corrected image
    """
    import cv2

    # Ensure float
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.copy()

    # Gamma
    if gamma != 1.0:
        img = np.power(img, 1.0 / gamma)

    # Brightness and contrast
    img = (img - 0.5) * contrast + 0.5 + brightness

    # Saturation
    if saturation != 1.0:
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        img[..., :3] = gray[..., None] + saturation * (img[..., :3] - gray[..., None])

    # Clip
    img = np.clip(img, 0, 1)

    return img
