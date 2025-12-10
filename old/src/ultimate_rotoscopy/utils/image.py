"""
Image Utilities for Ultimate Rotoscopy
=======================================
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image


def load_image(
    path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        path: Image file path
        target_size: Optional (width, height) to resize
        normalize: Normalize to [0, 1] range

    Returns:
        Image as numpy array (HxWxC or HxW)
    """
    path = Path(path)

    # Handle different formats
    if path.suffix.lower() == ".exr":
        return load_exr(path)

    img = Image.open(path)

    if target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)

    arr = np.array(img)

    if normalize:
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0

    return arr


def load_exr(path: Path) -> np.ndarray:
    """Load an EXR file."""
    try:
        import OpenEXR
        import Imath

        exr = OpenEXR.InputFile(str(path))
        header = exr.header()

        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1

        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        channels = list(header["channels"].keys())

        if "R" in channels and "G" in channels and "B" in channels:
            r = np.frombuffer(exr.channel("R", pt), dtype=np.float32).reshape(h, w)
            g = np.frombuffer(exr.channel("G", pt), dtype=np.float32).reshape(h, w)
            b = np.frombuffer(exr.channel("B", pt), dtype=np.float32).reshape(h, w)

            if "A" in channels:
                a = np.frombuffer(exr.channel("A", pt), dtype=np.float32).reshape(h, w)
                return np.stack([r, g, b, a], axis=-1)
            return np.stack([r, g, b], axis=-1)

        # Single channel
        ch = channels[0]
        return np.frombuffer(exr.channel(ch, pt), dtype=np.float32).reshape(h, w)

    except ImportError:
        import imageio
        return imageio.imread(str(path))


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    quality: int = 95,
) -> None:
    """
    Save an image to disk.

    Args:
        image: Image array
        path: Output path
        quality: JPEG quality (1-100)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle float images
    if image.dtype in (np.float32, np.float64):
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)

    img = Image.fromarray(image)

    if path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(path, quality=quality)
    else:
        img.save(path)


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    method: str = "lanczos",
) -> np.ndarray:
    """
    Resize an image.

    Args:
        image: Input image
        size: Target (width, height)
        method: Interpolation method (lanczos, bilinear, nearest)

    Returns:
        Resized image
    """
    import cv2

    method_map = {
        "lanczos": cv2.INTER_LANCZOS4,
        "bilinear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
    }

    interp = method_map.get(method, cv2.INTER_LANCZOS4)
    return cv2.resize(image, size, interpolation=interp)
