"""
Edge Operations for Ultimate Rotoscopy
=======================================

Pixel spread, edge extend, and edge blend operations for
professional matte integration.

The Edge Extend tool duplicates pixels outward - not the matte.
It essentially is "a negative simple choker that also clone brushes
the outer most layer of pixels."

Reference: Flame Pixel Spread, Sapphire Distort
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class EdgeOperation(Enum):
    """Edge operation types."""
    SPREAD = "spread"       # Extend edge pixels outward
    ERODE = "erode"         # Shrink matte
    DILATE = "dilate"       # Expand matte
    BLUR = "blur"           # Soften edges
    FEATHER = "feather"     # Gradient feather


@dataclass
class EdgeConfig:
    """Edge operation configuration."""
    operation: EdgeOperation = EdgeOperation.SPREAD
    amount: int = 5              # Pixels to spread/erode/dilate
    softness: float = 0.5        # Edge softness (0-1)
    falloff: str = "linear"      # linear, quadratic, gaussian
    preserve_detail: bool = True
    iterations: int = 1


class PixelSpread:
    """
    Pixel Spread / Edge Extend Operation.

    Extends edge pixels outward from the matte boundary,
    essential for straight alpha compositing.

    This is handy for creating already matted elements where you
    need to use a Straight alpha. Straight alpha needs the color
    elements to extend outside the matte area and then the alpha
    cleans up the edge.

    Example:
        >>> spreader = PixelSpread(amount=10, softness=0.5)
        >>> extended_fg = spreader.process(foreground, alpha)
    """

    def __init__(
        self,
        amount: int = 5,
        softness: float = 0.5,
        falloff: str = "linear",
    ):
        self.amount = amount
        self.softness = softness
        self.falloff = falloff

    def process(
        self,
        image: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """
        Spread edge pixels outward.

        Args:
            image: RGB image
            alpha: Alpha matte

        Returns:
            Image with extended edges
        """
        import cv2

        # Normalize
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if alpha.dtype == np.uint8:
            alpha = alpha.astype(np.float32) / 255.0

        if alpha.ndim == 3:
            alpha = alpha[..., 0]

        result = image.copy()

        # Create dilated alpha for spread region
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.amount * 2 + 1, self.amount * 2 + 1)
        )
        dilated_alpha = cv2.dilate(alpha, kernel)

        # Spread region is where dilated > original
        spread_region = (dilated_alpha > alpha) & (dilated_alpha > 0.01)

        # For each pixel in spread region, find nearest edge pixel color
        if np.any(spread_region):
            result = self._spread_pixels(image, alpha, spread_region, dilated_alpha)

        return result

    def _spread_pixels(
        self,
        image: np.ndarray,
        alpha: np.ndarray,
        spread_region: np.ndarray,
        dilated_alpha: np.ndarray,
    ) -> np.ndarray:
        """Spread edge pixels into the spread region."""
        import cv2
        from scipy import ndimage

        result = image.copy()
        h, w = alpha.shape

        # Find edge pixels (high alpha gradient)
        grad_x = cv2.Sobel(alpha, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(alpha, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Edge pixels are where gradient is high and alpha is significant
        edge_mask = (edge_magnitude > 0.1) & (alpha > 0.3)

        # Distance transform from edge
        distance = ndimage.distance_transform_edt(~edge_mask)

        # For spread region, sample from nearest edge pixel
        for c in range(3):
            channel = image[..., c].copy()

            # Use inpainting-like approach
            # Blur the edge pixels outward
            edge_colors = channel.copy()
            edge_colors[~edge_mask] = 0
            edge_weights = edge_mask.astype(np.float32)

            # Iterative blur to spread colors
            for _ in range(self.amount):
                edge_colors_blurred = cv2.blur(edge_colors, (3, 3))
                edge_weights_blurred = cv2.blur(edge_weights, (3, 3))

                # Update where we have new coverage
                new_coverage = (edge_weights_blurred > 0.01) & (edge_weights < 0.01)
                edge_colors[new_coverage] = edge_colors_blurred[new_coverage] / (edge_weights_blurred[new_coverage] + 1e-6)
                edge_weights[new_coverage] = edge_weights_blurred[new_coverage]

                edge_colors_blurred = edge_colors.copy()
                edge_weights_blurred = edge_weights.copy()

            # Apply to spread region only
            result[..., c] = np.where(
                spread_region & (edge_weights > 0.01),
                edge_colors,
                result[..., c]
            )

        # Apply softness falloff
        if self.softness > 0:
            falloff = self._compute_falloff(distance, self.amount)
            falloff_3ch = np.stack([falloff] * 3, axis=-1)
            result = result * falloff_3ch + image * (1 - falloff_3ch)

        return result

    def _compute_falloff(
        self,
        distance: np.ndarray,
        max_dist: float
    ) -> np.ndarray:
        """Compute falloff based on distance."""
        normalized = np.clip(distance / max_dist, 0, 1)

        if self.falloff == "linear":
            falloff = 1 - normalized
        elif self.falloff == "quadratic":
            falloff = (1 - normalized) ** 2
        elif self.falloff == "gaussian":
            falloff = np.exp(-normalized**2 * 3)
        else:
            falloff = 1 - normalized

        return falloff * self.softness


class EdgeExtend(PixelSpread):
    """Alias for PixelSpread - same operation, different name."""
    pass


class EdgeBlend:
    """
    Edge Blending for seamless compositing.

    Blends foreground edges with background for natural integration.
    """

    def __init__(
        self,
        blend_width: int = 10,
        blend_mode: str = "multiply",
        preserve_fg: bool = True,
    ):
        self.blend_width = blend_width
        self.blend_mode = blend_mode
        self.preserve_fg = preserve_fg

    def process(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """
        Blend foreground edges with background.

        Args:
            foreground: Foreground RGB
            background: Background RGB
            alpha: Alpha matte

        Returns:
            Blended composite
        """
        import cv2

        # Normalize
        if foreground.dtype == np.uint8:
            foreground = foreground.astype(np.float32) / 255.0
        if background.dtype == np.uint8:
            background = background.astype(np.float32) / 255.0
        if alpha.dtype == np.uint8:
            alpha = alpha.astype(np.float32) / 255.0

        if alpha.ndim == 3:
            alpha = alpha[..., 0]

        # Compute edge region
        edge_mask = self._compute_edge_mask(alpha)

        # Apply blend mode
        if self.blend_mode == "multiply":
            blended = foreground * background
        elif self.blend_mode == "screen":
            blended = 1 - (1 - foreground) * (1 - background)
        elif self.blend_mode == "overlay":
            blended = np.where(
                background < 0.5,
                2 * foreground * background,
                1 - 2 * (1 - foreground) * (1 - background)
            )
        else:  # normal blend
            blended = foreground

        # Create edge blend
        edge_3ch = np.stack([edge_mask] * 3, axis=-1)

        if self.preserve_fg:
            # Only blend at edges
            result = foreground * (1 - edge_3ch) + blended * edge_3ch
        else:
            result = blended

        # Final composite with alpha
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        composite = result * alpha_3ch + background * (1 - alpha_3ch)

        return composite

    def _compute_edge_mask(self, alpha: np.ndarray) -> np.ndarray:
        """Compute edge region mask."""
        import cv2

        alpha_8bit = (alpha * 255).astype(np.uint8)

        # Create edge detection
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.blend_width * 2 + 1, self.blend_width * 2 + 1)
        )

        dilated = cv2.dilate(alpha_8bit, kernel)
        eroded = cv2.erode(alpha_8bit, kernel)

        edge = (dilated.astype(np.float32) - eroded.astype(np.float32)) / 255.0

        # Smooth
        edge = cv2.GaussianBlur(edge, (self.blend_width * 2 + 1, self.blend_width * 2 + 1), 0)

        return edge


def edge_erode(alpha: np.ndarray, amount: int = 5) -> np.ndarray:
    """Erode (shrink) alpha matte."""
    import cv2

    if alpha.dtype == np.float32 or alpha.dtype == np.float64:
        alpha_8bit = (alpha * 255).astype(np.uint8)
    else:
        alpha_8bit = alpha

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (amount * 2 + 1, amount * 2 + 1)
    )

    eroded = cv2.erode(alpha_8bit, kernel)

    if alpha.dtype == np.float32 or alpha.dtype == np.float64:
        return eroded.astype(np.float32) / 255.0

    return eroded


def edge_dilate(alpha: np.ndarray, amount: int = 5) -> np.ndarray:
    """Dilate (expand) alpha matte."""
    import cv2

    if alpha.dtype == np.float32 or alpha.dtype == np.float64:
        alpha_8bit = (alpha * 255).astype(np.uint8)
    else:
        alpha_8bit = alpha

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (amount * 2 + 1, amount * 2 + 1)
    )

    dilated = cv2.dilate(alpha_8bit, kernel)

    if alpha.dtype == np.float32 or alpha.dtype == np.float64:
        return dilated.astype(np.float32) / 255.0

    return dilated
