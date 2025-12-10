"""
Light Wrap for Ultimate Rotoscopy
==================================

Professional light wrap implementation for seamless compositing.
Simulates the effect of background light wrapping around the
edges of the foreground subject.

"It simulates the light of the background affecting the subject,
creating a photorealistic result." - Digital Anarchy

Reference: Light Wrap Fantastic, Nuke Light Wrap nodes
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class WrapMode(Enum):
    """Light wrap modes."""
    SCREEN = "screen"         # Additive screen blend
    ADD = "add"               # Pure additive
    SOFT_LIGHT = "soft_light" # Soft light blend
    OVERLAY = "overlay"       # Overlay blend
    MULTIPLY = "multiply"     # Multiply for dark integration


@dataclass
class LightWrapConfig:
    """Light wrap configuration."""
    intensity: float = 0.5        # Overall intensity (0-1)
    wrap_width: int = 20          # Width of wrap effect in pixels
    blur_amount: float = 1.0      # Background blur multiplier
    saturation: float = 1.0       # Wrap saturation (0-2)
    luminance_only: bool = False  # Only use luminance for wrap
    wrap_mode: WrapMode = WrapMode.SCREEN
    inner_wrap: bool = False      # Also wrap inside (shadows)
    inner_intensity: float = 0.2  # Inner wrap intensity


class LightWrap:
    """
    Professional Light Wrap Effect.

    Creates the illusion that light from the background is
    bleeding onto the edges of the foreground subject, which
    is essential for photorealistic compositing.

    The effect works by:
    1. Blurring the background
    2. Masking to the edge region of the subject
    3. Blending onto the foreground

    Example:
        >>> light_wrap = LightWrap(LightWrapConfig(
        ...     intensity=0.6,
        ...     wrap_width=30,
        ...     wrap_mode=WrapMode.SCREEN
        ... ))
        >>>
        >>> wrapped = light_wrap.apply(foreground, background, alpha)
    """

    def __init__(self, config: Optional[LightWrapConfig] = None):
        self.config = config or LightWrapConfig()

    def apply(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """
        Apply light wrap effect.

        Args:
            foreground: Foreground RGB image
            background: Background RGB image
            alpha: Alpha matte

        Returns:
            Foreground with light wrap applied
        """
        import cv2

        # Normalize inputs
        if foreground.dtype == np.uint8:
            foreground = foreground.astype(np.float32) / 255.0
        if background.dtype == np.uint8:
            background = background.astype(np.float32) / 255.0
        if alpha.dtype == np.uint8:
            alpha = alpha.astype(np.float32) / 255.0

        if alpha.ndim == 3:
            alpha = alpha[..., 0]

        # Step 1: Create edge mask
        edge_mask = self._create_edge_mask(alpha)

        # Step 2: Blur the background
        blur_size = int(self.config.wrap_width * self.config.blur_amount)
        blur_size = blur_size * 2 + 1  # Ensure odd
        blurred_bg = cv2.GaussianBlur(background, (blur_size, blur_size), 0)

        # Step 3: Adjust wrap color
        wrap_color = self._adjust_wrap_color(blurred_bg)

        # Step 4: Apply blend mode
        wrapped = self._apply_blend(foreground, wrap_color, edge_mask)

        # Step 5: Inner wrap (shadows) if enabled
        if self.config.inner_wrap:
            inner_mask = self._create_inner_mask(alpha)
            inner_wrap = self._apply_inner_wrap(foreground, blurred_bg, inner_mask)
            wrapped = wrapped * (1 - inner_mask[..., None]) + inner_wrap * inner_mask[..., None]

        return np.clip(wrapped, 0, 1)

    def _create_edge_mask(self, alpha: np.ndarray) -> np.ndarray:
        """Create the edge mask for light wrap."""
        import cv2

        alpha_8bit = (alpha * 255).astype(np.uint8)

        # Dilate to get outer edge
        kernel_size = self.config.wrap_width * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        dilated = cv2.dilate(alpha_8bit, kernel)

        # Edge is the dilated minus original
        edge = (dilated.astype(np.float32) - alpha_8bit.astype(np.float32)) / 255.0

        # Multiply by alpha to feather
        edge = edge * (1 - alpha)  # Stronger at transparent edges

        # Apply intensity
        edge = edge * self.config.intensity

        # Smooth the mask
        blur_size = max(3, self.config.wrap_width // 2) * 2 + 1
        edge = cv2.GaussianBlur(edge, (blur_size, blur_size), 0)

        return edge

    def _create_inner_mask(self, alpha: np.ndarray) -> np.ndarray:
        """Create inner mask for shadow wrap."""
        import cv2

        alpha_8bit = (alpha * 255).astype(np.uint8)

        # Erode to get inner edge
        kernel_size = self.config.wrap_width * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        eroded = cv2.erode(alpha_8bit, kernel)

        # Inner edge is original minus eroded
        inner_edge = (alpha_8bit.astype(np.float32) - eroded.astype(np.float32)) / 255.0

        # Multiply by alpha
        inner_edge = inner_edge * alpha

        # Apply intensity
        inner_edge = inner_edge * self.config.inner_intensity

        # Smooth
        blur_size = max(3, self.config.wrap_width // 2) * 2 + 1
        inner_edge = cv2.GaussianBlur(inner_edge, (blur_size, blur_size), 0)

        return inner_edge

    def _adjust_wrap_color(self, blurred_bg: np.ndarray) -> np.ndarray:
        """Adjust wrap color properties."""
        wrap = blurred_bg.copy()

        # Adjust saturation
        if self.config.saturation != 1.0:
            gray = np.dot(wrap, [0.299, 0.587, 0.114])
            gray = np.stack([gray] * 3, axis=-1)
            wrap = gray + (wrap - gray) * self.config.saturation

        # Use luminance only if requested
        if self.config.luminance_only:
            lum = np.dot(wrap, [0.299, 0.587, 0.114])
            wrap = np.stack([lum] * 3, axis=-1)

        return np.clip(wrap, 0, 1)

    def _apply_blend(
        self,
        foreground: np.ndarray,
        wrap_color: np.ndarray,
        edge_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply blend mode for light wrap."""
        edge_3ch = np.stack([edge_mask] * 3, axis=-1)
        mode = self.config.wrap_mode

        if mode == WrapMode.SCREEN:
            # Screen blend: 1 - (1 - A) * (1 - B)
            blended = 1 - (1 - foreground) * (1 - wrap_color)

        elif mode == WrapMode.ADD:
            # Additive blend
            blended = foreground + wrap_color

        elif mode == WrapMode.SOFT_LIGHT:
            # Soft light blend
            blended = np.where(
                wrap_color <= 0.5,
                foreground - (1 - 2 * wrap_color) * foreground * (1 - foreground),
                foreground + (2 * wrap_color - 1) * (self._soft_light_d(foreground) - foreground)
            )

        elif mode == WrapMode.OVERLAY:
            # Overlay blend
            blended = np.where(
                foreground < 0.5,
                2 * foreground * wrap_color,
                1 - 2 * (1 - foreground) * (1 - wrap_color)
            )

        elif mode == WrapMode.MULTIPLY:
            # Multiply blend
            blended = foreground * wrap_color

        else:
            blended = foreground + wrap_color

        # Apply mask
        result = foreground * (1 - edge_3ch) + blended * edge_3ch

        return result

    def _soft_light_d(self, x: np.ndarray) -> np.ndarray:
        """Helper function for soft light blend."""
        return np.where(
            x <= 0.25,
            ((16 * x - 12) * x + 4) * x,
            np.sqrt(x)
        )

    def _apply_inner_wrap(
        self,
        foreground: np.ndarray,
        blurred_bg: np.ndarray,
        inner_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply inner/shadow wrap."""
        # Darken foreground edges based on background
        inner_3ch = np.stack([inner_mask] * 3, axis=-1)

        # Use multiply for shadow effect
        shadowed = foreground * (0.5 + 0.5 * blurred_bg)

        result = foreground * (1 - inner_3ch) + shadowed * inner_3ch

        return result


def apply_light_wrap(
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,
    intensity: float = 0.5,
    wrap_width: int = 20,
    mode: str = "screen",
) -> np.ndarray:
    """
    Quick light wrap function.

    Args:
        foreground: Foreground RGB
        background: Background RGB
        alpha: Alpha matte
        intensity: Effect intensity (0-1)
        wrap_width: Width in pixels
        mode: Blend mode (screen, add, overlay)

    Returns:
        Wrapped foreground
    """
    mode_map = {
        "screen": WrapMode.SCREEN,
        "add": WrapMode.ADD,
        "soft_light": WrapMode.SOFT_LIGHT,
        "overlay": WrapMode.OVERLAY,
        "multiply": WrapMode.MULTIPLY,
    }

    config = LightWrapConfig(
        intensity=intensity,
        wrap_width=wrap_width,
        wrap_mode=mode_map.get(mode, WrapMode.SCREEN),
    )

    light_wrap = LightWrap(config)
    return light_wrap.apply(foreground, background, alpha)
