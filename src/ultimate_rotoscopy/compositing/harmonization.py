"""
Color Harmonization for Ultimate Rotoscopy
===========================================

Advanced color harmonization techniques to match foreground
subjects to background plates for seamless compositing.

"The elements of the background (and their luminance) must be in place -
they are the most important part for judging the quality of the matte."
- Ben McEwan
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

import numpy as np


class HarmonizationMethod(Enum):
    """Color harmonization methods."""
    HISTOGRAM = "histogram"       # Histogram matching
    MEAN_STD = "mean_std"         # Mean/std transfer
    LAB_TRANSFER = "lab_transfer" # LAB color space transfer
    REINHARD = "reinhard"         # Reinhard color transfer
    NEURAL = "neural"             # AI-based harmonization
    ADAPTIVE = "adaptive"         # Context-aware adaptive


@dataclass
class HarmonizationConfig:
    """Color harmonization configuration."""
    method: HarmonizationMethod = HarmonizationMethod.LAB_TRANSFER
    strength: float = 0.5         # Harmonization strength (0-1)
    preserve_luminance: bool = True
    edge_aware: bool = True       # Only harmonize at edges
    edge_width: int = 50          # Edge region width
    match_contrast: bool = True
    match_saturation: bool = True
    protect_highlights: bool = True
    protect_shadows: bool = True


class ColorHarmonizer:
    """
    Professional Color Harmonization.

    Matches the color characteristics of the foreground subject
    to the background plate for seamless integration.

    Methods include:
    - Histogram matching
    - Mean/std color transfer
    - LAB color space transfer (Reinhard method)
    - Edge-aware harmonization

    Example:
        >>> harmonizer = ColorHarmonizer(HarmonizationConfig(
        ...     method=HarmonizationMethod.LAB_TRANSFER,
        ...     strength=0.6
        ... ))
        >>>
        >>> harmonized_fg = harmonizer.harmonize(foreground, background, alpha)
    """

    def __init__(self, config: Optional[HarmonizationConfig] = None):
        self.config = config or HarmonizationConfig()

    def harmonize(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        alpha: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Harmonize foreground colors to match background.

        Args:
            foreground: Foreground RGB image
            background: Background RGB image
            alpha: Optional alpha for edge-aware processing

        Returns:
            Color-harmonized foreground
        """
        # Normalize
        if foreground.dtype == np.uint8:
            foreground = foreground.astype(np.float32) / 255.0
        if background.dtype == np.uint8:
            background = background.astype(np.float32) / 255.0

        # Apply harmonization method
        method = self.config.method

        if method == HarmonizationMethod.HISTOGRAM:
            harmonized = self._histogram_match(foreground, background)
        elif method == HarmonizationMethod.MEAN_STD:
            harmonized = self._mean_std_transfer(foreground, background)
        elif method == HarmonizationMethod.LAB_TRANSFER:
            harmonized = self._lab_transfer(foreground, background)
        elif method == HarmonizationMethod.REINHARD:
            harmonized = self._reinhard_transfer(foreground, background)
        elif method == HarmonizationMethod.ADAPTIVE:
            harmonized = self._adaptive_harmonize(foreground, background, alpha)
        else:
            harmonized = self._lab_transfer(foreground, background)

        # Apply edge-aware blending
        if self.config.edge_aware and alpha is not None:
            harmonized = self._apply_edge_mask(foreground, harmonized, alpha)

        # Blend with original based on strength
        result = foreground * (1 - self.config.strength) + harmonized * self.config.strength

        return np.clip(result, 0, 1)

    def _histogram_match(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """Match histogram of source to target."""
        result = np.zeros_like(source)

        for c in range(3):
            src_channel = source[..., c].flatten()
            tgt_channel = target[..., c].flatten()

            # Compute CDFs
            src_values, src_indices, src_counts = np.unique(
                (src_channel * 255).astype(np.uint8),
                return_inverse=True,
                return_counts=True
            )
            src_cdf = np.cumsum(src_counts).astype(np.float64)
            src_cdf /= src_cdf[-1]

            tgt_values, tgt_counts = np.unique(
                (tgt_channel * 255).astype(np.uint8),
                return_counts=True
            )
            tgt_cdf = np.cumsum(tgt_counts).astype(np.float64)
            tgt_cdf /= tgt_cdf[-1]

            # Map source to target
            interp_values = np.interp(src_cdf, tgt_cdf, tgt_values)
            matched = interp_values[src_indices]

            result[..., c] = matched.reshape(source.shape[:2]) / 255.0

        return result

    def _mean_std_transfer(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """Transfer mean and std from target to source."""
        result = np.zeros_like(source)

        for c in range(3):
            src = source[..., c]
            tgt = target[..., c]

            src_mean, src_std = np.mean(src), np.std(src)
            tgt_mean, tgt_std = np.mean(tgt), np.std(tgt)

            # Normalize, scale, shift
            if src_std > 1e-6:
                result[..., c] = (src - src_mean) / src_std * tgt_std + tgt_mean
            else:
                result[..., c] = src

        return np.clip(result, 0, 1)

    def _lab_transfer(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """Color transfer in LAB color space."""
        import cv2

        # Convert to LAB
        src_8bit = (source * 255).astype(np.uint8)
        tgt_8bit = (target * 255).astype(np.uint8)

        src_lab = cv2.cvtColor(src_8bit, cv2.COLOR_RGB2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(tgt_8bit, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Transfer statistics for each LAB channel
        for c in range(3):
            src_mean, src_std = np.mean(src_lab[..., c]), np.std(src_lab[..., c])
            tgt_mean, tgt_std = np.mean(tgt_lab[..., c]), np.std(tgt_lab[..., c])

            if src_std > 1e-6:
                src_lab[..., c] = (src_lab[..., c] - src_mean) / src_std * tgt_std + tgt_mean

        # Clip to valid LAB range
        src_lab[..., 0] = np.clip(src_lab[..., 0], 0, 255)
        src_lab[..., 1] = np.clip(src_lab[..., 1], 0, 255)
        src_lab[..., 2] = np.clip(src_lab[..., 2], 0, 255)

        # Convert back to RGB
        result_8bit = cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        return result_8bit.astype(np.float32) / 255.0

    def _reinhard_transfer(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """
        Reinhard color transfer.

        Reference: Reinhard et al. "Color Transfer between Images"
        """
        import cv2

        # Convert to LAB (using lαβ approximation)
        src_lab = self._rgb_to_lab(source)
        tgt_lab = self._rgb_to_lab(target)

        # Compute statistics
        for c in range(3):
            src_mean = np.mean(src_lab[..., c])
            src_std = np.std(src_lab[..., c])
            tgt_mean = np.mean(tgt_lab[..., c])
            tgt_std = np.std(tgt_lab[..., c])

            # Transfer
            if src_std > 1e-6:
                src_lab[..., c] = (src_lab[..., c] - src_mean) * (tgt_std / src_std) + tgt_mean

        # Convert back
        result = self._lab_to_rgb(src_lab)

        return np.clip(result, 0, 1)

    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to lαβ color space."""
        # RGB to LMS
        lms_matrix = np.array([
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444]
        ])

        lms = np.einsum('ij,hwj->hwi', lms_matrix, rgb)
        lms = np.maximum(lms, 1e-10)  # Avoid log(0)
        lms = np.log10(lms)

        # LMS to lαβ
        lab_matrix = np.array([
            [1/np.sqrt(3), 0, 0],
            [0, 1/np.sqrt(6), 0],
            [0, 0, 1/np.sqrt(2)]
        ]) @ np.array([
            [1, 1, 1],
            [1, 1, -2],
            [1, -1, 0]
        ])

        lab = np.einsum('ij,hwj->hwi', lab_matrix, lms)

        return lab

    def _lab_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        """Convert lαβ back to RGB."""
        # lαβ to LMS
        lab_inv = np.linalg.inv(np.array([
            [1/np.sqrt(3), 0, 0],
            [0, 1/np.sqrt(6), 0],
            [0, 0, 1/np.sqrt(2)]
        ]) @ np.array([
            [1, 1, 1],
            [1, 1, -2],
            [1, -1, 0]
        ]))

        lms = np.einsum('ij,hwj->hwi', lab_inv, lab)
        lms = np.power(10, lms)

        # LMS to RGB
        rgb_matrix = np.linalg.inv(np.array([
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444]
        ]))

        rgb = np.einsum('ij,hwj->hwi', rgb_matrix, lms)

        return rgb

    def _adaptive_harmonize(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        alpha: Optional[np.ndarray],
    ) -> np.ndarray:
        """Context-aware adaptive harmonization."""
        import cv2

        # Analyze background regions near the foreground
        if alpha is not None:
            # Get edge region of alpha
            if alpha.dtype != np.uint8:
                alpha_8bit = (alpha * 255).astype(np.uint8)
            else:
                alpha_8bit = alpha

            if alpha_8bit.ndim == 3:
                alpha_8bit = alpha_8bit[..., 0]

            # Dilate to get nearby background region
            kernel = np.ones((51, 51), np.uint8)
            dilated = cv2.dilate(alpha_8bit, kernel)
            nearby_bg_mask = (dilated > 128) & (alpha_8bit < 128)

            # Get colors from nearby background
            if np.any(nearby_bg_mask):
                nearby_colors = background[nearby_bg_mask]
                tgt_mean = np.mean(nearby_colors, axis=0)
                tgt_std = np.std(nearby_colors, axis=0)
            else:
                tgt_mean = np.mean(background.reshape(-1, 3), axis=0)
                tgt_std = np.std(background.reshape(-1, 3), axis=0)
        else:
            tgt_mean = np.mean(background.reshape(-1, 3), axis=0)
            tgt_std = np.std(background.reshape(-1, 3), axis=0)

        # Transfer to foreground
        src_mean = np.mean(foreground.reshape(-1, 3), axis=0)
        src_std = np.std(foreground.reshape(-1, 3), axis=0)

        result = foreground.copy()
        for c in range(3):
            if src_std[c] > 1e-6:
                result[..., c] = (foreground[..., c] - src_mean[c]) / src_std[c] * tgt_std[c] + tgt_mean[c]

        return np.clip(result, 0, 1)

    def _apply_edge_mask(
        self,
        original: np.ndarray,
        harmonized: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """Apply edge-aware blending."""
        import cv2

        if alpha.dtype != np.uint8:
            alpha_8bit = (alpha * 255).astype(np.uint8)
        else:
            alpha_8bit = alpha

        if alpha_8bit.ndim == 3:
            alpha_8bit = alpha_8bit[..., 0]

        # Create edge mask
        kernel_size = self.config.edge_width * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        dilated = cv2.dilate(alpha_8bit, kernel)
        eroded = cv2.erode(alpha_8bit, kernel)

        edge_mask = (dilated.astype(np.float32) - eroded.astype(np.float32)) / 255.0
        edge_mask = cv2.GaussianBlur(edge_mask, (kernel_size, kernel_size), 0)

        # Apply mask
        edge_3ch = np.stack([edge_mask] * 3, axis=-1)
        result = original * (1 - edge_3ch) + harmonized * edge_3ch

        return result


def harmonize_colors(
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    method: str = "lab",
    strength: float = 0.5,
) -> np.ndarray:
    """
    Quick color harmonization function.

    Args:
        foreground: Foreground RGB
        background: Background RGB
        alpha: Optional alpha matte
        method: Harmonization method (histogram, mean_std, lab, reinhard)
        strength: Effect strength (0-1)

    Returns:
        Harmonized foreground
    """
    method_map = {
        "histogram": HarmonizationMethod.HISTOGRAM,
        "mean_std": HarmonizationMethod.MEAN_STD,
        "lab": HarmonizationMethod.LAB_TRANSFER,
        "reinhard": HarmonizationMethod.REINHARD,
        "adaptive": HarmonizationMethod.ADAPTIVE,
    }

    config = HarmonizationConfig(
        method=method_map.get(method, HarmonizationMethod.LAB_TRANSFER),
        strength=strength,
        edge_aware=alpha is not None,
    )

    harmonizer = ColorHarmonizer(config)
    return harmonizer.harmonize(foreground, background, alpha)


def match_histogram(
    source: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    """Quick histogram matching."""
    config = HarmonizationConfig(
        method=HarmonizationMethod.HISTOGRAM,
        strength=1.0,
        edge_aware=False,
    )

    harmonizer = ColorHarmonizer(config)
    return harmonizer.harmonize(source, target)
