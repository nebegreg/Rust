"""
Despill Algorithms for Ultimate Rotoscopy
==========================================

Professional despill techniques for removing green/blue screen spill
from keyed footage. Based on industry-standard algorithms.

Reference: Ben McEwan - Deconstructing Despill Algorithms
https://benmcewan.com/blog/understanding-despill-algorithms
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np


class DespillAlgorithm(Enum):
    """Available despill algorithms."""
    AVERAGE = "average"           # (R + B) / 2 for green spill
    MAXIMUM = "maximum"           # max(R, B) for green spill
    MINIMUM = "minimum"           # min(R, B) - more aggressive
    DOUBLE_AVERAGE = "double_avg" # Flame-style double average
    LUMINANCE = "luminance"       # Based on luminance preservation
    ADAPTIVE = "adaptive"         # Context-aware adaptive
    AI_BASED = "ai"               # Neural network based


class SpillChannel(Enum):
    """Spill color channel."""
    GREEN = "green"
    BLUE = "blue"
    RED = "red"  # Rare but possible


@dataclass
class DespillConfig:
    """Despill configuration."""
    algorithm: DespillAlgorithm = DespillAlgorithm.AVERAGE
    spill_channel: SpillChannel = SpillChannel.GREEN
    strength: float = 1.0           # 0-1, despill intensity
    preserve_luminance: bool = True
    edge_only: bool = False         # Only despill at edges
    edge_softness: float = 0.5      # Edge mask softness
    protect_skin: bool = True       # Protect skin tones
    skin_hue_range: Tuple[float, float] = (0.0, 60.0)  # Hue range for skin


class Despill:
    """
    Professional Despill Processing.

    Removes color spill from chroma key footage using various
    industry-standard algorithms.

    "Despilling is arguably the most important step to get right when
    pulling a key. A great despill can often hide imperfections in your
    alpha channel." - Ben McEwan

    Example:
        >>> despill = Despill(DespillConfig(
        ...     algorithm=DespillAlgorithm.AVERAGE,
        ...     spill_channel=SpillChannel.GREEN,
        ...     strength=0.8
        ... ))
        >>>
        >>> result = despill.process(image, alpha)
    """

    def __init__(self, config: Optional[DespillConfig] = None):
        self.config = config or DespillConfig()

    def process(
        self,
        image: np.ndarray,
        alpha: Optional[np.ndarray] = None,
        background: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply despill to image.

        Args:
            image: RGB image (0-1 or 0-255)
            alpha: Optional alpha matte for edge detection
            background: Optional background for context-aware despill

        Returns:
            Despilled image
        """
        # Normalize to 0-1
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        # Store original luminance if needed
        if self.config.preserve_luminance:
            original_lum = self._compute_luminance(image)

        # Compute despill map
        despill_map = self._compute_despill_map(image)

        # Apply edge mask if requested
        if self.config.edge_only and alpha is not None:
            edge_mask = self._compute_edge_mask(alpha)
            despill_map = despill_map * edge_mask

        # Protect skin tones
        if self.config.protect_skin:
            skin_mask = self._compute_skin_mask(image)
            despill_map = despill_map * (1 - skin_mask)

        # Apply strength
        despill_map = despill_map * self.config.strength

        # Apply despill based on algorithm
        result = self._apply_despill(image, despill_map)

        # Restore luminance if requested
        if self.config.preserve_luminance:
            result = self._restore_luminance(result, original_lum)

        return np.clip(result, 0, 1)

    def _compute_despill_map(self, image: np.ndarray) -> np.ndarray:
        """Compute the despill correction map."""
        r, g, b = image[..., 0], image[..., 1], image[..., 2]

        if self.config.spill_channel == SpillChannel.GREEN:
            return self._compute_green_despill_map(r, g, b)
        elif self.config.spill_channel == SpillChannel.BLUE:
            return self._compute_blue_despill_map(r, g, b)
        else:
            return self._compute_red_despill_map(r, g, b)

    def _compute_green_despill_map(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray
    ) -> np.ndarray:
        """Compute green spill map."""
        algo = self.config.algorithm

        if algo == DespillAlgorithm.AVERAGE:
            # G - (R + B) / 2
            limit = (r + b) / 2
            spill = np.maximum(0, g - limit)

        elif algo == DespillAlgorithm.MAXIMUM:
            # G - max(R, B)
            limit = np.maximum(r, b)
            spill = np.maximum(0, g - limit)

        elif algo == DespillAlgorithm.MINIMUM:
            # G - min(R, B) - more aggressive
            limit = np.minimum(r, b)
            spill = np.maximum(0, g - limit)

        elif algo == DespillAlgorithm.DOUBLE_AVERAGE:
            # Flame-style: G - ((R + B) / 2 + max(R, B)) / 2
            avg = (r + b) / 2
            mx = np.maximum(r, b)
            limit = (avg + mx) / 2
            spill = np.maximum(0, g - limit)

        elif algo == DespillAlgorithm.LUMINANCE:
            # Luminance-preserving despill
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            expected_g = lum / 0.587 * 0.587  # Neutral green
            spill = np.maximum(0, g - expected_g)

        elif algo == DespillAlgorithm.ADAPTIVE:
            # Adaptive based on local context
            avg = (r + b) / 2
            mx = np.maximum(r, b)
            # Blend between average and max based on saturation
            sat = (mx - np.minimum(r, b)) / (mx + 1e-6)
            limit = sat * mx + (1 - sat) * avg
            spill = np.maximum(0, g - limit)

        else:
            # Default to average
            limit = (r + b) / 2
            spill = np.maximum(0, g - limit)

        return spill

    def _compute_blue_despill_map(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray
    ) -> np.ndarray:
        """Compute blue spill map."""
        algo = self.config.algorithm

        if algo == DespillAlgorithm.AVERAGE:
            limit = (r + g) / 2
            spill = np.maximum(0, b - limit)

        elif algo == DespillAlgorithm.MAXIMUM:
            limit = np.maximum(r, g)
            spill = np.maximum(0, b - limit)

        elif algo == DespillAlgorithm.MINIMUM:
            limit = np.minimum(r, g)
            spill = np.maximum(0, b - limit)

        elif algo == DespillAlgorithm.DOUBLE_AVERAGE:
            avg = (r + g) / 2
            mx = np.maximum(r, g)
            limit = (avg + mx) / 2
            spill = np.maximum(0, b - limit)

        else:
            limit = (r + g) / 2
            spill = np.maximum(0, b - limit)

        return spill

    def _compute_red_despill_map(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray
    ) -> np.ndarray:
        """Compute red spill map (rare case)."""
        limit = (g + b) / 2
        spill = np.maximum(0, r - limit)
        return spill

    def _apply_despill(
        self,
        image: np.ndarray,
        despill_map: np.ndarray
    ) -> np.ndarray:
        """Apply the despill correction."""
        result = image.copy()

        if self.config.spill_channel == SpillChannel.GREEN:
            # Reduce green channel
            result[..., 1] = result[..., 1] - despill_map
            # Optionally add to other channels to maintain luminance
            if self.config.preserve_luminance:
                compensation = despill_map * 0.587 / (0.299 + 0.114)
                result[..., 0] = result[..., 0] + compensation * 0.299 / 0.587
                result[..., 2] = result[..., 2] + compensation * 0.114 / 0.587

        elif self.config.spill_channel == SpillChannel.BLUE:
            result[..., 2] = result[..., 2] - despill_map
            if self.config.preserve_luminance:
                compensation = despill_map * 0.114 / (0.299 + 0.587)
                result[..., 0] = result[..., 0] + compensation * 0.299 / 0.114
                result[..., 1] = result[..., 1] + compensation * 0.587 / 0.114

        elif self.config.spill_channel == SpillChannel.RED:
            result[..., 0] = result[..., 0] - despill_map

        return result

    def _compute_edge_mask(self, alpha: np.ndarray) -> np.ndarray:
        """Compute edge mask from alpha."""
        import cv2

        if alpha.dtype != np.uint8:
            alpha_8bit = (alpha * 255).astype(np.uint8)
        else:
            alpha_8bit = alpha

        # Dilate and erode
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(alpha_8bit, kernel, iterations=2)
        eroded = cv2.erode(alpha_8bit, kernel, iterations=2)

        # Edge is the difference
        edge = (dilated.astype(np.float32) - eroded.astype(np.float32)) / 255.0

        # Apply softness
        if self.config.edge_softness > 0:
            blur_size = int(self.config.edge_softness * 20) * 2 + 1
            edge = cv2.GaussianBlur(edge, (blur_size, blur_size), 0)

        return edge

    def _compute_skin_mask(self, image: np.ndarray) -> np.ndarray:
        """Compute skin tone protection mask."""
        import cv2

        # Convert to HSV
        if image.max() <= 1:
            img_8bit = (image * 255).astype(np.uint8)
        else:
            img_8bit = image.astype(np.uint8)

        hsv = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2HSV)

        # Skin hue range (typically 0-50 degrees)
        h_min, h_max = self.config.skin_hue_range
        hue = hsv[..., 0].astype(np.float32)

        # Create mask
        mask = ((hue >= h_min) & (hue <= h_max)).astype(np.float32)

        # Add saturation constraint (skin is moderately saturated)
        sat = hsv[..., 1].astype(np.float32) / 255.0
        sat_mask = ((sat > 0.1) & (sat < 0.7)).astype(np.float32)

        mask = mask * sat_mask

        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (11, 11), 0)

        return mask

    def _compute_luminance(self, image: np.ndarray) -> np.ndarray:
        """Compute luminance."""
        return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

    def _restore_luminance(
        self,
        image: np.ndarray,
        target_lum: np.ndarray
    ) -> np.ndarray:
        """Restore original luminance."""
        current_lum = self._compute_luminance(image)

        # Avoid division by zero
        ratio = target_lum / (current_lum + 1e-6)
        ratio = np.clip(ratio, 0.5, 2.0)  # Limit correction

        result = image.copy()
        for c in range(3):
            result[..., c] = result[..., c] * ratio

        return result


def despill_green(
    image: np.ndarray,
    strength: float = 1.0,
    algorithm: str = "average"
) -> np.ndarray:
    """Quick green despill function."""
    algo_map = {
        "average": DespillAlgorithm.AVERAGE,
        "maximum": DespillAlgorithm.MAXIMUM,
        "minimum": DespillAlgorithm.MINIMUM,
        "double": DespillAlgorithm.DOUBLE_AVERAGE,
    }

    config = DespillConfig(
        algorithm=algo_map.get(algorithm, DespillAlgorithm.AVERAGE),
        spill_channel=SpillChannel.GREEN,
        strength=strength,
    )

    despill = Despill(config)
    return despill.process(image)


def despill_blue(
    image: np.ndarray,
    strength: float = 1.0,
    algorithm: str = "average"
) -> np.ndarray:
    """Quick blue despill function."""
    algo_map = {
        "average": DespillAlgorithm.AVERAGE,
        "maximum": DespillAlgorithm.MAXIMUM,
        "minimum": DespillAlgorithm.MINIMUM,
    }

    config = DespillConfig(
        algorithm=algo_map.get(algorithm, DespillAlgorithm.AVERAGE),
        spill_channel=SpillChannel.BLUE,
        strength=strength,
    )

    despill = Despill(config)
    return despill.process(image)


def despill_average(image: np.ndarray, channel: str = "green") -> np.ndarray:
    """Simple average despill."""
    channel_map = {
        "green": SpillChannel.GREEN,
        "blue": SpillChannel.BLUE,
    }

    config = DespillConfig(
        algorithm=DespillAlgorithm.AVERAGE,
        spill_channel=channel_map.get(channel, SpillChannel.GREEN),
    )

    despill = Despill(config)
    return despill.process(image)
