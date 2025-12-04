"""
Ultimate Compositor for Ultimate Rotoscopy
============================================

The complete compositing pipeline that combines:
- SAM3 segmentation
- MatAnyone/Matte refinement
- Despill processing
- Pixel spread / edge extend
- Light wrap
- Color harmonization

This creates the "ultimate pipeline" for cinema-quality compositing.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ultimate_rotoscopy.compositing.despill import Despill, DespillConfig, DespillAlgorithm, SpillChannel
from ultimate_rotoscopy.compositing.edge_operations import PixelSpread, EdgeBlend, edge_erode, edge_dilate
from ultimate_rotoscopy.compositing.light_wrap import LightWrap, LightWrapConfig, WrapMode
from ultimate_rotoscopy.compositing.harmonization import ColorHarmonizer, HarmonizationConfig, HarmonizationMethod


class CompositeMode(Enum):
    """Composite blend modes."""
    OVER = "over"           # Standard over composite
    PREMULTIPLIED = "premult"  # Premultiplied alpha
    ADDITIVE = "additive"   # Additive blend


@dataclass
class CompositorConfig:
    """Ultimate compositor configuration."""
    # Despill
    enable_despill: bool = True
    despill_algorithm: DespillAlgorithm = DespillAlgorithm.AVERAGE
    despill_channel: SpillChannel = SpillChannel.GREEN
    despill_strength: float = 0.8

    # Edge operations
    enable_edge_operations: bool = True
    edge_erode_amount: int = 0        # Shrink matte
    edge_dilate_amount: int = 0       # Expand matte
    pixel_spread_amount: int = 5      # Edge extension
    edge_blend_width: int = 10

    # Light wrap
    enable_light_wrap: bool = True
    light_wrap_intensity: float = 0.5
    light_wrap_width: int = 20
    light_wrap_mode: WrapMode = WrapMode.SCREEN

    # Color harmonization
    enable_harmonization: bool = True
    harmonization_method: HarmonizationMethod = HarmonizationMethod.LAB_TRANSFER
    harmonization_strength: float = 0.4

    # Composite
    composite_mode: CompositeMode = CompositeMode.OVER
    output_premultiplied: bool = False

    # Processing order
    processing_order: List[str] = field(default_factory=lambda: [
        "despill",
        "edge_operations",
        "harmonization",
        "light_wrap",
        "composite"
    ])


@dataclass
class CompositeResult:
    """Result from compositor."""
    composite: np.ndarray            # Final composite
    foreground: np.ndarray           # Processed foreground
    alpha: np.ndarray                # Final alpha matte
    despilled: Optional[np.ndarray] = None
    edge_extended: Optional[np.ndarray] = None
    harmonized: Optional[np.ndarray] = None
    light_wrapped: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UltimateCompositor:
    """
    Ultimate Compositing Pipeline.

    Combines all professional compositing techniques for
    cinema-quality results:

    1. Despill - Remove green/blue screen contamination
    2. Edge Operations - Pixel spread, erode/dilate
    3. Color Harmonization - Match FG to BG colors
    4. Light Wrap - Integrate BG light onto FG edges
    5. Final Composite - Alpha over with proper blending

    This follows the professional VFX pipeline used at studios
    like ILM, where "despilling is arguably the most important step"
    and the background must be in place for proper evaluation.

    Example:
        >>> compositor = UltimateCompositor(CompositorConfig(
        ...     enable_despill=True,
        ...     enable_light_wrap=True,
        ...     light_wrap_intensity=0.6
        ... ))
        >>>
        >>> result = compositor.composite(foreground, background, alpha)
        >>> final = result.composite
    """

    def __init__(self, config: Optional[CompositorConfig] = None):
        self.config = config or CompositorConfig()

        # Initialize components
        self.despill = Despill(DespillConfig(
            algorithm=self.config.despill_algorithm,
            spill_channel=self.config.despill_channel,
            strength=self.config.despill_strength,
        )) if self.config.enable_despill else None

        self.pixel_spread = PixelSpread(
            amount=self.config.pixel_spread_amount,
        ) if self.config.enable_edge_operations else None

        self.edge_blend = EdgeBlend(
            blend_width=self.config.edge_blend_width,
        ) if self.config.enable_edge_operations else None

        self.light_wrap = LightWrap(LightWrapConfig(
            intensity=self.config.light_wrap_intensity,
            wrap_width=self.config.light_wrap_width,
            wrap_mode=self.config.light_wrap_mode,
        )) if self.config.enable_light_wrap else None

        self.harmonizer = ColorHarmonizer(HarmonizationConfig(
            method=self.config.harmonization_method,
            strength=self.config.harmonization_strength,
        )) if self.config.enable_harmonization else None

    def composite(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        alpha: np.ndarray,
        clean_plate: Optional[np.ndarray] = None,
    ) -> CompositeResult:
        """
        Perform full compositing pipeline.

        Args:
            foreground: Foreground RGB image
            background: Background RGB image
            alpha: Alpha matte
            clean_plate: Optional clean background plate (no subject)

        Returns:
            CompositeResult with all intermediate and final outputs
        """
        # Normalize inputs
        foreground = self._normalize(foreground)
        background = self._normalize(background)
        alpha = self._normalize_alpha(alpha)

        if clean_plate is not None:
            clean_plate = self._normalize(clean_plate)

        # Track intermediate results
        result = CompositeResult(
            composite=np.zeros_like(foreground),
            foreground=foreground.copy(),
            alpha=alpha.copy(),
        )

        # Process according to order
        current_fg = foreground.copy()
        current_alpha = alpha.copy()

        for step in self.config.processing_order:
            if step == "despill" and self.config.enable_despill:
                current_fg = self._apply_despill(current_fg, current_alpha, background)
                result.despilled = current_fg.copy()

            elif step == "edge_operations" and self.config.enable_edge_operations:
                current_fg, current_alpha = self._apply_edge_operations(
                    current_fg, current_alpha
                )
                result.edge_extended = current_fg.copy()

            elif step == "harmonization" and self.config.enable_harmonization:
                current_fg = self._apply_harmonization(current_fg, background, current_alpha)
                result.harmonized = current_fg.copy()

            elif step == "light_wrap" and self.config.enable_light_wrap:
                current_fg = self._apply_light_wrap(current_fg, background, current_alpha)
                result.light_wrapped = current_fg.copy()

            elif step == "composite":
                pass  # Done at the end

        # Final composite
        result.foreground = current_fg
        result.alpha = current_alpha
        result.composite = self._final_composite(current_fg, background, current_alpha)

        # Add metadata
        result.metadata = {
            "processing_order": self.config.processing_order,
            "despill_enabled": self.config.enable_despill,
            "light_wrap_enabled": self.config.enable_light_wrap,
            "harmonization_enabled": self.config.enable_harmonization,
        }

        return result

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 float."""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)

    def _normalize_alpha(self, alpha: np.ndarray) -> np.ndarray:
        """Normalize alpha to 0-1 float, single channel."""
        alpha = self._normalize(alpha)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        return alpha

    def _apply_despill(
        self,
        foreground: np.ndarray,
        alpha: np.ndarray,
        background: np.ndarray,
    ) -> np.ndarray:
        """Apply despill processing."""
        return self.despill.process(foreground, alpha, background)

    def _apply_edge_operations(
        self,
        foreground: np.ndarray,
        alpha: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply edge operations."""
        # Erode if requested
        if self.config.edge_erode_amount > 0:
            alpha = edge_erode(alpha, self.config.edge_erode_amount)

        # Dilate if requested
        if self.config.edge_dilate_amount > 0:
            alpha = edge_dilate(alpha, self.config.edge_dilate_amount)

        # Pixel spread
        if self.config.pixel_spread_amount > 0:
            foreground = self.pixel_spread.process(foreground, alpha)

        return foreground, alpha

    def _apply_harmonization(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """Apply color harmonization."""
        return self.harmonizer.harmonize(foreground, background, alpha)

    def _apply_light_wrap(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """Apply light wrap."""
        return self.light_wrap.apply(foreground, background, alpha)

    def _final_composite(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """Perform final alpha composite."""
        alpha_3ch = np.stack([alpha] * 3, axis=-1)

        if self.config.composite_mode == CompositeMode.OVER:
            # Standard alpha over
            composite = foreground * alpha_3ch + background * (1 - alpha_3ch)

        elif self.config.composite_mode == CompositeMode.PREMULTIPLIED:
            # Premultiplied composite
            fg_premult = foreground * alpha_3ch
            composite = fg_premult + background * (1 - alpha_3ch)

        elif self.config.composite_mode == CompositeMode.ADDITIVE:
            # Additive composite
            composite = foreground * alpha_3ch + background

        else:
            composite = foreground * alpha_3ch + background * (1 - alpha_3ch)

        return np.clip(composite, 0, 1)

    def composite_video_frame(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        alpha: np.ndarray,
        prev_composite: Optional[np.ndarray] = None,
        temporal_blend: float = 0.0,
    ) -> CompositeResult:
        """
        Composite a video frame with optional temporal blending.

        Args:
            foreground: Current frame foreground
            background: Current frame background
            alpha: Current frame alpha
            prev_composite: Previous frame composite for temporal consistency
            temporal_blend: Blend factor with previous frame (0-1)

        Returns:
            CompositeResult for current frame
        """
        result = self.composite(foreground, background, alpha)

        # Apply temporal blending if requested
        if prev_composite is not None and temporal_blend > 0:
            prev_composite = self._normalize(prev_composite)
            result.composite = (
                result.composite * (1 - temporal_blend) +
                prev_composite * temporal_blend
            )

        return result


def quick_composite(
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,
    despill: bool = True,
    light_wrap: bool = True,
    harmonize: bool = True,
) -> np.ndarray:
    """
    Quick compositing function with sensible defaults.

    Args:
        foreground: Foreground RGB
        background: Background RGB
        alpha: Alpha matte
        despill: Enable despill
        light_wrap: Enable light wrap
        harmonize: Enable color harmonization

    Returns:
        Final composite image
    """
    config = CompositorConfig(
        enable_despill=despill,
        enable_light_wrap=light_wrap,
        enable_harmonization=harmonize,
    )

    compositor = UltimateCompositor(config)
    result = compositor.composite(foreground, background, alpha)

    return result.composite
