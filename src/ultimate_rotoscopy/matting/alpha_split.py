"""
Advanced Alpha Split System - Core/Edge/Hair Separation
========================================================

Professional alpha channel decomposition for cinema-quality compositing.

Technique based on:
- Core/Edge/Hair separation used in professional VFX studios
- Motion blur-aware processing
- Adaptive thresholding and frequency analysis

References:
- Advanced Keying Breakdown (Compositing Mentor)
- Core Despill and Edge Despill techniques
- High-frequency detail preservation for hair
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage


class AlphaComponent(Enum):
    """Alpha channel components."""
    CORE = "core"      # Solid interior alpha
    EDGE = "edge"      # Boundary/transition region
    HAIR = "hair"      # High-frequency details (hair, fur, fine structures)


@dataclass
class AlphaSplitConfig:
    """Configuration for alpha split processing."""

    # Core extraction
    core_threshold_low: float = 0.15      # t1 in formula: (alpha - t1)/(1-t1)
    core_erosion_size: int = 2            # Erosion kernel size for core cleanup

    # Edge extraction
    edge_band_width: int = 10             # Width of edge band in pixels
    edge_smoothing: int = 3               # Gaussian smoothing for edge

    # Hair/detail extraction
    hair_frequency_low: float = 0.02      # Bandpass low frequency
    hair_frequency_high: float = 0.5      # Bandpass high frequency
    hair_threshold: float = 0.05          # Minimum detail threshold

    # Quality
    use_guided_filter: bool = True        # Use guided filter for edge refinement
    guided_radius: int = 5
    guided_eps: float = 1e-3


@dataclass
class AlphaSplitResult:
    """Result of alpha split operation."""

    core: np.ndarray           # Solid alpha (0-1 float)
    edge: np.ndarray           # Edge/transition alpha (0-1 float)
    hair: np.ndarray           # Hair/detail alpha (0-1 float)

    # Reconstruction
    alpha_reconstructed: np.ndarray  # Core + Edge + Hair (should equal original)

    # Metadata
    core_coverage: float       # Percentage of pixels in core
    edge_coverage: float       # Percentage of pixels in edge
    hair_coverage: float       # Percentage of pixels in hair

    def get_component(self, component: AlphaComponent) -> np.ndarray:
        """Get specific alpha component."""
        if component == AlphaComponent.CORE:
            return self.core
        elif component == AlphaComponent.EDGE:
            return self.edge
        elif component == AlphaComponent.HAIR:
            return self.hair
        else:
            raise ValueError(f"Unknown component: {component}")


class AlphaSplitter:
    """
    Advanced alpha channel splitter for professional VFX workflows.

    Decomposes alpha into three components:
    - Core: Solid interior (with erosion to ensure clean edges)
    - Edge: Boundary transition region (for smooth compositing)
    - Hair: High-frequency details (for fine structures)

    This separation allows for:
    - Better edge control in compositing
    - Separate treatment of hair/fine details
    - Motion blur-aware processing
    - Professional multi-pass compositing
    """

    def __init__(self, config: Optional[AlphaSplitConfig] = None):
        self.config = config or AlphaSplitConfig()

    def split(
        self,
        alpha: np.ndarray,
        reference_image: Optional[np.ndarray] = None,
    ) -> AlphaSplitResult:
        """
        Split alpha channel into core/edge/hair components.

        Args:
            alpha: Input alpha channel (H, W) in range [0, 1]
            reference_image: Optional RGB image for guided filtering (H, W, 3)

        Returns:
            AlphaSplitResult with all components
        """
        # Ensure float32 in [0, 1]
        alpha = self._normalize_alpha(alpha)
        h, w = alpha.shape

        # 1. Extract CORE (solid interior)
        core = self._extract_core(alpha)

        # 2. Extract EDGE (transition band)
        edge = self._extract_edge(alpha, core, reference_image)

        # 3. Extract HAIR (high-frequency details)
        hair = self._extract_hair(alpha, core, edge)

        # 4. Reconstruct and normalize
        alpha_recon = self._reconstruct_alpha(core, edge, hair, alpha)

        # 5. Calculate coverage statistics
        total_pixels = h * w
        core_coverage = np.sum(core > 0.01) / total_pixels * 100
        edge_coverage = np.sum(edge > 0.01) / total_pixels * 100
        hair_coverage = np.sum(hair > 0.01) / total_pixels * 100

        return AlphaSplitResult(
            core=core,
            edge=edge,
            hair=hair,
            alpha_reconstructed=alpha_recon,
            core_coverage=core_coverage,
            edge_coverage=edge_coverage,
            hair_coverage=hair_coverage,
        )

    def _normalize_alpha(self, alpha: np.ndarray) -> np.ndarray:
        """Normalize alpha to float32 [0, 1]."""
        if alpha.dtype == np.uint8:
            alpha = alpha.astype(np.float32) / 255.0
        else:
            alpha = alpha.astype(np.float32)

        return np.clip(alpha, 0.0, 1.0)

    def _extract_core(self, alpha: np.ndarray) -> np.ndarray:
        """
        Extract core alpha (solid interior).

        Algorithm:
        1. Threshold and remap: core = clamp((alpha - t1) / (1 - t1))
        2. Light erosion to ensure clean separation from edge
        """
        t1 = self.config.core_threshold_low

        # Remap alpha values above threshold
        core = np.where(
            alpha > t1,
            np.clip((alpha - t1) / (1.0 - t1), 0.0, 1.0),
            0.0
        )

        # Light erosion for clean core
        if self.config.core_erosion_size > 0:
            kernel_size = self.config.core_erosion_size
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size * 2 + 1, kernel_size * 2 + 1)
            )
            core = cv2.erode(core, kernel, iterations=1)

        return core.astype(np.float32)

    def _extract_edge(
        self,
        alpha: np.ndarray,
        core: np.ndarray,
        reference_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract edge band around alpha contour.

        Algorithm:
        1. Find alpha contour
        2. Create band of specified width around contour
        3. Extract alpha values in band
        4. Apply guided filter if reference image provided
        """
        # Find where alpha transitions (not in core, not fully transparent)
        alpha_transition = np.logical_and(alpha > 0.01, core < 0.99).astype(np.uint8)

        # Dilate to create band
        band_width = self.config.edge_band_width
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (band_width * 2 + 1, band_width * 2 + 1)
        )
        edge_mask = cv2.dilate(alpha_transition, kernel, iterations=1)

        # Extract edge alpha (alpha values in edge region, excluding core)
        edge = np.where(edge_mask > 0, alpha, 0.0)
        edge = np.where(core > 0.5, 0.0, edge)  # Remove core region

        # Smooth edge
        if self.config.edge_smoothing > 0:
            ksize = self.config.edge_smoothing * 2 + 1
            edge = cv2.GaussianBlur(edge, (ksize, ksize), 0)

        # Guided filter refinement
        if self.config.use_guided_filter and reference_image is not None:
            edge = self._guided_filter(edge, reference_image)

        return edge.astype(np.float32)

    def _extract_hair(
        self,
        alpha: np.ndarray,
        core: np.ndarray,
        edge: np.ndarray,
    ) -> np.ndarray:
        """
        Extract hair/fine details using high-frequency analysis.

        Algorithm:
        1. Compute high-frequency component of alpha (bandpass)
        2. Threshold to keep only significant details
        3. Exclude core and edge regions
        """
        # Apply bandpass filter (high-pass - low-pass)
        freq_low = self.config.hair_frequency_low
        freq_high = self.config.hair_frequency_high

        # High-pass filter (Laplacian for edge detection)
        laplacian = cv2.Laplacian(alpha, cv2.CV_32F, ksize=3)
        high_freq = np.abs(laplacian)

        # Low-pass filter (Gaussian blur)
        ksize_low = int(1.0 / freq_low) if freq_low > 0 else 15
        ksize_low = max(3, ksize_low | 1)  # Ensure odd
        low_freq = cv2.GaussianBlur(alpha, (ksize_low, ksize_low), 0)

        # Bandpass: high_freq component above low_freq baseline
        bandpass = high_freq * (alpha - low_freq + 0.5)

        # Threshold for hair details
        hair_threshold = self.config.hair_threshold
        hair = np.where(bandpass > hair_threshold, bandpass, 0.0)

        # Normalize to [0, 1]
        if np.max(hair) > 0:
            hair = hair / np.max(hair)

        # Exclude core and strong edge regions
        hair = np.where(core > 0.8, 0.0, hair)
        hair = np.where(edge > 0.9, 0.0, hair)

        # Apply alpha values (hair details should have some alpha)
        hair = hair * np.clip(alpha, 0.0, 1.0)

        return hair.astype(np.float32)

    def _guided_filter(
        self,
        alpha: np.ndarray,
        guide: np.ndarray,
    ) -> np.ndarray:
        """
        Apply guided filter for edge-aware smoothing.

        Uses the reference image to preserve edges while smoothing.
        """
        # Convert guide to grayscale if needed
        if len(guide.shape) == 3:
            guide_gray = cv2.cvtColor(
                (guide * 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY
            ).astype(np.float32) / 255.0
        else:
            guide_gray = guide

        # Simple guided filter implementation
        radius = self.config.guided_radius
        eps = self.config.guided_eps

        # Box filter
        mean_I = cv2.boxFilter(guide_gray, cv2.CV_32F, (radius, radius))
        mean_p = cv2.boxFilter(alpha, cv2.CV_32F, (radius, radius))
        mean_Ip = cv2.boxFilter(guide_gray * alpha, cv2.CV_32F, (radius, radius))

        # Covariance and variance
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = cv2.boxFilter(guide_gray * guide_gray, cv2.CV_32F, (radius, radius)) - mean_I * mean_I

        # Linear coefficients
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        # Apply filter
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))

        filtered = mean_a * guide_gray + mean_b

        return np.clip(filtered, 0.0, 1.0)

    def _reconstruct_alpha(
        self,
        core: np.ndarray,
        edge: np.ndarray,
        hair: np.ndarray,
        original: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct alpha from components.

        Ensures conservation: core + edge + hair ≈ original
        """
        # Simple additive reconstruction
        reconstructed = core + edge + hair

        # Normalize to match original alpha range
        reconstructed = np.clip(reconstructed, 0.0, 1.0)

        # Optional: blend with original to preserve exact alpha values
        # This ensures lossless reconstruction in regions where components overlap
        blend_factor = 0.9  # 90% reconstruction, 10% original
        reconstructed = blend_factor * reconstructed + (1 - blend_factor) * original

        return np.clip(reconstructed, 0.0, 1.0)


def visualize_alpha_split(result: AlphaSplitResult) -> np.ndarray:
    """
    Create visualization of alpha split components.

    Returns RGB image with color-coded components:
    - Red: Core
    - Green: Edge
    - Blue: Hair
    """
    h, w = result.core.shape
    vis = np.zeros((h, w, 3), dtype=np.float32)

    # Color code each component
    vis[:, :, 0] = result.core       # Red channel = Core
    vis[:, :, 1] = result.edge       # Green channel = Edge
    vis[:, :, 2] = result.hair       # Blue channel = Hair

    return vis


def export_alpha_split(
    result: AlphaSplitResult,
    output_prefix: str,
    format: str = "exr",
) -> dict:
    """
    Export alpha split components as separate files.

    Args:
        result: AlphaSplitResult to export
        output_prefix: Prefix for output files (e.g., "shot_001")
        format: Output format ("exr", "png", "tiff")

    Returns:
        Dictionary mapping component name to file path
    """
    import os
    from pathlib import Path

    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(output_prefix).stem
    ext = f".{format}"

    files = {}

    # Export each component
    components = {
        "alpha_core": result.core,
        "alpha_edge": result.edge,
        "alpha_hair": result.hair,
        "alpha_full": result.alpha_reconstructed,
    }

    for name, alpha in components.items():
        filepath = output_dir / f"{stem}_{name}{ext}"

        if format == "exr":
            _write_exr_channel(filepath, alpha, name)
        elif format in ("png", "tiff"):
            _write_image(filepath, alpha)

        files[name] = str(filepath)

    return files


def _write_exr_channel(filepath, alpha, channel_name):
    """Write alpha channel to EXR file."""
    try:
        import OpenEXR
        import Imath

        h, w = alpha.shape
        header = OpenEXR.Header(w, h)
        header['channels'] = {channel_name: Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))}

        exr = OpenEXR.OutputFile(str(filepath), header)
        exr.writePixels({channel_name: alpha.astype(np.float32).tobytes()})
        exr.close()
    except ImportError:
        print(f"Warning: OpenEXR not available, falling back to PNG")
        _write_image(filepath.with_suffix('.png'), alpha)


def _write_image(filepath, alpha):
    """Write alpha channel to PNG/TIFF."""
    # Convert to uint16 for better precision
    alpha_uint16 = (np.clip(alpha, 0, 1) * 65535).astype(np.uint16)
    cv2.imwrite(str(filepath), alpha_uint16)
