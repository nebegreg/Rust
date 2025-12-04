"""
HDR Processing for Ultimate Rotoscopy
=====================================

High Dynamic Range image processing including:
- HDR merging from exposure brackets
- Tone mapping for SDR displays
- HDR10/Dolby Vision output
- Linear workflow support
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ToneMappingOperator(Enum):
    """Tone mapping operators."""
    REINHARD = "reinhard"               # Classic Reinhard
    REINHARD_EXTENDED = "reinhard_ext"  # Extended Reinhard
    FILMIC = "filmic"                   # Filmic curve (ACES-style)
    HABLE = "hable"                     # Uncharted 2 tone map
    ACES = "aces"                       # ACES RRT approximation
    AGXBASE = "agxbase"                 # AgX base contrast


class HDRFormat(Enum):
    """HDR output formats."""
    HDR10 = "hdr10"           # HDR10 (PQ + Rec.2020)
    HLG = "hlg"               # Hybrid Log-Gamma
    DOLBY_VISION = "dolby"    # Dolby Vision (metadata)
    LINEAR = "linear"          # Linear HDR (EXR)


@dataclass
class HDRConfig:
    """HDR processing configuration."""
    # Tone mapping
    tone_mapper: ToneMappingOperator = ToneMappingOperator.FILMIC
    exposure: float = 0.0           # EV adjustment
    gamma: float = 2.2              # Display gamma

    # Dynamic range
    white_point: float = 4.0        # Scene white point
    black_point: float = 0.0        # Scene black point

    # HDR output
    output_format: HDRFormat = HDRFormat.LINEAR
    peak_luminance: float = 1000.0  # Peak nits for HDR
    reference_white: float = 203.0   # SDR white in nits

    # Merging settings
    response_curve: str = "linear"   # linear, log, gamma
    alignment: bool = True           # Align exposure brackets
    ghost_removal: bool = True       # Remove ghosting artifacts


@dataclass
class HDRResult:
    """Result from HDR processing."""
    hdr: np.ndarray                 # HDR image (linear)
    tonemapped: np.ndarray          # Tone mapped for display
    luminance: np.ndarray           # Luminance map
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExposureMerger:
    """
    Merge multiple exposures into HDR.

    Uses Debevec's method or similar for
    recovering high dynamic range from LDR brackets.
    """

    def __init__(self, config: HDRConfig):
        self.config = config

    def merge(
        self,
        images: List[np.ndarray],
        exposures: List[float],
    ) -> np.ndarray:
        """
        Merge exposure bracket into HDR.

        Args:
            images: List of LDR images at different exposures
            exposures: Exposure times for each image

        Returns:
            HDR image in linear space
        """
        import cv2

        # Convert to proper format
        images_8bit = []
        for img in images:
            if img.dtype == np.float32:
                img = (img * 255).astype(np.uint8)
            images_8bit.append(img)

        exposures = np.array(exposures, dtype=np.float32)

        # Align if enabled
        if self.config.alignment:
            images_8bit = self._align_images(images_8bit)

        # Create merger
        merge_debevec = cv2.createMergeDebevec()

        # Estimate response function
        calibrate = cv2.createCalibrateDebevec()
        response = calibrate.process(images_8bit, exposures)

        # Merge
        hdr = merge_debevec.process(images_8bit, exposures, response)

        # Remove ghosts if enabled
        if self.config.ghost_removal:
            hdr = self._remove_ghosts(hdr, images_8bit, exposures)

        return hdr

    def _align_images(
        self,
        images: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Align exposure bracket images."""
        import cv2

        alignMTB = cv2.createAlignMTB()
        aligned = images.copy()
        alignMTB.process(aligned, aligned)

        return aligned

    def _remove_ghosts(
        self,
        hdr: np.ndarray,
        images: List[np.ndarray],
        exposures: np.ndarray,
    ) -> np.ndarray:
        """Remove ghosting artifacts from HDR."""
        import cv2

        # Use median image as reference
        median_idx = len(images) // 2

        # Convert HDR to reference exposure
        ref_ldr = hdr * exposures[median_idx] * 255

        # Find ghost regions
        diff = np.abs(ref_ldr - images[median_idx].astype(np.float32))
        ghost_mask = np.mean(diff, axis=-1) > 50

        # Dilate mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        ghost_mask = cv2.dilate(ghost_mask.astype(np.uint8), kernel)

        # Replace ghost regions with single exposure
        ghost_mask_3ch = ghost_mask[..., np.newaxis].astype(np.float32)
        single_hdr = images[median_idx].astype(np.float32) / (exposures[median_idx] * 255)

        hdr = hdr * (1 - ghost_mask_3ch) + single_hdr * ghost_mask_3ch

        return hdr


class ToneMapper:
    """
    Tone mapping for HDR to SDR conversion.

    Implements multiple tone mapping operators for
    different aesthetic and technical requirements.
    """

    def __init__(self, config: HDRConfig):
        self.config = config

    def tonemap(self, hdr: np.ndarray) -> np.ndarray:
        """
        Tone map HDR image to displayable range.

        Args:
            hdr: HDR image in linear space

        Returns:
            Tone mapped image in gamma space
        """
        # Apply exposure
        exposed = hdr * (2 ** self.config.exposure)

        # Apply tone mapping operator
        if self.config.tone_mapper == ToneMappingOperator.REINHARD:
            mapped = self._reinhard(exposed)
        elif self.config.tone_mapper == ToneMappingOperator.REINHARD_EXTENDED:
            mapped = self._reinhard_extended(exposed)
        elif self.config.tone_mapper == ToneMappingOperator.FILMIC:
            mapped = self._filmic(exposed)
        elif self.config.tone_mapper == ToneMappingOperator.HABLE:
            mapped = self._hable(exposed)
        elif self.config.tone_mapper == ToneMappingOperator.ACES:
            mapped = self._aces(exposed)
        elif self.config.tone_mapper == ToneMappingOperator.AGXBASE:
            mapped = self._agx_base(exposed)
        else:
            mapped = np.clip(exposed, 0, 1)

        # Apply gamma
        mapped = np.power(np.clip(mapped, 0, 1), 1 / self.config.gamma)

        return mapped

    def _reinhard(self, hdr: np.ndarray) -> np.ndarray:
        """Simple Reinhard operator."""
        return hdr / (1 + hdr)

    def _reinhard_extended(self, hdr: np.ndarray) -> np.ndarray:
        """Extended Reinhard with white point."""
        white = self.config.white_point
        return hdr * (1 + hdr / (white * white)) / (1 + hdr)

    def _filmic(self, hdr: np.ndarray) -> np.ndarray:
        """Filmic tone mapping (ACES-inspired)."""
        # Simplified filmic curve
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14

        return np.clip(
            (hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e),
            0, 1
        )

    def _hable(self, hdr: np.ndarray) -> np.ndarray:
        """Hable/Uncharted 2 tone mapping."""
        def hable_partial(x):
            A = 0.15
            B = 0.50
            C = 0.10
            D = 0.20
            E = 0.02
            F = 0.30
            return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

        exposure_bias = 2.0
        curr = hable_partial(exposure_bias * hdr)
        white_scale = 1.0 / hable_partial(self.config.white_point)

        return curr * white_scale

    def _aces(self, hdr: np.ndarray) -> np.ndarray:
        """ACES RRT/ODT approximation."""
        # Narkowicz ACES fit
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14

        return np.clip(
            (hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e),
            0, 1
        )

    def _agx_base(self, hdr: np.ndarray) -> np.ndarray:
        """AgX base contrast."""
        # Simplified AgX
        # Apply log encoding
        log_encoded = np.log2(np.maximum(hdr, 1e-6)) / 10 + 0.5

        # Apply sigmoid
        sigmoid = 1 / (1 + np.exp(-6 * (log_encoded - 0.5)))

        return np.clip(sigmoid, 0, 1)


class HDREncoder:
    """
    Encode HDR for various output formats.

    Supports HDR10, HLG, and Dolby Vision encoding.
    """

    def __init__(self, config: HDRConfig):
        self.config = config

    def encode(
        self,
        hdr: np.ndarray,
        format: Optional[HDRFormat] = None,
    ) -> np.ndarray:
        """
        Encode HDR for output format.

        Args:
            hdr: Linear HDR image
            format: Output format (uses config default if None)

        Returns:
            Encoded image
        """
        format = format or self.config.output_format

        if format == HDRFormat.HDR10:
            return self._encode_hdr10(hdr)
        elif format == HDRFormat.HLG:
            return self._encode_hlg(hdr)
        elif format == HDRFormat.DOLBY_VISION:
            return self._encode_dolby(hdr)
        else:  # LINEAR
            return hdr

    def _encode_hdr10(self, hdr: np.ndarray) -> np.ndarray:
        """Encode to HDR10 (PQ curve)."""
        # Normalize to peak luminance
        normalized = hdr * self.config.reference_white / self.config.peak_luminance

        # Apply PQ (ST.2084) EOTF
        return self._linear_to_pq(normalized)

    def _encode_hlg(self, hdr: np.ndarray) -> np.ndarray:
        """Encode to HLG."""
        # HLG OETF
        a = 0.17883277
        b = 0.28466892
        c = 0.55991073

        # Normalize
        normalized = hdr * self.config.reference_white / 1000

        hlg = np.where(
            normalized <= 1/12,
            np.sqrt(3 * normalized),
            a * np.log(12 * normalized - b) + c
        )

        return np.clip(hlg, 0, 1)

    def _encode_dolby(self, hdr: np.ndarray) -> np.ndarray:
        """Encode for Dolby Vision (simplified)."""
        # Use PQ as base
        pq = self._encode_hdr10(hdr)

        # Dolby Vision would add metadata here
        # This is a simplified version

        return pq

    def _linear_to_pq(self, linear: np.ndarray) -> np.ndarray:
        """Convert linear to PQ (ST.2084)."""
        m1 = 0.1593017578125
        m2 = 78.84375
        c1 = 0.8359375
        c2 = 18.8515625
        c3 = 18.6875

        linear = np.clip(linear, 0, 1)
        Lm = np.power(linear, m1)
        pq = np.power((c1 + c2 * Lm) / (1 + c3 * Lm), m2)

        return pq


class HDRPipeline:
    """
    Complete HDR processing pipeline.

    Handles:
    - HDR merging from brackets
    - Tone mapping for SDR
    - HDR encoding for output

    Example:
        >>> hdr = HDRPipeline(HDRConfig(
        ...     tone_mapper=ToneMappingOperator.FILMIC,
        ...     output_format=HDRFormat.HDR10,
        ... ))
        >>>
        >>> # Merge exposure bracket
        >>> result = hdr.merge_exposures(images, exposures)
        >>>
        >>> # Or process single HDR image
        >>> result = hdr.process(hdr_image)
    """

    def __init__(self, config: Optional[HDRConfig] = None):
        self.config = config or HDRConfig()
        self.merger = ExposureMerger(self.config)
        self.tonemapper = ToneMapper(self.config)
        self.encoder = HDREncoder(self.config)

    def merge_exposures(
        self,
        images: List[np.ndarray],
        exposures: List[float],
    ) -> HDRResult:
        """
        Merge exposure bracket into HDR.

        Args:
            images: LDR images at different exposures
            exposures: Exposure times

        Returns:
            HDRResult with merged HDR and tone mapped image
        """
        hdr = self.merger.merge(images, exposures)
        tonemapped = self.tonemapper.tonemap(hdr)
        luminance = self._compute_luminance(hdr)

        return HDRResult(
            hdr=hdr,
            tonemapped=tonemapped,
            luminance=luminance,
            metadata={
                "num_exposures": len(images),
                "exposure_range": [min(exposures), max(exposures)],
            }
        )

    def process(
        self,
        hdr: np.ndarray,
        encode_output: bool = False,
    ) -> HDRResult:
        """
        Process HDR image.

        Args:
            hdr: Linear HDR image
            encode_output: Also encode for output format

        Returns:
            HDRResult
        """
        tonemapped = self.tonemapper.tonemap(hdr)
        luminance = self._compute_luminance(hdr)

        if encode_output:
            encoded = self.encoder.encode(hdr)
        else:
            encoded = hdr

        return HDRResult(
            hdr=encoded,
            tonemapped=tonemapped,
            luminance=luminance,
            metadata={
                "tone_mapper": self.config.tone_mapper.value,
                "output_format": self.config.output_format.value,
            }
        )

    def _compute_luminance(self, hdr: np.ndarray) -> np.ndarray:
        """Compute luminance from HDR."""
        return np.dot(hdr, [0.2126, 0.7152, 0.0722])


def tonemap_hdr(
    hdr: np.ndarray,
    operator: str = "filmic",
    exposure: float = 0.0,
) -> np.ndarray:
    """
    Quick tone mapping.

    Args:
        hdr: Linear HDR image
        operator: Tone mapping operator name
        exposure: EV exposure adjustment

    Returns:
        Tone mapped image
    """
    config = HDRConfig(
        tone_mapper=ToneMappingOperator(operator),
        exposure=exposure,
    )
    tonemapper = ToneMapper(config)
    return tonemapper.tonemap(hdr)


def merge_exposures(
    images: List[np.ndarray],
    exposures: List[float],
) -> np.ndarray:
    """
    Quick exposure merging.

    Args:
        images: LDR images
        exposures: Exposure times

    Returns:
        HDR image
    """
    config = HDRConfig()
    merger = ExposureMerger(config)
    return merger.merge(images, exposures)
