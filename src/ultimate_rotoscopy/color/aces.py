"""
ACES Color Management for Ultimate Rotoscopy
=============================================

Professional ACES (Academy Color Encoding System) workflow support
for cinema-quality color management.

ACES provides:
- Standardized color interchange
- Wide color gamut preservation
- Consistent look across displays
- Future-proof archival

Key color spaces:
- ACES2065-1: Linear, AP0 primaries (archival)
- ACEScg: Linear, AP1 primaries (CG/compositing)
- ACEScc: Log, AP1 primaries (grading)
- ACEScct: Log with toe, AP1 primaries (grading)

Reference: https://acescentral.com/
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ACESColorSpace(Enum):
    """ACES color space variants."""
    ACES2065_1 = "aces2065_1"   # AP0, linear (archival master)
    ACESCG = "acescg"           # AP1, linear (CG working space)
    ACESCC = "acescc"           # AP1, log (color correction)
    ACESCCT = "acescct"         # AP1, log with toe (grading)


class InputColorSpace(Enum):
    """Common input color spaces."""
    SRGB = "srgb"
    LINEAR_SRGB = "linear_srgb"
    REC709 = "rec709"
    REC2020 = "rec2020"
    LINEAR_REC2020 = "linear_rec2020"
    P3_D65 = "p3_d65"
    LINEAR_P3 = "linear_p3"
    ARRI_LOGC3 = "arri_logc3"
    ARRI_LOGC4 = "arri_logc4"
    RED_LOG3G10 = "red_log3g10"
    SONY_SLOG3 = "sony_slog3"
    BMD_FILM_GEN5 = "bmd_film_gen5"


class OutputColorSpace(Enum):
    """Common output color spaces."""
    SRGB = "srgb"
    REC709 = "rec709"
    REC2020 = "rec2020"
    P3_D65 = "p3_d65"
    P3_DCI = "p3_dci"


@dataclass
class ACESConfig:
    """ACES color management configuration."""
    working_space: ACESColorSpace = ACESColorSpace.ACESCG
    input_space: InputColorSpace = InputColorSpace.SRGB
    output_space: OutputColorSpace = OutputColorSpace.SRGB

    # OCIO config path (optional)
    ocio_config: Optional[str] = None

    # Gamut mapping
    enable_gamut_compression: bool = True
    gamut_compression_threshold: float = 0.8

    # HDR settings
    hdr_enabled: bool = False
    peak_luminance: float = 1000.0   # nits
    reference_white: float = 100.0    # nits for SDR

    # Tone mapping for SDR output
    tone_mapping: str = "aces_rrt"   # aces_rrt, reinhard, filmic


# Color space primaries (CIE xy)
COLOR_PRIMARIES = {
    "ap0": {  # ACES 2065-1
        "red": (0.7347, 0.2653),
        "green": (-0.0001, 1.0),  # Imaginary
        "blue": (0.0001, -0.0770),  # Imaginary
        "white": (0.32168, 0.33767),
    },
    "ap1": {  # ACEScg
        "red": (0.713, 0.293),
        "green": (0.165, 0.830),
        "blue": (0.128, 0.044),
        "white": (0.32168, 0.33767),
    },
    "srgb": {
        "red": (0.64, 0.33),
        "green": (0.30, 0.60),
        "blue": (0.15, 0.06),
        "white": (0.3127, 0.3290),
    },
    "rec2020": {
        "red": (0.708, 0.292),
        "green": (0.170, 0.797),
        "blue": (0.131, 0.046),
        "white": (0.3127, 0.3290),
    },
    "p3_d65": {
        "red": (0.680, 0.320),
        "green": (0.265, 0.690),
        "blue": (0.150, 0.060),
        "white": (0.3127, 0.3290),
    },
}


def _xy_to_XYZ(x: float, y: float) -> np.ndarray:
    """Convert CIE xy to XYZ (Y=1)."""
    return np.array([x / y, 1.0, (1 - x - y) / y])


def _compute_rgb_to_xyz_matrix(primaries: dict) -> np.ndarray:
    """Compute RGB to XYZ matrix from primaries."""
    # Get XYZ for primaries
    Xr, Yr, Zr = _xy_to_XYZ(*primaries["red"])
    Xg, Yg, Zg = _xy_to_XYZ(*primaries["green"])
    Xb, Yb, Zb = _xy_to_XYZ(*primaries["blue"])

    # White point
    Xw, Yw, Zw = _xy_to_XYZ(*primaries["white"])

    # Solve for scaling factors
    M = np.array([
        [Xr, Xg, Xb],
        [Yr, Yg, Yb],
        [Zr, Zg, Zb],
    ])

    S = np.linalg.solve(M, np.array([Xw, Yw, Zw]))

    # Build final matrix
    return M * S


# Pre-computed matrices
SRGB_TO_XYZ = _compute_rgb_to_xyz_matrix(COLOR_PRIMARIES["srgb"])
XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ)

AP1_TO_XYZ = _compute_rgb_to_xyz_matrix(COLOR_PRIMARIES["ap1"])
XYZ_TO_AP1 = np.linalg.inv(AP1_TO_XYZ)

AP0_TO_XYZ = _compute_rgb_to_xyz_matrix(COLOR_PRIMARIES["ap0"])
XYZ_TO_AP0 = np.linalg.inv(AP0_TO_XYZ)

# Direct conversion matrices
SRGB_TO_AP1 = XYZ_TO_AP1 @ SRGB_TO_XYZ
AP1_TO_SRGB = XYZ_TO_SRGB @ AP1_TO_XYZ

SRGB_TO_AP0 = XYZ_TO_AP0 @ SRGB_TO_XYZ
AP0_TO_SRGB = XYZ_TO_SRGB @ AP0_TO_XYZ

AP1_TO_AP0 = XYZ_TO_AP0 @ AP1_TO_XYZ
AP0_TO_AP1 = XYZ_TO_AP1 @ AP0_TO_XYZ


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear RGB."""
    # sRGB EOTF
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4)
    )
    return linear


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB."""
    # sRGB inverse EOTF
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1/2.4) - 0.055
    )
    return np.clip(srgb, 0, 1)


def acescc_to_linear(acescc: np.ndarray) -> np.ndarray:
    """Convert ACEScc to linear ACEScg."""
    # ACEScc encoding
    linear = np.where(
        acescc < -0.3014,
        (np.power(2, acescc * 17.52 - 9.72) - np.power(2, -16)) * 2,
        np.power(2, acescc * 17.52 - 9.72)
    )
    return linear


def linear_to_acescc(linear: np.ndarray) -> np.ndarray:
    """Convert linear ACEScg to ACEScc."""
    # Clamp negatives
    linear = np.maximum(linear, 0)

    acescc = np.where(
        linear < 0.00003051757,
        (np.log2(0.00001525878 + linear * 0.5) + 9.72) / 17.52,
        (np.log2(linear) + 9.72) / 17.52
    )
    return acescc


def acescct_to_linear(acescct: np.ndarray) -> np.ndarray:
    """Convert ACEScct to linear ACEScg."""
    CUT = 0.155251141552511
    A = 10.5402377416545
    B = 0.0729055341958355

    linear = np.where(
        acescct <= CUT,
        (acescct - B) / A,
        np.power(2, acescct * 17.52 - 9.72)
    )
    return linear


def linear_to_acescct(linear: np.ndarray) -> np.ndarray:
    """Convert linear ACEScg to ACEScct."""
    CUT_LINEAR = 0.0078125
    A = 10.5402377416545
    B = 0.0729055341958355

    acescct = np.where(
        linear <= CUT_LINEAR,
        A * linear + B,
        (np.log2(np.maximum(linear, 1e-10)) + 9.72) / 17.52
    )
    return acescct


def apply_matrix(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply 3x3 color matrix to RGB image."""
    shape = rgb.shape
    rgb_flat = rgb.reshape(-1, 3)
    result = rgb_flat @ matrix.T
    return result.reshape(shape)


class ACESPipeline:
    """
    ACES Color Management Pipeline.

    Provides professional color management for VFX workflows:
    - Input transform from camera/source color space
    - Working in ACEScg (linear, wide gamut)
    - Output transform to display color space

    Example:
        >>> aces = ACESPipeline(ACESConfig(
        ...     input_space=InputColorSpace.SRGB,
        ...     working_space=ACESColorSpace.ACESCG,
        ...     output_space=OutputColorSpace.REC709,
        ... ))
        >>>
        >>> # Convert input to working space
        >>> working = aces.input_to_working(srgb_image)
        >>>
        >>> # ... do compositing in ACEScg ...
        >>>
        >>> # Convert to output for display
        >>> display = aces.working_to_output(composited)
    """

    def __init__(self, config: Optional[ACESConfig] = None):
        self.config = config or ACESConfig()
        self._ocio = None
        self._init_ocio()

    def _init_ocio(self):
        """Initialize OpenColorIO if available."""
        try:
            import PyOpenColorIO as OCIO

            if self.config.ocio_config:
                self._ocio = OCIO.Config.CreateFromFile(self.config.ocio_config)
            else:
                # Try to find default ACES config
                try:
                    self._ocio = OCIO.Config.CreateFromEnv()
                except Exception:
                    self._ocio = None

        except ImportError:
            self._ocio = None

    def input_to_working(self, image: np.ndarray) -> np.ndarray:
        """
        Convert input image to ACES working space.

        Args:
            image: Input image in input_space

        Returns:
            Image in ACEScg (or configured working space)
        """
        # Normalize
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Use OCIO if available
        if self._ocio is not None:
            return self._ocio_transform(
                image,
                self._input_space_name(),
                self._working_space_name()
            )

        # Fallback to manual conversion
        return self._manual_input_to_working(image)

    def working_to_output(self, image: np.ndarray) -> np.ndarray:
        """
        Convert working space image to output display space.

        Args:
            image: Image in ACEScg

        Returns:
            Image in output_space ready for display
        """
        # Use OCIO if available
        if self._ocio is not None:
            return self._ocio_transform(
                image,
                self._working_space_name(),
                self._output_space_name()
            )

        # Fallback to manual conversion
        return self._manual_working_to_output(image)

    def _manual_input_to_working(self, image: np.ndarray) -> np.ndarray:
        """Manual input to working space conversion."""
        # Step 1: Linearize if needed
        if self.config.input_space == InputColorSpace.SRGB:
            linear = srgb_to_linear(image)
        elif self.config.input_space == InputColorSpace.LINEAR_SRGB:
            linear = image
        elif self.config.input_space == InputColorSpace.ARRI_LOGC3:
            linear = self._logc3_to_linear(image)
        else:
            # Assume already linear
            linear = image

        # Step 2: Convert to AP1 (ACEScg)
        if self.config.input_space in (InputColorSpace.SRGB, InputColorSpace.LINEAR_SRGB):
            acescg = apply_matrix(linear, SRGB_TO_AP1)
        else:
            # Assume already in ACEScg
            acescg = linear

        # Step 3: Gamut compression if enabled
        if self.config.enable_gamut_compression:
            acescg = self._gamut_compress(acescg)

        # Step 4: Convert to working space if not ACEScg
        if self.config.working_space == ACESColorSpace.ACES2065_1:
            return apply_matrix(acescg, AP1_TO_AP0)
        elif self.config.working_space == ACESColorSpace.ACESCC:
            return linear_to_acescc(acescg)
        elif self.config.working_space == ACESColorSpace.ACESCCT:
            return linear_to_acescct(acescg)

        return acescg

    def _manual_working_to_output(self, image: np.ndarray) -> np.ndarray:
        """Manual working space to output conversion."""
        # Step 1: Convert to linear ACEScg
        if self.config.working_space == ACESColorSpace.ACES2065_1:
            acescg = apply_matrix(image, AP0_TO_AP1)
        elif self.config.working_space == ACESColorSpace.ACESCC:
            acescg = acescc_to_linear(image)
        elif self.config.working_space == ACESColorSpace.ACESCCT:
            acescg = acescct_to_linear(image)
        else:
            acescg = image

        # Step 2: Apply RRT+ODT (simplified)
        if self.config.tone_mapping == "aces_rrt":
            display = self._aces_rrt_odt(acescg)
        elif self.config.tone_mapping == "reinhard":
            display = self._reinhard_tonemap(acescg)
        else:
            # Simple clamp
            display = np.clip(acescg, 0, 1)

        # Step 3: Convert to output color space
        if self.config.output_space == OutputColorSpace.SRGB:
            # ACEScg to sRGB
            linear_srgb = apply_matrix(display, AP1_TO_SRGB)
            return linear_to_srgb(np.clip(linear_srgb, 0, 1))
        elif self.config.output_space == OutputColorSpace.REC709:
            linear_srgb = apply_matrix(display, AP1_TO_SRGB)
            return linear_to_srgb(np.clip(linear_srgb, 0, 1))  # Same as sRGB for SDR

        return np.clip(display, 0, 1)

    def _aces_rrt_odt(self, acescg: np.ndarray) -> np.ndarray:
        """
        Simplified ACES RRT + sRGB ODT.

        This is a simplified approximation of the full ACES pipeline.
        For production use, use OpenColorIO with official ACES transforms.
        """
        # Simplified S-curve tone mapping
        # Based on ACES RRT characteristics

        # Apply exposure
        exposed = acescg * 0.6  # Exposure adjustment

        # Simplified filmic curve
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14

        tonemapped = (exposed * (a * exposed + b)) / (exposed * (c * exposed + d) + e)

        return np.clip(tonemapped, 0, 1)

    def _reinhard_tonemap(self, linear: np.ndarray) -> np.ndarray:
        """Reinhard tone mapping."""
        # Extended Reinhard
        white = 4.0  # White point
        tonemapped = linear * (1 + linear / (white * white)) / (1 + linear)
        return np.clip(tonemapped, 0, 1)

    def _gamut_compress(self, acescg: np.ndarray) -> np.ndarray:
        """Apply ACES gamut compression."""
        # Simplified gamut compression
        # Compress out-of-gamut colors back into AP1

        # Check for negative values
        compressed = acescg.copy()

        # Soft clip negatives
        threshold = -0.001
        compressed = np.where(
            compressed < threshold,
            threshold * (1 - np.exp(-(compressed - threshold) / 0.1)),
            compressed
        )

        return compressed

    def _logc3_to_linear(self, logc: np.ndarray) -> np.ndarray:
        """Convert ARRI LogC3 to linear."""
        cut = 0.010591
        a = 5.555556
        b = 0.052272
        c = 0.247190
        d = 0.385537
        e = 5.367655
        f = 0.092809

        linear = np.where(
            logc > e * cut + f,
            (np.power(10, (logc - d) / c) - b) / a,
            (logc - f) / e
        )
        return linear

    def _ocio_transform(
        self,
        image: np.ndarray,
        src_space: str,
        dst_space: str,
    ) -> np.ndarray:
        """Apply OCIO color transform."""
        import PyOpenColorIO as OCIO

        processor = self._ocio.getProcessor(src_space, dst_space)
        cpu = processor.getDefaultCPUProcessor()

        # Process image
        result = image.copy()
        if result.dtype != np.float32:
            result = result.astype(np.float32)

        # OCIO expects pixels in contiguous array
        h, w, c = result.shape
        pixels = result.reshape(-1, c)

        cpu.applyRGB(pixels)

        return pixels.reshape(h, w, c)

    def _input_space_name(self) -> str:
        """Get OCIO-compatible input space name."""
        mapping = {
            InputColorSpace.SRGB: "sRGB - Texture",
            InputColorSpace.LINEAR_SRGB: "Linear Rec.709 (sRGB)",
            InputColorSpace.REC709: "Rec.709 - Display",
            InputColorSpace.ARRI_LOGC3: "ARRI LogC3 EI800",
        }
        return mapping.get(self.config.input_space, "sRGB - Texture")

    def _working_space_name(self) -> str:
        """Get OCIO-compatible working space name."""
        mapping = {
            ACESColorSpace.ACESCG: "ACEScg",
            ACESColorSpace.ACES2065_1: "ACES2065-1",
            ACESColorSpace.ACESCC: "ACEScc",
            ACESColorSpace.ACESCCT: "ACEScct",
        }
        return mapping.get(self.config.working_space, "ACEScg")

    def _output_space_name(self) -> str:
        """Get OCIO-compatible output space name."""
        mapping = {
            OutputColorSpace.SRGB: "sRGB - Display",
            OutputColorSpace.REC709: "Rec.709 - Display",
            OutputColorSpace.REC2020: "Rec.2020 - Display",
            OutputColorSpace.P3_D65: "P3-D65 - Display",
        }
        return mapping.get(self.config.output_space, "sRGB - Display")


class HDRPipeline:
    """
    HDR processing pipeline for ACES.

    Handles high dynamic range content with proper
    tone mapping for various output formats.
    """

    def __init__(self, config: ACESConfig):
        self.config = config
        self.aces = ACESPipeline(config)

    def process(
        self,
        image: np.ndarray,
        target_nits: float = 100.0,
    ) -> np.ndarray:
        """
        Process HDR image for target display.

        Args:
            image: HDR image in ACEScg
            target_nits: Target display peak luminance

        Returns:
            Tone-mapped image for display
        """
        if target_nits > 100:
            # HDR output
            return self._hdr_output(image, target_nits)
        else:
            # SDR output
            return self.aces.working_to_output(image)

    def _hdr_output(
        self,
        image: np.ndarray,
        peak_nits: float,
    ) -> np.ndarray:
        """Process for HDR display."""
        # Scale to nits
        nits = image * self.config.reference_white

        # Apply PQ (ST.2084) curve for HDR10
        pq = self._linear_to_pq(nits / peak_nits)

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


def convert_to_acescg(
    image: np.ndarray,
    from_space: str = "srgb",
) -> np.ndarray:
    """
    Quick conversion to ACEScg.

    Args:
        image: Input image
        from_space: Source color space ('srgb', 'linear_srgb', etc.)

    Returns:
        Image in ACEScg
    """
    config = ACESConfig(
        input_space=InputColorSpace(from_space),
        working_space=ACESColorSpace.ACESCG,
    )
    aces = ACESPipeline(config)
    return aces.input_to_working(image)


def convert_from_acescg(
    image: np.ndarray,
    to_space: str = "srgb",
) -> np.ndarray:
    """
    Quick conversion from ACEScg.

    Args:
        image: Image in ACEScg
        to_space: Target color space

    Returns:
        Converted image
    """
    config = ACESConfig(
        working_space=ACESColorSpace.ACESCG,
        output_space=OutputColorSpace(to_space),
    )
    aces = ACESPipeline(config)
    return aces.working_to_output(image)
