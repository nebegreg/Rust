"""
Color Management for Ultimate Rotoscopy
=======================================

Professional color management including:
- ACES color pipeline (ACEScg, ACEScc, ACEScct)
- HDR processing and tone mapping
- Color space conversions
"""

from ultimate_rotoscopy.color.aces import (
    ACESPipeline,
    ACESConfig,
    ACESColorSpace,
    InputColorSpace,
    OutputColorSpace,
    HDRPipeline as ACESHDRPipeline,
    convert_to_acescg,
    convert_from_acescg,
    srgb_to_linear,
    linear_to_srgb,
)

from ultimate_rotoscopy.color.hdr import (
    HDRPipeline,
    HDRConfig,
    HDRResult,
    ToneMappingOperator,
    HDRFormat,
    ToneMapper,
    ExposureMerger,
    HDREncoder,
    tonemap_hdr,
    merge_exposures,
)

__all__ = [
    # ACES
    "ACESPipeline",
    "ACESConfig",
    "ACESColorSpace",
    "InputColorSpace",
    "OutputColorSpace",
    "ACESHDRPipeline",
    "convert_to_acescg",
    "convert_from_acescg",
    "srgb_to_linear",
    "linear_to_srgb",
    # HDR
    "HDRPipeline",
    "HDRConfig",
    "HDRResult",
    "ToneMappingOperator",
    "HDRFormat",
    "ToneMapper",
    "ExposureMerger",
    "HDREncoder",
    "tonemap_hdr",
    "merge_exposures",
]
