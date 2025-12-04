"""
Compositing Module for Ultimate Rotoscopy
==========================================

Professional compositing tools including:
- Despill algorithms
- Pixel spread / edge extend
- Light wrap
- Color harmonization
- Edge integration
"""

from ultimate_rotoscopy.compositing.despill import (
    DespillAlgorithm,
    Despill,
    despill_green,
    despill_blue,
    despill_average,
)
from ultimate_rotoscopy.compositing.edge_operations import (
    PixelSpread,
    EdgeExtend,
    EdgeBlend,
    edge_erode,
    edge_dilate,
)
from ultimate_rotoscopy.compositing.light_wrap import (
    LightWrap,
    apply_light_wrap,
)
from ultimate_rotoscopy.compositing.harmonization import (
    ColorHarmonizer,
    harmonize_colors,
    match_histogram,
)
from ultimate_rotoscopy.compositing.compositor import (
    UltimateCompositor,
    CompositeResult,
)

__all__ = [
    "DespillAlgorithm",
    "Despill",
    "despill_green",
    "despill_blue",
    "despill_average",
    "PixelSpread",
    "EdgeExtend",
    "EdgeBlend",
    "edge_erode",
    "edge_dilate",
    "LightWrap",
    "apply_light_wrap",
    "ColorHarmonizer",
    "harmonize_colors",
    "match_histogram",
    "UltimateCompositor",
    "CompositeResult",
]
