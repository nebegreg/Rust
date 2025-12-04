"""
Plugin System for Ultimate Rotoscopy
====================================

OpenFX and other plugin interfaces for integration
with professional compositing applications:
- Nuke
- Flame
- DaVinci Resolve
- Natron
"""

from ultimate_rotoscopy.plugins.openfx import (
    OFXPluginBundle,
    OFXImageEffect,
    OFXParameter,
    OFXParamType,
    OFXClip,
    OFXPluginInfo,
    OFXContext,
    UltimateMatteOFX,
    DepthExtractOFX,
    get_bundle,
    OfxGetNumberOfPlugins,
    OfxGetPlugin,
)

__all__ = [
    "OFXPluginBundle",
    "OFXImageEffect",
    "OFXParameter",
    "OFXParamType",
    "OFXClip",
    "OFXPluginInfo",
    "OFXContext",
    "UltimateMatteOFX",
    "DepthExtractOFX",
    "get_bundle",
    "OfxGetNumberOfPlugins",
    "OfxGetPlugin",
]
