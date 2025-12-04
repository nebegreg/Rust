"""
Ultimate Rotoscopy - AI-Powered Professional VFX Tool
======================================================

A comprehensive rotoscopy application leveraging cutting-edge AI models:
- SAM3 (Segment Anything Model 3) for precise object segmentation
- RobustSAM for degraded/motion-blurred footage
- Depth Anything V3 for depth estimation, normals, and 3D reconstruction
- Matte Anything for professional-grade alpha matting
- MatAnyone for consistent video matting with memory propagation (CVPR 2025)
- GVM (Generative Video Matting) for diffusion-based fine detail
- ViTMatte for transformer-based matting with automatic trimap
- OmniMatte for video layer decomposition with effects
- Background Matting V2 for real-time matting without green screen

Professional compositing pipeline:
- Advanced despill algorithms (Average, Maximum, Double Average, Adaptive)
- Pixel spread / edge extend
- Light wrap for photorealistic integration
- Color harmonization (LAB transfer, Reinhard, Adaptive)
- Shadow and AO extraction from depth
- ACES color management (ACEScg, ACEScc)
- HDR processing and tone mapping

Performance & Integration:
- OpenFX plugins for Nuke/Flame/Resolve
- ONNX/TensorRT acceleration
- Multi-GPU distributed processing
- Intelligent caching system

Designed for professional VFX artists working with tools like Autodesk Flame.
Following ILM StageCraft-style workflows for cinema-quality compositing.
"""

__version__ = "3.0.0"
__author__ = "Ultimate Rotoscopy Team"

# Core
from ultimate_rotoscopy.core.engine import RotoscopyEngine
from ultimate_rotoscopy.core.ultimate_pipeline import (
    UltimatePipeline,
    UltimatePipelineConfig,
    ultimate_composite,
)
from ultimate_rotoscopy.pipeline.unified import UnifiedPipeline

# Models
from ultimate_rotoscopy.models.sam3 import SAM3Segmentor, SAM3
from ultimate_rotoscopy.models.robust_sam import RobustSAM
from ultimate_rotoscopy.models.depth_anything import DepthAnythingV3
from ultimate_rotoscopy.models.matte_anything import MatteAnything
from ultimate_rotoscopy.models.matanyone import MatAnyone
from ultimate_rotoscopy.models.gvm import GVM

# Compositing
from ultimate_rotoscopy.compositing.compositor import UltimateCompositor
from ultimate_rotoscopy.compositing.despill import Despill, despill_green, despill_blue
from ultimate_rotoscopy.compositing.light_wrap import LightWrap, apply_light_wrap
from ultimate_rotoscopy.compositing.harmonization import ColorHarmonizer, harmonize_colors
from ultimate_rotoscopy.compositing.edge_operations import PixelSpread, edge_erode, edge_dilate

# Export
from ultimate_rotoscopy.export.exr_writer import EXRWriter
from ultimate_rotoscopy.export.aov_manager import AOVManager

__all__ = [
    # Core
    "RotoscopyEngine",
    "UltimatePipeline",
    "UltimatePipelineConfig",
    "ultimate_composite",
    "UnifiedPipeline",
    # AI Models
    "SAM3Segmentor",
    "SAM3",
    "RobustSAM",
    "DepthAnythingV3",
    "MatteAnything",
    "MatAnyone",
    "GVM",
    # Compositing
    "UltimateCompositor",
    "Despill",
    "despill_green",
    "despill_blue",
    "LightWrap",
    "apply_light_wrap",
    "ColorHarmonizer",
    "harmonize_colors",
    "PixelSpread",
    "edge_erode",
    "edge_dilate",
    # Export
    "EXRWriter",
    "AOVManager",
    # Meta
    "__version__",
]
