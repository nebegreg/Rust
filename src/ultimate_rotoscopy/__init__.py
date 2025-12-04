"""
Ultimate Rotoscopy - AI-Powered Professional VFX Tool
======================================================

A comprehensive rotoscopy application leveraging cutting-edge AI models:
- SAM3 (Segment Anything Model 3) for precise object segmentation
- Depth Anything V3 for depth estimation, normals, and 3D reconstruction
- Matte Anything for professional-grade alpha matting

Designed for professional VFX artists working with tools like Autodesk Flame.
"""

__version__ = "1.0.0"
__author__ = "Ultimate Rotoscopy Team"

from ultimate_rotoscopy.core.engine import RotoscopyEngine
from ultimate_rotoscopy.pipeline.unified import UnifiedPipeline
from ultimate_rotoscopy.models.sam3 import SAM3Segmentor
from ultimate_rotoscopy.models.depth_anything import DepthAnythingV3
from ultimate_rotoscopy.models.matte_anything import MatteAnything
from ultimate_rotoscopy.export.exr_writer import EXRWriter
from ultimate_rotoscopy.export.aov_manager import AOVManager

__all__ = [
    "RotoscopyEngine",
    "UnifiedPipeline",
    "SAM3Segmentor",
    "DepthAnythingV3",
    "MatteAnything",
    "EXRWriter",
    "AOVManager",
    "__version__",
]
