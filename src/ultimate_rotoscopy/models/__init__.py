"""
AI Models for Ultimate Rotoscopy
================================

This module contains integrations for cutting-edge AI models:
- SAM3: Segment Anything Model 3 for object segmentation
- Depth Anything V3: Monocular depth estimation
- Matte Anything: Alpha matte generation
"""

from ultimate_rotoscopy.models.sam3 import SAM3Segmentor
from ultimate_rotoscopy.models.depth_anything import DepthAnythingV3
from ultimate_rotoscopy.models.matte_anything import MatteAnything
from ultimate_rotoscopy.models.base import BaseModel

__all__ = [
    "SAM3Segmentor",
    "DepthAnythingV3",
    "MatteAnything",
    "BaseModel",
]
