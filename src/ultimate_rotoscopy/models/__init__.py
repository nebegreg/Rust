"""
AI Models for Ultimate Rotoscopy
================================

This module contains integrations for cutting-edge AI models:
- SAM3: Segment Anything Model 3 for object segmentation
- RobustSAM: SAM for degraded/motion-blurred images
- Depth Anything V3: Monocular depth estimation
- Matte Anything: Alpha matte generation
- MatAnyone: Consistent memory-based video matting (CVPR 2025)
- GVM: Generative Video Matting with diffusion
"""

from ultimate_rotoscopy.models.sam3 import SAM3Segmentor, SAM3, SAM3Config
from ultimate_rotoscopy.models.robust_sam import RobustSAM, RobustSAMConfig, DegradationType
from ultimate_rotoscopy.models.depth_anything import DepthAnythingV3
from ultimate_rotoscopy.models.matte_anything import MatteAnything
from ultimate_rotoscopy.models.matanyone import MatAnyone, MatAnyoneConfig
from ultimate_rotoscopy.models.gvm import GVM, GVMConfig, GVMMode
from ultimate_rotoscopy.models.base import BaseModel

__all__ = [
    # SAM3
    "SAM3Segmentor",
    "SAM3",
    "SAM3Config",
    # RobustSAM
    "RobustSAM",
    "RobustSAMConfig",
    "DegradationType",
    # Depth
    "DepthAnythingV3",
    # Matting
    "MatteAnything",
    "MatAnyone",
    "MatAnyoneConfig",
    "GVM",
    "GVMConfig",
    "GVMMode",
    # Base
    "BaseModel",
]
