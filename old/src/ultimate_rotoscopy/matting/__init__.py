"""
Advanced Alpha Matting System
==============================

Professional matting with alpha split and motion blur awareness.

Modules:
- alpha_split: Core/Edge/Hair decomposition
- motion_blur_aware: Motion blur detection and handling
- professional_matting: Integrated pipeline

Key Features:
- 3-way alpha split (core/edge/hair)
- Motion blur detection (Laplacian + optical flow)
- Adaptive alpha mixing
- Temporal consistency
- Professional multi-layer export
"""

from ultimate_rotoscopy.matting.alpha_split import (
    AlphaComponent,
    AlphaSplitConfig,
    AlphaSplitResult,
    AlphaSplitter,
    visualize_alpha_split,
    export_alpha_split,
)

from ultimate_rotoscopy.matting.motion_blur_aware import (
    MotionBlurLevel,
    MotionBlurConfig,
    MotionBlurResult,
    MotionBlurDetector,
    MotionBlurAwareMatting,
    visualize_motion_blur_result,
)

from ultimate_rotoscopy.matting.professional_matting import (
    ProfessionalMattingConfig,
    ProfessionalMattingResult,
    ProfessionalMatting,
)

__all__ = [
    # Alpha Split
    "AlphaComponent",
    "AlphaSplitConfig",
    "AlphaSplitResult",
    "AlphaSplitter",
    "visualize_alpha_split",
    "export_alpha_split",
    # Motion Blur
    "MotionBlurLevel",
    "MotionBlurConfig",
    "MotionBlurResult",
    "MotionBlurDetector",
    "MotionBlurAwareMatting",
    "visualize_motion_blur_result",
    # Professional Matting
    "ProfessionalMattingConfig",
    "ProfessionalMattingResult",
    "ProfessionalMatting",
]
