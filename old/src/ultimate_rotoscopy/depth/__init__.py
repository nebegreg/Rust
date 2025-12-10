"""
Depth Processing for Ultimate Rotoscopy
=======================================

Depth-based effects and utilities:
- Shadow generation from depth
- Ambient occlusion (SSAO)
- Contact shadows
- Depth-based compositing
"""

from ultimate_rotoscopy.depth.shadow_ao import (
    ShadowAOGenerator,
    ShadowAOResult,
    ShadowConfig,
    AOConfig,
    ShadowType,
    AOType,
    ContactShadowGenerator,
    ProjectedShadowGenerator,
    SelfShadowGenerator,
    SSAOGenerator,
    DepthUtils,
    generate_shadow,
    generate_ao,
)

__all__ = [
    "ShadowAOGenerator",
    "ShadowAOResult",
    "ShadowConfig",
    "AOConfig",
    "ShadowType",
    "AOType",
    "ContactShadowGenerator",
    "ProjectedShadowGenerator",
    "SelfShadowGenerator",
    "SSAOGenerator",
    "DepthUtils",
    "generate_shadow",
    "generate_ao",
]
