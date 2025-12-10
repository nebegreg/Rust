"""
Core Engine for Ultimate Rotoscopy
==================================

Provides the main orchestration layer for all AI models and processing.
"""

from ultimate_rotoscopy.core.engine import RotoscopyEngine
from ultimate_rotoscopy.core.session import Session, SessionConfig
from ultimate_rotoscopy.core.ultimate_pipeline import (
    UltimatePipeline,
    UltimatePipelineConfig,
    UltimatePipelineResult,
    PipelineStage,
    MattingBackend,
    ultimate_composite,
)

__all__ = [
    "RotoscopyEngine",
    "Session",
    "SessionConfig",
    # Ultimate Pipeline
    "UltimatePipeline",
    "UltimatePipelineConfig",
    "UltimatePipelineResult",
    "PipelineStage",
    "MattingBackend",
    "ultimate_composite",
]
