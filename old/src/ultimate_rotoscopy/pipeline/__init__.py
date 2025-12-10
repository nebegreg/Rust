"""
Ultimate Rotoscopy Pipeline
============================

Professional rotoscopy pipeline for cinema-quality results.
Integrates SAM3, Depth Anything 3, and advanced edge refinement.
"""

from ultimate_rotoscopy.pipeline.roto_pipeline import (
    UltimateRotoPipeline,
    RotoConfig,
    RotoResult,
    RotoMode,
    EdgeMode,
    OutputFormat,
    MatteChannel,
    RotoObject,
    Keyframe,
)

# Legacy imports for compatibility
try:
    from ultimate_rotoscopy.pipeline.unified import UnifiedPipeline
    from ultimate_rotoscopy.pipeline.video import VideoPipeline
    from ultimate_rotoscopy.pipeline.batch import BatchProcessor
except ImportError:
    UnifiedPipeline = None
    VideoPipeline = None
    BatchProcessor = None

__all__ = [
    # New Ultimate Pipeline
    "UltimateRotoPipeline",
    "RotoConfig",
    "RotoResult",
    "RotoMode",
    "EdgeMode",
    "OutputFormat",
    "MatteChannel",
    "RotoObject",
    "Keyframe",
    # Legacy
    "UnifiedPipeline",
    "VideoPipeline",
    "BatchProcessor",
]
