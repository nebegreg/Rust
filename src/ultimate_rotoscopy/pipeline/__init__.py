"""
Processing Pipelines for Ultimate Rotoscopy
============================================

Provides unified pipelines for batch processing and video sequences.
"""

from ultimate_rotoscopy.pipeline.unified import UnifiedPipeline
from ultimate_rotoscopy.pipeline.video import VideoPipeline
from ultimate_rotoscopy.pipeline.batch import BatchProcessor

__all__ = [
    "UnifiedPipeline",
    "VideoPipeline",
    "BatchProcessor",
]
