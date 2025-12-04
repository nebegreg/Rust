"""
Core Engine for Ultimate Rotoscopy
==================================

Provides the main orchestration layer for all AI models and processing.
"""

from ultimate_rotoscopy.core.engine import RotoscopyEngine
from ultimate_rotoscopy.core.session import Session, SessionConfig

__all__ = [
    "RotoscopyEngine",
    "Session",
    "SessionConfig",
]
