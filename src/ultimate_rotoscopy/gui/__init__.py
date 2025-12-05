"""
Ultimate Rotoscopy GUI
======================

Professional graphical user interface for the Ultimate Rotoscopy application.

Provides:
- Interactive canvas with SAM point/box prompts
- Parameter panels for all AI models
- Timeline for video editing
- Layer management
- Real-time processing with progress feedback
"""

from ultimate_rotoscopy.gui.main_window import MainWindow, main

# Alias for entry point
launch = main

try:
    from ultimate_rotoscopy.gui.backend import (
        ProcessingBackend,
        ProcessingWorker,
        ProcessingStage,
        ProcessingRequest,
        ProcessingResult,
    )
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    ProcessingBackend = None
    ProcessingWorker = None
    ProcessingStage = None
    ProcessingRequest = None
    ProcessingResult = None

__all__ = [
    "MainWindow",
    "main",
    "launch",
    "ProcessingBackend",
    "ProcessingWorker",
    "ProcessingStage",
    "ProcessingRequest",
    "ProcessingResult",
    "BACKEND_AVAILABLE",
]
