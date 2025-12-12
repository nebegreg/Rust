#!/usr/bin/env python3
"""
Ultimate Rotoscopy GUI
======================

Professional PySide6 GUI for the Ultimate Rotoscopy Pipeline.

Features:
- Interactive viewport with zoom/pan
- SAM3 text, point, and box prompts
- Real-time trimap preview
- Alpha matte visualization
- Depth map display
- Layer decomposition view
- Video timeline for video matting
- Export to multiple formats

Usage:
    python ultimate_roto_gui.py
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import json
import tempfile

import numpy as np
import cv2
from PIL import Image

# Qt imports
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QPushButton, QLineEdit, QTextEdit,
        QFileDialog, QSlider, QComboBox, QCheckBox, QGroupBox,
        QTabWidget, QProgressBar, QStatusBar, QMenuBar, QMenu,
        QToolBar, QSplitter, QScrollArea, QFrame, QSpinBox,
        QDoubleSpinBox, QMessageBox, QInputDialog
    )
    from PySide6.QtCore import Qt, QThread, Signal, QPoint, QRect, QSize, QTimer
    from PySide6.QtGui import (
        QImage, QPixmap, QPainter, QPen, QColor, QBrush,
        QAction, QKeySequence, QFont, QPainterPath, QCursor
    )
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False
    print("PySide6 not available. Install with: pip install PySide6")

# PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# VIEWPORT WIDGET
# =============================================================================

class ImageViewport(QLabel):
    """
    Interactive image viewport with:
    - Zoom and pan
    - Point annotation
    - Box annotation
    - Mask overlay
    """

    point_added = Signal(int, int)  # x, y
    box_drawn = Signal(int, int, int, int)  # x1, y1, x2, y2
    view_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setMouseTracking(True)

        # Image data
        self.original_image: Optional[np.ndarray] = None
        self.display_image: Optional[QPixmap] = None
        self.mask_overlay: Optional[np.ndarray] = None
        self.trimap_overlay: Optional[np.ndarray] = None
        self.depth_overlay: Optional[np.ndarray] = None

        # View state
        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.pan_start = QPoint()

        # Annotation state
        self.annotation_mode = "point"  # point, box, none
        self.points: List[Tuple[int, int]] = []
        self.point_labels: List[int] = []  # 1 = foreground, 0 = background
        self.box_start: Optional[QPoint] = None
        self.box_end: Optional[QPoint] = None
        self.current_box: Optional[Tuple[int, int, int, int]] = None

        # Display options
        self.show_mask = True
        self.show_trimap = False
        self.show_depth = False
        self.mask_opacity = 0.5
        self.mask_color = QColor(0, 255, 0, 128)

    def set_image(self, image: np.ndarray):
        """Set the base image."""
        self.original_image = image.copy()
        self.clear_annotations()
        self._update_display()

    def set_mask(self, mask: np.ndarray):
        """Set mask overlay."""
        self.mask_overlay = mask.copy()
        self._update_display()

    def set_trimap(self, trimap: np.ndarray):
        """Set trimap overlay."""
        self.trimap_overlay = trimap.copy()
        self._update_display()

    def set_depth(self, depth: np.ndarray):
        """Set depth overlay."""
        self.depth_overlay = depth.copy()
        self._update_display()

    def clear_annotations(self):
        """Clear all annotations."""
        self.points.clear()
        self.point_labels.clear()
        self.current_box = None
        self.box_start = None
        self.box_end = None
        self._update_display()

    def set_annotation_mode(self, mode: str):
        """Set annotation mode (point, box, none)."""
        self.annotation_mode = mode
        if mode == "point":
            self.setCursor(QCursor(Qt.CrossCursor))
        elif mode == "box":
            self.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))

    def _screen_to_image(self, pos: QPoint) -> Tuple[int, int]:
        """Convert screen coordinates to image coordinates."""
        if self.original_image is None:
            return (0, 0)

        # Get widget center
        center_x = self.width() / 2
        center_y = self.height() / 2

        # Calculate image position
        img_h, img_w = self.original_image.shape[:2]
        scaled_w = img_w * self.zoom_level
        scaled_h = img_h * self.zoom_level

        img_x = center_x - scaled_w / 2 + self.pan_offset.x()
        img_y = center_y - scaled_h / 2 + self.pan_offset.y()

        # Convert to image coordinates
        x = int((pos.x() - img_x) / self.zoom_level)
        y = int((pos.y() - img_y) / self.zoom_level)

        # Clamp to image bounds
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))

        return (x, y)

    def _update_display(self):
        """Update the display with current image and overlays."""
        if self.original_image is None:
            self.clear()
            return

        # Start with original image
        display = self.original_image.copy()

        # Apply overlays
        if self.show_mask and self.mask_overlay is not None:
            display = self._apply_mask_overlay(display, self.mask_overlay)
        elif self.show_trimap and self.trimap_overlay is not None:
            display = self._apply_trimap_overlay(display, self.trimap_overlay)
        elif self.show_depth and self.depth_overlay is not None:
            display = self._apply_depth_overlay(display, self.depth_overlay)

        # Draw annotations
        display = self._draw_annotations(display)

        # Convert to QPixmap
        h, w = display.shape[:2]
        if len(display.shape) == 3:
            bytes_per_line = 3 * w
            q_image = QImage(display.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            bytes_per_line = w
            q_image = QImage(display.data, w, h, bytes_per_line, QImage.Format_Grayscale8)

        self.display_image = QPixmap.fromImage(q_image)
        self._render()

    def _apply_mask_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask overlay to image."""
        result = image.copy()

        # Normalize mask to 0-1
        if mask.max() > 1:
            mask_norm = mask.astype(np.float32) / 255.0
        else:
            mask_norm = mask.astype(np.float32)

        # Create colored overlay
        overlay_color = np.array([0, 255, 0], dtype=np.float32)  # Green
        overlay = np.zeros_like(result, dtype=np.float32)
        overlay[:] = overlay_color

        # Blend
        alpha = mask_norm[:, :, np.newaxis] * self.mask_opacity
        result = result.astype(np.float32) * (1 - alpha) + overlay * alpha

        return result.astype(np.uint8)

    def _apply_trimap_overlay(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Apply trimap overlay to image."""
        result = image.copy()

        # Color code: 0=red (bg), 128=gray (unknown), 255=green (fg)
        fg_mask = trimap > 200
        unknown_mask = (trimap > 50) & (trimap <= 200)
        bg_mask = trimap <= 50

        overlay = np.zeros_like(result)
        overlay[fg_mask] = [0, 255, 0]  # Green for foreground
        overlay[unknown_mask] = [128, 128, 128]  # Gray for unknown
        overlay[bg_mask] = [255, 0, 0]  # Red for background

        # Blend
        alpha = 0.3
        result = (result.astype(np.float32) * (1 - alpha) +
                  overlay.astype(np.float32) * alpha).astype(np.uint8)

        return result

    def _apply_depth_overlay(self, image: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Apply depth colormap overlay."""
        # Normalize depth
        if depth.max() > 1:
            depth_norm = depth.astype(np.float32) / depth.max()
        else:
            depth_norm = depth.astype(np.float32)

        # Apply turbo colormap
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        # Blend with original
        alpha = 0.5
        result = (image.astype(np.float32) * (1 - alpha) +
                  depth_colored.astype(np.float32) * alpha).astype(np.uint8)

        return result

    def _draw_annotations(self, image: np.ndarray) -> np.ndarray:
        """Draw points and boxes on image."""
        result = image.copy()

        # Draw points
        for (x, y), label in zip(self.points, self.point_labels):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(result, (x, y), 5, color, -1)
            cv2.circle(result, (x, y), 7, (255, 255, 255), 2)

        # Draw current box
        if self.box_start is not None and self.box_end is not None:
            x1, y1 = self._screen_to_image(self.box_start)
            x2, y2 = self._screen_to_image(self.box_end)
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Draw saved box
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return result

    def _render(self):
        """Render the display image with current zoom and pan."""
        if self.display_image is None:
            return

        # Scale
        scaled = self.display_image.scaled(
            int(self.display_image.width() * self.zoom_level),
            int(self.display_image.height() * self.zoom_level),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.setPixmap(scaled)

    def wheelEvent(self, event):
        """Handle zoom with mouse wheel."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_level = min(self.zoom_level * 1.1, 10.0)
        else:
            self.zoom_level = max(self.zoom_level / 1.1, 0.1)
        self._render()
        self.view_changed.emit()

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.pan_start = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        elif event.button() == Qt.LeftButton:
            if self.annotation_mode == "point":
                x, y = self._screen_to_image(event.pos())
                self.points.append((x, y))
                self.point_labels.append(1)  # Foreground
                self.point_added.emit(x, y)
                self._update_display()
            elif self.annotation_mode == "box":
                self.box_start = event.pos()
                self.box_end = event.pos()
        elif event.button() == Qt.RightButton:
            if self.annotation_mode == "point":
                x, y = self._screen_to_image(event.pos())
                self.points.append((x, y))
                self.point_labels.append(0)  # Background
                self.point_added.emit(x, y)
                self._update_display()

    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if self.is_panning:
            delta = event.pos() - self.pan_start
            self.pan_offset += delta
            self.pan_start = event.pos()
            self._render()
        elif self.annotation_mode == "box" and self.box_start is not None:
            self.box_end = event.pos()
            self._update_display()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
            if self.annotation_mode == "point":
                self.setCursor(QCursor(Qt.CrossCursor))
            else:
                self.setCursor(QCursor(Qt.ArrowCursor))
        elif event.button() == Qt.LeftButton:
            if self.annotation_mode == "box" and self.box_start is not None:
                x1, y1 = self._screen_to_image(self.box_start)
                x2, y2 = self._screen_to_image(self.box_end)
                self.current_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                self.box_drawn.emit(*self.current_box)
                self.box_start = None
                self.box_end = None
                self._update_display()

    def reset_view(self):
        """Reset zoom and pan."""
        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)
        self._render()


# =============================================================================
# WORKER THREAD
# =============================================================================

class RotoWorker(QThread):
    """Background worker for rotoscopy processing."""

    progress = Signal(int, str)  # progress %, message
    finished = Signal(object)  # result
    error = Signal(str)  # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.task = None
        self.params = {}

    def set_task(self, task: str, **params):
        """Set the task to execute."""
        self.task = task
        self.params = params

    def run(self):
        """Execute the task."""
        try:
            if self.task == "process_image":
                self._process_image()
            elif self.task == "process_video":
                self._process_video()
            elif self.task == "generate_trimap":
                self._generate_trimap()
        except Exception as e:
            self.error.emit(str(e))

    def _process_image(self):
        """Process image for rotoscopy."""
        from ultimate_roto import UltimateRoto, RotoConfig

        self.progress.emit(10, "Loading models...")

        config = RotoConfig(
            device=self.params.get('device', 'cuda'),
            trimap_erosion=self.params.get('erosion', 15),
            trimap_dilation=self.params.get('dilation', 30),
            depth_enabled=self.params.get('depth', False),
        )

        roto = UltimateRoto(config)

        self.progress.emit(30, "SAM3 Segmentation...")

        result = roto.process_image(
            self.params['image_path'],
            prompt=self.params.get('text'),
            points=self.params.get('points'),
            box=self.params.get('box'),
            estimate_depth=self.params.get('depth', False)
        )

        self.progress.emit(90, "Saving results...")

        if 'output_dir' in self.params:
            roto.save_result(result, self.params['output_dir'])

        self.progress.emit(100, "Complete!")
        self.finished.emit(result)

    def _process_video(self):
        """Process video for rotoscopy."""
        from ultimate_roto import UltimateRoto, RotoConfig

        self.progress.emit(10, "Loading models...")

        config = RotoConfig(
            device=self.params.get('device', 'cuda'),
            video_warmup_frames=self.params.get('warmup', 10),
        )

        roto = UltimateRoto(config)

        self.progress.emit(20, "Processing video...")

        fg_path, alpha_path = roto.process_video(
            self.params['video_path'],
            self.params['text'],
            self.params['output_dir']
        )

        self.progress.emit(100, "Complete!")
        self.finished.emit({'foreground': fg_path, 'alpha': alpha_path})

    def _generate_trimap(self):
        """Generate trimap from mask."""
        from ultimate_roto import TrimapGenerator, RotoConfig

        config = RotoConfig(
            trimap_erosion=self.params.get('erosion', 15),
            trimap_dilation=self.params.get('dilation', 30),
        )

        generator = TrimapGenerator(config)

        mask = self.params['mask']
        image = self.params.get('image')

        trimap = generator.generate_adaptive(mask, image)

        self.finished.emit(trimap)


# =============================================================================
# MAIN WINDOW
# =============================================================================

class UltimateRotoGUI(QMainWindow):
    """Main window for Ultimate Rotoscopy GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultimate Rotoscopy - Professional Alpha Matting")
        self.setMinimumSize(1400, 900)

        # State
        self.current_image: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.current_trimap: Optional[np.ndarray] = None
        self.current_alpha: Optional[np.ndarray] = None
        self.current_depth: Optional[np.ndarray] = None
        self.current_result = None
        self.image_path: Optional[Path] = None

        # Worker
        self.worker = RotoWorker(self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._apply_style()

    def _setup_ui(self):
        """Setup the main UI."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Left panel - Controls
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # Center - Viewport
        self.viewport = ImageViewport()
        self.viewport.point_added.connect(self._on_point_added)
        self.viewport.box_drawn.connect(self._on_box_drawn)
        main_layout.addWidget(self.viewport, 4)

        # Right panel - Results
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 1)

    def _create_left_panel(self) -> QWidget:
        """Create left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)

        # File controls
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self._load_image)
        file_layout.addWidget(load_btn)

        load_video_btn = QPushButton("Load Video")
        load_video_btn.clicked.connect(self._load_video)
        file_layout.addWidget(load_video_btn)

        layout.addWidget(file_group)

        # Prompt controls
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_group)

        prompt_layout.addWidget(QLabel("Text Prompt:"))
        self.text_prompt = QLineEdit()
        self.text_prompt.setPlaceholderText("e.g., person, cat, car...")
        prompt_layout.addWidget(self.text_prompt)

        prompt_layout.addWidget(QLabel("Annotation Mode:"))
        mode_layout = QHBoxLayout()
        self.mode_point = QPushButton("Point")
        self.mode_point.setCheckable(True)
        self.mode_point.setChecked(True)
        self.mode_point.clicked.connect(lambda: self._set_mode("point"))
        mode_layout.addWidget(self.mode_point)

        self.mode_box = QPushButton("Box")
        self.mode_box.setCheckable(True)
        self.mode_box.clicked.connect(lambda: self._set_mode("box"))
        mode_layout.addWidget(self.mode_box)

        prompt_layout.addLayout(mode_layout)

        clear_btn = QPushButton("Clear Annotations")
        clear_btn.clicked.connect(self._clear_annotations)
        prompt_layout.addWidget(clear_btn)

        layout.addWidget(prompt_group)

        # Trimap controls
        trimap_group = QGroupBox("Trimap Settings")
        trimap_layout = QGridLayout(trimap_group)

        trimap_layout.addWidget(QLabel("Erosion:"), 0, 0)
        self.erosion_spin = QSpinBox()
        self.erosion_spin.setRange(1, 100)
        self.erosion_spin.setValue(15)
        trimap_layout.addWidget(self.erosion_spin, 0, 1)

        trimap_layout.addWidget(QLabel("Dilation:"), 1, 0)
        self.dilation_spin = QSpinBox()
        self.dilation_spin.setRange(1, 100)
        self.dilation_spin.setValue(30)
        trimap_layout.addWidget(self.dilation_spin, 1, 1)

        self.hair_check = QCheckBox("Hair Refinement")
        self.hair_check.setChecked(True)
        trimap_layout.addWidget(self.hair_check, 2, 0, 1, 2)

        layout.addWidget(trimap_group)

        # Process controls
        process_group = QGroupBox("Process")
        process_layout = QVBoxLayout(process_group)

        self.depth_check = QCheckBox("Estimate Depth")
        process_layout.addWidget(self.depth_check)

        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        device_layout.addWidget(self.device_combo)
        process_layout.addLayout(device_layout)

        self.process_btn = QPushButton("Process")
        self.process_btn.setStyleSheet("background-color: #4CAF50; font-weight: bold;")
        self.process_btn.clicked.connect(self._process)
        process_layout.addWidget(self.process_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        process_layout.addWidget(self.progress_bar)

        layout.addWidget(process_group)

        layout.addStretch()

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create right results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)

        # View controls
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)

        self.view_original = QPushButton("Original")
        self.view_original.setCheckable(True)
        self.view_original.setChecked(True)
        self.view_original.clicked.connect(lambda: self._set_view("original"))
        view_layout.addWidget(self.view_original)

        self.view_mask = QPushButton("Mask")
        self.view_mask.setCheckable(True)
        self.view_mask.clicked.connect(lambda: self._set_view("mask"))
        view_layout.addWidget(self.view_mask)

        self.view_trimap = QPushButton("Trimap")
        self.view_trimap.setCheckable(True)
        self.view_trimap.clicked.connect(lambda: self._set_view("trimap"))
        view_layout.addWidget(self.view_trimap)

        self.view_alpha = QPushButton("Alpha")
        self.view_alpha.setCheckable(True)
        self.view_alpha.clicked.connect(lambda: self._set_view("alpha"))
        view_layout.addWidget(self.view_alpha)

        self.view_depth = QPushButton("Depth")
        self.view_depth.setCheckable(True)
        self.view_depth.clicked.connect(lambda: self._set_view("depth"))
        view_layout.addWidget(self.view_depth)

        layout.addWidget(view_group)

        # Opacity slider
        opacity_group = QGroupBox("Overlay Opacity")
        opacity_layout = QVBoxLayout(opacity_group)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)

        layout.addWidget(opacity_group)

        # Export controls
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        export_alpha_btn = QPushButton("Export Alpha")
        export_alpha_btn.clicked.connect(lambda: self._export("alpha"))
        export_layout.addWidget(export_alpha_btn)

        export_fg_btn = QPushButton("Export Foreground")
        export_fg_btn.clicked.connect(lambda: self._export("foreground"))
        export_layout.addWidget(export_fg_btn)

        export_all_btn = QPushButton("Export All")
        export_all_btn.clicked.connect(lambda: self._export("all"))
        export_layout.addWidget(export_all_btn)

        layout.addWidget(export_group)

        # Info
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(200)
        layout.addWidget(self.info_text)

        layout.addStretch()

        return panel

    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Image", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._load_image)
        file_menu.addAction(open_action)

        open_video_action = QAction("Open Video", self)
        open_video_action.triggered.connect(self._load_video)
        file_menu.addAction(open_video_action)

        file_menu.addSeparator()

        save_action = QAction("Save Result", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(lambda: self._export("all"))
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")

        clear_action = QAction("Clear Annotations", self)
        clear_action.setShortcut("C")
        clear_action.triggered.connect(self._clear_annotations)
        edit_menu.addAction(clear_action)

        reset_view_action = QAction("Reset View", self)
        reset_view_action.setShortcut("R")
        reset_view_action.triggered.connect(self.viewport.reset_view)
        edit_menu.addAction(reset_view_action)

        # View menu
        view_menu = menubar.addMenu("View")

        original_action = QAction("Original", self)
        original_action.setShortcut("1")
        original_action.triggered.connect(lambda: self._set_view("original"))
        view_menu.addAction(original_action)

        mask_action = QAction("Mask", self)
        mask_action.setShortcut("2")
        mask_action.triggered.connect(lambda: self._set_view("mask"))
        view_menu.addAction(mask_action)

        trimap_action = QAction("Trimap", self)
        trimap_action.setShortcut("3")
        trimap_action.triggered.connect(lambda: self._set_view("trimap"))
        view_menu.addAction(trimap_action)

        alpha_action = QAction("Alpha", self)
        alpha_action.setShortcut("4")
        alpha_action.triggered.connect(lambda: self._set_view("alpha"))
        view_menu.addAction(alpha_action)

        depth_action = QAction("Depth", self)
        depth_action.setShortcut("5")
        depth_action.triggered.connect(lambda: self._set_view("depth"))
        view_menu.addAction(depth_action)

    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        toolbar.addAction("Open", self._load_image)
        toolbar.addAction("Process", self._process)
        toolbar.addSeparator()
        toolbar.addAction("Point Mode", lambda: self._set_mode("point"))
        toolbar.addAction("Box Mode", lambda: self._set_mode("box"))
        toolbar.addSeparator()
        toolbar.addAction("Clear", self._clear_annotations)
        toolbar.addAction("Reset View", self.viewport.reset_view)

    def _setup_statusbar(self):
        """Setup status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

    def _apply_style(self):
        """Apply dark theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QPushButton:checked {
                background-color: #0d47a1;
            }
            QLineEdit, QTextEdit {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #3d3d3d;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0d47a1;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
            QMenuBar {
                background-color: #2b2b2b;
            }
            QMenuBar::item:selected {
                background-color: #3d3d3d;
            }
            QMenu {
                background-color: #2b2b2b;
            }
            QMenu::item:selected {
                background-color: #0d47a1;
            }
            QToolBar {
                background-color: #2b2b2b;
                border: none;
                spacing: 5px;
            }
            QStatusBar {
                background-color: #1a1a1a;
            }
        """)

    # Event handlers

    def _load_image(self):
        """Load image file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image",
            "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if path:
            self.image_path = Path(path)
            self.current_image = np.array(Image.open(path).convert("RGB"))
            self.viewport.set_image(self.current_image)
            self.current_mask = None
            self.current_trimap = None
            self.current_alpha = None
            self.current_depth = None
            self.statusbar.showMessage(f"Loaded: {self.image_path.name}")
            self._update_info()

    def _load_video(self):
        """Load video file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video",
            "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            # Get first frame for preview
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                self.image_path = Path(path)
                self.current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.viewport.set_image(self.current_image)
                self.statusbar.showMessage(f"Loaded video: {self.image_path.name}")

    def _set_mode(self, mode: str):
        """Set annotation mode."""
        self.mode_point.setChecked(mode == "point")
        self.mode_box.setChecked(mode == "box")
        self.viewport.set_annotation_mode(mode)

    def _set_view(self, view: str):
        """Set current view."""
        # Update buttons
        self.view_original.setChecked(view == "original")
        self.view_mask.setChecked(view == "mask")
        self.view_trimap.setChecked(view == "trimap")
        self.view_alpha.setChecked(view == "alpha")
        self.view_depth.setChecked(view == "depth")

        # Update viewport
        self.viewport.show_mask = view == "mask"
        self.viewport.show_trimap = view == "trimap"
        self.viewport.show_depth = view == "depth"

        if view == "alpha" and self.current_alpha is not None:
            alpha_vis = (self.current_alpha * 255).astype(np.uint8)
            alpha_rgb = np.stack([alpha_vis] * 3, axis=-1)
            self.viewport.set_image(alpha_rgb)
        elif view == "original" and self.current_image is not None:
            self.viewport.set_image(self.current_image)

        self.viewport._update_display()

    def _clear_annotations(self):
        """Clear all annotations."""
        self.viewport.clear_annotations()

    def _on_point_added(self, x: int, y: int):
        """Handle point added."""
        self.statusbar.showMessage(f"Point added: ({x}, {y})")

    def _on_box_drawn(self, x1: int, y1: int, x2: int, y2: int):
        """Handle box drawn."""
        self.statusbar.showMessage(f"Box: ({x1}, {y1}) - ({x2}, {y2})")

    def _on_opacity_changed(self, value: int):
        """Handle opacity change."""
        self.viewport.mask_opacity = value / 100.0
        self.viewport._update_display()

    def _process(self):
        """Start processing."""
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "No image loaded")
            return

        # Check prompts
        text = self.text_prompt.text().strip()
        points = self.viewport.points if self.viewport.points else None
        box = self.viewport.current_box

        if not text and not points and not box:
            QMessageBox.warning(
                self, "Error",
                "Please provide a text prompt, add points, or draw a box"
            )
            return

        # Setup worker
        self.worker.set_task(
            "process_image",
            image_path=str(self.image_path),
            text=text if text else None,
            points=list(zip(self.viewport.points, self.viewport.point_labels)) if points else None,
            box=box,
            device=self.device_combo.currentText(),
            erosion=self.erosion_spin.value(),
            dilation=self.dilation_spin.value(),
            depth=self.depth_check.isChecked()
        )

        # Update UI
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start
        self.worker.start()

    def _on_progress(self, value: int, message: str):
        """Handle progress update."""
        self.progress_bar.setValue(value)
        self.statusbar.showMessage(message)

    def _on_finished(self, result):
        """Handle processing finished."""
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if hasattr(result, 'alpha'):
            # Image result
            self.current_result = result
            self.current_alpha = result.alpha
            self.current_trimap = result.trimap
            self.current_mask = (result.alpha > 0.5).astype(np.uint8) * 255

            if result.depth is not None:
                self.current_depth = result.depth
                self.viewport.set_depth(result.depth)

            self.viewport.set_mask(self.current_mask)
            if self.current_trimap is not None:
                self.viewport.set_trimap(self.current_trimap)

            self._set_view("mask")
            self._update_info()

            self.statusbar.showMessage("Processing complete!")
        else:
            # Video result
            QMessageBox.information(
                self, "Complete",
                f"Video processing complete!\n\n"
                f"Foreground: {result['foreground']}\n"
                f"Alpha: {result['alpha']}"
            )

    def _on_error(self, message: str):
        """Handle error."""
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", message)

    def _export(self, what: str):
        """Export results."""
        if self.current_result is None:
            QMessageBox.warning(self, "Error", "No results to export")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if not output_dir:
            return

        from ultimate_roto import UltimateRoto, RotoConfig
        roto = UltimateRoto(RotoConfig())
        roto.save_result(
            self.current_result,
            output_dir,
            prefix=self.image_path.stem if self.image_path else "roto"
        )

        QMessageBox.information(
            self, "Success",
            f"Results saved to {output_dir}"
        )

    def _update_info(self):
        """Update info panel."""
        info = []

        if self.current_image is not None:
            h, w = self.current_image.shape[:2]
            info.append(f"Image: {w}x{h}")

        if self.image_path:
            info.append(f"File: {self.image_path.name}")

        if self.current_alpha is not None:
            info.append(f"Alpha: min={self.current_alpha.min():.3f}, max={self.current_alpha.max():.3f}")

        if self.viewport.points:
            info.append(f"Points: {len(self.viewport.points)}")

        if self.viewport.current_box:
            x1, y1, x2, y2 = self.viewport.current_box
            info.append(f"Box: ({x1},{y1})-({x2},{y2})")

        self.info_text.setText("\n".join(info))


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not PYSIDE_AVAILABLE:
        print("PySide6 required. Install: pip install PySide6")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("Ultimate Rotoscopy")

    window = UltimateRotoGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
