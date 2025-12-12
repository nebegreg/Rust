#!/usr/bin/env python3
"""
Ultimate Rotoscopy GUI
======================

Professional PySide6 GUI for the Ultimate Rotoscopy Pipeline.

Features:
- Interactive viewport with zoom/pan
- SAM3 text, point, and box prompts
- Real-time trimap preview with adjustable erosion/dilation
- Multiple view modes: Original, SAM Mask, Trimap, Alpha, Depth, Layers
- Layer visualization (core/edge/hair)
- Composite preview on custom background
- Video timeline for MatAnyone
- Full configuration panel
- Export to PNG/EXR

Usage:
    python ultimate_roto_gui.py
    python ultimate_roto_gui.py --image photo.jpg
    python ultimate_roto_gui.py --video clip.mp4
"""

import os
os.environ['QT_QPA_PLATFORMTHEME'] = ''
os.environ['GIO_USE_VFS'] = 'local'
os.environ['QT_FILE_DIALOG_NO_NATIVE'] = '1'

import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from enum import Enum
import json

import numpy as np
import cv2
from PIL import Image

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QPushButton, QLineEdit, QTextEdit,
        QFileDialog, QSlider, QComboBox, QCheckBox, QGroupBox,
        QTabWidget, QProgressBar, QStatusBar, QMenuBar, QMenu,
        QToolBar, QSplitter, QScrollArea, QFrame, QSpinBox,
        QDoubleSpinBox, QMessageBox, QColorDialog, QSizePolicy,
        QButtonGroup, QRadioButton
    )
    from PySide6.QtCore import Qt, QThread, Signal, QPoint, QTimer, QSize
    from PySide6.QtGui import (
        QImage, QPixmap, QPainter, QPen, QColor, QBrush,
        QAction, QKeySequence, QCursor, QWheelEvent
    )
except ImportError:
    print("PySide6 required. Install: pip install PySide6")
    sys.exit(1)

import torch


# =============================================================================
# VIEW MODE ENUM
# =============================================================================

class ViewMode(Enum):
    ORIGINAL = "Original"
    SAM_MASK = "SAM Mask"
    TRIMAP = "Trimap"
    ALPHA = "Alpha"
    FOREGROUND = "Foreground"
    COMPOSITE = "Composite"
    DEPTH = "Depth"
    LAYER_CORE = "Core Layer"
    LAYER_EDGE = "Edge Layer"
    LAYER_HAIR = "Hair Layer"


# =============================================================================
# INTERACTIVE VIEWPORT
# =============================================================================

class Viewport(QLabel):
    """Interactive image viewport with zoom, pan, and annotation."""

    point_added = Signal(int, int, int)  # x, y, label (1=fg, 0=bg)
    box_drawn = Signal(int, int, int, int)  # x1, y1, x2, y2
    mouse_moved = Signal(int, int)  # x, y in image coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Image data
        self.image: Optional[np.ndarray] = None
        self.display_pixmap: Optional[QPixmap] = None

        # View state
        self.zoom = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.pan_start = QPoint()

        # Annotation state
        self.mode = "point"  # point, box, pan
        self.points: List[Tuple[int, int, int]] = []  # (x, y, label)
        self.box_start: Optional[QPoint] = None
        self.box_end: Optional[QPoint] = None
        self.current_box: Optional[Tuple[int, int, int, int]] = None

    def set_image(self, image: np.ndarray):
        """Set display image (RGB numpy array)."""
        self.image = image.copy()
        self._update_display()

    def set_mode(self, mode: str):
        """Set interaction mode: point, box, pan."""
        self.mode = mode
        cursors = {
            "point": Qt.CrossCursor,
            "box": Qt.CrossCursor,
            "pan": Qt.OpenHandCursor
        }
        self.setCursor(QCursor(cursors.get(mode, Qt.ArrowCursor)))

    def clear_annotations(self):
        """Clear all annotations."""
        self.points.clear()
        self.current_box = None
        self.box_start = None
        self.box_end = None
        self._update_display()

    def get_points(self) -> List[Tuple[int, int]]:
        """Get foreground points."""
        return [(x, y) for x, y, label in self.points if label == 1]

    def get_point_labels(self) -> List[int]:
        """Get point labels."""
        return [label for _, _, label in self.points]

    def get_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Get current box."""
        return self.current_box

    def _screen_to_image(self, pos: QPoint) -> Tuple[int, int]:
        """Convert screen coords to image coords."""
        if self.image is None:
            return (0, 0)

        # Widget center
        cx, cy = self.width() / 2, self.height() / 2

        # Image dimensions scaled
        h, w = self.image.shape[:2]
        sw, sh = w * self.zoom, h * self.zoom

        # Image top-left in widget coords
        ix = cx - sw / 2 + self.pan_offset.x()
        iy = cy - sh / 2 + self.pan_offset.y()

        # Convert to image coords
        x = int((pos.x() - ix) / self.zoom)
        y = int((pos.y() - iy) / self.zoom)

        # Clamp
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        return (x, y)

    def _update_display(self):
        """Update display with annotations."""
        if self.image is None:
            self.clear()
            return

        display = self.image.copy()

        # Draw points
        for x, y, label in self.points:
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(display, (x, y), 6, color, -1)
            cv2.circle(display, (x, y), 8, (255, 255, 255), 2)

        # Draw box being drawn
        if self.box_start and self.box_end:
            x1, y1 = self._screen_to_image(self.box_start)
            x2, y2 = self._screen_to_image(self.box_end)
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Draw saved box
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Convert to QPixmap
        h, w = display.shape[:2]
        qimg = QImage(display.data, w, h, 3 * w, QImage.Format_RGB888)
        self.display_pixmap = QPixmap.fromImage(qimg)
        self._render()

    def _render(self):
        """Render with current zoom."""
        if self.display_pixmap is None:
            return

        scaled = self.display_pixmap.scaled(
            int(self.display_pixmap.width() * self.zoom),
            int(self.display_pixmap.height() * self.zoom),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def wheelEvent(self, event: QWheelEvent):
        """Zoom with mouse wheel."""
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self.zoom = max(0.1, min(10.0, self.zoom * factor))
        self._render()

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MiddleButton or self.mode == "pan":
            self.is_panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            if self.mode == "point":
                x, y = self._screen_to_image(event.pos())
                self.points.append((x, y, 1))
                self.point_added.emit(x, y, 1)
                self._update_display()
            elif self.mode == "box":
                self.box_start = event.pos()
                self.box_end = event.pos()
        elif event.button() == Qt.RightButton:
            if self.mode == "point":
                x, y = self._screen_to_image(event.pos())
                self.points.append((x, y, 0))
                self.point_added.emit(x, y, 0)
                self._update_display()

    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if self.image is not None:
            x, y = self._screen_to_image(event.pos())
            self.mouse_moved.emit(x, y)

        if self.is_panning:
            delta = event.pos() - self.pan_start
            self.pan_offset += delta
            self.pan_start = event.pos()
            self._render()
        elif self.mode == "box" and self.box_start:
            self.box_end = event.pos()
            self._update_display()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self.mode == "pan"):
            self.is_panning = False
            cursor = Qt.OpenHandCursor if self.mode == "pan" else Qt.CrossCursor
            self.setCursor(cursor)
        elif event.button() == Qt.LeftButton and self.mode == "box" and self.box_start:
            x1, y1 = self._screen_to_image(self.box_start)
            x2, y2 = self._screen_to_image(self.box_end)
            self.current_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.box_drawn.emit(*self.current_box)
            self.box_start = None
            self.box_end = None
            self._update_display()

    def reset_view(self):
        """Reset zoom and pan."""
        self.zoom = 1.0
        self.pan_offset = QPoint(0, 0)
        self._render()


# =============================================================================
# WORKER THREAD
# =============================================================================

class RotoWorker(QThread):
    """Background worker for processing."""

    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.task = None
        self.params = {}

    def setup(self, task: str, **params):
        """Setup task."""
        self.task = task
        self.params = params

    def run(self):
        """Execute task."""
        try:
            if self.task == "process":
                self._process_image()
            elif self.task == "trimap":
                self._generate_trimap()
            elif self.task == "depth":
                self._estimate_depth()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def _process_image(self):
        """Full pipeline processing."""
        from ultimate_roto import UltimateRoto, RotoConfig

        self.progress.emit(5, "Initializing...")

        config = RotoConfig(
            device=self.params.get('device', 'cuda'),
            trimap_erosion=self.params.get('erosion', 15),
            trimap_dilation=self.params.get('dilation', 30),
        )

        roto = UltimateRoto(config)

        self.progress.emit(20, "SAM3 Segmentation...")

        result = roto.process_image(
            self.params['image_path'],
            prompt=self.params.get('text'),
            points=self.params.get('points'),
            box=self.params.get('box'),
            estimate_depth=self.params.get('depth', False)
        )

        self.progress.emit(100, "Complete!")
        self.finished.emit(result)

    def _generate_trimap(self):
        """Generate trimap only."""
        from ultimate_roto import TrimapGenerator, RotoConfig

        config = RotoConfig(
            trimap_erosion=self.params['erosion'],
            trimap_dilation=self.params['dilation'],
            hair_refinement=self.params.get('hair', True)
        )

        gen = TrimapGenerator(config)
        trimap = gen.generate_adaptive(
            self.params['mask'],
            self.params.get('image')
        )

        self.finished.emit({'trimap': trimap})

    def _estimate_depth(self):
        """Estimate depth only."""
        from ultimate_roto import DepthAnything3Processor, RotoConfig

        config = RotoConfig(device=self.params.get('device', 'cuda'))
        processor = DepthAnything3Processor(config)
        depth = processor.estimate(self.params['image'])

        self.finished.emit({'depth': depth})


# =============================================================================
# MAIN WINDOW
# =============================================================================

class UltimateRotoGUI(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultimate Rotoscopy - Professional Alpha Matting")
        self.setMinimumSize(1600, 1000)

        # State
        self.image_path: Optional[Path] = None
        self.original_image: Optional[np.ndarray] = None
        self.sam_mask: Optional[np.ndarray] = None
        self.trimap: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.foreground: Optional[np.ndarray] = None
        self.core_mask: Optional[np.ndarray] = None
        self.edge_mask: Optional[np.ndarray] = None
        self.hair_mask: Optional[np.ndarray] = None
        self.result = None

        self.background_color = np.array([0, 177, 64], dtype=np.uint8)  # Green screen
        self.view_mode = ViewMode.ORIGINAL

        # Worker
        self.worker = RotoWorker()
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        # UI
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._apply_style()

    def _setup_ui(self):
        """Setup main UI."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Left panel - Controls
        left = self._create_controls_panel()
        layout.addWidget(left, 1)

        # Center - Viewport
        self.viewport = Viewport()
        self.viewport.point_added.connect(self._on_point_added)
        self.viewport.box_drawn.connect(self._on_box_drawn)
        self.viewport.mouse_moved.connect(self._on_mouse_moved)
        layout.addWidget(self.viewport, 4)

        # Right panel - View & Export
        right = self._create_view_panel()
        layout.addWidget(right, 1)

    def _create_controls_panel(self) -> QWidget:
        """Create left controls panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMaximumWidth(320)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # === FILE ===
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)

        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self._load_image)
        file_layout.addWidget(btn_load)

        btn_load_video = QPushButton("Load Video")
        btn_load_video.clicked.connect(self._load_video)
        file_layout.addWidget(btn_load_video)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #888;")
        file_layout.addWidget(self.file_label)

        layout.addWidget(file_group)

        # === PROMPT ===
        prompt_group = QGroupBox("SAM3 Prompt")
        prompt_layout = QVBoxLayout(prompt_group)

        prompt_layout.addWidget(QLabel("Text Prompt:"))
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("e.g., person, dog, car...")
        prompt_layout.addWidget(self.text_input)

        prompt_layout.addWidget(QLabel("Annotation Mode:"))
        mode_layout = QHBoxLayout()

        self.btn_point = QPushButton("Point")
        self.btn_point.setCheckable(True)
        self.btn_point.setChecked(True)
        self.btn_point.clicked.connect(lambda: self._set_mode("point"))

        self.btn_box = QPushButton("Box")
        self.btn_box.setCheckable(True)
        self.btn_box.clicked.connect(lambda: self._set_mode("box"))

        self.btn_pan = QPushButton("Pan")
        self.btn_pan.setCheckable(True)
        self.btn_pan.clicked.connect(lambda: self._set_mode("pan"))

        mode_layout.addWidget(self.btn_point)
        mode_layout.addWidget(self.btn_box)
        mode_layout.addWidget(self.btn_pan)
        prompt_layout.addLayout(mode_layout)

        btn_clear = QPushButton("Clear Annotations")
        btn_clear.clicked.connect(self._clear_annotations)
        prompt_layout.addWidget(btn_clear)

        prompt_layout.addWidget(QLabel("Left-click: Foreground | Right-click: Background"))

        layout.addWidget(prompt_group)

        # === TRIMAP ===
        trimap_group = QGroupBox("Trimap Settings")
        trimap_layout = QGridLayout(trimap_group)

        trimap_layout.addWidget(QLabel("Erosion:"), 0, 0)
        self.erosion_spin = QSpinBox()
        self.erosion_spin.setRange(1, 100)
        self.erosion_spin.setValue(15)
        self.erosion_spin.valueChanged.connect(self._on_trimap_changed)
        trimap_layout.addWidget(self.erosion_spin, 0, 1)

        trimap_layout.addWidget(QLabel("Dilation:"), 1, 0)
        self.dilation_spin = QSpinBox()
        self.dilation_spin.setRange(1, 100)
        self.dilation_spin.setValue(30)
        self.dilation_spin.valueChanged.connect(self._on_trimap_changed)
        trimap_layout.addWidget(self.dilation_spin, 1, 1)

        self.hair_check = QCheckBox("Hair Refinement")
        self.hair_check.setChecked(True)
        self.hair_check.stateChanged.connect(self._on_trimap_changed)
        trimap_layout.addWidget(self.hair_check, 2, 0, 1, 2)

        btn_update_trimap = QPushButton("Update Trimap")
        btn_update_trimap.clicked.connect(self._update_trimap)
        trimap_layout.addWidget(btn_update_trimap, 3, 0, 1, 2)

        layout.addWidget(trimap_group)

        # === PROCESSING ===
        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout(process_group)

        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        if not torch.cuda.is_available():
            self.device_combo.setCurrentText("cpu")
        device_layout.addWidget(self.device_combo)
        process_layout.addLayout(device_layout)

        self.depth_check = QCheckBox("Estimate Depth")
        process_layout.addWidget(self.depth_check)

        self.btn_process = QPushButton("Process")
        self.btn_process.setStyleSheet("background-color: #4CAF50; font-weight: bold; font-size: 14px; padding: 10px;")
        self.btn_process.clicked.connect(self._process)
        process_layout.addWidget(self.btn_process)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        process_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #888;")
        process_layout.addWidget(self.progress_label)

        layout.addWidget(process_group)

        # === COMPOSITE ===
        composite_group = QGroupBox("Composite Background")
        composite_layout = QVBoxLayout(composite_group)

        bg_layout = QHBoxLayout()
        self.bg_preview = QLabel()
        self.bg_preview.setFixedSize(30, 30)
        self.bg_preview.setStyleSheet(f"background-color: rgb(0, 177, 64); border: 1px solid #555;")
        bg_layout.addWidget(self.bg_preview)

        btn_pick_color = QPushButton("Pick Color")
        btn_pick_color.clicked.connect(self._pick_background_color)
        bg_layout.addWidget(btn_pick_color)

        btn_load_bg = QPushButton("Load Image")
        btn_load_bg.clicked.connect(self._load_background_image)
        bg_layout.addWidget(btn_load_bg)

        composite_layout.addLayout(bg_layout)

        layout.addWidget(composite_group)

        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _create_view_panel(self) -> QWidget:
        """Create right view/export panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMaximumWidth(280)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # === VIEW MODE ===
        view_group = QGroupBox("View Mode")
        view_layout = QVBoxLayout(view_group)

        self.view_buttons = {}
        for mode in ViewMode:
            btn = QPushButton(mode.value)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, m=mode: self._set_view(m))
            view_layout.addWidget(btn)
            self.view_buttons[mode] = btn

        self.view_buttons[ViewMode.ORIGINAL].setChecked(True)

        layout.addWidget(view_group)

        # === OVERLAY ===
        overlay_group = QGroupBox("Overlay Settings")
        overlay_layout = QVBoxLayout(overlay_group)

        overlay_layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        overlay_layout.addWidget(self.opacity_slider)

        self.opacity_label = QLabel("50%")
        overlay_layout.addWidget(self.opacity_label)

        layout.addWidget(overlay_group)

        # === INFO ===
        info_group = QGroupBox("Info")
        info_layout = QVBoxLayout(info_group)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        info_layout.addWidget(self.info_text)

        layout.addWidget(info_group)

        # === EXPORT ===
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        btn_export_alpha = QPushButton("Export Alpha")
        btn_export_alpha.clicked.connect(lambda: self._export("alpha"))
        export_layout.addWidget(btn_export_alpha)

        btn_export_fg = QPushButton("Export Foreground (RGBA)")
        btn_export_fg.clicked.connect(lambda: self._export("foreground"))
        export_layout.addWidget(btn_export_fg)

        btn_export_all = QPushButton("Export All")
        btn_export_all.clicked.connect(lambda: self._export("all"))
        export_layout.addWidget(btn_export_all)

        layout.addWidget(export_group)

        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open Image", self._load_image, QKeySequence.Open)
        file_menu.addAction("Open Video", self._load_video)
        file_menu.addSeparator()
        file_menu.addAction("Export All", lambda: self._export("all"), QKeySequence.Save)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close, QKeySequence.Quit)

        # Edit
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Clear Annotations", self._clear_annotations, "C")
        edit_menu.addAction("Reset View", self.viewport.reset_view, "R")

        # View
        view_menu = menubar.addMenu("View")
        for i, mode in enumerate(ViewMode):
            if i < 9:
                view_menu.addAction(mode.value, lambda m=mode: self._set_view(m), str(i + 1))

    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction("Open", self._load_image)
        toolbar.addSeparator()
        toolbar.addAction("Point", lambda: self._set_mode("point"))
        toolbar.addAction("Box", lambda: self._set_mode("box"))
        toolbar.addAction("Pan", lambda: self._set_mode("pan"))
        toolbar.addSeparator()
        toolbar.addAction("Clear", self._clear_annotations)
        toolbar.addAction("Process", self._process)

    def _setup_statusbar(self):
        """Setup status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready - Load an image to start")

    def _apply_style(self):
        """Apply dark theme."""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { background-color: #2b2b2b; color: #fff; font-family: 'Segoe UI', Arial; }
            QGroupBox {
                border: 1px solid #444; border-radius: 5px;
                margin-top: 10px; padding-top: 10px; font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton {
                background-color: #3d3d3d; border: 1px solid #555;
                border-radius: 4px; padding: 6px 12px;
            }
            QPushButton:hover { background-color: #4d4d4d; }
            QPushButton:pressed { background-color: #2d2d2d; }
            QPushButton:checked { background-color: #0d47a1; }
            QLineEdit, QTextEdit, QComboBox, QSpinBox {
                background-color: #3d3d3d; border: 1px solid #555;
                border-radius: 4px; padding: 5px;
            }
            QSlider::groove:horizontal { height: 6px; background: #3d3d3d; border-radius: 3px; }
            QSlider::handle:horizontal { background: #0d47a1; width: 16px; margin: -5px 0; border-radius: 8px; }
            QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background-color: #4CAF50; }
            QScrollArea { border: none; }
            QMenuBar { background-color: #2b2b2b; }
            QMenuBar::item:selected { background-color: #3d3d3d; }
            QMenu { background-color: #2b2b2b; }
            QMenu::item:selected { background-color: #0d47a1; }
            QToolBar { background-color: #2b2b2b; border: none; spacing: 5px; }
            QStatusBar { background-color: #1a1a1a; }
        """)

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def _load_image(self):
        """Load image file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if path:
            self.image_path = Path(path)
            self.original_image = np.array(Image.open(path).convert("RGB"))
            self.viewport.set_image(self.original_image)

            # Reset state
            self.sam_mask = None
            self.trimap = None
            self.alpha = None
            self.depth = None
            self.foreground = None
            self.result = None
            self.viewport.clear_annotations()

            self.file_label.setText(f"{self.image_path.name}\n{self.original_image.shape[1]}x{self.original_image.shape[0]}")
            self.statusbar.showMessage(f"Loaded: {self.image_path.name}")
            self._update_info()
            self._set_view(ViewMode.ORIGINAL)

    def _load_video(self):
        """Load video file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if path:
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if ret:
                self.image_path = Path(path)
                self.original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.viewport.set_image(self.original_image)

                self.sam_mask = None
                self.trimap = None
                self.alpha = None
                self.result = None
                self.viewport.clear_annotations()

                self.file_label.setText(f"VIDEO: {self.image_path.name}\n{total} frames @ {fps:.1f}fps")
                self.statusbar.showMessage(f"Loaded video: {self.image_path.name} ({total} frames)")
                self._update_info()

    def _set_mode(self, mode: str):
        """Set annotation mode."""
        self.btn_point.setChecked(mode == "point")
        self.btn_box.setChecked(mode == "box")
        self.btn_pan.setChecked(mode == "pan")
        self.viewport.set_mode(mode)

    def _set_view(self, mode: ViewMode):
        """Set view mode."""
        self.view_mode = mode

        # Update buttons
        for m, btn in self.view_buttons.items():
            btn.setChecked(m == mode)

        # Update display
        self._update_display()

    def _update_display(self):
        """Update viewport based on current view mode."""
        if self.original_image is None:
            return

        mode = self.view_mode
        display = None

        if mode == ViewMode.ORIGINAL:
            display = self.original_image

        elif mode == ViewMode.SAM_MASK:
            if self.sam_mask is not None:
                display = self._overlay_mask(self.original_image, self.sam_mask, (0, 255, 0))
            else:
                display = self.original_image

        elif mode == ViewMode.TRIMAP:
            if self.trimap is not None:
                display = self._colorize_trimap(self.trimap)
            else:
                display = self.original_image

        elif mode == ViewMode.ALPHA:
            if self.alpha is not None:
                alpha_uint8 = (self.alpha * 255).astype(np.uint8)
                display = np.stack([alpha_uint8] * 3, axis=-1)
            else:
                display = self.original_image

        elif mode == ViewMode.FOREGROUND:
            if self.foreground is not None:
                # Show RGBA on checkerboard
                display = self._show_rgba_on_checker(self.foreground)
            else:
                display = self.original_image

        elif mode == ViewMode.COMPOSITE:
            if self.alpha is not None:
                display = self._composite(self.original_image, self.alpha)
            else:
                display = self.original_image

        elif mode == ViewMode.DEPTH:
            if self.depth is not None:
                depth_uint8 = (self.depth * 255).astype(np.uint8)
                display = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
                display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            else:
                display = self.original_image

        elif mode == ViewMode.LAYER_CORE:
            if self.core_mask is not None:
                display = self._overlay_mask(self.original_image, (self.core_mask * 255).astype(np.uint8), (0, 255, 0))
            else:
                display = self.original_image

        elif mode == ViewMode.LAYER_EDGE:
            if self.edge_mask is not None:
                display = self._overlay_mask(self.original_image, (self.edge_mask * 255).astype(np.uint8), (255, 255, 0))
            else:
                display = self.original_image

        elif mode == ViewMode.LAYER_HAIR:
            if self.hair_mask is not None:
                display = self._overlay_mask(self.original_image, (self.hair_mask * 255).astype(np.uint8), (255, 0, 255))
            else:
                display = self.original_image

        if display is not None:
            self.viewport.set_image(display)

    def _overlay_mask(self, image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        """Overlay mask on image."""
        result = image.copy()
        opacity = self.opacity_slider.value() / 100.0

        if mask.max() > 1:
            mask_norm = mask.astype(np.float32) / 255.0
        else:
            mask_norm = mask.astype(np.float32)

        overlay = np.zeros_like(result, dtype=np.float32)
        overlay[:] = color

        alpha = mask_norm[:, :, np.newaxis] * opacity
        result = result.astype(np.float32) * (1 - alpha) + overlay * alpha

        return result.astype(np.uint8)

    def _colorize_trimap(self, trimap: np.ndarray) -> np.ndarray:
        """Colorize trimap: FG=green, Unknown=gray, BG=red."""
        h, w = trimap.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)

        result[trimap > 200] = [0, 255, 0]  # Foreground - Green
        result[(trimap > 50) & (trimap <= 200)] = [128, 128, 128]  # Unknown - Gray
        result[trimap <= 50] = [255, 0, 0]  # Background - Red

        return result

    def _show_rgba_on_checker(self, rgba: np.ndarray) -> np.ndarray:
        """Show RGBA image on checkerboard."""
        h, w = rgba.shape[:2]
        checker = np.zeros((h, w, 3), dtype=np.uint8)

        block = 20
        for y in range(0, h, block):
            for x in range(0, w, block):
                if ((x // block) + (y // block)) % 2 == 0:
                    checker[y:y+block, x:x+block] = [200, 200, 200]
                else:
                    checker[y:y+block, x:x+block] = [150, 150, 150]

        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        rgb = rgba[:, :, :3].astype(np.float32)

        result = rgb * alpha + checker.astype(np.float32) * (1 - alpha)
        return result.astype(np.uint8)

    def _composite(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Composite image over background color."""
        h, w = image.shape[:2]
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:] = self.background_color

        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        result = image.astype(np.float32) * alpha_3ch + bg.astype(np.float32) * (1 - alpha_3ch)

        return result.astype(np.uint8)

    def _clear_annotations(self):
        """Clear annotations."""
        self.viewport.clear_annotations()
        self.statusbar.showMessage("Annotations cleared")

    def _on_point_added(self, x: int, y: int, label: int):
        """Handle point added."""
        label_str = "FG" if label == 1 else "BG"
        self.statusbar.showMessage(f"Point added: ({x}, {y}) [{label_str}]")
        self._update_info()

    def _on_box_drawn(self, x1: int, y1: int, x2: int, y2: int):
        """Handle box drawn."""
        self.statusbar.showMessage(f"Box: ({x1}, {y1}) - ({x2}, {y2})")
        self._update_info()

    def _on_mouse_moved(self, x: int, y: int):
        """Handle mouse move - show coordinates."""
        info = f"({x}, {y})"
        if self.alpha is not None and 0 <= y < self.alpha.shape[0] and 0 <= x < self.alpha.shape[1]:
            info += f" Alpha: {self.alpha[y, x]:.3f}"
        # Update in status bar without clearing main message
        # self.statusbar.showMessage(info)

    def _on_opacity_changed(self, value: int):
        """Handle opacity change."""
        self.opacity_label.setText(f"{value}%")
        self._update_display()

    def _on_trimap_changed(self):
        """Handle trimap settings change."""
        # Could auto-update trimap here if desired
        pass

    def _update_trimap(self):
        """Regenerate trimap with current settings."""
        if self.sam_mask is None:
            QMessageBox.warning(self, "Error", "No mask available. Run processing first.")
            return

        from ultimate_roto import TrimapGenerator, RotoConfig

        config = RotoConfig(
            trimap_erosion=self.erosion_spin.value(),
            trimap_dilation=self.dilation_spin.value(),
            hair_refinement=self.hair_check.isChecked()
        )

        gen = TrimapGenerator(config)
        self.trimap = gen.generate_adaptive(self.sam_mask, self.original_image)

        self._set_view(ViewMode.TRIMAP)
        self.statusbar.showMessage("Trimap updated")

    def _pick_background_color(self):
        """Pick composite background color."""
        color = QColorDialog.getColor(
            QColor(*self.background_color),
            self, "Select Background Color"
        )
        if color.isValid():
            self.background_color = np.array([color.red(), color.green(), color.blue()], dtype=np.uint8)
            self.bg_preview.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); border: 1px solid #555;")
            if self.view_mode == ViewMode.COMPOSITE:
                self._update_display()

    def _load_background_image(self):
        """Load background image for composite."""
        # TODO: Implement background image loading
        QMessageBox.information(self, "Info", "Background image loading - coming soon!")

    def _process(self):
        """Run full processing pipeline."""
        if self.original_image is None:
            QMessageBox.warning(self, "Error", "No image loaded")
            return

        # Get prompts
        text = self.text_input.text().strip()
        points = self.viewport.get_points()
        box = self.viewport.get_box()

        if not text and not points and not box:
            QMessageBox.warning(self, "Error", "Provide a text prompt, add points, or draw a box")
            return

        # Prepare points with labels
        point_data = None
        if points:
            point_data = points  # Just coordinates, labels handled separately

        # Setup worker
        self.worker.setup(
            "process",
            image_path=str(self.image_path),
            text=text if text else None,
            points=point_data,
            box=box,
            device=self.device_combo.currentText(),
            erosion=self.erosion_spin.value(),
            dilation=self.dilation_spin.value(),
            depth=self.depth_check.isChecked()
        )

        # UI update
        self.btn_process.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker.start()

    def _on_progress(self, value: int, message: str):
        """Handle progress update."""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        self.statusbar.showMessage(message)

    def _on_finished(self, result):
        """Handle processing complete."""
        self.btn_process.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")

        if hasattr(result, 'alpha'):
            # Full result
            self.result = result
            self.alpha = result.alpha
            self.trimap = result.trimap
            self.foreground = result.foreground
            self.sam_mask = result.sam_mask
            self.depth = result.depth
            self.core_mask = result.core_mask
            self.edge_mask = result.edge_mask
            self.hair_mask = result.hair_mask

            self._set_view(ViewMode.ALPHA)
            self.statusbar.showMessage("Processing complete!")

        elif 'trimap' in result:
            self.trimap = result['trimap']
            self._set_view(ViewMode.TRIMAP)
            self.statusbar.showMessage("Trimap generated")

        elif 'depth' in result:
            self.depth = result['depth']
            self._set_view(ViewMode.DEPTH)
            self.statusbar.showMessage("Depth estimated")

        self._update_info()

    def _on_error(self, message: str):
        """Handle error."""
        self.btn_process.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")

        QMessageBox.critical(self, "Error", message)
        self.statusbar.showMessage(f"Error: {message}")

    def _update_info(self):
        """Update info panel."""
        lines = []

        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            lines.append(f"Image: {w} x {h}")

        if self.image_path:
            lines.append(f"File: {self.image_path.name}")

        points = self.viewport.get_points()
        if points:
            lines.append(f"Points: {len(points)}")

        box = self.viewport.get_box()
        if box:
            lines.append(f"Box: {box}")

        if self.alpha is not None:
            lines.append(f"Alpha: {self.alpha.min():.3f} - {self.alpha.max():.3f}")

        if self.depth is not None:
            lines.append("Depth: Available")

        self.info_text.setText("\n".join(lines))

    def _export(self, what: str):
        """Export results."""
        if self.result is None:
            QMessageBox.warning(self, "Error", "No results to export. Run processing first.")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if not output_dir:
            return

        from ultimate_roto import UltimateRoto, RotoConfig

        roto = UltimateRoto(RotoConfig())
        saved = roto.save_result(
            self.result,
            output_dir,
            prefix=self.image_path.stem if self.image_path else "roto"
        )

        QMessageBox.information(
            self, "Export Complete",
            f"Saved {len(saved)} files to:\n{output_dir}"
        )
        self.statusbar.showMessage(f"Exported to {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ultimate Rotoscopy GUI")
    parser.add_argument("--image", type=str, help="Load image directly")
    parser.add_argument("--video", type=str, help="Load video directly")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Ultimate Rotoscopy")

    window = UltimateRotoGUI()

    # Load file if specified
    if args.image:
        path = Path(args.image)
        if path.exists():
            window.image_path = path
            window.original_image = np.array(Image.open(path).convert("RGB"))
            window.viewport.set_image(window.original_image)
            window.file_label.setText(f"{path.name}")
            window.statusbar.showMessage(f"Loaded: {path.name}")

    elif args.video:
        path = Path(args.video)
        if path.exists():
            cap = cv2.VideoCapture(str(path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                window.image_path = path
                window.original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                window.viewport.set_image(window.original_image)
                window.file_label.setText(f"VIDEO: {path.name}")

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
