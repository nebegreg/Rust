#!/usr/bin/env python3
"""
Ultimate Rotoscopy GUI - Professional Unified Interface
========================================================

All-in-one interface combining:
- SAM3 Segmentation (text prompts, video tracking)
- Depth Anything V3 (depth, normals, point cloud)
- Matte Anything (professional alpha matting)
- Flame Export (multi-layer EXR, AOV)

For Autodesk Flame artists and VFX professionals.

Requirements:
    pip install PySide6 opencv-python numpy pillow

Usage:
    python ultimate_gui.py
    python ultimate_gui.py --image photo.jpg
    python ultimate_gui.py --video clip.mp4
"""

# Set environment variables BEFORE importing Qt
import os
os.environ['QT_QPA_PLATFORMTHEME'] = ''
os.environ['GIO_USE_VFS'] = 'local'
os.environ['QT_FILE_DIALOG_NO_NATIVE'] = '1'

import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QLineEdit, QComboBox, QSlider, QFileDialog,
        QToolBar, QStatusBar, QGroupBox, QCheckBox, QTabWidget,
        QSpinBox, QDoubleSpinBox, QTextEdit, QProgressBar, QSplitter,
        QListWidget, QFrame, QScrollArea, QMessageBox
    )
    from PySide6.QtCore import Qt, Signal, QThread, QTimer
    from PySide6.QtGui import QImage, QPixmap, QAction, QFont
except ImportError as e:
    print(f"Error: PySide6 not installed - {e}")
    print("Install: pip install PySide6")
    sys.exit(1)


class ImageViewport(QLabel):
    """Shared image viewport component."""

    clicked = Signal(int, int)  # x, y coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #1a1a1a; border: 1px solid #444; }")

        self.image: Optional[np.ndarray] = None
        self.overlay: Optional[np.ndarray] = None
        self.overlay_alpha: float = 0.5
        self.pixmap: Optional[QPixmap] = None

    def set_image(self, image: np.ndarray):
        """Set base image (RGB)."""
        self.image = image.copy()
        self._update_display()

    def set_overlay(self, overlay: np.ndarray, alpha: float = 0.5):
        """Set overlay image."""
        self.overlay = overlay.copy()
        self.overlay_alpha = alpha
        self._update_display()

    def clear_overlay(self):
        """Clear overlay."""
        self.overlay = None
        self._update_display()

    def clear(self):
        """Clear viewport."""
        self.image = None
        self.overlay = None
        self.pixmap = None
        self.setText("No image loaded")

    def _update_display(self):
        """Update display with image and overlay."""
        if self.image is None:
            return

        display = self.image.copy()

        # Apply overlay if exists
        if self.overlay is not None and self.overlay.shape[:2] == display.shape[:2]:
            if len(self.overlay.shape) == 2:
                # Grayscale overlay (mask)
                overlay_rgb = np.stack([self.overlay] * 3, axis=-1)
                if self.overlay.max() > 1:
                    overlay_rgb = overlay_rgb / 255.0
                overlay_rgb = (overlay_rgb * 255).astype(np.uint8)
                overlay_rgb[:, :, 1] = np.clip(overlay_rgb[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
            else:
                overlay_rgb = self.overlay

            display = cv2.addWeighted(
                display, 1 - self.overlay_alpha,
                overlay_rgb, self.overlay_alpha, 0
            )

        # Convert to QPixmap
        h, w = display.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(display.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(q_image)

        # Scale to fit
        scaled = self.pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)

    def mousePressEvent(self, event):
        """Handle mouse click for point selection."""
        if self.pixmap and self.image is not None:
            # Calculate image coordinates
            label_size = self.size()
            pixmap_size = self.pixmap.size()

            # Offset for centered image
            x_offset = (label_size.width() - pixmap_size.width()) // 2
            y_offset = (label_size.height() - pixmap_size.height()) // 2

            click_x = event.position().x() - x_offset
            click_y = event.position().y() - y_offset

            if 0 <= click_x < pixmap_size.width() and 0 <= click_y < pixmap_size.height():
                # Scale to original image coordinates
                scale_x = self.image.shape[1] / pixmap_size.width()
                scale_y = self.image.shape[0] / pixmap_size.height()

                img_x = int(click_x * scale_x)
                img_y = int(click_y * scale_y)

                self.clicked.emit(img_x, img_y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap:
            self._update_display()


class ProcessingWorker(QThread):
    """Unified worker thread for all processing tasks."""

    result_ready = Signal(str, object)  # task_type, result
    error_occurred = Signal(str)
    progress_updated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.task = None

        # Processors (lazy loaded)
        self.sam3_processor = None
        self.da3_processor = None
        self.matte_processor = None

    def run_task(self, task_type: str, **kwargs):
        """Queue a task for processing."""
        self.task = (task_type, kwargs)
        self.start()

    def run(self):
        """Execute queued task."""
        if self.task is None:
            return

        task_type, kwargs = self.task

        try:
            if task_type == "sam3_segment":
                result = self._run_sam3(**kwargs)
                self.result_ready.emit("sam3_segment", result)

            elif task_type == "da3_depth":
                result = self._run_da3(**kwargs)
                self.result_ready.emit("da3_depth", result)

            elif task_type == "matte":
                result = self._run_matte(**kwargs)
                self.result_ready.emit("matte", result)

            elif task_type == "full_pipeline":
                result = self._run_full_pipeline(**kwargs)
                self.result_ready.emit("full_pipeline", result)

        except Exception as e:
            self.error_occurred.emit(f"{task_type} failed: {str(e)}")

    def _run_sam3(self, image_path, text_prompt):
        """Run SAM3 segmentation."""
        self.progress_updated.emit("Loading SAM3...")

        if self.sam3_processor is None:
            from sam3_complete import SAM3ImageProcessor
            self.sam3_processor = SAM3ImageProcessor(device="cuda")

        self.progress_updated.emit(f"Segmenting: {text_prompt}")
        result = self.sam3_processor.segment_with_text(image_path, text_prompt)

        return result

    def _run_da3(self, image_path, compute_normals, compute_pointcloud, fov):
        """Run Depth Anything V3."""
        self.progress_updated.emit("Loading Depth Anything V3...")

        if self.da3_processor is None:
            from da3_complete import DepthAnything3
            self.da3_processor = DepthAnything3(device="cuda")

        self.progress_updated.emit("Estimating depth...")
        result = self.da3_processor.process(
            image_path,
            compute_normals=compute_normals,
            compute_pointcloud=compute_pointcloud,
            fov_degrees=fov
        )

        return result

    def _run_matte(self, image_path, mask, text_prompt):
        """Run matting."""
        self.progress_updated.emit("Loading matting models...")

        if self.matte_processor is None:
            from matte_anything import MatteAnything
            self.matte_processor = MatteAnything(device="cuda")

        self.progress_updated.emit("Computing alpha matte...")
        result = self.matte_processor.process_image(
            image_path,
            mask=mask,
            text_prompt=text_prompt
        )

        return result

    def _run_full_pipeline(self, image_path, text_prompt, compute_depth, compute_matte):
        """Run full pipeline: SAM3 -> Depth -> Matte."""
        results = {}

        # SAM3 Segmentation
        self.progress_updated.emit("Step 1/3: SAM3 Segmentation...")
        sam3_result = self._run_sam3(image_path, text_prompt)
        results['sam3'] = sam3_result

        # Get best mask
        mask, score = sam3_result.get_best_mask()

        # Depth estimation
        if compute_depth:
            self.progress_updated.emit("Step 2/3: Depth Estimation...")
            da3_result = self._run_da3(image_path, True, True, 60.0)
            results['depth'] = da3_result

        # Matting
        if compute_matte:
            self.progress_updated.emit("Step 3/3: Alpha Matting...")
            matte_result = self._run_matte(image_path, mask, None)
            results['matte'] = matte_result

        self.progress_updated.emit("Pipeline complete!")
        return results


class UltimateMainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ultimate Rotoscopy - Professional VFX Tool")
        self.setMinimumSize(1600, 1000)

        # Data
        self.current_image_path: Optional[Path] = None
        self.current_image: Optional[np.ndarray] = None
        self.results: Dict[str, Any] = {}

        # Video data
        self.video_path: Optional[Path] = None
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.video_frames: List[np.ndarray] = []
        self.video_masks: Dict[int, List[np.ndarray]] = {}
        self.current_frame_idx: int = 0
        self.total_frames: int = 0
        self.fps: float = 0.0
        self.is_playing: bool = False
        self.play_timer: Optional[QTimer] = None
        self.is_video_mode: bool = False

        # Worker
        self.worker = ProcessingWorker()
        self.worker.result_ready.connect(self._on_result_ready)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.progress_updated.connect(self._on_progress)

        # Setup UI
        self._setup_ui()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_statusbar()
        self._apply_dark_theme()

    def _setup_ui(self):
        """Setup main UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel: Tools
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Center: Viewports
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, stretch=4)

        # Right panel: Results
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)

    def _create_left_panel(self) -> QWidget:
        """Create left tools panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(350)

        panel = QWidget()
        layout = QVBoxLayout(panel)

        # File controls
        file_group = QGroupBox("Input")
        file_layout = QVBoxLayout(file_group)

        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self._load_image)
        file_layout.addWidget(self.load_image_btn)

        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self._load_video)
        file_layout.addWidget(self.load_video_btn)

        layout.addWidget(file_group)

        # SAM3 Controls
        sam3_group = QGroupBox("SAM3 Segmentation")
        sam3_layout = QVBoxLayout(sam3_group)

        sam3_layout.addWidget(QLabel("Text Prompt:"))
        self.sam3_prompt = QLineEdit()
        self.sam3_prompt.setPlaceholderText("e.g., person, car, dog...")
        sam3_layout.addWidget(self.sam3_prompt)

        self.sam3_btn = QPushButton("Segment with SAM3")
        self.sam3_btn.clicked.connect(self._run_sam3)
        sam3_layout.addWidget(self.sam3_btn)

        layout.addWidget(sam3_group)

        # Depth Controls
        depth_group = QGroupBox("Depth Anything V3")
        depth_layout = QVBoxLayout(depth_group)

        self.depth_normals_cb = QCheckBox("Compute Normals")
        self.depth_normals_cb.setChecked(True)
        depth_layout.addWidget(self.depth_normals_cb)

        self.depth_pointcloud_cb = QCheckBox("Generate Point Cloud")
        self.depth_pointcloud_cb.setChecked(True)
        depth_layout.addWidget(self.depth_pointcloud_cb)

        depth_layout.addWidget(QLabel("FOV (degrees):"))
        self.depth_fov = QDoubleSpinBox()
        self.depth_fov.setRange(10, 180)
        self.depth_fov.setValue(60.0)
        depth_layout.addWidget(self.depth_fov)

        self.depth_btn = QPushButton("Estimate Depth")
        self.depth_btn.clicked.connect(self._run_depth)
        depth_layout.addWidget(self.depth_btn)

        layout.addWidget(depth_group)

        # Matte Controls
        matte_group = QGroupBox("Alpha Matting")
        matte_layout = QVBoxLayout(matte_group)

        self.matte_use_sam3_cb = QCheckBox("Use SAM3 mask")
        self.matte_use_sam3_cb.setChecked(True)
        matte_layout.addWidget(self.matte_use_sam3_cb)

        self.matte_refine_cb = QCheckBox("Refine edges")
        self.matte_refine_cb.setChecked(True)
        matte_layout.addWidget(self.matte_refine_cb)

        self.matte_btn = QPushButton("Generate Matte")
        self.matte_btn.clicked.connect(self._run_matte)
        matte_layout.addWidget(self.matte_btn)

        layout.addWidget(matte_group)

        # Full Pipeline
        pipeline_group = QGroupBox("Full Pipeline")
        pipeline_layout = QVBoxLayout(pipeline_group)

        pipeline_layout.addWidget(QLabel("Run complete workflow:"))
        pipeline_layout.addWidget(QLabel("SAM3 -> Depth -> Matte"))

        self.pipeline_depth_cb = QCheckBox("Include Depth")
        self.pipeline_depth_cb.setChecked(True)
        pipeline_layout.addWidget(self.pipeline_depth_cb)

        self.pipeline_matte_cb = QCheckBox("Include Matte")
        self.pipeline_matte_cb.setChecked(True)
        pipeline_layout.addWidget(self.pipeline_matte_cb)

        self.pipeline_btn = QPushButton("Run Full Pipeline")
        self.pipeline_btn.setStyleSheet("QPushButton { background-color: #28a745; font-weight: bold; padding: 12px; }")
        self.pipeline_btn.clicked.connect(self._run_full_pipeline)
        pipeline_layout.addWidget(self.pipeline_btn)

        layout.addWidget(pipeline_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _create_center_panel(self) -> QWidget:
        """Create center viewport panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tab widget for views
        self.view_tabs = QTabWidget()

        # Original
        self.original_viewport = ImageViewport()
        self.original_viewport.setText("Load an image to start")
        self.view_tabs.addTab(self.original_viewport, "Original")

        # SAM3 Result
        self.sam3_viewport = ImageViewport()
        self.sam3_viewport.setText("Run SAM3 segmentation")
        self.view_tabs.addTab(self.sam3_viewport, "Segmentation")

        # Depth
        self.depth_viewport = ImageViewport()
        self.depth_viewport.setText("Run depth estimation")
        self.view_tabs.addTab(self.depth_viewport, "Depth")

        # Normals
        self.normals_viewport = ImageViewport()
        self.normals_viewport.setText("Enable normals computation")
        self.view_tabs.addTab(self.normals_viewport, "Normals")

        # Matte
        self.matte_viewport = ImageViewport()
        self.matte_viewport.setText("Run matting")
        self.view_tabs.addTab(self.matte_viewport, "Alpha Matte")

        # Composite
        self.composite_viewport = ImageViewport()
        self.composite_viewport.setText("Run full pipeline for composite")
        self.view_tabs.addTab(self.composite_viewport, "Composite")

        layout.addWidget(self.view_tabs)

        # Video timeline controls
        self.timeline_widget = self._create_video_timeline()
        layout.addWidget(self.timeline_widget)

        # Visualization controls
        viz_layout = QHBoxLayout()

        viz_layout.addWidget(QLabel("Overlay:"))
        self.overlay_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_slider.setRange(0, 100)
        self.overlay_slider.setValue(50)
        self.overlay_slider.valueChanged.connect(self._update_overlay_alpha)
        viz_layout.addWidget(self.overlay_slider)

        viz_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["turbo", "viridis", "plasma", "jet", "gray"])
        self.colormap_combo.currentTextChanged.connect(self._update_depth_colormap)
        viz_layout.addWidget(self.colormap_combo)

        viz_layout.addStretch()

        layout.addLayout(viz_layout)

        return panel

    def _create_video_timeline(self) -> QWidget:
        """Create video timeline with playback controls."""
        widget = QWidget()
        widget.setMaximumHeight(100)
        layout = QVBoxLayout(widget)

        # Timeline slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame:"))

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setValue(0)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.valueChanged.connect(self._on_timeline_changed)
        slider_layout.addWidget(self.timeline_slider)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(100)
        slider_layout.addWidget(self.frame_label)

        layout.addLayout(slider_layout)

        # Playback controls
        controls_layout = QHBoxLayout()

        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._toggle_play)
        controls_layout.addWidget(self.play_btn)

        self.prev_frame_btn = QPushButton("Prev")
        self.prev_frame_btn.setEnabled(False)
        self.prev_frame_btn.clicked.connect(self._prev_frame)
        controls_layout.addWidget(self.prev_frame_btn)

        self.next_frame_btn = QPushButton("Next")
        self.next_frame_btn.setEnabled(False)
        self.next_frame_btn.clicked.connect(self._next_frame)
        controls_layout.addWidget(self.next_frame_btn)

        self.process_video_btn = QPushButton("Process Video")
        self.process_video_btn.setEnabled(False)
        self.process_video_btn.clicked.connect(self._process_video)
        self.process_video_btn.setStyleSheet("QPushButton { background-color: #28a745; }")
        controls_layout.addWidget(self.process_video_btn)

        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Initially hide timeline
        widget.setVisible(False)

        return widget

    def _create_right_panel(self) -> QWidget:
        """Create right results panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(350)

        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Results info
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(300)
        results_layout.addWidget(self.results_text)

        layout.addWidget(results_group)

        # Mask list
        masks_group = QGroupBox("Detected Masks")
        masks_layout = QVBoxLayout(masks_group)

        self.mask_list = QListWidget()
        self.mask_list.setMaximumHeight(150)
        self.mask_list.itemClicked.connect(self._on_mask_selected)
        masks_layout.addWidget(self.mask_list)

        layout.addWidget(masks_group)

        # Export
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        self.export_masks_btn = QPushButton("Export Masks")
        self.export_masks_btn.clicked.connect(self._export_masks)
        export_layout.addWidget(self.export_masks_btn)

        self.export_depth_btn = QPushButton("Export Depth")
        self.export_depth_btn.clicked.connect(self._export_depth)
        export_layout.addWidget(self.export_depth_btn)

        self.export_matte_btn = QPushButton("Export Alpha Matte")
        self.export_matte_btn.clicked.connect(self._export_matte)
        export_layout.addWidget(self.export_matte_btn)

        self.export_all_btn = QPushButton("Export All (Flame)")
        self.export_all_btn.setStyleSheet("QPushButton { background-color: #17a2b8; font-weight: bold; }")
        self.export_all_btn.clicked.connect(self._export_all_flame)
        export_layout.addWidget(self.export_all_btn)

        layout.addWidget(export_group)

        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _setup_menubar(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open Image...", self._load_image)
        file_menu.addAction("Open Video...", self._load_video)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # Process
        process_menu = menubar.addMenu("Process")
        process_menu.addAction("SAM3 Segment", self._run_sam3)
        process_menu.addAction("Depth Estimation", self._run_depth)
        process_menu.addAction("Alpha Matting", self._run_matte)
        process_menu.addSeparator()
        process_menu.addAction("Full Pipeline", self._run_full_pipeline)

        # Export
        export_menu = menubar.addMenu("Export")
        export_menu.addAction("Export All...", self._export_all_flame)

        # Help
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self._show_about)

    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        toolbar.addAction("Open", self._load_image)
        toolbar.addSeparator()
        toolbar.addAction("SAM3", self._run_sam3)
        toolbar.addAction("Depth", self._run_depth)
        toolbar.addAction("Matte", self._run_matte)
        toolbar.addSeparator()
        toolbar.addAction("Full Pipeline", self._run_full_pipeline)
        toolbar.addSeparator()
        toolbar.addAction("Export", self._export_all_flame)

    def _setup_statusbar(self):
        """Setup status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load an image to start")

    def _apply_dark_theme(self):
        """Apply dark theme."""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { background-color: #2b2b2b; color: #ffffff; font-family: 'Segoe UI', Arial; font-size: 10pt; }
            QPushButton { background-color: #0e639c; color: white; border: none; padding: 8px 16px; border-radius: 4px; }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:pressed { background-color: #0d5589; }
            QPushButton:disabled { background-color: #555; color: #888; }
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox { background-color: #3c3c3c; border: 1px solid #555; border-radius: 4px; padding: 6px; }
            QGroupBox { border: 1px solid #555; border-radius: 6px; margin-top: 12px; padding-top: 12px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QComboBox { background-color: #3c3c3c; border: 1px solid #555; border-radius: 4px; padding: 6px; }
            QTabWidget::pane { border: 1px solid #555; }
            QTabBar::tab { background-color: #3c3c3c; padding: 8px 16px; margin-right: 2px; }
            QTabBar::tab:selected { background-color: #0e639c; }
            QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background-color: #0e639c; }
            QListWidget { background-color: #3c3c3c; border: 1px solid #555; }
            QListWidget::item:selected { background-color: #0e639c; }
            QScrollArea { border: none; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QCheckBox::indicator:unchecked { border: 1px solid #555; background-color: #3c3c3c; }
            QCheckBox::indicator:checked { background-color: #0e639c; border: 1px solid #0e639c; }
        """)

    # Slots
    def _load_image(self):
        """Load image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_path:
            self.current_image_path = Path(file_path)
            image = cv2.imread(str(self.current_image_path))
            self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            self.original_viewport.set_image(self.current_image)
            self.view_tabs.setCurrentIndex(0)

            # Clear previous results
            self.results = {}
            self.mask_list.clear()
            self.results_text.clear()

            self.status_bar.showMessage(f"Loaded: {self.current_image_path.name}")

    def _load_video(self):
        """Load video file with full timeline support."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_path:
            self.status_bar.showMessage(f"Loading video: {Path(file_path).name}")

            try:
                # Close previous video if any
                if self.video_cap is not None:
                    self.video_cap.release()

                # Open video with OpenCV
                self.video_cap = cv2.VideoCapture(str(file_path))

                if not self.video_cap.isOpened():
                    self.status_bar.showMessage(f"Error: Cannot open video {Path(file_path).name}")
                    return

                # Get video info
                self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Store video path
                self.video_path = Path(file_path)
                self.is_video_mode = True

                # Reset video state
                self.video_frames = []
                self.video_masks = {}
                self.current_frame_idx = 0

                # Load first frame
                self._load_frame(0)

                # Setup timeline
                self.timeline_slider.setMaximum(self.total_frames - 1)
                self.timeline_slider.setValue(0)
                self.timeline_slider.setEnabled(True)
                self.frame_label.setText(f"0 / {self.total_frames}")

                # Enable controls
                self.play_btn.setEnabled(True)
                self.prev_frame_btn.setEnabled(True)
                self.next_frame_btn.setEnabled(True)
                self.process_video_btn.setEnabled(True)

                # Show timeline
                self.timeline_widget.setVisible(True)

                # Clear previous results
                self.results = {}
                self.mask_list.clear()
                self.results_text.clear()

                self.status_bar.showMessage(
                    f"Video loaded: {Path(file_path).name} - "
                    f"{width}x{height}, {self.total_frames} frames @ {self.fps:.1f}fps"
                )

            except Exception as e:
                self.status_bar.showMessage(f"Error loading video: {str(e)}")
                import traceback
                traceback.print_exc()

    def _load_frame(self, frame_idx: int):
        """Load and display a specific frame."""
        if self.video_cap is None:
            return

        # Seek to frame
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()

        if not ret:
            self.status_bar.showMessage(f"Error reading frame {frame_idx}")
            return

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update current image
        self.current_image = frame_rgb
        self.current_frame_idx = frame_idx

        # Display frame
        self.original_viewport.set_image(frame_rgb)

        # Display masks if available for this frame
        if frame_idx in self.video_masks:
            masks = self.video_masks[frame_idx]
            if masks:
                # Combine masks for visualization
                combined_mask = np.zeros(frame_rgb.shape[:2], dtype=np.float32)
                for mask in masks:
                    combined_mask = np.maximum(combined_mask, mask)
                self.sam3_viewport.set_image(frame_rgb)
                self.sam3_viewport.set_overlay((combined_mask * 255).astype(np.uint8))

    def _on_timeline_changed(self, frame_idx: int):
        """Handle timeline slider change."""
        if self.video_cap is not None:
            self._load_frame(frame_idx)
            self.frame_label.setText(f"{frame_idx} / {self.total_frames}")

    def _prev_frame(self):
        """Go to previous frame."""
        if self.current_frame_idx > 0:
            new_idx = self.current_frame_idx - 1
            self.timeline_slider.setValue(new_idx)

    def _next_frame(self):
        """Go to next frame."""
        if self.current_frame_idx < self.total_frames - 1:
            new_idx = self.current_frame_idx + 1
            self.timeline_slider.setValue(new_idx)

    def _toggle_play(self):
        """Toggle video playback."""
        if self.is_playing:
            # Stop playing
            self.is_playing = False
            self.play_btn.setText("Play")
            if self.play_timer:
                self.play_timer.stop()
        else:
            # Start playing
            self.is_playing = True
            self.play_btn.setText("Pause")

            # Create timer for playback
            if self.play_timer is None:
                self.play_timer = QTimer()
                self.play_timer.timeout.connect(self._play_next_frame)

            # Calculate interval from FPS
            interval = int(1000 / self.fps) if self.fps > 0 else 33
            self.play_timer.start(interval)

    def _play_next_frame(self):
        """Advance to next frame during playback."""
        if self.current_frame_idx < self.total_frames - 1:
            self._next_frame()
        else:
            # End of video, stop playing
            self._toggle_play()
            self.timeline_slider.setValue(0)

    def _process_video(self):
        """Process entire video with current settings."""
        if not self.is_video_mode or self.video_path is None:
            self.status_bar.showMessage("Error: No video loaded")
            return

        prompt = self.sam3_prompt.text().strip()
        if not prompt:
            self.status_bar.showMessage("Error: Enter a text prompt for video processing")
            return

        # Save current frame as temp file for processing
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        temp_frame = temp_dir / f"ultimate_frame_{self.current_frame_idx}.png"

        if self.current_image is not None:
            cv2.imwrite(str(temp_frame), cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
            self.current_image_path = temp_frame

            # Run SAM3 on current frame
            self._set_processing(True)
            self.worker.run_task("sam3_segment",
                image_path=temp_frame,
                text_prompt=prompt
            )

            self.status_bar.showMessage(f"Processing frame {self.current_frame_idx}...")

    def _run_sam3(self):
        """Run SAM3 segmentation."""
        if self.current_image_path is None:
            self.status_bar.showMessage("Error: Load an image first")
            return

        prompt = self.sam3_prompt.text().strip()
        if not prompt:
            self.status_bar.showMessage("Error: Enter a text prompt")
            return

        self._set_processing(True)
        self.worker.run_task("sam3_segment",
            image_path=self.current_image_path,
            text_prompt=prompt
        )

    def _run_depth(self):
        """Run depth estimation."""
        if self.current_image_path is None:
            self.status_bar.showMessage("Error: Load an image first")
            return

        self._set_processing(True)
        self.worker.run_task("da3_depth",
            image_path=self.current_image_path,
            compute_normals=self.depth_normals_cb.isChecked(),
            compute_pointcloud=self.depth_pointcloud_cb.isChecked(),
            fov=self.depth_fov.value()
        )

    def _run_matte(self):
        """Run matting."""
        if self.current_image_path is None:
            self.status_bar.showMessage("Error: Load an image first")
            return

        mask = None
        if self.matte_use_sam3_cb.isChecked() and 'sam3' in self.results:
            mask, _ = self.results['sam3'].get_best_mask()

        self._set_processing(True)
        self.worker.run_task("matte",
            image_path=self.current_image_path,
            mask=mask,
            text_prompt=self.sam3_prompt.text().strip() if not mask else None
        )

    def _run_full_pipeline(self):
        """Run full pipeline."""
        if self.current_image_path is None:
            self.status_bar.showMessage("Error: Load an image first")
            return

        prompt = self.sam3_prompt.text().strip()
        if not prompt:
            self.status_bar.showMessage("Error: Enter a text prompt")
            return

        self._set_processing(True)
        self.worker.run_task("full_pipeline",
            image_path=self.current_image_path,
            text_prompt=prompt,
            compute_depth=self.pipeline_depth_cb.isChecked(),
            compute_matte=self.pipeline_matte_cb.isChecked()
        )

    def _on_result_ready(self, task_type: str, result):
        """Handle processing result."""
        self._set_processing(False)

        if task_type == "sam3_segment":
            self.results['sam3'] = result
            self._display_sam3_result(result)

        elif task_type == "da3_depth":
            self.results['depth'] = result
            self._display_depth_result(result)

        elif task_type == "matte":
            self.results['matte'] = result
            self._display_matte_result(result)

        elif task_type == "full_pipeline":
            if 'sam3' in result:
                self.results['sam3'] = result['sam3']
                self._display_sam3_result(result['sam3'])
            if 'depth' in result:
                self.results['depth'] = result['depth']
                self._display_depth_result(result['depth'])
            if 'matte' in result:
                self.results['matte'] = result['matte']
                self._display_matte_result(result['matte'])

            self._create_composite()

    def _display_sam3_result(self, result):
        """Display SAM3 result."""
        # Update mask list
        self.mask_list.clear()
        for i, (box, score) in enumerate(zip(result.boxes, result.scores)):
            self.mask_list.addItem(f"Mask {i+1}: score={score:.3f}")

        # Display best mask
        mask, score = result.get_best_mask()
        self.sam3_viewport.set_image(self.current_image)
        self.sam3_viewport.set_overlay((mask * 255).astype(np.uint8))

        self.view_tabs.setCurrentIndex(1)

        # Update info
        info = f"SAM3 Segmentation\n{'='*20}\nMasks found: {len(result.masks)}\nBest score: {score:.3f}"
        self.results_text.setText(info)

    def _display_depth_result(self, result):
        """Display depth result."""
        from da3_complete import DepthColormap

        colormap = DepthColormap(self.colormap_combo.currentText())
        depth_vis = result.get_depth_colormap(colormap)
        self.depth_viewport.set_image(depth_vis)

        if result.normals is not None:
            normals_vis = result.get_normals_colormap()
            self.normals_viewport.set_image(normals_vis)

        self.view_tabs.setCurrentIndex(2)

        # Update info
        info = f"Depth Estimation\n{'='*20}\n"
        info += f"Depth range: {result.depth_min:.3f} - {result.depth_max:.3f}\n"
        info += f"Has normals: {result.normals is not None}\n"
        info += f"Point cloud: {len(result.point_cloud) if result.point_cloud is not None else 0} points\n"
        info += f"Time: {result.processing_time_ms:.0f}ms"
        self.results_text.setText(info)

    def _display_matte_result(self, result):
        """Display matte result."""
        alpha_vis = result.get_alpha_visualization()
        self.matte_viewport.set_image(alpha_vis)

        self.view_tabs.setCurrentIndex(4)

        # Update info
        info = f"Alpha Matting\n{'='*20}\n"
        info += f"Model: {result.model_used}\n"
        info += f"Has layers: {result.alpha_core is not None}\n"
        info += f"Time: {result.processing_time_ms:.0f}ms"
        self.results_text.setText(info)

    def _create_composite(self):
        """Create composite visualization."""
        if 'matte' not in self.results or self.current_image is None:
            return

        matte = self.results['matte']

        # Create checkerboard background
        h, w = self.current_image.shape[:2]
        checker = np.zeros((h, w, 3), dtype=np.uint8)
        checker_size = 20
        for i in range(0, h, checker_size):
            for j in range(0, w, checker_size):
                if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                    checker[i:i+checker_size, j:j+checker_size] = [100, 100, 100]
                else:
                    checker[i:i+checker_size, j:j+checker_size] = [150, 150, 150]

        # Composite
        composite = matte.get_composite(checker)
        self.composite_viewport.set_image(composite)

    def _on_mask_selected(self, item):
        """Handle mask selection."""
        if 'sam3' not in self.results:
            return

        idx = self.mask_list.row(item)
        result = self.results['sam3']

        if idx < len(result.masks):
            mask = result.masks[idx]
            self.sam3_viewport.set_image(self.current_image)
            self.sam3_viewport.set_overlay((mask * 255).astype(np.uint8))

    def _update_overlay_alpha(self, value):
        """Update overlay transparency."""
        alpha = value / 100.0
        self.sam3_viewport.overlay_alpha = alpha
        self.sam3_viewport._update_display()

    def _update_depth_colormap(self, colormap_name):
        """Update depth colormap."""
        if 'depth' in self.results:
            self._display_depth_result(self.results['depth'])

    def _on_error(self, error_msg: str):
        """Handle error."""
        self._set_processing(False)
        self.status_bar.showMessage(f"Error: {error_msg}")
        self.results_text.setText(f"ERROR:\n{error_msg}")

    def _on_progress(self, message: str):
        """Handle progress update."""
        self.status_bar.showMessage(message)

    def _set_processing(self, processing: bool):
        """Enable/disable UI during processing."""
        self.progress_bar.setVisible(processing)
        if processing:
            self.progress_bar.setRange(0, 0)

        self.sam3_btn.setEnabled(not processing)
        self.depth_btn.setEnabled(not processing)
        self.matte_btn.setEnabled(not processing)
        self.pipeline_btn.setEnabled(not processing)

    def _export_masks(self):
        """Export masks."""
        if 'sam3' not in self.results:
            self.status_bar.showMessage("Error: No masks to export")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if output_dir:
            result = self.results['sam3']
            for i, mask in enumerate(result.masks):
                mask_path = Path(output_dir) / f"mask_{i+1}.png"
                cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

            self.status_bar.showMessage(f"Exported {len(result.masks)} masks")

    def _export_depth(self):
        """Export depth."""
        if 'depth' not in self.results:
            self.status_bar.showMessage("Error: No depth to export")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if output_dir:
            self.results['depth'].save_all(output_dir, self.current_image_path.stem)
            self.status_bar.showMessage(f"Depth exported to: {output_dir}")

    def _export_matte(self):
        """Export matte."""
        if 'matte' not in self.results:
            self.status_bar.showMessage("Error: No matte to export")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if output_dir:
            self.results['matte'].save_all(output_dir, self.current_image_path.stem)
            self.status_bar.showMessage(f"Matte exported to: {output_dir}")

    def _export_all_flame(self):
        """Export all for Flame."""
        if not self.results:
            self.status_bar.showMessage("Error: No results to export")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Flame Export",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if output_dir:
            output_path = Path(output_dir)
            base_name = self.current_image_path.stem

            # Export each result type
            if 'sam3' in self.results:
                masks_dir = output_path / "masks"
                masks_dir.mkdir(exist_ok=True)
                result = self.results['sam3']
                for i, mask in enumerate(result.masks):
                    cv2.imwrite(str(masks_dir / f"{base_name}_mask_{i+1}.png"), (mask * 255).astype(np.uint8))

            if 'depth' in self.results:
                depth_dir = output_path / "depth"
                depth_dir.mkdir(exist_ok=True)
                self.results['depth'].save_all(depth_dir, base_name)

            if 'matte' in self.results:
                matte_dir = output_path / "matte"
                matte_dir.mkdir(exist_ok=True)
                self.results['matte'].save_all(matte_dir, base_name)

            self.status_bar.showMessage(f"All results exported to: {output_dir}")

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Ultimate Rotoscopy",
            "Ultimate Rotoscopy v1.0\n\n"
            "Professional VFX tool combining:\n"
            "- SAM3 Segmentation\n"
            "- Depth Anything V3\n"
            "- Professional Alpha Matting\n\n"
            "For Autodesk Flame artists and VFX professionals."
        )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ultimate Rotoscopy GUI")
    parser.add_argument("--image", type=str, help="Load image directly")
    parser.add_argument("--video", type=str, help="Load video directly")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Ultimate Rotoscopy")

    window = UltimateMainWindow()

    # Load file if provided
    if args.image:
        path = Path(args.image)
        if path.exists():
            window.current_image_path = path
            image = cv2.imread(str(path))
            if image is not None:
                window.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                window.original_viewport.set_image(window.current_image)
                window.status_bar.showMessage(f"Loaded: {path.name}")

    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
