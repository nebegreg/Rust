#!/usr/bin/env python3
"""
SAM3 GUI - Modern PySide6 Interface for Professional Segmentation
=================================================================

Professional interface for SAM3 with:
- Interactive viewport with mask overlays
- Text and visual prompting
- Video timeline and tracking
- Real-time visualization
- Export controls

Requirements:
    pip install PySide6 opencv-python numpy pillow

Usage:
    python sam3_gui.py
"""

# IMPORTANT: Set environment variables BEFORE importing Qt to prevent GVFS hang
import os
os.environ['QT_QPA_PLATFORMTHEME'] = ''  # Disable native platform theme
os.environ['GIO_USE_VFS'] = 'local'      # Disable GVFS
os.environ['QT_FILE_DIALOG_NO_NATIVE'] = '1'  # Force Qt dialogs

import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QLineEdit, QComboBox, QSlider, QFileDialog,
        QToolBar, QStatusBar, QSplitter, QListWidget, QGroupBox, QCheckBox,
        QSpinBox, QDoubleSpinBox, QTextEdit, QTabWidget, QFrame, QScrollArea
    )
    from PySide6.QtCore import Qt, Signal, QTimer, QThread, QRect, QPoint, QSize
    from PySide6.QtGui import (
        QImage, QPixmap, QPainter, QColor, QPen, QBrush, QIcon,
        QAction, QMouseEvent, QPalette
    )
except ImportError as e:
    print(f"Error: PySide6 not installed - {e}")
    print("Install: pip install PySide6")
    sys.exit(1)

# Import SAM3 complete wrapper
try:
    from sam3_complete import (
        SAM3ImageProcessor, SAM3VideoTracker,
        SegmentationResult, VideoTrackingSession,
        PromptType
    )
except ImportError:
    print("Error: sam3_complete.py not found!")
    print("Make sure sam3_complete.py is in the same directory.")
    sys.exit(1)


class ImageViewport(QLabel):
    """
    Interactive image viewport with annotation capabilities.

    Features:
    - Display image with mask overlay
    - Interactive point/box annotation
    - Zoom and pan
    - Mask transparency control
    """

    # Signals
    point_added = Signal(int, int)  # x, y
    box_drawn = Signal(int, int, int, int)  # x1, y1, x2, y2

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #2b2b2b; }")

        # Image data
        self.image: Optional[np.ndarray] = None  # RGB image
        self.mask: Optional[np.ndarray] = None  # Binary mask
        self.overlay_alpha = 0.5

        # Annotation mode
        self.annotation_mode = "point"  # "point" or "box"
        self.points: List[Tuple[int, int]] = []
        self.point_labels: List[int] = []  # 1=FG, 0=BG

        # Box drawing
        self.box_start: Optional[Tuple[int, int]] = None
        self.box_end: Optional[Tuple[int, int]] = None
        self.drawing_box = False

        # Display
        self.pixmap: Optional[QPixmap] = None

        self.setMouseTracking(True)

    def set_image(self, image: np.ndarray):
        """Set image to display (RGB numpy array)."""
        self.image = image.copy()
        self.points = []
        self.point_labels = []
        self.box_start = None
        self.box_end = None
        self._update_display()

    def set_mask(self, mask: np.ndarray, alpha: float = 0.5):
        """Set mask overlay (binary numpy array)."""
        # Convert PyTorch tensor to numpy if needed
        import torch
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        # Ensure mask is 2D by squeezing extra dimensions
        while mask.ndim > 2:
            mask = mask.squeeze(0)

        self.mask = mask.copy()
        self.overlay_alpha = alpha
        self._update_display()

    def set_overlay_alpha(self, alpha: float):
        """Set mask overlay transparency."""
        self.overlay_alpha = alpha
        self._update_display()

    def set_annotation_mode(self, mode: str):
        """Set annotation mode: 'point' or 'box'."""
        self.annotation_mode = mode
        self.box_start = None
        self.box_end = None
        self.drawing_box = False

    def clear_annotations(self):
        """Clear all annotations."""
        self.points = []
        self.point_labels = []
        self.box_start = None
        self.box_end = None
        self.drawing_box = False
        self._update_display()

    def _update_display(self):
        """Update the displayed image with overlays."""
        if self.image is None:
            return

        # Start with original image
        display = self.image.copy()

        # Add mask overlay
        if self.mask is not None:
            overlay = display.copy()
            overlay[self.mask > 0] = [0, 255, 0]  # Green overlay
            display = cv2.addWeighted(
                display, 1 - self.overlay_alpha,
                overlay, self.overlay_alpha, 0
            )

        # Draw points
        for (x, y), label in zip(self.points, self.point_labels):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)  # Green=FG, Red=BG
            cv2.circle(display, (x, y), 5, color, -1)
            cv2.circle(display, (x, y), 7, (255, 255, 255), 2)

        # Draw box
        if self.box_start and self.box_end:
            cv2.rectangle(
                display,
                self.box_start, self.box_end,
                (0, 255, 255), 2
            )

        # Convert to QPixmap
        height, width, channel = display.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            display.data, width, height, bytes_per_line,
            QImage.Format.Format_RGB888
        )
        self.pixmap = QPixmap.fromImage(q_image)
        self.setPixmap(self.pixmap)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for annotations."""
        if self.image is None or self.pixmap is None:
            return

        # Get click position in image coordinates
        pos = event.pos()
        pixmap_rect = self.pixmap.rect()
        label_rect = self.rect()

        # Calculate offset (image is centered)
        x_offset = (label_rect.width() - pixmap_rect.width()) // 2
        y_offset = (label_rect.height() - pixmap_rect.height()) // 2

        # Map to image coordinates
        x = pos.x() - x_offset
        y = pos.y() - y_offset

        # Check if click is within image bounds
        if x < 0 or y < 0 or x >= pixmap_rect.width() or y >= pixmap_rect.height():
            return

        if self.annotation_mode == "point":
            # Add point
            label = 1  # Foreground by default (use Ctrl for background)
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                label = 0  # Background

            self.points.append((x, y))
            self.point_labels.append(label)
            self._update_display()

            self.point_added.emit(x, y)

        elif self.annotation_mode == "box":
            # Start box drawing
            self.box_start = (x, y)
            self.box_end = (x, y)
            self.drawing_box = True

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for box drawing."""
        if not self.drawing_box:
            return

        pos = event.pos()
        pixmap_rect = self.pixmap.rect()
        label_rect = self.rect()

        x_offset = (label_rect.width() - pixmap_rect.width()) // 2
        y_offset = (label_rect.height() - pixmap_rect.height()) // 2

        x = pos.x() - x_offset
        y = pos.y() - y_offset

        # Clamp to image bounds
        x = max(0, min(x, pixmap_rect.width() - 1))
        y = max(0, min(y, pixmap_rect.height() - 1))

        self.box_end = (x, y)
        self._update_display()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release for box completion."""
        if self.drawing_box and self.box_start and self.box_end:
            self.drawing_box = False

            # Normalize box coordinates
            x1, y1 = self.box_start
            x2, y2 = self.box_end
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            self.box_drawn.emit(x1, y1, x2, y2)
            self._update_display()


class SAM3Worker(QThread):
    """Worker thread for SAM3 processing."""

    # Signals
    result_ready = Signal(object)  # SegmentationResult
    error_occurred = Signal(str)
    progress_updated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.processor: Optional[SAM3ImageProcessor] = None
        self.task = None

    def initialize_processor(self, device="cuda"):
        """Initialize SAM3 processor in background."""
        try:
            self.progress_updated.emit("Loading SAM3 image model...")
            self.processor = SAM3ImageProcessor(device=device)
            self.progress_updated.emit("SAM3 ready")
        except Exception as e:
            self.error_occurred.emit(f"Failed to load SAM3: {str(e)}")

    def process_image_text(self, image_path: Path, text_prompt: str):
        """Process image with text prompt."""
        self.task = ("text", image_path, text_prompt)
        self.start()

    def process_image_points(self, image_path: Path, points: List, labels: List):
        """Process image with point prompts."""
        self.task = ("points", image_path, points, labels)
        self.start()

    def process_image_box(self, image_path: Path, box: Tuple):
        """Process image with box prompt."""
        self.task = ("box", image_path, box)
        self.start()

    def run(self):
        """Execute processing task."""
        if self.processor is None:
            self.error_occurred.emit("SAM3 not initialized")
            return

        try:
            task_type = self.task[0]

            if task_type == "text":
                _, image_path, text_prompt = self.task
                self.progress_updated.emit(f"Segmenting with text: {text_prompt}")
                result = self.processor.segment_with_text(image_path, text_prompt)

            elif task_type == "points":
                _, image_path, points, labels = self.task
                self.progress_updated.emit(f"Segmenting with {len(points)} points")
                result = self.processor.segment_with_points(image_path, points, labels)

            elif task_type == "box":
                _, image_path, box = self.task
                self.progress_updated.emit("Segmenting with box")
                result = self.processor.segment_with_box(image_path, box)

            else:
                raise ValueError(f"Unknown task type: {task_type}")

            self.result_ready.emit(result)
            self.progress_updated.emit("Segmentation complete")

        except Exception as e:
            self.error_occurred.emit(f"Processing failed: {str(e)}")


class SAM3MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("SAM3 Professional - Segmentation & Tracking")
        self.setMinimumSize(1400, 900)

        # Data
        self.current_image_path: Optional[Path] = None
        self.current_result: Optional[SegmentationResult] = None

        # Video data
        self.video_path: Optional[Path] = None
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.video_frames: List[np.ndarray] = []  # Cached frames
        self.video_masks: Dict[int, np.ndarray] = {}  # Frame index -> mask
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 0.0
        self.is_playing = False
        self.play_timer: Optional[QTimer] = None

        # Worker thread
        self.worker = SAM3Worker()
        self.worker.result_ready.connect(self._on_result_ready)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.progress_updated.connect(self._on_progress)

        # Setup UI
        self._setup_ui()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_statusbar()

        # Initialize SAM3 in background
        QTimer.singleShot(100, lambda: self.worker.initialize_processor("cuda"))

        # Apply modern dark theme
        self._apply_dark_theme()

    def _setup_ui(self):
        """Setup main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel: Controls
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Center: Viewport
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, stretch=4)

        # Right panel: Results
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)

    def _create_left_panel(self) -> QWidget:
        """Create left control panel."""
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

        # Prompt controls
        prompt_group = QGroupBox("Prompting")
        prompt_layout = QVBoxLayout(prompt_group)

        # Text prompt
        prompt_layout.addWidget(QLabel("Text Prompt:"))
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("e.g., red baseball cap")
        prompt_layout.addWidget(self.text_input)

        self.text_segment_btn = QPushButton("Segment with Text")
        self.text_segment_btn.clicked.connect(self._segment_with_text)
        prompt_layout.addWidget(self.text_segment_btn)

        prompt_layout.addWidget(QLabel("---"))

        # Visual prompting
        prompt_layout.addWidget(QLabel("Visual Prompting:"))

        self.annotation_mode_combo = QComboBox()
        self.annotation_mode_combo.addItems(["Point Mode", "Box Mode"])
        self.annotation_mode_combo.currentTextChanged.connect(self._change_annotation_mode)
        prompt_layout.addWidget(self.annotation_mode_combo)

        self.visual_segment_btn = QPushButton("Segment with Visual Prompt")
        self.visual_segment_btn.clicked.connect(self._segment_with_visual)
        prompt_layout.addWidget(self.visual_segment_btn)

        self.clear_annotations_btn = QPushButton("Clear Annotations")
        self.clear_annotations_btn.clicked.connect(self._clear_annotations)
        prompt_layout.addWidget(self.clear_annotations_btn)

        layout.addWidget(prompt_group)

        # Visualization controls
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)

        viz_layout.addWidget(QLabel("Mask Opacity:"))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(self._update_alpha)
        viz_layout.addWidget(self.alpha_slider)

        self.alpha_label = QLabel("50%")
        viz_layout.addWidget(self.alpha_label)

        layout.addWidget(viz_group)

        layout.addStretch()

        return panel

    def _create_center_panel(self) -> QWidget:
        """Create center viewport panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Viewport
        self.viewport = ImageViewport()
        self.viewport.point_added.connect(self._on_point_added)
        self.viewport.box_drawn.connect(self._on_box_drawn)
        layout.addWidget(self.viewport)

        # Video timeline controls
        timeline_widget = self._create_video_timeline()
        layout.addWidget(timeline_widget)

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

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._toggle_play)
        controls_layout.addWidget(self.play_btn)

        self.prev_frame_btn = QPushButton("◀ Prev")
        self.prev_frame_btn.setEnabled(False)
        self.prev_frame_btn.clicked.connect(self._prev_frame)
        controls_layout.addWidget(self.prev_frame_btn)

        self.next_frame_btn = QPushButton("Next ▶")
        self.next_frame_btn.setEnabled(False)
        self.next_frame_btn.clicked.connect(self._next_frame)
        controls_layout.addWidget(self.next_frame_btn)

        self.track_video_btn = QPushButton("🎬 Track Video")
        self.track_video_btn.setEnabled(False)
        self.track_video_btn.clicked.connect(self._track_video)
        controls_layout.addWidget(self.track_video_btn)

        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Initially hide timeline
        widget.setVisible(False)
        self.timeline_widget = widget

        return widget

    def _create_right_panel(self) -> QWidget:
        """Create right results panel."""
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

        # Export controls
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        self.export_mask_btn = QPushButton("Export Mask")
        self.export_mask_btn.clicked.connect(self._export_mask)
        export_layout.addWidget(self.export_mask_btn)

        self.export_viz_btn = QPushButton("Export Visualization")
        self.export_viz_btn.clicked.connect(self._export_visualization)
        export_layout.addWidget(self.export_viz_btn)

        # Video export button
        self.export_video_btn = QPushButton("Export Video Masks")
        self.export_video_btn.setEnabled(False)
        self.export_video_btn.clicked.connect(self._export_video_masks)
        export_layout.addWidget(self.export_video_btn)

        layout.addWidget(export_group)

        layout.addStretch()

        return panel

    def _setup_menubar(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Image...", self)
        open_action.triggered.connect(self._load_image)
        file_menu.addAction(open_action)

        open_video_action = QAction("Open Video...", self)
        open_video_action.triggered.connect(self._load_video)
        file_menu.addAction(open_video_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        zoom_in_action = QAction("Zoom In", self)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        view_menu.addAction(zoom_out_action)

    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        toolbar.addAction("Open", self._load_image)
        toolbar.addSeparator()
        toolbar.addAction("Point Mode", lambda: self.annotation_mode_combo.setCurrentIndex(0))
        toolbar.addAction("Box Mode", lambda: self.annotation_mode_combo.setCurrentIndex(1))

    def _setup_statusbar(self):
        """Setup status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load an image to start")

    def _apply_dark_theme(self):
        """Apply modern dark theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5589;
            }
            QLineEdit, QTextEdit {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px;
            }
            QSlider::groove:horizontal {
                background: #3c3c3c;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0e639c;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)

    # Slots
    def _load_image(self):
        """Load image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image",
            "", "Images (*.png *.jpg *.jpeg *.bmp)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_path:
            self.current_image_path = Path(file_path)
            image = cv2.imread(str(self.current_image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.viewport.set_image(image_rgb)
            self.status_bar.showMessage(f"Loaded: {self.current_image_path.name}")

    def _load_video(self):
        """Load video file with full timeline support."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video",
            "", "Videos (*.mp4 *.avi *.mov *.mkv)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_path:
            self.status_bar.showMessage(f"Loading video: {Path(file_path).name}")

            try:
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
                self.track_video_btn.setEnabled(True)

                # Show timeline
                self.timeline_widget.setVisible(True)

                self.status_bar.showMessage(
                    f"Video loaded: {Path(file_path).name} - "
                    f"{width}x{height}, {self.total_frames} frames @ {self.fps:.1f}fps"
                )

            except Exception as e:
                self.status_bar.showMessage(f"Error loading video: {str(e)}")

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

        # Display frame
        self.viewport.set_image(frame_rgb)

        # Display mask if available for this frame
        if frame_idx in self.video_masks:
            self.viewport.set_mask(self.video_masks[frame_idx])
        else:
            self.viewport.mask = None
            self.viewport._update_display()

        self.current_frame_idx = frame_idx

    def _segment_with_text(self):
        """Segment with text prompt."""
        # Check if we have an image or video loaded
        if self.current_image_path is None and self.video_path is None:
            self.status_bar.showMessage("Error: Load an image or video first")
            return

        text_prompt = self.text_input.text().strip()
        if not text_prompt:
            self.status_bar.showMessage("Error: Enter a text prompt")
            return

        # If video mode, save current frame to temp file
        if self.video_path is not None:
            import tempfile
            temp_dir = Path(tempfile.gettempdir())
            temp_frame = temp_dir / f"sam3_current_frame_{self.current_frame_idx}.png"

            # Get current frame from viewport
            if self.video_cap is not None:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                ret, frame = self.video_cap.read()
                if ret:
                    cv2.imwrite(str(temp_frame), frame)
                    self.current_image_path = temp_frame
                else:
                    self.status_bar.showMessage("Error: Cannot read current frame")
                    return

        self.status_bar.showMessage(f"Segmenting: '{text_prompt}'...")
        self.worker.process_image_text(self.current_image_path, text_prompt)

    def _segment_with_visual(self):
        """Segment with visual prompts (points or box)."""
        # Check if we have an image or video loaded
        if self.current_image_path is None and self.video_path is None:
            self.status_bar.showMessage("Error: Load an image or video first")
            return

        # If video mode, save current frame to temp file
        if self.video_path is not None and self.current_image_path is None:
            import tempfile
            temp_dir = Path(tempfile.gettempdir())
            temp_frame = temp_dir / f"sam3_current_frame_{self.current_frame_idx}.png"

            if self.video_cap is not None:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                ret, frame = self.video_cap.read()
                if ret:
                    cv2.imwrite(str(temp_frame), frame)
                    self.current_image_path = temp_frame
                else:
                    self.status_bar.showMessage("Error: Cannot read current frame")
                    return

        if self.viewport.points:
            # Use points
            self.status_bar.showMessage(f"Segmenting with {len(self.viewport.points)} points...")
            self.worker.process_image_points(
                self.current_image_path,
                self.viewport.points,
                self.viewport.point_labels
            )
        elif self.viewport.box_start and self.viewport.box_end:
            # Use box
            x1, y1 = self.viewport.box_start
            x2, y2 = self.viewport.box_end
            box = (x1, y1, x2, y2)
            self.status_bar.showMessage("Segmenting with box...")
            self.worker.process_image_box(self.current_image_path, box)
        else:
            self.status_bar.showMessage("Error: Add points or draw a box first")

    def _clear_annotations(self):
        """Clear all annotations."""
        self.viewport.clear_annotations()
        self.status_bar.showMessage("Annotations cleared")

    def _change_annotation_mode(self, mode_text: str):
        """Change annotation mode."""
        if "Point" in mode_text:
            self.viewport.set_annotation_mode("point")
            self.status_bar.showMessage("Point annotation mode (Ctrl+click for background)")
        else:
            self.viewport.set_annotation_mode("box")
            self.status_bar.showMessage("Box annotation mode")

    def _update_alpha(self, value: int):
        """Update mask overlay alpha."""
        alpha = value / 100.0
        self.alpha_label.setText(f"{value}%")
        self.viewport.set_overlay_alpha(alpha)

    def _on_point_added(self, x: int, y: int):
        """Handle point added."""
        num_points = len(self.viewport.points)
        self.status_bar.showMessage(f"Point {num_points} added at ({x}, {y})")

    def _on_box_drawn(self, x1: int, y1: int, x2: int, y2: int):
        """Handle box drawn."""
        self.status_bar.showMessage(f"Box drawn: ({x1}, {y1}) -> ({x2}, {y2})")

    def _on_result_ready(self, result: SegmentationResult):
        """Handle segmentation result."""
        self.current_result = result

        # Display mask
        best_mask, score = result.get_best_mask()
        self.viewport.set_mask(best_mask)

        # Update results text
        info = f"""Segmentation Complete
━━━━━━━━━━━━━━━━━━━━
Prompt Type: {result.prompt_type.value}
Masks Found: {len(result.masks)}
Best Score: {score:.3f}

Bounding Boxes:
"""
        for i, (box, s) in enumerate(zip(result.boxes, result.scores)):
            x1, y1, x2, y2 = box
            info += f"  {i+1}. [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}] (score: {s:.3f})\n"

        self.results_text.setText(info)
        self.status_bar.showMessage(f"Segmentation complete - Score: {score:.3f}")

    def _on_error(self, error_msg: str):
        """Handle error."""
        self.status_bar.showMessage(f"Error: {error_msg}")
        self.results_text.setText(f"ERROR:\n{error_msg}")

    def _on_progress(self, message: str):
        """Handle progress update."""
        self.status_bar.showMessage(message)

    def _export_mask(self):
        """Export mask to file."""
        if self.current_result is None:
            self.status_bar.showMessage("Error: No mask to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask",
            "mask.png", "Images (*.png)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_path:
            best_mask, _ = self.current_result.get_best_mask()
            mask_uint8 = (best_mask * 255).astype(np.uint8)
            cv2.imwrite(file_path, mask_uint8)
            self.status_bar.showMessage(f"Mask saved: {Path(file_path).name}")

    def _export_visualization(self):
        """Export visualization to file."""
        if self.current_result is None or self.current_image_path is None:
            self.status_bar.showMessage("Error: No result to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Visualization",
            "visualization.png", "Images (*.png)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_path:
            # Create visualization
            image = cv2.imread(str(self.current_image_path))
            best_mask, score = self.current_result.get_best_mask()

            overlay = image.copy()
            overlay[best_mask > 0] = [0, 255, 0]
            blended = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

            # Draw box
            best_idx = self.current_result.scores.argmax()
            box = self.current_result.boxes[best_idx]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                blended, f"Score: {score:.3f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.imwrite(file_path, blended)
            self.status_bar.showMessage(f"Visualization saved: {Path(file_path).name}")

    def _export_video_masks(self):
        """Export all video masks to a directory."""
        if not self.video_masks:
            self.status_bar.showMessage("Error: No video masks to export")
            return

        # Ask user for output directory
        from PySide6.QtWidgets import QFileDialog
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Video Masks",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if not output_dir:
            return

        output_path = Path(output_dir)
        self.status_bar.showMessage(f"Exporting {len(self.video_masks)} masks...")

        try:
            # Create subdirectory for masks
            masks_dir = output_path / f"{self.video_path.stem}_masks"
            masks_dir.mkdir(exist_ok=True)

            # Export each mask
            exported_count = 0
            for frame_idx in sorted(self.video_masks.keys()):
                mask = self.video_masks[frame_idx]

                # Ensure mask is 2D and valid
                if mask.ndim > 2:
                    mask = mask.squeeze()

                # Convert to uint8
                if mask.dtype != np.uint8:
                    mask_uint8 = (mask * 255).astype(np.uint8)
                else:
                    mask_uint8 = mask

                # Save mask
                mask_filename = masks_dir / f"mask_{frame_idx:04d}.png"

                try:
                    cv2.imwrite(str(mask_filename), mask_uint8)
                    exported_count += 1

                    # Update status every 10 frames
                    if exported_count % 10 == 0:
                        self.status_bar.showMessage(f"Exported {exported_count}/{len(self.video_masks)} masks...")
                except Exception as e:
                    print(f"Error exporting frame {frame_idx}: {e}")
                    continue

            # Create a summary file
            summary_file = masks_dir / "export_info.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Video: {self.video_path.name}\n")
                f.write(f"Total frames: {self.total_frames}\n")
                f.write(f"Tracked frames: {len(self.video_masks)}\n")
                f.write(f"Exported masks: {exported_count}\n")
                f.write(f"Frame indices: {sorted(self.video_masks.keys())}\n")

            self.status_bar.showMessage(
                f"✓ Exported {exported_count} masks to {masks_dir.name}"
            )

        except Exception as e:
            self.status_bar.showMessage(f"Error during export: {str(e)}")
            print(f"Export error: {e}")
            import traceback
            traceback.print_exc()

    # Video timeline controls
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
            self.play_btn.setText("▶ Play")
            if self.play_timer:
                self.play_timer.stop()
        else:
            # Start playing
            self.is_playing = True
            self.play_btn.setText("⏸ Pause")

            # Create timer for playback
            if self.play_timer is None:
                self.play_timer = QTimer()
                self.play_timer.timeout.connect(self._play_next_frame)

            # Calculate interval from FPS (convert to milliseconds)
            interval = int(1000 / self.fps) if self.fps > 0 else 33  # Default 30fps
            self.play_timer.start(interval)

    def _play_next_frame(self):
        """Advance to next frame during playback."""
        if self.current_frame_idx < self.total_frames - 1:
            self._next_frame()
        else:
            # End of video, stop playing
            self._toggle_play()
            self.timeline_slider.setValue(0)  # Reset to start

    def _track_video(self):
        """Track object through entire video using SAM3VideoTracker."""
        if self.video_path is None:
            self.status_bar.showMessage("Error: No video loaded")
            return

        text_prompt = self.text_input.text().strip()
        if not text_prompt:
            self.status_bar.showMessage("Error: Enter a text prompt for tracking")
            return

        # Disable controls during tracking
        self.track_video_btn.setEnabled(False)
        self.status_bar.showMessage(f"Tracking '{text_prompt}' through {self.total_frames} frames...")

        try:
            # Import SAM3VideoTracker
            from sam3_complete import SAM3VideoTracker

            # Initialize tracker
            tracker = SAM3VideoTracker(device="cuda")

            # Start session with video
            session = tracker.start_session(self.video_path)

            # Add text prompt to first frame
            result = tracker.add_text_prompt(
                session=session,
                frame_index=0,
                text_prompt=text_prompt
            )

            # Store mask for first frame
            best_mask, score = result.get_best_mask()
            self.video_masks[0] = best_mask

            # Propagate tracking across all frames
            self.status_bar.showMessage(f"Propagating tracking to all frames...")

            results = tracker.propagate_tracking(
                session=session,
                start_frame=0,
                end_frame=self.total_frames - 1
            )

            # Store all masks
            for frame_idx, frame_result in session.frame_results.items():
                mask, _ = frame_result.get_best_mask()
                self.video_masks[frame_idx] = mask

            # Update current frame to show mask
            self._load_frame(self.current_frame_idx)

            # Enable video export button
            self.export_video_btn.setEnabled(True)

            self.status_bar.showMessage(
                f"✓ Tracking complete! Processed {len(self.video_masks)} frames - Ready to export"
            )

        except Exception as e:
            self.status_bar.showMessage(f"Error during tracking: {str(e)}")
            print(f"Tracking error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Re-enable controls
            self.track_video_btn.setEnabled(True)


def main():
    """Main entry point."""
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="SAM3 GUI - Professional Segmentation")
    parser.add_argument("--image", type=str, help="Load image file directly (bypasses file dialog)")
    parser.add_argument("--video", type=str, help="Load video file directly (bypasses file dialog)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("SAM3 Professional")

    window = SAM3MainWindow()

    # Load file directly if provided
    if args.image:
        image_path = Path(args.image)
        if image_path.exists():
            window.current_image_path = image_path
            image = cv2.imread(str(image_path))
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                window.viewport.set_image(image_rgb)
                window.status_bar.showMessage(f"Loaded: {image_path.name}")
            else:
                print(f"Error: Cannot read image {image_path}")
        else:
            print(f"Error: Image file not found: {image_path}")

    elif args.video:
        video_path = Path(args.video)
        if video_path.exists():
            window.video_cap = cv2.VideoCapture(str(video_path))
            if window.video_cap.isOpened():
                window.fps = window.video_cap.get(cv2.CAP_PROP_FPS)
                window.total_frames = int(window.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                window.video_path = video_path
                window.video_frames = []
                window.video_masks = {}
                window.current_frame_idx = 0

                # Load first frame
                window._load_frame(0)

                # Setup timeline
                window.timeline_slider.setMaximum(window.total_frames - 1)
                window.timeline_slider.setValue(0)
                window.timeline_slider.setEnabled(True)
                window.frame_label.setText(f"0 / {window.total_frames}")
                window.play_btn.setEnabled(True)
                window.prev_frame_btn.setEnabled(True)
                window.next_frame_btn.setEnabled(True)
                window.track_video_btn.setEnabled(True)
                window.timeline_widget.setVisible(True)

                window.status_bar.showMessage(f"Video loaded: {video_path.name} - {window.total_frames} frames")
            else:
                print(f"Error: Cannot open video {video_path}")
        else:
            print(f"Error: Video file not found: {video_path}")

    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
