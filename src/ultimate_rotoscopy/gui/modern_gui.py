#!/usr/bin/env python3
"""
Ultimate Rotoscopy - Modern Professional GUI
=============================================

Cinema-quality interface with clear tabbed workflow:
1. Media - Load video/sequence
2. Segmentation - SAM3 prompts and masks
3. Matting - Professional alpha refinement
4. Composite - Final result
5. Export - Multi-layer output

Complete integration with all backend models.
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2

# GUI Framework
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QToolBar, QStatusBar, QMenuBar,
    QMenu, QFileDialog, QMessageBox, QProgressBar, QLabel,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QPushButton, QGroupBox, QScrollArea, QFrame, QSizePolicy,
    QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QLineEdit, QTextEdit, QRadioButton,
    QButtonGroup, QGridLayout,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, Slot, QTimer, QSize, QPoint, QRect,
    QPointF, QRectF, QSettings, QRunnable, QThreadPool,
)
from PySide6.QtGui import (
    QAction, QIcon, QPixmap, QImage, QPainter, QPen, QBrush,
    QColor, QKeySequence, QCursor, QFont, QTransform,
)

# Processing backend
try:
    from ultimate_rotoscopy.gui.backend import (
        ProcessingBackend, ProcessingStage, ProcessingResult, ProcessingRequest
    )
    # Import models directly for proper connection
    from ultimate_rotoscopy.models.sam3 import SAM3Segmentor, SAM3Config
    from ultimate_rotoscopy.matting.professional_matting import (
        ProfessionalMatting, ProfessionalMattingConfig
    )
    from ultimate_rotoscopy.export.aov_manager import AOVManager
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Backend import failed: {e}")
    BACKEND_AVAILABLE = False


class PromptType(Enum):
    """SAM3 prompt types."""
    POINT_FG = "foreground_point"
    POINT_BG = "background_point"
    BOX = "bounding_box"
    TEXT = "text_prompt"


@dataclass
class Prompt:
    """SAM3 prompt data."""
    type: PromptType
    data: Any  # Points: (x, y), Box: (x1, y1, x2, y2), Text: str
    label: int = 1  # 1=foreground, 0=background


class InteractiveCanvas(QGraphicsView):
    """
    Interactive canvas for image display and SAM3 prompting.

    Features:
    - Pan/zoom
    - Point prompts (left=FG, right=BG)
    - Box prompts (drag rectangle)
    - Overlay visualization with opacity
    """

    prompt_added = Signal(object)  # Emits Prompt

    def __init__(self, parent=None):
        super().__init__(parent)

        # Setup scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Configure view
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # Image layers
        self.image_item = None
        self.overlay_item = None

        # Prompts
        self.prompts: List[Prompt] = []
        self.prompt_items = []  # Visual representations

        # Interaction state
        self.prompt_mode = PromptType.POINT_FG
        self.overlay_opacity = 0.5
        self.box_start = None
        self.box_rect_item = None

        # Zoom
        self.zoom_factor = 1.0

    def set_image(self, image: np.ndarray):
        """Set main image."""
        self.scene.clear()
        self.image_item = None
        self.overlay_item = None
        self.prompt_items.clear()

        # Convert to QImage
        if len(image.shape) == 2:
            # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Ensure RGB
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # Convert to uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        h, w = image.shape[:2]
        qimage = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        self.image_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(0, 0, w, h)
        self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def set_overlay(self, overlay: np.ndarray, opacity: float = 0.5):
        """Set overlay (mask/matte/depth)."""
        if self.overlay_item:
            self.scene.removeItem(self.overlay_item)
            self.overlay_item = None

        self.overlay_opacity = opacity

        # Convert to RGB
        if len(overlay.shape) == 2:
            # Single channel - colorize green for masks
            overlay_rgb = np.zeros((*overlay.shape, 3), dtype=np.uint8)
            if overlay.dtype == np.float32 or overlay.dtype == np.float64:
                overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
            overlay_rgb[:, :, 1] = overlay  # Green channel
        else:
            if overlay.dtype == np.float32 or overlay.dtype == np.float64:
                overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
            overlay_rgb = overlay[:, :, :3]

        h, w = overlay_rgb.shape[:2]
        qimage = QImage(overlay_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        self.overlay_item = self.scene.addPixmap(pixmap)
        self.overlay_item.setOpacity(opacity)
        self.overlay_item.setZValue(1)  # Above image

    def update_overlay_opacity(self, opacity: float):
        """Update overlay opacity."""
        self.overlay_opacity = opacity
        if self.overlay_item:
            self.overlay_item.setOpacity(opacity)

    def set_prompt_mode(self, mode: PromptType):
        """Set current prompt mode."""
        self.prompt_mode = mode

    def clear_prompts(self):
        """Clear all prompts."""
        self.prompts.clear()
        for item in self.prompt_items:
            self.scene.removeItem(item)
        self.prompt_items.clear()

    def mousePressEvent(self, event):
        """Handle mouse press for prompts."""
        if self.image_item is None:
            return super().mousePressEvent(event)

        # Get scene position
        scene_pos = self.mapToScene(event.pos())
        x, y = int(scene_pos.x()), int(scene_pos.y())

        # Check bounds
        rect = self.image_item.boundingRect()
        if not rect.contains(scene_pos):
            return super().mousePressEvent(event)

        if self.prompt_mode == PromptType.POINT_FG:
            # Left click = foreground point
            if event.button() == Qt.LeftButton:
                prompt = Prompt(PromptType.POINT_FG, (x, y), label=1)
                self.prompts.append(prompt)
                self._draw_point(x, y, QColor(0, 255, 0))  # Green
                self.prompt_added.emit(prompt)

        elif self.prompt_mode == PromptType.POINT_BG:
            # Right click or left click = background point
            prompt = Prompt(PromptType.POINT_BG, (x, y), label=0)
            self.prompts.append(prompt)
            self._draw_point(x, y, QColor(255, 0, 0))  # Red
            self.prompt_added.emit(prompt)

        elif self.prompt_mode == PromptType.BOX:
            # Start box drawing
            if event.button() == Qt.LeftButton:
                self.box_start = (x, y)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for box drawing."""
        if self.box_start and self.prompt_mode == PromptType.BOX:
            scene_pos = self.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())

            # Draw temporary rectangle
            if self.box_rect_item:
                self.scene.removeItem(self.box_rect_item)

            x1, y1 = self.box_start
            rect = QRectF(
                min(x1, x),
                min(y1, y),
                abs(x - x1),
                abs(y - y1)
            )

            pen = QPen(QColor(0, 255, 255), 2, Qt.DashLine)
            self.box_rect_item = self.scene.addRect(rect, pen)
            self.box_rect_item.setZValue(2)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release for box completion."""
        if self.box_start and self.prompt_mode == PromptType.BOX:
            scene_pos = self.mapToScene(event.pos())
            x2, y2 = int(scene_pos.x()), int(scene_pos.y())
            x1, y1 = self.box_start

            # Create box prompt
            box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            prompt = Prompt(PromptType.BOX, box, label=1)
            self.prompts.append(prompt)

            # Draw final box
            rect = QRectF(box[0], box[1], box[2] - box[0], box[3] - box[1])
            pen = QPen(QColor(0, 255, 255), 2)
            box_item = self.scene.addRect(rect, pen)
            box_item.setZValue(2)
            self.prompt_items.append(box_item)

            self.prompt_added.emit(prompt)

            # Reset
            self.box_start = None
            if self.box_rect_item:
                self.scene.removeItem(self.box_rect_item)
                self.box_rect_item = None

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Handle zoom."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self.zoom_factor *= factor

    def _draw_point(self, x: int, y: int, color: QColor):
        """Draw a point prompt."""
        radius = 5
        pen = QPen(color, 2)
        brush = QBrush(color)

        circle = self.scene.addEllipse(
            x - radius, y - radius, radius * 2, radius * 2,
            pen, brush
        )
        circle.setZValue(2)
        self.prompt_items.append(circle)


class MediaTab(QWidget):
    """
    Tab 1: Media Loading

    Load video files (mov, mp4) or image sequences (exr, png, tiff).
    """

    media_loaded = Signal(object, object)  # frames, metadata

    def __init__(self, parent=None):
        super().__init__(parent)

        self.frames = []
        self.current_frame_idx = 0
        self.metadata = {}

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("<h2>Load Media</h2>")
        layout.addWidget(header)

        # Load buttons
        btn_layout = QHBoxLayout()

        self.btn_load_video = QPushButton("📹 Load Video (mov/mp4/avi)")
        self.btn_load_video.clicked.connect(self._load_video)
        self.btn_load_video.setMinimumHeight(50)
        btn_layout.addWidget(self.btn_load_video)

        self.btn_load_sequence = QPushButton("🎞️ Load Sequence (exr/png/tiff)")
        self.btn_load_sequence.clicked.connect(self._load_sequence)
        self.btn_load_sequence.setMinimumHeight(50)
        btn_layout.addWidget(self.btn_load_sequence)

        self.btn_load_image = QPushButton("🖼️ Load Single Image")
        self.btn_load_image.clicked.connect(self._load_image)
        self.btn_load_image.setMinimumHeight(50)
        btn_layout.addWidget(self.btn_load_image)

        layout.addLayout(btn_layout)

        # Info display
        info_group = QGroupBox("Media Info")
        info_layout = QGridLayout()

        self.lbl_path = QLabel("Path: -")
        self.lbl_resolution = QLabel("Resolution: -")
        self.lbl_fps = QLabel("FPS: -")
        self.lbl_frames = QLabel("Frames: -")
        self.lbl_duration = QLabel("Duration: -")

        info_layout.addWidget(QLabel("<b>Path:</b>"), 0, 0)
        info_layout.addWidget(self.lbl_path, 0, 1)
        info_layout.addWidget(QLabel("<b>Resolution:</b>"), 1, 0)
        info_layout.addWidget(self.lbl_resolution, 1, 1)
        info_layout.addWidget(QLabel("<b>FPS:</b>"), 2, 0)
        info_layout.addWidget(self.lbl_fps, 2, 1)
        info_layout.addWidget(QLabel("<b>Frames:</b>"), 3, 0)
        info_layout.addWidget(self.lbl_frames, 3, 1)
        info_layout.addWidget(QLabel("<b>Duration:</b>"), 4, 0)
        info_layout.addWidget(self.lbl_duration, 4, 1)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Frame navigator
        nav_group = QGroupBox("Frame Navigator")
        nav_layout = QVBoxLayout()

        slider_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        slider_layout.addWidget(QLabel("Frame:"))
        slider_layout.addWidget(self.frame_slider)

        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.setMaximum(0)
        self.frame_spinbox.valueChanged.connect(self._on_frame_changed)
        slider_layout.addWidget(self.frame_spinbox)

        nav_layout.addLayout(slider_layout)
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)

        layout.addStretch()

    def _load_video(self):
        """Load video file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Video",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)"
        )

        if not filepath:
            return

        try:
            # Load with OpenCV
            cap = cv2.VideoCapture(filepath)

            if not cap.isOpened():
                QMessageBox.critical(self, "Error", f"Could not open video: {filepath}")
                return

            # Get metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            # Load all frames (for now - TODO: optimize with caching)
            self.frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_float = frame_rgb.astype(np.float32) / 255.0
                self.frames.append(frame_float)

            cap.release()

            # Store metadata
            self.metadata = {
                'path': filepath,
                'type': 'video',
                'resolution': (width, height),
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
            }

            # Update UI
            self._update_info()
            self.frame_slider.setMaximum(frame_count - 1)
            self.frame_spinbox.setMaximum(frame_count - 1)
            self.frame_slider.setValue(0)

            # Emit signal
            self.media_loaded.emit(self.frames, self.metadata)

            QMessageBox.information(
                self,
                "Success",
                f"Loaded {frame_count} frames from video\n{width}x{height} @ {fps:.2f} FPS"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video:\n{str(e)}")

    def _load_sequence(self):
        """Load image sequence."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load First Frame of Sequence",
            "",
            "Image Files (*.exr *.png *.tif *.tiff *.jpg *.jpeg);;All Files (*)"
        )

        if not filepath:
            return

        try:
            path = Path(filepath)
            directory = path.parent
            ext = path.suffix

            # Find all files with same extension
            files = sorted(directory.glob(f"*{ext}"))

            if not files:
                QMessageBox.warning(self, "Warning", "No sequence files found")
                return

            # Load all frames
            self.frames = []
            for file in files:
                img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                # Handle different formats
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Normalize to float32
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                elif img.dtype == np.uint16:
                    img = img.astype(np.float32) / 65535.0

                self.frames.append(img)

            if not self.frames:
                QMessageBox.warning(self, "Warning", "No valid frames loaded")
                return

            # Metadata
            h, w = self.frames[0].shape[:2]
            self.metadata = {
                'path': str(directory),
                'type': 'sequence',
                'resolution': (w, h),
                'fps': 24.0,  # Default
                'frame_count': len(self.frames),
                'duration': len(self.frames) / 24.0,
            }

            # Update UI
            self._update_info()
            self.frame_slider.setMaximum(len(self.frames) - 1)
            self.frame_spinbox.setMaximum(len(self.frames) - 1)
            self.frame_slider.setValue(0)

            # Emit signal
            self.media_loaded.emit(self.frames, self.metadata)

            QMessageBox.information(
                self,
                "Success",
                f"Loaded {len(self.frames)} frames from sequence\n{w}x{h}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load sequence:\n{str(e)}")

    def _load_image(self):
        """Load single image."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.exr);;All Files (*)"
        )

        if not filepath:
            return

        try:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

            if img is None:
                QMessageBox.critical(self, "Error", f"Could not load image: {filepath}")
                return

            # Handle formats
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0

            self.frames = [img]
            h, w = img.shape[:2]

            self.metadata = {
                'path': filepath,
                'type': 'image',
                'resolution': (w, h),
                'fps': 24.0,
                'frame_count': 1,
                'duration': 0.0,
            }

            # Update UI
            self._update_info()
            self.frame_slider.setMaximum(0)
            self.frame_spinbox.setMaximum(0)

            # Emit signal
            self.media_loaded.emit(self.frames, self.metadata)

            QMessageBox.information(self, "Success", f"Loaded image\n{w}x{h}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    def _update_info(self):
        """Update info labels."""
        self.lbl_path.setText(self.metadata.get('path', '-'))
        w, h = self.metadata.get('resolution', (0, 0))
        self.lbl_resolution.setText(f"{w}x{h}")
        self.lbl_fps.setText(f"{self.metadata.get('fps', 0):.2f}")
        self.lbl_frames.setText(str(self.metadata.get('frame_count', 0)))
        self.lbl_duration.setText(f"{self.metadata.get('duration', 0):.2f}s")

    def _on_frame_changed(self, value):
        """Handle frame slider/spinbox change."""
        # Sync slider and spinbox
        self.frame_slider.blockSignals(True)
        self.frame_spinbox.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_spinbox.setValue(value)
        self.frame_slider.blockSignals(False)
        self.frame_spinbox.blockSignals(False)

        self.current_frame_idx = value

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame."""
        if self.frames and 0 <= self.current_frame_idx < len(self.frames):
            return self.frames[self.current_frame_idx]
        return None


class SegmentationTab(QWidget):
    """
    Tab 2: Segmentation with SAM3

    Add prompts and generate masks.
    """

    mask_generated = Signal(object)  # mask

    def __init__(self, canvas: InteractiveCanvas, parent=None):
        super().__init__(parent)

        self.canvas = canvas
        self.current_image = None
        self.current_mask = None
        self.backend = None
        self.layers = []  # List of (name, mask) tuples

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("<h2>SAM3 Segmentation</h2>")
        layout.addWidget(header)

        # Prompt tools
        tools_group = QGroupBox("Prompt Tools")
        tools_layout = QVBoxLayout()

        self.btn_group = QButtonGroup()

        self.radio_point_fg = QRadioButton("🟢 Foreground Point (Left Click)")
        self.radio_point_fg.setChecked(True)
        self.radio_point_fg.toggled.connect(lambda: self.canvas.set_prompt_mode(PromptType.POINT_FG))
        self.btn_group.addButton(self.radio_point_fg)
        tools_layout.addWidget(self.radio_point_fg)

        self.radio_point_bg = QRadioButton("🔴 Background Point (Click)")
        self.radio_point_bg.toggled.connect(lambda: self.canvas.set_prompt_mode(PromptType.POINT_BG))
        self.btn_group.addButton(self.radio_point_bg)
        tools_layout.addWidget(self.radio_point_bg)

        self.radio_box = QRadioButton("📦 Bounding Box (Drag)")
        self.radio_box.toggled.connect(lambda: self.canvas.set_prompt_mode(PromptType.BOX))
        self.btn_group.addButton(self.radio_box)
        tools_layout.addWidget(self.radio_box)

        tools_group.setLayout(tools_layout)
        layout.addWidget(tools_group)

        # Text prompt (SAM3 feature)
        text_group = QGroupBox("Text Prompt (SAM3)")
        text_layout = QVBoxLayout()

        self.text_prompt = QLineEdit()
        self.text_prompt.setPlaceholderText("e.g., 'person', 'car', 'background'...")
        text_layout.addWidget(self.text_prompt)

        text_group.setLayout(text_layout)
        layout.addWidget(text_group)

        # Prompts list
        prompts_group = QGroupBox("Current Prompts")
        prompts_layout = QVBoxLayout()

        self.prompts_list = QListWidget()
        prompts_layout.addWidget(self.prompts_list)

        btn_clear = QPushButton("🗑️ Clear Prompts")
        btn_clear.clicked.connect(self._clear_prompts)
        prompts_layout.addWidget(btn_clear)

        prompts_group.setLayout(prompts_layout)
        layout.addWidget(prompts_group)

        # Generate mask
        self.btn_generate = QPushButton("✨ Generate Mask")
        self.btn_generate.setMinimumHeight(50)
        self.btn_generate.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_generate.clicked.connect(self._generate_mask)
        layout.addWidget(self.btn_generate)

        # Mask layers
        layers_group = QGroupBox("Mask Layers")
        layers_layout = QVBoxLayout()

        self.layers_list = QListWidget()
        layers_layout.addWidget(self.layers_list)

        btn_layout = QHBoxLayout()
        btn_add_layer = QPushButton("➕ New Layer")
        btn_add_layer.clicked.connect(self._new_layer)
        btn_layout.addWidget(btn_add_layer)

        btn_combine = QPushButton("🔗 Combine Layers")
        btn_combine.clicked.connect(self._combine_layers)
        btn_layout.addWidget(btn_combine)

        layers_layout.addLayout(btn_layout)
        layers_group.setLayout(layers_layout)
        layout.addWidget(layers_group)

        # Connect canvas signal
        self.canvas.prompt_added.connect(self._on_prompt_added)

        layout.addStretch()

    def set_image(self, image: np.ndarray):
        """Set current image."""
        self.current_image = image

    def set_backend(self, backend):
        """Set processing backend."""
        self.backend = backend

    def _on_prompt_added(self, prompt: Prompt):
        """Handle prompt added from canvas."""
        # Add to list
        if prompt.type == PromptType.POINT_FG:
            text = f"🟢 FG Point: ({prompt.data[0]}, {prompt.data[1]})"
        elif prompt.type == PromptType.POINT_BG:
            text = f"🔴 BG Point: ({prompt.data[0]}, {prompt.data[1]})"
        elif prompt.type == PromptType.BOX:
            text = f"📦 Box: {prompt.data}"
        else:
            text = str(prompt)

        self.prompts_list.addItem(text)

    def _clear_prompts(self):
        """Clear all prompts."""
        self.canvas.clear_prompts()
        self.prompts_list.clear()

    def _generate_mask(self):
        """Generate mask with SAM3."""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return

        if not self.canvas.prompts and not self.text_prompt.text():
            QMessageBox.warning(self, "Warning", "Add at least one prompt")
            return

        if not BACKEND_AVAILABLE or self.backend is None:
            QMessageBox.critical(self, "Error", "Backend not available")
            return

        try:
            # Prepare prompts
            points = []
            point_labels = []
            boxes = []

            for prompt in self.canvas.prompts:
                if prompt.type in (PromptType.POINT_FG, PromptType.POINT_BG):
                    points.append(prompt.data)
                    point_labels.append(prompt.label)
                elif prompt.type == PromptType.BOX:
                    boxes.append(prompt.data)

            # Convert to numpy
            points_array = np.array(points) if points else None
            labels_array = np.array(point_labels) if point_labels else None
            box_array = np.array(boxes[0]) if boxes else None

            # Create request
            request = ProcessingRequest(
                stage=ProcessingStage.SEGMENTATION,
                image=self.current_image,
                points=points_array,
                point_labels=labels_array,
                box=box_array,
                parameters={'text_prompt': self.text_prompt.text()}
            )

            # Process (synchronous for now - TODO: async)
            self.btn_generate.setEnabled(False)
            self.btn_generate.setText("Processing...")

            QApplication.processEvents()

            result = self.backend.process_sync(request)

            if result.success and result.mask is not None:
                self.current_mask = result.mask

                # Show overlay
                self.canvas.set_overlay(result.mask, opacity=0.5)

                # Emit signal
                self.mask_generated.emit(result.mask)

                QMessageBox.information(self, "Success", "Mask generated successfully!")
            else:
                QMessageBox.warning(self, "Warning", f"Mask generation failed:\n{result.error}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate mask:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_generate.setEnabled(True)
            self.btn_generate.setText("✨ Generate Mask")

    def _new_layer(self):
        """Create new mask layer."""
        if self.current_mask is not None:
            name = f"Layer {len(self.layers) + 1}"
            self.layers.append((name, self.current_mask.copy()))
            self.layers_list.addItem(name)
            self._clear_prompts()

    def _combine_layers(self):
        """Combine all layers."""
        if not self.layers:
            QMessageBox.warning(self, "Warning", "No layers to combine")
            return

        # Combine with max
        combined = np.zeros_like(self.layers[0][1])
        for name, mask in self.layers:
            combined = np.maximum(combined, mask)

        self.current_mask = combined
        self.canvas.set_overlay(combined, opacity=0.5)
        self.mask_generated.emit(combined)


class ModernMainWindow(QMainWindow):
    """
    Main Window for Ultimate Rotoscopy.

    5-tab professional workflow:
    1. Media - Load content
    2. Segmentation - SAM3 masks
    3. Matting - Alpha refinement
    4. Composite - Final result
    5. Export - Multi-layer output
    """

    def __init__(self):
        super().__init__()

        self.frames = []
        self.metadata = {}
        self.current_mask = None
        self.backend = None

        self._setup_ui()
        self._setup_backend()
        self._connect_signals()

        self.setWindowTitle("Ultimate Rotoscopy - Professional Matting System")
        self.resize(1600, 900)

    def _setup_ui(self):
        """Setup UI components."""
        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # Main splitter: canvas (left) | tabs (right)
        splitter = QSplitter(Qt.Horizontal)

        # Left: Interactive canvas
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)

        canvas_header = QLabel("<h2>Canvas</h2>")
        canvas_layout.addWidget(canvas_header)

        self.canvas = InteractiveCanvas()
        canvas_layout.addWidget(self.canvas)

        # Overlay controls
        overlay_controls = QHBoxLayout()
        overlay_controls.addWidget(QLabel("Overlay Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(
            lambda v: self.canvas.update_overlay_opacity(v / 100.0)
        )
        overlay_controls.addWidget(self.opacity_slider)
        canvas_layout.addLayout(overlay_controls)

        splitter.addWidget(canvas_widget)

        # Right: Tab widget
        self.tabs = QTabWidget()

        # Tab 1: Media
        self.media_tab = MediaTab()
        self.tabs.addTab(self.media_tab, "📹 Media")

        # Tab 2: Segmentation
        self.seg_tab = SegmentationTab(self.canvas)
        self.tabs.addTab(self.seg_tab, "✂️ Segmentation")

        # Tab 3-6: Import from modern_gui_tabs
        try:
            from ultimate_rotoscopy.gui.modern_gui_tabs import (
                DepthTab, MattingTab, CompositeTab, ExportTab
            )

            # Tab 3: Depth Anything V3
            self.depth_tab = DepthTab(self.canvas)
            self.tabs.addTab(self.depth_tab, "🌊 Depth")

            # Tab 4: Matting
            self.matting_tab = MattingTab(self.canvas)
            self.tabs.addTab(self.matting_tab, "🎨 Matting")

            # Tab 5: Composite
            self.composite_tab = CompositeTab(self.canvas)
            self.tabs.addTab(self.composite_tab, "🖼️ Composite")

            # Tab 6: Export
            self.export_tab = ExportTab()
            self.tabs.addTab(self.export_tab, "💾 Export")

        except ImportError as e:
            print(f"Warning: Could not load additional tabs: {e}")
            self.depth_tab = None
            self.matting_tab = None
            self.composite_tab = None
            self.export_tab = None

        splitter.addWidget(self.tabs)

        # Set splitter proportions (60% canvas, 40% tabs)
        splitter.setSizes([960, 640])

        main_layout.addWidget(splitter)

        # Menu bar
        self._setup_menu_bar()

        # Toolbar
        self._setup_toolbar()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _setup_menu_bar(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        load_video_action = QAction("Load Video...", self)
        load_video_action.setShortcut("Ctrl+O")
        load_video_action.triggered.connect(self.media_tab._load_video)
        file_menu.addAction(load_video_action)

        load_seq_action = QAction("Load Sequence...", self)
        load_seq_action.setShortcut("Ctrl+Shift+O")
        load_seq_action.triggered.connect(self.media_tab._load_sequence)
        file_menu.addAction(load_seq_action)

        file_menu.addSeparator()

        if self.export_tab:
            export_action = QAction("Export...", self)
            export_action.setShortcut("Ctrl+E")
            export_action.triggered.connect(lambda: self.tabs.setCurrentWidget(self.export_tab))
            file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        fit_action = QAction("Fit to View", self)
        fit_action.setShortcut("F")
        fit_action.triggered.connect(
            lambda: self.canvas.fitInView(self.canvas.image_item, Qt.KeepAspectRatio) if self.canvas.image_item else None
        )
        view_menu.addAction(fit_action)

        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.setShortcut("Ctrl+0")
        reset_zoom_action.triggered.connect(self._reset_zoom)
        view_menu.addAction(reset_zoom_action)

        # Process menu
        process_menu = menubar.addMenu("&Process")

        segment_action = QAction("Generate Mask", self)
        segment_action.setShortcut("Ctrl+G")
        segment_action.triggered.connect(self.seg_tab._generate_mask)
        process_menu.addAction(segment_action)

        if self.matting_tab:
            refine_action = QAction("Refine Alpha", self)
            refine_action.setShortcut("Ctrl+R")
            refine_action.triggered.connect(self.matting_tab._process_matting)
            process_menu.addAction(refine_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(32, 32))
        self.addToolBar(toolbar)

        # Load actions
        load_video_btn = QAction("📹 Video", self)
        load_video_btn.setToolTip("Load video file")
        load_video_btn.triggered.connect(self.media_tab._load_video)
        toolbar.addAction(load_video_btn)

        load_seq_btn = QAction("🎞️ Sequence", self)
        load_seq_btn.setToolTip("Load image sequence")
        load_seq_btn.triggered.connect(self.media_tab._load_sequence)
        toolbar.addAction(load_seq_btn)

        toolbar.addSeparator()

        # Process actions
        generate_btn = QAction("✨ Mask", self)
        generate_btn.setToolTip("Generate mask with SAM3")
        generate_btn.triggered.connect(self.seg_tab._generate_mask)
        toolbar.addAction(generate_btn)

        if self.matting_tab:
            refine_btn = QAction("🎨 Refine", self)
            refine_btn.setToolTip("Refine alpha with professional matting")
            refine_btn.triggered.connect(self.matting_tab._process_matting)
            toolbar.addAction(refine_btn)

        toolbar.addSeparator()

        # View actions
        fit_btn = QAction("🔍 Fit", self)
        fit_btn.setToolTip("Fit to view")
        fit_btn.triggered.connect(
            lambda: self.canvas.fitInView(self.canvas.image_item, Qt.KeepAspectRatio) if self.canvas.image_item else None
        )
        toolbar.addAction(fit_btn)

    def _setup_backend(self):
        """Initialize processing backend."""
        if not BACKEND_AVAILABLE:
            self.status_label.setText("⚠️ Backend not available")
            return

        try:
            self.status_label.setText("Loading models...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate

            QApplication.processEvents()

            # Initialize backend
            self.backend = ProcessingBackend()

            # Set backend on tabs
            self.seg_tab.set_backend(self.backend)
            if self.matting_tab:
                self.matting_tab.set_backend(self.backend)

            # Initialize and set Depth Anything V3
            if self.depth_tab:
                try:
                    from ultimate_rotoscopy.models.depth_anything import DepthAnythingV3, DepthConfig, DepthModelSize
                    depth_config = DepthConfig(
                        model_size=DepthModelSize.LARGE,
                        generate_normals=True,
                        estimate_intrinsics=True,
                        sky_segmentation=True,
                    )
                    depth_model = DepthAnythingV3(depth_config)
                    # Note: Model will be loaded on first use (lazy loading)
                    self.depth_tab.set_depth_model(depth_model)
                    print("Depth Anything V3 model configured (lazy loading)")
                except ImportError as e:
                    print(f"Warning: Could not load Depth Anything V3: {e}")

            self.progress_bar.setVisible(False)
            self.status_label.setText("✓ Ready - Models loaded")

        except Exception as e:
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"⚠️ Backend initialization failed: {e}")
            print(f"Backend initialization error: {e}")
            import traceback
            traceback.print_exc()

    def _connect_signals(self):
        """Connect signals between tabs."""
        # Media loaded -> update canvas and segmentation tab
        self.media_tab.media_loaded.connect(self._on_media_loaded)

        # Frame changed -> update canvas
        self.media_tab.frame_slider.valueChanged.connect(self._on_frame_changed)

        # Mask generated -> enable matting
        self.seg_tab.mask_generated.connect(self._on_mask_generated)

        # Depth generated -> can be used for compositing
        if self.depth_tab:
            self.depth_tab.depth_generated.connect(self._on_depth_generated)

        # Matting result -> enable composite
        if self.matting_tab:
            self.matting_tab.matte_generated.connect(self._on_matting_complete)

        # Composite ready -> enable export
        if self.composite_tab:
            self.composite_tab.composite_generated.connect(self._on_composite_ready)

    def _on_media_loaded(self, frames, metadata):
        """Handle media loaded."""
        self.frames = frames
        self.metadata = metadata

        # Display first frame
        if frames:
            self.canvas.set_image(frames[0])
            self.seg_tab.set_image(frames[0])
            if self.depth_tab:
                self.depth_tab.set_image(frames[0])

        self.status_label.setText(f"✓ Loaded {len(frames)} frames - {metadata.get('resolution', '?')}")

        # Switch to segmentation tab
        self.tabs.setCurrentIndex(1)

    def _on_frame_changed(self, frame_idx):
        """Handle frame change."""
        if self.frames and 0 <= frame_idx < len(self.frames):
            frame = self.frames[frame_idx]
            self.canvas.set_image(frame)
            self.seg_tab.set_image(frame)
            if self.depth_tab:
                self.depth_tab.set_image(frame)

    def _on_mask_generated(self, mask):
        """Handle mask generated."""
        self.current_mask = mask
        self.status_label.setText("✓ Mask generated - Ready for matting")

        # Enable matting tab
        if self.matting_tab:
            frame = self.media_tab.get_current_frame()
            if frame is not None:
                self.matting_tab.set_image(frame)
            self.matting_tab.set_mask(mask)

    def _on_matting_complete(self, alpha_final):
        """Handle matting complete."""
        self.status_label.setText("✓ Matting complete - Ready for compositing")

        # Enable composite tab
        if self.composite_tab:
            # Get current frame as foreground
            frame = self.media_tab.get_current_frame()
            if frame is not None:
                self.composite_tab.set_foreground(frame)
            self.composite_tab.set_alpha(alpha_final)

    def _on_composite_ready(self, composite):
        """Handle composite ready."""
        self.status_label.setText("✓ Composite ready - Ready for export")

        # Enable export tab
        if self.export_tab:
            self.export_tab.set_layer_data("composite", composite)

    def _on_depth_generated(self, depth_result):
        """Handle depth estimation complete."""
        self.status_label.setText("✓ Depth estimated - Available for compositing")

        # Depth maps can be used for:
        # - 3D compositing with depth-aware effects
        # - Relighting based on normals
        # - Point cloud export for 3D reconstruction
        # Store for later use if needed
        pass

    def _reset_zoom(self):
        """Reset zoom to 100%."""
        if self.canvas.image_item:
            self.canvas.resetTransform()
            self.canvas.zoom_factor = 1.0

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Ultimate Rotoscopy",
            """<h2>Ultimate Rotoscopy</h2>
            <p><b>Professional Matting System</b></p>
            <p>Version 1.0</p>
            <hr>
            <p>Features:</p>
            <ul>
                <li>SAM3 Segmentation with prompts</li>
                <li>Professional alpha matting (core/edge/hair)</li>
                <li>Motion blur awareness</li>
                <li>Multi-layer EXR export</li>
                <li>Cinema-quality compositing</li>
            </ul>
            <hr>
            <p>Powered by SAM3, Depth Anything V3, ViTMatte</p>
            """
        )


def main():
    """Application entry point."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Dark theme
    palette = app.palette()
    palette.setColor(palette.Window, QColor(53, 53, 53))
    palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, QColor(35, 35, 35))
    palette.setColor(palette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(palette.ToolTipText, Qt.white)
    palette.setColor(palette.Text, Qt.white)
    palette.setColor(palette.Button, QColor(53, 53, 53))
    palette.setColor(palette.ButtonText, Qt.white)
    palette.setColor(palette.BrightText, Qt.red)
    palette.setColor(palette.Link, QColor(42, 130, 218))
    palette.setColor(palette.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.HighlightedText, QColor(35, 35, 35))
    app.setPalette(palette)

    # Create main window
    window = ModernMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
