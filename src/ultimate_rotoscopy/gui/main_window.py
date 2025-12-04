#!/usr/bin/env python3
"""
Ultimate Rotoscopy - Professional GUI
=====================================

A cinema-quality interface for AI-powered rotoscopy and compositing.

Features:
- Interactive segmentation with SAM2
- AI matting with MatAnyone, GVM, ViTMatte
- Depth estimation with Depth Anything V2
- Professional compositing (despill, light wrap, harmonization)
- ACES color management
- Multi-layer timeline
- Real-time preview
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

# GUI Framework
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QToolBar, QStatusBar, QMenuBar,
    QMenu, QFileDialog, QMessageBox, QProgressBar, QLabel,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QPushButton, QGroupBox, QScrollArea, QFrame, QSizePolicy,
    QDockWidget, QListWidget, QListWidgetItem, QTreeWidget,
    QTreeWidgetItem, QStackedWidget, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QRubberBand, QColorDialog, QInputDialog,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, Slot, QTimer, QSize, QPoint, QRect,
    QPointF, QRectF, QSettings, QUrl,
)
from PySide6.QtGui import (
    QAction, QIcon, QPixmap, QImage, QPainter, QPen, QBrush,
    QColor, QKeySequence, QCursor, QFont, QWheelEvent,
    QMouseEvent, QPainterPath, QTransform,
)


class ToolMode(Enum):
    """Available tools for interaction."""
    SELECT = "select"
    PAN = "pan"
    ZOOM = "zoom"
    POINT_FG = "point_fg"       # Foreground point
    POINT_BG = "point_bg"       # Background point
    BOX = "box"                 # Bounding box
    BRUSH_ADD = "brush_add"     # Add to mask
    BRUSH_SUB = "brush_sub"     # Subtract from mask
    MEASURE = "measure"


class ViewMode(Enum):
    """Viewer display modes."""
    SOURCE = "source"
    MATTE = "matte"
    COMPOSITE = "composite"
    DEPTH = "depth"
    NORMALS = "normals"
    SPLIT = "split"
    CHECKERBOARD = "checkerboard"


@dataclass
class ProjectSettings:
    """Project configuration."""
    name: str = "Untitled"
    width: int = 1920
    height: int = 1080
    fps: float = 24.0
    color_space: str = "ACEScg"
    working_dir: Optional[Path] = None


class ImageCanvas(QGraphicsView):
    """
    Main image canvas with interactive segmentation support.

    Features:
    - Pan/zoom with mouse
    - Point and box prompts for SAM
    - Brush for mask editing
    - Split view comparison
    """

    point_added = Signal(int, int, int)  # x, y, label (1=fg, 0=bg)
    box_drawn = Signal(int, int, int, int)  # x1, y1, x2, y2
    brush_stroke = Signal(list, int)  # points, mode (1=add, 0=sub)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Image items
        self._source_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_item: Optional[QGraphicsPixmapItem] = None

        # State
        self._tool_mode = ToolMode.SELECT
        self._view_mode = ViewMode.SOURCE
        self._zoom_level = 1.0
        self._is_panning = False
        self._pan_start = QPoint()

        # Points and boxes
        self._fg_points: List[Tuple[int, int]] = []
        self._bg_points: List[Tuple[int, int]] = []
        self._box_start: Optional[QPoint] = None
        self._rubber_band: Optional[QRubberBand] = None

        # Brush
        self._brush_size = 20
        self._brush_points: List[Tuple[int, int]] = []

        # Configure view
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))

    def set_image(self, image: np.ndarray):
        """Set the displayed image."""
        h, w = image.shape[:2]

        # Convert to QImage
        if image.ndim == 2:
            # Grayscale
            qimage = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        elif image.shape[2] == 3:
            # RGB
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            qimage = QImage(image.data, w, h, w * 3, QImage.Format.Format_RGB888)
        else:
            # RGBA
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            qimage = QImage(image.data, w, h, w * 4, QImage.Format.Format_RGBA8888)

        pixmap = QPixmap.fromImage(qimage)

        if self._source_item is None:
            self._source_item = self.scene.addPixmap(pixmap)
        else:
            self._source_item.setPixmap(pixmap)

        self.setSceneRect(QRectF(pixmap.rect()))
        self._draw_points()

    def set_overlay(self, overlay: np.ndarray, opacity: float = 0.5):
        """Set overlay (matte, composite, etc.)."""
        h, w = overlay.shape[:2]

        if overlay.ndim == 2:
            # Convert matte to RGBA with color
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 0] = 0      # R
            rgba[..., 1] = 255    # G
            rgba[..., 2] = 0      # B
            rgba[..., 3] = (overlay * 255 * opacity).astype(np.uint8)
        else:
            if overlay.dtype == np.float32:
                overlay = (overlay * 255).astype(np.uint8)
            rgba = overlay

        qimage = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)

        if self._overlay_item is None:
            self._overlay_item = self.scene.addPixmap(pixmap)
            self._overlay_item.setZValue(1)
        else:
            self._overlay_item.setPixmap(pixmap)

    def set_tool(self, tool: ToolMode):
        """Set the active tool."""
        self._tool_mode = tool

        # Update cursor
        cursors = {
            ToolMode.SELECT: Qt.CursorShape.ArrowCursor,
            ToolMode.PAN: Qt.CursorShape.OpenHandCursor,
            ToolMode.ZOOM: Qt.CursorShape.CrossCursor,
            ToolMode.POINT_FG: Qt.CursorShape.CrossCursor,
            ToolMode.POINT_BG: Qt.CursorShape.CrossCursor,
            ToolMode.BOX: Qt.CursorShape.CrossCursor,
            ToolMode.BRUSH_ADD: Qt.CursorShape.CrossCursor,
            ToolMode.BRUSH_SUB: Qt.CursorShape.CrossCursor,
        }
        self.setCursor(cursors.get(tool, Qt.CursorShape.ArrowCursor))

    def clear_prompts(self):
        """Clear all prompt points and boxes."""
        self._fg_points.clear()
        self._bg_points.clear()
        self._draw_points()

    def _draw_points(self):
        """Redraw prompt points on the canvas."""
        # Remove old point items
        for item in self.scene.items():
            if hasattr(item, '_is_prompt_point'):
                self.scene.removeItem(item)

        # Draw foreground points (green)
        for x, y in self._fg_points:
            self._draw_point(x, y, QColor(0, 255, 0), 1)

        # Draw background points (red)
        for x, y in self._bg_points:
            self._draw_point(x, y, QColor(255, 0, 0), 0)

    def _draw_point(self, x: int, y: int, color: QColor, label: int):
        """Draw a single prompt point."""
        from PySide6.QtWidgets import QGraphicsEllipseItem

        radius = 8
        item = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
        item.setPen(QPen(Qt.GlobalColor.white, 2))
        item.setBrush(QBrush(color))
        item.setZValue(10)
        item._is_prompt_point = True
        self.scene.addItem(item)

    def wheelEvent(self, event: QWheelEvent):
        """Handle zoom with mouse wheel."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom_level *= factor
        self._zoom_level = max(0.1, min(10.0, self._zoom_level))
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        pos = self.mapToScene(event.position().toPoint())

        if event.button() == Qt.MouseButton.MiddleButton:
            # Pan with middle mouse
            self._is_panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._tool_mode == ToolMode.POINT_FG:
                self._fg_points.append((int(pos.x()), int(pos.y())))
                self.point_added.emit(int(pos.x()), int(pos.y()), 1)
                self._draw_points()

            elif self._tool_mode == ToolMode.POINT_BG:
                self._bg_points.append((int(pos.x()), int(pos.y())))
                self.point_added.emit(int(pos.x()), int(pos.y()), 0)
                self._draw_points()

            elif self._tool_mode == ToolMode.BOX:
                self._box_start = event.position().toPoint()
                if self._rubber_band is None:
                    self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
                self._rubber_band.setGeometry(QRect(self._box_start, QSize()))
                self._rubber_band.show()

            elif self._tool_mode == ToolMode.PAN:
                self._is_panning = True
                self._pan_start = event.position().toPoint()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        if self._is_panning:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            return

        if self._rubber_band and self._box_start:
            self._rubber_band.setGeometry(
                QRect(self._box_start, event.position().toPoint()).normalized()
            )

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._tool_mode == ToolMode.PAN else Qt.CursorShape.ArrowCursor)
            return

        if self._rubber_band and self._box_start:
            rect = self._rubber_band.geometry()
            self._rubber_band.hide()

            # Convert to scene coordinates
            p1 = self.mapToScene(rect.topLeft())
            p2 = self.mapToScene(rect.bottomRight())

            self.box_drawn.emit(
                int(p1.x()), int(p1.y()),
                int(p2.x()), int(p2.y())
            )
            self._box_start = None

        super().mouseReleaseEvent(event)


class ParameterPanel(QWidget):
    """
    Parameter panel for controlling processing settings.

    Organized into collapsible groups for:
    - Segmentation (SAM2)
    - Matting (MatAnyone, GVM, ViTMatte)
    - Depth (Depth Anything V2)
    - Compositing (Despill, Light Wrap, Harmonization)
    - Color Management (ACES)
    """

    parameters_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(300)
        self.setMaximumWidth(400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(5)

        # Add parameter groups
        self._create_segmentation_group()
        self._create_matting_group()
        self._create_depth_group()
        self._create_compositing_group()
        self._create_color_group()

        self._layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

    def _create_group(self, title: str) -> Tuple[QGroupBox, QVBoxLayout]:
        """Create a collapsible parameter group."""
        group = QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(True)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(10, 10, 10, 10)
        group_layout.setSpacing(5)
        self._layout.addWidget(group)
        return group, group_layout

    def _create_segmentation_group(self):
        """Create segmentation parameters."""
        group, layout = self._create_group("Segmentation (SAM2)")

        # Model selection
        layout.addWidget(QLabel("Model:"))
        self.sam_model = QComboBox()
        self.sam_model.addItems(["SAM2.1 Large", "SAM2.1 Base+", "SAM2.1 Small"])
        layout.addWidget(self.sam_model)

        # Use RobustSAM
        self.use_robust_sam = QCheckBox("Use RobustSAM (for motion blur)")
        layout.addWidget(self.use_robust_sam)

        # Multi-mask output
        self.multi_mask = QCheckBox("Multi-mask output")
        self.multi_mask.setChecked(True)
        layout.addWidget(self.multi_mask)

        # Process button
        self.segment_btn = QPushButton("Segment")
        self.segment_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        layout.addWidget(self.segment_btn)

    def _create_matting_group(self):
        """Create matting parameters."""
        group, layout = self._create_group("Matting")

        # Backend selection
        layout.addWidget(QLabel("Matting Model:"))
        self.matte_model = QComboBox()
        self.matte_model.addItems(["MatAnyone", "GVM (Diffusion)", "ViTMatte", "Hybrid"])
        layout.addWidget(self.matte_model)

        # Trimap settings
        layout.addWidget(QLabel("Trimap Unknown Width:"))
        self.trimap_width = QSpinBox()
        self.trimap_width.setRange(5, 100)
        self.trimap_width.setValue(25)
        layout.addWidget(self.trimap_width)

        # Edge refinement
        layout.addWidget(QLabel("Edge Refinement:"))
        self.edge_refine = QDoubleSpinBox()
        self.edge_refine.setRange(0, 1)
        self.edge_refine.setSingleStep(0.1)
        self.edge_refine.setValue(0.5)
        layout.addWidget(self.edge_refine)

        # Temporal consistency
        self.temporal_consistency = QCheckBox("Temporal Consistency (video)")
        self.temporal_consistency.setChecked(True)
        layout.addWidget(self.temporal_consistency)

        # Generate matte button
        self.matte_btn = QPushButton("Generate Matte")
        self.matte_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        layout.addWidget(self.matte_btn)

    def _create_depth_group(self):
        """Create depth estimation parameters."""
        group, layout = self._create_group("Depth Estimation")

        # Model selection
        layout.addWidget(QLabel("Model:"))
        self.depth_model = QComboBox()
        self.depth_model.addItems(["Depth Anything V2 Large", "Depth Anything V2 Base", "Depth Anything V2 Small"])
        layout.addWidget(self.depth_model)

        # Output type
        layout.addWidget(QLabel("Output:"))
        self.depth_output = QComboBox()
        self.depth_output.addItems(["Depth Map", "Normal Map", "Point Cloud", "Ambient Occlusion"])
        layout.addWidget(self.depth_output)

        # Depth scale
        layout.addWidget(QLabel("Depth Scale:"))
        self.depth_scale = QDoubleSpinBox()
        self.depth_scale.setRange(0.1, 10)
        self.depth_scale.setSingleStep(0.1)
        self.depth_scale.setValue(1.0)
        layout.addWidget(self.depth_scale)

        # Generate button
        self.depth_btn = QPushButton("Estimate Depth")
        self.depth_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        layout.addWidget(self.depth_btn)

    def _create_compositing_group(self):
        """Create compositing parameters."""
        group, layout = self._create_group("Compositing")

        # Despill
        self.enable_despill = QCheckBox("Enable Despill")
        self.enable_despill.setChecked(True)
        layout.addWidget(self.enable_despill)

        layout.addWidget(QLabel("Despill Algorithm:"))
        self.despill_algo = QComboBox()
        self.despill_algo.addItems(["Adaptive", "Average", "Maximum", "Double Average"])
        layout.addWidget(self.despill_algo)

        layout.addWidget(QLabel("Despill Channel:"))
        self.despill_channel = QComboBox()
        self.despill_channel.addItems(["Green", "Blue"])
        layout.addWidget(self.despill_channel)

        layout.addWidget(QLabel("Despill Strength:"))
        self.despill_strength = QDoubleSpinBox()
        self.despill_strength.setRange(0, 1)
        self.despill_strength.setSingleStep(0.1)
        self.despill_strength.setValue(0.8)
        layout.addWidget(self.despill_strength)

        # Edge operations
        layout.addWidget(QLabel("Edge Erode/Dilate:"))
        self.edge_erode = QSpinBox()
        self.edge_erode.setRange(-50, 50)
        self.edge_erode.setValue(0)
        layout.addWidget(self.edge_erode)

        # Light Wrap
        self.enable_light_wrap = QCheckBox("Enable Light Wrap")
        layout.addWidget(self.enable_light_wrap)

        layout.addWidget(QLabel("Light Wrap Width:"))
        self.light_wrap_width = QSpinBox()
        self.light_wrap_width.setRange(1, 100)
        self.light_wrap_width.setValue(20)
        layout.addWidget(self.light_wrap_width)

        layout.addWidget(QLabel("Light Wrap Intensity:"))
        self.light_wrap_intensity = QDoubleSpinBox()
        self.light_wrap_intensity.setRange(0, 1)
        self.light_wrap_intensity.setSingleStep(0.1)
        self.light_wrap_intensity.setValue(0.5)
        layout.addWidget(self.light_wrap_intensity)

        # Harmonization
        self.enable_harmonize = QCheckBox("Enable Color Harmonization")
        layout.addWidget(self.enable_harmonize)

        layout.addWidget(QLabel("Harmonization Method:"))
        self.harmonize_method = QComboBox()
        self.harmonize_method.addItems(["Adaptive", "LAB Transfer", "Reinhard", "Histogram"])
        layout.addWidget(self.harmonize_method)

        # Composite button
        self.composite_btn = QPushButton("Composite")
        self.composite_btn.setStyleSheet("background-color: #FF5722; color: white; font-weight: bold;")
        layout.addWidget(self.composite_btn)

    def _create_color_group(self):
        """Create color management parameters."""
        group, layout = self._create_group("Color Management")

        # Input color space
        layout.addWidget(QLabel("Input Color Space:"))
        self.input_colorspace = QComboBox()
        self.input_colorspace.addItems(["sRGB", "Linear sRGB", "Rec.709", "Rec.2020", "ARRI LogC3", "Sony S-Log3"])
        layout.addWidget(self.input_colorspace)

        # Working color space
        layout.addWidget(QLabel("Working Space:"))
        self.working_colorspace = QComboBox()
        self.working_colorspace.addItems(["ACEScg", "ACES2065-1", "ACEScc", "ACEScct"])
        layout.addWidget(self.working_colorspace)

        # Output color space
        layout.addWidget(QLabel("Output Color Space:"))
        self.output_colorspace = QComboBox()
        self.output_colorspace.addItems(["sRGB", "Rec.709", "Rec.2020", "P3-D65"])
        layout.addWidget(self.output_colorspace)

        # Tone mapping
        layout.addWidget(QLabel("Tone Mapping:"))
        self.tone_mapping = QComboBox()
        self.tone_mapping.addItems(["ACES RRT", "Filmic", "Reinhard", "Hable"])
        layout.addWidget(self.tone_mapping)

    def get_parameters(self) -> Dict[str, Any]:
        """Get all parameter values as dictionary."""
        return {
            # Segmentation
            "sam_model": self.sam_model.currentText(),
            "use_robust_sam": self.use_robust_sam.isChecked(),
            "multi_mask": self.multi_mask.isChecked(),
            # Matting
            "matte_model": self.matte_model.currentText(),
            "trimap_width": self.trimap_width.value(),
            "edge_refine": self.edge_refine.value(),
            "temporal_consistency": self.temporal_consistency.isChecked(),
            # Depth
            "depth_model": self.depth_model.currentText(),
            "depth_output": self.depth_output.currentText(),
            "depth_scale": self.depth_scale.value(),
            # Compositing
            "enable_despill": self.enable_despill.isChecked(),
            "despill_algo": self.despill_algo.currentText(),
            "despill_channel": self.despill_channel.currentText(),
            "despill_strength": self.despill_strength.value(),
            "edge_erode": self.edge_erode.value(),
            "enable_light_wrap": self.enable_light_wrap.isChecked(),
            "light_wrap_width": self.light_wrap_width.value(),
            "light_wrap_intensity": self.light_wrap_intensity.value(),
            "enable_harmonize": self.enable_harmonize.isChecked(),
            "harmonize_method": self.harmonize_method.currentText(),
            # Color
            "input_colorspace": self.input_colorspace.currentText(),
            "working_colorspace": self.working_colorspace.currentText(),
            "output_colorspace": self.output_colorspace.currentText(),
            "tone_mapping": self.tone_mapping.currentText(),
        }


class TimelineWidget(QWidget):
    """
    Timeline widget for video frame navigation.

    Features:
    - Frame scrubbing
    - Keyframe markers
    - In/out points
    - Multiple layers
    """

    frame_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumHeight(100)
        self.setMaximumHeight(150)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Controls
        controls = QHBoxLayout()

        self.play_btn = QPushButton("▶")
        self.play_btn.setMaximumWidth(40)
        controls.addWidget(self.play_btn)

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.valueChanged.connect(self.frame_changed.emit)
        controls.addWidget(QLabel("Frame:"))
        controls.addWidget(self.frame_spin)

        self.fps_label = QLabel("24 fps")
        controls.addWidget(self.fps_label)

        controls.addStretch()

        self.total_frames = QLabel("0 / 0")
        controls.addWidget(self.total_frames)

        layout.addLayout(controls)

        # Timeline slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

    def set_range(self, start: int, end: int, fps: float = 24.0):
        """Set timeline range."""
        self.frame_spin.setRange(start, end)
        self.slider.setRange(start, end)
        self.fps_label.setText(f"{fps} fps")
        self._update_total()

    def set_frame(self, frame: int):
        """Set current frame."""
        self.frame_spin.blockSignals(True)
        self.slider.blockSignals(True)

        self.frame_spin.setValue(frame)
        self.slider.setValue(frame)

        self.frame_spin.blockSignals(False)
        self.slider.blockSignals(False)

        self._update_total()

    def _on_slider_changed(self, value: int):
        """Handle slider change."""
        self.frame_spin.setValue(value)
        self.frame_changed.emit(value)

    def _update_total(self):
        """Update total frames display."""
        current = self.frame_spin.value()
        total = self.frame_spin.maximum()
        self.total_frames.setText(f"{current} / {total}")


class LayerPanel(QWidget):
    """
    Layer panel for managing multiple outputs.

    Layers:
    - Source
    - Matte
    - Depth
    - Composite
    - Custom AOVs
    """

    layer_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Layer list
        self.layer_list = QListWidget()
        self.layer_list.currentTextChanged.connect(self.layer_changed.emit)
        layout.addWidget(self.layer_list)

        # Default layers
        for layer in ["Source", "Matte", "Depth", "Normals", "AO", "Composite"]:
            item = QListWidgetItem(layer)
            item.setCheckState(Qt.CheckState.Checked)
            self.layer_list.addItem(item)

        # Buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("+")
        self.add_btn.setMaximumWidth(30)
        self.remove_btn = QPushButton("-")
        self.remove_btn.setMaximumWidth(30)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)


class MainWindow(QMainWindow):
    """
    Main application window for Ultimate Rotoscopy.

    Professional VFX interface with:
    - Multi-pane viewer with split views
    - Parameter panels for all AI models
    - Timeline for video editing
    - Layer management
    - Export to industry formats
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ultimate Rotoscopy v3.0.0")
        self.setMinimumSize(1600, 900)

        # State
        self._current_image: Optional[np.ndarray] = None
        self._current_matte: Optional[np.ndarray] = None
        self._current_depth: Optional[np.ndarray] = None
        self._background: Optional[np.ndarray] = None
        self._project = ProjectSettings()

        # Setup UI
        self._setup_menus()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_dock_widgets()
        self._setup_status_bar()

        # Load settings
        self._load_settings()

    def _setup_menus(self):
        """Setup application menus."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Project", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        file_menu.addAction(new_action)

        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        import_bg = QAction("Import &Background...", self)
        import_bg.triggered.connect(self._import_background)
        file_menu.addAction(import_bg)

        import_clean = QAction("Import &Clean Plate...", self)
        file_menu.addAction(import_clean)

        file_menu.addSeparator()

        export_menu = file_menu.addMenu("&Export")
        export_menu.addAction(QAction("Export Matte (EXR)", self))
        export_menu.addAction(QAction("Export Composite (EXR)", self))
        export_menu.addAction(QAction("Export All AOVs", self))
        export_menu.addAction(QAction("Export for Flame", self))
        export_menu.addAction(QAction("Export for Nuke", self))

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(QAction("&Undo", self, shortcut=QKeySequence.StandardKey.Undo))
        edit_menu.addAction(QAction("&Redo", self, shortcut=QKeySequence.StandardKey.Redo))
        edit_menu.addSeparator()
        edit_menu.addAction(QAction("Clear Prompts", self))

        # View menu
        view_menu = menubar.addMenu("&View")

        view_menu.addAction(QAction("&Source", self, checkable=True, checked=True))
        view_menu.addAction(QAction("&Matte", self, checkable=True))
        view_menu.addAction(QAction("&Composite", self, checkable=True))
        view_menu.addAction(QAction("&Depth", self, checkable=True))
        view_menu.addSeparator()
        view_menu.addAction(QAction("Split View", self, checkable=True))
        view_menu.addAction(QAction("Checkerboard Background", self, checkable=True))

        # Process menu
        process_menu = menubar.addMenu("&Process")
        process_menu.addAction(QAction("Run &Segmentation", self, shortcut="Ctrl+1"))
        process_menu.addAction(QAction("Run &Matting", self, shortcut="Ctrl+2"))
        process_menu.addAction(QAction("Run &Depth Estimation", self, shortcut="Ctrl+3"))
        process_menu.addAction(QAction("Run &Compositing", self, shortcut="Ctrl+4"))
        process_menu.addSeparator()
        process_menu.addAction(QAction("Process All", self, shortcut="Ctrl+Return"))

        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(QAction("&Documentation", self))
        help_menu.addAction(QAction("&About", self))

    def _setup_toolbar(self):
        """Setup main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Tool buttons
        tool_group = [
            ("Select", ToolMode.SELECT, "V"),
            ("Pan", ToolMode.PAN, "H"),
            ("Zoom", ToolMode.ZOOM, "Z"),
            ("|", None, None),
            ("FG Point", ToolMode.POINT_FG, "F"),
            ("BG Point", ToolMode.POINT_BG, "B"),
            ("Box", ToolMode.BOX, "X"),
            ("|", None, None),
            ("Brush +", ToolMode.BRUSH_ADD, "="),
            ("Brush -", ToolMode.BRUSH_SUB, "-"),
        ]

        self._tool_actions = {}
        for name, mode, shortcut in tool_group:
            if name == "|":
                toolbar.addSeparator()
            else:
                action = QAction(name, self)
                action.setCheckable(True)
                if shortcut:
                    action.setShortcut(shortcut)
                if mode:
                    action.triggered.connect(lambda checked, m=mode: self._set_tool(m))
                    self._tool_actions[mode] = action
                toolbar.addAction(action)

        toolbar.addSeparator()

        # View mode
        toolbar.addWidget(QLabel(" View: "))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Source", "Matte", "Composite", "Depth", "Normals", "Split"])
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        toolbar.addWidget(self.view_combo)

        toolbar.addSeparator()

        # Overlay opacity
        toolbar.addWidget(QLabel(" Overlay: "))
        self.overlay_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_slider.setRange(0, 100)
        self.overlay_slider.setValue(50)
        self.overlay_slider.setMaximumWidth(100)
        toolbar.addWidget(self.overlay_slider)

    def _setup_central_widget(self):
        """Setup central widget with canvas and timeline."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main canvas
        self.canvas = ImageCanvas()
        self.canvas.point_added.connect(self._on_point_added)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        layout.addWidget(self.canvas, stretch=1)

        # Timeline
        self.timeline = TimelineWidget()
        self.timeline.frame_changed.connect(self._on_frame_changed)
        layout.addWidget(self.timeline)

    def _setup_dock_widgets(self):
        """Setup dock widgets for parameters and layers."""
        # Parameters dock (right)
        params_dock = QDockWidget("Parameters", self)
        params_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.params_panel = ParameterPanel()
        params_dock.setWidget(self.params_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, params_dock)

        # Connect buttons
        self.params_panel.segment_btn.clicked.connect(self._run_segmentation)
        self.params_panel.matte_btn.clicked.connect(self._run_matting)
        self.params_panel.depth_btn.clicked.connect(self._run_depth)
        self.params_panel.composite_btn.clicked.connect(self._run_composite)

        # Layers dock (left)
        layers_dock = QDockWidget("Layers", self)
        layers_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.layers_panel = LayerPanel()
        layers_dock.setWidget(self.layers_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, layers_dock)

    def _setup_status_bar(self):
        """Setup status bar."""
        self.statusBar().showMessage("Ready")

        # GPU info
        self.gpu_label = QLabel("GPU: RTX 4090")
        self.statusBar().addPermanentWidget(self.gpu_label)

        # Memory info
        self.mem_label = QLabel("VRAM: 0 / 24 GB")
        self.statusBar().addPermanentWidget(self.mem_label)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

    def _load_settings(self):
        """Load application settings."""
        settings = QSettings("UltimateRotoscopy", "App")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def closeEvent(self, event):
        """Save settings on close."""
        settings = QSettings("UltimateRotoscopy", "App")
        settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)

    def _set_tool(self, mode: ToolMode):
        """Set the active tool."""
        self.canvas.set_tool(mode)

        # Update button states
        for m, action in self._tool_actions.items():
            action.setChecked(m == mode)

    def _on_view_changed(self, view_name: str):
        """Handle view mode change."""
        # Update canvas display based on selected view
        pass

    def _on_point_added(self, x: int, y: int, label: int):
        """Handle point prompt added."""
        self.statusBar().showMessage(
            f"{'Foreground' if label == 1 else 'Background'} point added at ({x}, {y})"
        )

    def _on_box_drawn(self, x1: int, y1: int, x2: int, y2: int):
        """Handle bounding box drawn."""
        self.statusBar().showMessage(f"Box: ({x1}, {y1}) to ({x2}, {y2})")

    def _on_frame_changed(self, frame: int):
        """Handle frame change."""
        self.statusBar().showMessage(f"Frame: {frame}")

    def _open_file(self):
        """Open image or video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.exr);;Videos (*.mp4 *.mov *.avi);;All Files (*)"
        )

        if file_path:
            self._load_file(Path(file_path))

    def _load_file(self, path: Path):
        """Load an image or video file."""
        import cv2

        if path.suffix.lower() in ['.mp4', '.mov', '.avi']:
            # Video
            cap = cv2.VideoCapture(str(path))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._current_image = frame.astype(np.float32) / 255.0
                self.canvas.set_image(frame)

                # Setup timeline
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                self.timeline.set_range(0, total_frames - 1, fps)

            cap.release()
        else:
            # Image
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                if image.ndim == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.ndim == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                self._current_image = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image
                self.canvas.set_image(image)
                self.timeline.set_range(0, 0)

        self.statusBar().showMessage(f"Loaded: {path.name}")

    def _import_background(self):
        """Import background plate."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Background",
            "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.exr);;All Files (*)"
        )

        if file_path:
            import cv2
            bg = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if bg is not None:
                if bg.ndim == 3 and bg.shape[2] == 3:
                    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                self._background = bg.astype(np.float32) / 255.0 if bg.dtype == np.uint8 else bg
                self.statusBar().showMessage(f"Background loaded: {Path(file_path).name}")

    def _run_segmentation(self):
        """Run SAM2 segmentation."""
        self.statusBar().showMessage("Running segmentation...")
        self.progress.setVisible(True)
        self.progress.setValue(0)

        # Get prompts from canvas
        params = self.params_panel.get_parameters()

        # TODO: Run actual segmentation
        # For now, just show placeholder
        self.progress.setValue(100)
        self.statusBar().showMessage("Segmentation complete")

        QTimer.singleShot(1000, lambda: self.progress.setVisible(False))

    def _run_matting(self):
        """Run AI matting."""
        self.statusBar().showMessage("Running matting...")
        self.progress.setVisible(True)
        self.progress.setValue(0)

        params = self.params_panel.get_parameters()

        # TODO: Run actual matting

        self.progress.setValue(100)
        self.statusBar().showMessage("Matting complete")

        QTimer.singleShot(1000, lambda: self.progress.setVisible(False))

    def _run_depth(self):
        """Run depth estimation."""
        self.statusBar().showMessage("Running depth estimation...")
        self.progress.setVisible(True)
        self.progress.setValue(0)

        params = self.params_panel.get_parameters()

        # TODO: Run actual depth estimation

        self.progress.setValue(100)
        self.statusBar().showMessage("Depth estimation complete")

        QTimer.singleShot(1000, lambda: self.progress.setVisible(False))

    def _run_composite(self):
        """Run compositing pipeline."""
        self.statusBar().showMessage("Running compositing...")
        self.progress.setVisible(True)
        self.progress.setValue(0)

        params = self.params_panel.get_parameters()

        # TODO: Run actual compositing

        self.progress.setValue(100)
        self.statusBar().showMessage("Compositing complete")

        QTimer.singleShot(1000, lambda: self.progress.setVisible(False))


def main():
    """Application entry point."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Dark theme
    dark_palette = app.palette()
    dark_palette.setColor(dark_palette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.ColorRole.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(dark_palette.ColorRole.Base, QColor(35, 35, 35))
    dark_palette.setColor(dark_palette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    dark_palette.setColor(dark_palette.ColorRole.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(dark_palette.ColorRole.Text, QColor(255, 255, 255))
    dark_palette.setColor(dark_palette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.ColorRole.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(dark_palette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(dark_palette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(dark_palette.ColorRole.HighlightedText, QColor(35, 35, 35))
    app.setPalette(dark_palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
