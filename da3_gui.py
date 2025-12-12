#!/usr/bin/env python3
"""
Depth Anything V3 GUI - Professional PySide6 Interface
=======================================================

Professional interface for Depth Anything V3 with:
- Interactive depth visualization
- Normal map display
- 3D point cloud preview
- Multiple colormap options
- Export controls (PLY, OBJ, EXR)
- Batch processing

Requirements:
    pip install PySide6 opencv-python numpy pillow

Usage:
    python da3_gui.py
    python da3_gui.py --image photo.jpg
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
from typing import Optional, List
from dataclasses import dataclass

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QComboBox, QSlider, QFileDialog,
        QToolBar, QStatusBar, QGroupBox, QCheckBox,
        QSpinBox, QDoubleSpinBox, QTextEdit, QTabWidget, QFrame,
        QProgressBar, QSplitter
    )
    from PySide6.QtCore import Qt, Signal, QThread, QTimer
    from PySide6.QtGui import QImage, QPixmap, QAction
except ImportError as e:
    print(f"Error: PySide6 not installed - {e}")
    print("Install: pip install PySide6")
    sys.exit(1)

# Import DA3 wrapper
try:
    from da3_complete import (
        DepthAnything3, DepthResult, DepthColormap, PointCloudFormat, CameraIntrinsics
    )
except ImportError:
    print("Error: da3_complete.py not found!")
    print("Make sure da3_complete.py is in the same directory.")
    sys.exit(1)


class ImageViewport(QLabel):
    """Image viewport for displaying images with overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #2b2b2b; border: 1px solid #555; }")

        self.image: Optional[np.ndarray] = None
        self.pixmap: Optional[QPixmap] = None

    def set_image(self, image: np.ndarray):
        """Set image to display (RGB numpy array)."""
        self.image = image.copy()
        self._update_display()

    def clear(self):
        """Clear the viewport."""
        self.image = None
        self.pixmap = None
        self.setText("No image loaded")

    def _update_display(self):
        """Update the displayed image."""
        if self.image is None:
            return

        # Convert to QPixmap
        if len(self.image.shape) == 2:
            # Grayscale
            h, w = self.image.shape
            bytes_per_line = w
            q_image = QImage(
                self.image.data, w, h, bytes_per_line,
                QImage.Format.Format_Grayscale8
            )
        else:
            # RGB
            h, w, c = self.image.shape
            bytes_per_line = 3 * w
            q_image = QImage(
                self.image.data, w, h, bytes_per_line,
                QImage.Format.Format_RGB888
            )

        self.pixmap = QPixmap.fromImage(q_image)

        # Scale to fit while maintaining aspect ratio
        scaled = self.pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        """Handle resize to rescale image."""
        super().resizeEvent(event)
        if self.pixmap:
            self._update_display()


class DA3Worker(QThread):
    """Worker thread for depth estimation."""

    result_ready = Signal(object)  # DepthResult
    error_occurred = Signal(str)
    progress_updated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.processor: Optional[DepthAnything3] = None
        self.task = None

    def initialize_processor(self, device="cuda", model_size="base"):
        """Initialize DA3 processor in background."""
        try:
            self.progress_updated.emit(f"Loading Depth Anything V3 ({model_size})...")
            self.processor = DepthAnything3(device=device, model_size=model_size)
            self.processor._load_model()
            self.progress_updated.emit("Depth Anything V3 ready")
        except Exception as e:
            self.error_occurred.emit(f"Failed to load DA3: {str(e)}")

    def process_image(self, image_path: Path, compute_normals: bool, compute_pointcloud: bool, fov: float):
        """Process image with depth estimation."""
        self.task = ("process", image_path, compute_normals, compute_pointcloud, fov)
        self.start()

    def run(self):
        """Execute processing task."""
        if self.processor is None:
            self.error_occurred.emit("DA3 not initialized")
            return

        try:
            task_type = self.task[0]

            if task_type == "process":
                _, image_path, compute_normals, compute_pointcloud, fov = self.task
                self.progress_updated.emit(f"Processing: {image_path.name}")

                result = self.processor.process(
                    image_path,
                    compute_normals=compute_normals,
                    compute_pointcloud=compute_pointcloud,
                    fov_degrees=fov,
                )

                self.result_ready.emit(result)
                self.progress_updated.emit("Processing complete")

        except Exception as e:
            self.error_occurred.emit(f"Processing failed: {str(e)}")


class DA3MainWindow(QMainWindow):
    """Main application window for Depth Anything V3 GUI."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Depth Anything V3 - Professional Depth Estimation")
        self.setMinimumSize(1400, 900)

        # Data
        self.current_image_path: Optional[Path] = None
        self.current_image: Optional[np.ndarray] = None
        self.current_result: Optional[DepthResult] = None

        # Worker thread
        self.worker = DA3Worker()
        self.worker.result_ready.connect(self._on_result_ready)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.progress_updated.connect(self._on_progress)

        # Setup UI
        self._setup_ui()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_statusbar()

        # Apply dark theme
        self._apply_dark_theme()

        # Initialize DA3 in background
        QTimer.singleShot(100, lambda: self.worker.initialize_processor("cuda", "base"))

    def _setup_ui(self):
        """Setup main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel: Controls
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Center: Viewports with tabs
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, stretch=4)

        # Right panel: Results and export
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

        self.load_batch_btn = QPushButton("Load Batch...")
        self.load_batch_btn.clicked.connect(self._load_batch)
        file_layout.addWidget(self.load_batch_btn)

        layout.addWidget(file_group)

        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout(model_group)

        model_layout.addWidget(QLabel("Model Size:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["small", "base", "large"])
        self.model_combo.setCurrentText("base")
        model_layout.addWidget(self.model_combo)

        model_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        model_layout.addWidget(self.device_combo)

        layout.addWidget(model_group)

        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout(options_group)

        self.normals_checkbox = QCheckBox("Compute Normals")
        self.normals_checkbox.setChecked(True)
        options_layout.addWidget(self.normals_checkbox)

        self.pointcloud_checkbox = QCheckBox("Generate Point Cloud")
        self.pointcloud_checkbox.setChecked(True)
        options_layout.addWidget(self.pointcloud_checkbox)

        options_layout.addWidget(QLabel("Estimated FOV (degrees):"))
        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(10, 180)
        self.fov_spin.setValue(60.0)
        self.fov_spin.setSingleStep(5.0)
        options_layout.addWidget(self.fov_spin)

        layout.addWidget(options_group)

        # Process button
        self.process_btn = QPushButton("Process Image")
        self.process_btn.clicked.connect(self._process_image)
        self.process_btn.setStyleSheet("QPushButton { background-color: #28a745; font-weight: bold; padding: 12px; }")
        layout.addWidget(self.process_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        return panel

    def _create_center_panel(self) -> QWidget:
        """Create center viewport panel with tabs."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tab widget for different views
        self.view_tabs = QTabWidget()

        # Original image tab
        self.original_viewport = ImageViewport()
        self.original_viewport.setText("Load an image to start")
        self.view_tabs.addTab(self.original_viewport, "Original")

        # Depth visualization tab
        self.depth_viewport = ImageViewport()
        self.depth_viewport.setText("Process image to see depth")
        self.view_tabs.addTab(self.depth_viewport, "Depth")

        # Normals tab
        self.normals_viewport = ImageViewport()
        self.normals_viewport.setText("Enable normals to see normal map")
        self.view_tabs.addTab(self.normals_viewport, "Normals")

        # Side by side comparison
        comparison_widget = QWidget()
        comparison_layout = QHBoxLayout(comparison_widget)
        self.compare_left = ImageViewport()
        self.compare_right = ImageViewport()
        comparison_layout.addWidget(self.compare_left)
        comparison_layout.addWidget(self.compare_right)
        self.view_tabs.addTab(comparison_widget, "Compare")

        layout.addWidget(self.view_tabs)

        # Visualization controls
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout(viz_group)

        viz_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["turbo", "viridis", "plasma", "inferno", "magma", "jet", "gray"])
        self.colormap_combo.currentTextChanged.connect(self._update_depth_visualization)
        viz_layout.addWidget(self.colormap_combo)

        viz_layout.addStretch()

        layout.addWidget(viz_group)

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create right results and export panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Results info
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(250)
        results_layout.addWidget(self.results_text)

        layout.addWidget(results_group)

        # Camera intrinsics
        intrinsics_group = QGroupBox("Camera Intrinsics")
        intrinsics_layout = QVBoxLayout(intrinsics_group)

        self.intrinsics_text = QTextEdit()
        self.intrinsics_text.setReadOnly(True)
        self.intrinsics_text.setMaximumHeight(120)
        intrinsics_layout.addWidget(self.intrinsics_text)

        layout.addWidget(intrinsics_group)

        # Export controls
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        self.export_depth_btn = QPushButton("Export Depth Map")
        self.export_depth_btn.clicked.connect(self._export_depth)
        export_layout.addWidget(self.export_depth_btn)

        self.export_normals_btn = QPushButton("Export Normals")
        self.export_normals_btn.clicked.connect(self._export_normals)
        export_layout.addWidget(self.export_normals_btn)

        self.export_pointcloud_btn = QPushButton("Export Point Cloud")
        self.export_pointcloud_btn.clicked.connect(self._export_pointcloud)
        export_layout.addWidget(self.export_pointcloud_btn)

        export_layout.addWidget(QLabel("Point Cloud Format:"))
        self.pointcloud_format_combo = QComboBox()
        self.pointcloud_format_combo.addItems(["PLY", "OBJ", "XYZ", "NPY"])
        export_layout.addWidget(self.pointcloud_format_combo)

        self.export_all_btn = QPushButton("Export All")
        self.export_all_btn.clicked.connect(self._export_all)
        self.export_all_btn.setStyleSheet("QPushButton { background-color: #17a2b8; font-weight: bold; }")
        export_layout.addWidget(self.export_all_btn)

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

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        show_original = QAction("Show Original", self)
        show_original.triggered.connect(lambda: self.view_tabs.setCurrentIndex(0))
        view_menu.addAction(show_original)

        show_depth = QAction("Show Depth", self)
        show_depth.triggered.connect(lambda: self.view_tabs.setCurrentIndex(1))
        view_menu.addAction(show_depth)

        show_normals = QAction("Show Normals", self)
        show_normals.triggered.connect(lambda: self.view_tabs.setCurrentIndex(2))
        view_menu.addAction(show_normals)

    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        toolbar.addAction("Open", self._load_image)
        toolbar.addAction("Process", self._process_image)
        toolbar.addSeparator()
        toolbar.addAction("Export All", self._export_all)

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
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox {
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
            QComboBox::drop-down {
                border: none;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0e639c;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #555555;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #0e639c;
                border: 1px solid #0e639c;
            }
        """)

    # Slots
    def _load_image(self):
        """Load image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image",
            "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_path:
            self.current_image_path = Path(file_path)
            image = cv2.imread(str(self.current_image_path))
            self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display original
            self.original_viewport.set_image(self.current_image)
            self.compare_left.set_image(self.current_image)

            # Switch to original tab
            self.view_tabs.setCurrentIndex(0)

            self.status_bar.showMessage(f"Loaded: {self.current_image_path.name}")

            # Clear previous results
            self.current_result = None
            self.depth_viewport.clear()
            self.normals_viewport.clear()
            self.results_text.clear()
            self.intrinsics_text.clear()

    def _load_batch(self):
        """Load multiple images for batch processing."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Images",
            "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_paths:
            self.status_bar.showMessage(f"Batch: {len(file_paths)} images selected")
            # TODO: Implement batch processing UI

    def _process_image(self):
        """Process current image with depth estimation."""
        if self.current_image_path is None:
            self.status_bar.showMessage("Error: Load an image first")
            return

        # Disable button during processing
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        # Start processing
        self.worker.process_image(
            self.current_image_path,
            compute_normals=self.normals_checkbox.isChecked(),
            compute_pointcloud=self.pointcloud_checkbox.isChecked(),
            fov=self.fov_spin.value()
        )

    def _on_result_ready(self, result: DepthResult):
        """Handle depth estimation result."""
        self.current_result = result

        # Update depth visualization
        self._update_depth_visualization()

        # Update normals visualization
        if result.normals is not None:
            normals_vis = result.get_normals_colormap()
            self.normals_viewport.set_image(normals_vis)

        # Update comparison views
        if self.current_image is not None:
            self.compare_left.set_image(self.current_image)
        depth_vis = result.get_depth_colormap(DepthColormap(self.colormap_combo.currentText()))
        self.compare_right.set_image(depth_vis)

        # Update results text
        info = f"""Depth Estimation Complete
{'='*30}
Depth Range: {result.depth_min:.3f} - {result.depth_max:.3f}
Processing Time: {result.processing_time_ms:.0f}ms

Has Normals: {result.normals is not None}
Has Point Cloud: {result.point_cloud is not None}
"""
        if result.point_cloud is not None:
            info += f"Point Count: {len(result.point_cloud):,}\n"

        self.results_text.setText(info)

        # Update intrinsics
        if result.intrinsics is not None:
            intr = result.intrinsics
            intr_text = f"""Focal Length: fx={intr.fx:.1f}, fy={intr.fy:.1f}
Principal Point: cx={intr.cx:.1f}, cy={intr.cy:.1f}
Image Size: {intr.width}x{intr.height}"""
            self.intrinsics_text.setText(intr_text)

        # Switch to depth tab
        self.view_tabs.setCurrentIndex(1)

        # Re-enable UI
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.status_bar.showMessage(f"Depth estimation complete - {result.processing_time_ms:.0f}ms")

    def _on_error(self, error_msg: str):
        """Handle error."""
        self.status_bar.showMessage(f"Error: {error_msg}")
        self.results_text.setText(f"ERROR:\n{error_msg}")
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _on_progress(self, message: str):
        """Handle progress update."""
        self.status_bar.showMessage(message)

    def _update_depth_visualization(self):
        """Update depth visualization with selected colormap."""
        if self.current_result is None:
            return

        colormap = DepthColormap(self.colormap_combo.currentText())
        depth_vis = self.current_result.get_depth_colormap(colormap)
        self.depth_viewport.set_image(depth_vis)
        self.compare_right.set_image(depth_vis)

    def _export_depth(self):
        """Export depth map."""
        if self.current_result is None:
            self.status_bar.showMessage("Error: No depth data to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Depth Map",
            f"{self.current_image_path.stem}_depth.png",
            "PNG (*.png);;TIFF (*.tiff);;NumPy (*.npy);;EXR (*.exr)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_path:
            path = Path(file_path)
            if path.suffix == ".png":
                self.current_result.save_depth_visualization(path)
            elif path.suffix == ".npy":
                self.current_result.save_depth(path, format="npy")
            else:
                self.current_result.save_depth(path, format=path.suffix[1:])

            self.status_bar.showMessage(f"Depth saved: {path.name}")

    def _export_normals(self):
        """Export normal map."""
        if self.current_result is None or self.current_result.normals is None:
            self.status_bar.showMessage("Error: No normals to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Normal Map",
            f"{self.current_image_path.stem}_normals.png",
            "PNG (*.png);;NumPy (*.npy);;EXR (*.exr)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_path:
            path = Path(file_path)
            self.current_result.save_normals(path, format=path.suffix[1:])
            self.status_bar.showMessage(f"Normals saved: {path.name}")

    def _export_pointcloud(self):
        """Export point cloud."""
        if self.current_result is None or self.current_result.point_cloud is None:
            self.status_bar.showMessage("Error: No point cloud to export")
            return

        format_map = {"PLY": "ply", "OBJ": "obj", "XYZ": "xyz", "NPY": "npy"}
        selected_format = format_map[self.pointcloud_format_combo.currentText()]

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Point Cloud",
            f"{self.current_image_path.stem}_pointcloud.{selected_format}",
            f"{selected_format.upper()} (*.{selected_format})",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_path:
            path = Path(file_path)
            self.current_result.save_point_cloud(path, format=PointCloudFormat(selected_format))
            self.status_bar.showMessage(f"Point cloud saved: {path.name}")

    def _export_all(self):
        """Export all outputs."""
        if self.current_result is None:
            self.status_bar.showMessage("Error: No results to export")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if output_dir:
            self.current_result.save_all(output_dir, self.current_image_path.stem)
            self.status_bar.showMessage(f"All outputs saved to: {output_dir}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Depth Anything V3 GUI")
    parser.add_argument("--image", type=str, help="Load image directly")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Depth Anything V3")

    window = DA3MainWindow()

    # Load image directly if provided
    if args.image:
        image_path = Path(args.image)
        if image_path.exists():
            window.current_image_path = image_path
            image = cv2.imread(str(image_path))
            if image is not None:
                window.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                window.original_viewport.set_image(window.current_image)
                window.compare_left.set_image(window.current_image)
                window.status_bar.showMessage(f"Loaded: {image_path.name}")

    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
