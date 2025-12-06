"""
Modern GUI - Tabs Part 2
=========================

Matting, Composite, and Export tabs.
"""

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import numpy as np

try:
    from ultimate_rotoscopy.gui.backend import ProcessingRequest, ProcessingStage
    from ultimate_rotoscopy.matting.professional_matting import ProfessionalMatting, ProfessionalMattingConfig
    BACKEND_AVAILABLE = True
except:
    BACKEND_AVAILABLE = False


class MattingTab(QWidget):
    """
    Tab 3: Professional Matting

    Alpha refinement with core/edge/hair split.
    """

    matte_generated = Signal(object)  # alpha

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_image = None
        self.current_mask = None
        self.current_alpha = None
        self.professional_matting = None
        self.backend = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("<h2>Professional Matting</h2>")
        layout.addWidget(header)

        # Model selection
        model_group = QGroupBox("Matting Model")
        model_layout = QVBoxLayout()

        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "Professional Matting (Alpha Split + Motion Blur)",
            "MatAnyone",
            "ViTMatte",
            "Simple (Morphological)"
        ])
        model_layout.addWidget(self.combo_model)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Alpha Split options
        split_group = QGroupBox("Alpha Split (Core/Edge/Hair)")
        split_layout = QVBoxLayout()

        self.check_enable_split = QCheckBox("Enable Alpha Split")
        self.check_enable_split.setChecked(True)
        split_layout.addWidget(self.check_enable_split)

        # Core threshold
        core_layout = QHBoxLayout()
        core_layout.addWidget(QLabel("Core Threshold:"))
        self.spin_core_threshold = QDoubleSpinBox()
        self.spin_core_threshold.setRange(0.0, 1.0)
        self.spin_core_threshold.setSingleStep(0.05)
        self.spin_core_threshold.setValue(0.15)
        core_layout.addWidget(self.spin_core_threshold)
        split_layout.addLayout(core_layout)

        # Edge width
        edge_layout = QHBoxLayout()
        edge_layout.addWidget(QLabel("Edge Band Width:"))
        self.spin_edge_width = QSpinBox()
        self.spin_edge_width.setRange(1, 50)
        self.spin_edge_width.setValue(10)
        edge_layout.addWidget(self.spin_edge_width)
        split_layout.addLayout(edge_layout)

        split_group.setLayout(split_layout)
        layout.addWidget(split_group)

        # Motion Blur options
        blur_group = QGroupBox("Motion Blur Awareness")
        blur_layout = QVBoxLayout()

        self.check_motion_blur = QCheckBox("Enable Motion Blur Detection")
        self.check_motion_blur.setChecked(True)
        blur_layout.addWidget(self.check_motion_blur)

        blur_group.setLayout(blur_layout)
        layout.addWidget(blur_group)

        # Generate button
        self.btn_generate = QPushButton("✨ Generate Professional Matte")
        self.btn_generate.setMinimumHeight(50)
        self.btn_generate.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_generate.clicked.connect(self._generate_matte)
        layout.addWidget(self.btn_generate)

        # Results
        results_group = QGroupBox("Matte Components")
        results_layout = QVBoxLayout()

        self.check_show_core = QCheckBox("Show Core")
        self.check_show_edge = QCheckBox("Show Edge")
        self.check_show_hair = QCheckBox("Show Hair")
        self.check_show_final = QCheckBox("Show Final")
        self.check_show_final.setChecked(True)

        results_layout.addWidget(self.check_show_core)
        results_layout.addWidget(self.check_show_edge)
        results_layout.addWidget(self.check_show_hair)
        results_layout.addWidget(self.check_show_final)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        layout.addStretch()

    def set_image(self, image: np.ndarray):
        """Set current image."""
        self.current_image = image

    def set_mask(self, mask: np.ndarray):
        """Set segmentation mask."""
        self.current_mask = mask

    def set_backend(self, backend):
        """Set processing backend."""
        self.backend = backend

    def _generate_matte(self):
        """Generate professional matte."""
        if self.current_image is None or self.current_mask is None:
            QMessageBox.warning(self, "Warning", "Need image and mask first")
            return

        try:
            # Initialize professional matting if needed
            if self.professional_matting is None:
                from ultimate_rotoscopy.matting.professional_matting import (
                    ProfessionalMatting, ProfessionalMattingConfig
                )
                from ultimate_rotoscopy.matting.alpha_split import AlphaSplitConfig
                from ultimate_rotoscopy.matting.motion_blur_aware import MotionBlurConfig

                config = ProfessionalMattingConfig(
                    alpha_split=AlphaSplitConfig(
                        core_threshold_low=self.spin_core_threshold.value(),
                        edge_band_width=self.spin_edge_width.value(),
                    ),
                    motion_blur=MotionBlurConfig(),
                    enable_motion_blur=self.check_motion_blur.isChecked(),
                )

                self.professional_matting = ProfessionalMatting(config)

            # Process
            self.btn_generate.setEnabled(False)
            self.btn_generate.setText("Processing...")
            QApplication.processEvents()

            # Use mask as initial alpha
            alpha_initial = self.current_mask.astype(np.float32)

            result = self.professional_matting.process_frame(
                alpha_initial,
                self.current_image,
                prev_frame=None
            )

            self.current_alpha = result.alpha_final

            # Emit signal
            self.matte_generated.emit(result.alpha_final)

            # Store results for visualization
            self.result = result

            QMessageBox.information(
                self,
                "Success",
                f"Professional matte generated!\n"
                f"Motion blur detected: {result.has_motion_blur}\n"
                f"Blur level: {result.blur_level.value if result.has_motion_blur else 'None'}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Matting failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_generate.setEnabled(True)
            self.btn_generate.setText("✨ Generate Professional Matte")


class CompositeTab(QWidget):
    """
    Tab 4: Compositing

    Final composite with background.
    """

    composite_generated = Signal(object)  # composite

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_fg = None
        self.current_alpha = None
        self.current_bg = None
        self.current_composite = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("<h2>Compositing</h2>")
        layout.addWidget(header)

        # Load background
        bg_group = QGroupBox("Background Plate")
        bg_layout = QVBoxLayout()

        self.btn_load_bg = QPushButton("📂 Load Background")
        self.btn_load_bg.clicked.connect(self._load_background)
        bg_layout.addWidget(self.btn_load_bg)

        self.lbl_bg_info = QLabel("No background loaded")
        bg_layout.addWidget(self.lbl_bg_info)

        bg_group.setLayout(bg_layout)
        layout.addWidget(bg_group)

        # Composite options
        opts_group = QGroupBox("Composite Options")
        opts_layout = QVBoxLayout()

        self.check_premult = QCheckBox("Premultiply Alpha")
        self.check_premult.setChecked(True)
        opts_layout.addWidget(self.check_premult)

        opts_group.setLayout(opts_layout)
        layout.addWidget(opts_group)

        # Generate composite
        self.btn_composite = QPushButton("🎬 Generate Composite")
        self.btn_composite.setMinimumHeight(50)
        self.btn_composite.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_composite.clicked.connect(self._generate_composite)
        layout.addWidget(self.btn_composite)

        layout.addStretch()

    def set_foreground(self, fg: np.ndarray):
        """Set foreground image."""
        self.current_fg = fg

    def set_alpha(self, alpha: np.ndarray):
        """Set alpha matte."""
        self.current_alpha = alpha

    def _load_background(self):
        """Load background image."""
        import cv2

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Background",
            "",
            "Image Files (*.png *.jpg *.jpeg *.exr *.tif *.tiff);;All Files (*)"
        )

        if not filepath:
            return

        try:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

            if img is None:
                QMessageBox.critical(self, "Error", "Could not load image")
                return

            # Convert to RGB
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

            self.current_bg = img
            h, w = img.shape[:2]
            self.lbl_bg_info.setText(f"Loaded: {w}x{h}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{str(e)}")

    def _generate_composite(self):
        """Generate final composite."""
        if self.current_fg is None or self.current_alpha is None or self.current_bg is None:
            QMessageBox.warning(self, "Warning", "Need foreground, alpha, and background")
            return

        try:
            # Resize background to match foreground
            if self.current_bg.shape[:2] != self.current_fg.shape[:2]:
                import cv2
                h, w = self.current_fg.shape[:2]
                self.current_bg = cv2.resize(self.current_bg, (w, h))

            # Composite: fg * alpha + bg * (1 - alpha)
            alpha_3ch = self.current_alpha[..., np.newaxis] if len(self.current_alpha.shape) == 2 else self.current_alpha

            if self.check_premult.isChecked():
                # Premultiply
                fg_premult = self.current_fg * alpha_3ch
                composite = fg_premult + self.current_bg * (1 - alpha_3ch)
            else:
                # Over operation
                composite = self.current_fg * alpha_3ch + self.current_bg * (1 - alpha_3ch)

            self.current_composite = np.clip(composite, 0, 1)

            # Emit signal
            self.composite_generated.emit(self.current_composite)

            QMessageBox.information(self, "Success", "Composite generated!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Composite failed:\n{str(e)}")


class ExportTab(QWidget):
    """
    Tab 5: Export

    Multi-layer export to EXR, PNG, etc.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.layers_data = {}

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("<h2>Export</h2>")
        layout.addWidget(header)

        # Layers to export
        layers_group = QGroupBox("Layers to Export")
        layers_layout = QVBoxLayout()

        self.check_alpha_core = QCheckBox("Alpha Core")
        self.check_alpha_edge = QCheckBox("Alpha Edge")
        self.check_alpha_hair = QCheckBox("Alpha Hair")
        self.check_alpha_final = QCheckBox("Alpha Final")
        self.check_alpha_final.setChecked(True)
        self.check_composite = QCheckBox("Composite")
        self.check_depth = QCheckBox("Depth Map")
        self.check_normals = QCheckBox("Normal Map")

        layers_layout.addWidget(self.check_alpha_core)
        layers_layout.addWidget(self.check_alpha_edge)
        layers_layout.addWidget(self.check_alpha_hair)
        layers_layout.addWidget(self.check_alpha_final)
        layers_layout.addWidget(self.check_composite)
        layers_layout.addWidget(self.check_depth)
        layers_layout.addWidget(self.check_normals)

        layers_group.setLayout(layers_layout)
        layout.addWidget(layers_group)

        # Format selection
        format_group = QGroupBox("Export Format")
        format_layout = QVBoxLayout()

        self.combo_format = QComboBox()
        self.combo_format.addItems([
            "Multi-layer EXR",
            "Separate EXR files",
            "PNG 16-bit",
            "TIFF 16-bit"
        ])
        format_layout.addWidget(self.combo_format)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Export button
        self.btn_export = QPushButton("💾 Export Layers")
        self.btn_export.setMinimumHeight(50)
        self.btn_export.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        self.btn_export.clicked.connect(self._export)
        layout.addWidget(self.btn_export)

        layout.addStretch()

    def set_layer_data(self, name: str, data: np.ndarray):
        """Set layer data for export."""
        self.layers_data[name] = data

    def _export(self):
        """Export selected layers."""
        # Get output path
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Layers",
            "",
            "EXR Files (*.exr);;PNG Files (*.png);;TIFF Files (*.tiff);;All Files (*)"
        )

        if not output_path:
            return

        try:
            from pathlib import Path
            import cv2

            output_path = Path(output_path)
            format_type = self.combo_format.currentText()

            if "Multi-layer EXR" in format_type:
                self._export_multilayer_exr(output_path)
            else:
                self._export_separate_files(output_path)

            QMessageBox.information(self, "Success", f"Exported to:\n{output_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def _export_multilayer_exr(self, output_path):
        """Export all layers to single EXR."""
        try:
            import OpenEXR
            import Imath

            # Collect selected layers
            layers_to_export = {}

            if self.check_alpha_core.isChecked() and 'alpha_core' in self.layers_data:
                layers_to_export['alpha_core.A'] = self.layers_data['alpha_core']
            if self.check_alpha_edge.isChecked() and 'alpha_edge' in self.layers_data:
                layers_to_export['alpha_edge.A'] = self.layers_data['alpha_edge']
            if self.check_alpha_hair.isChecked() and 'alpha_hair' in self.layers_data:
                layers_to_export['alpha_hair.A'] = self.layers_data['alpha_hair']
            if self.check_alpha_final.isChecked() and 'alpha_final' in self.layers_data:
                layers_to_export['alpha_final.A'] = self.layers_data['alpha_final']

            if not layers_to_export:
                raise ValueError("No layers selected for export")

            # Get dimensions from first layer
            first_layer = list(layers_to_export.values())[0]
            h, w = first_layer.shape[:2]

            # Create EXR header
            header = OpenEXR.Header(w, h)
            header['channels'] = {
                name: Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
                for name in layers_to_export.keys()
            }

            # Write EXR
            exr = OpenEXR.OutputFile(str(output_path), header)
            pixel_data = {
                name: data.astype(np.float32).tobytes()
                for name, data in layers_to_export.items()
            }
            exr.writePixels(pixel_data)
            exr.close()

        except ImportError:
            raise ImportError("OpenEXR not available. Install with: pip install OpenEXR")

    def _export_separate_files(self, output_path):
        """Export layers as separate files."""
        import cv2
        from pathlib import Path

        output_dir = output_path.parent
        stem = output_path.stem
        ext = output_path.suffix

        exported = []

        # Export selected layers
        if self.check_alpha_final.isChecked() and 'alpha_final' in self.layers_data:
            filepath = output_dir / f"{stem}_alpha_final{ext}"
            self._write_image(filepath, self.layers_data['alpha_final'])
            exported.append(filepath.name)

        if self.check_composite.isChecked() and 'composite' in self.layers_data:
            filepath = output_dir / f"{stem}_composite{ext}"
            self._write_image(filepath, self.layers_data['composite'])
            exported.append(filepath.name)

        return exported

    def _write_image(self, filepath, data):
        """Write image to file."""
        import cv2

        # Convert to uint16 for PNG/TIFF
        if filepath.suffix in ['.png', '.tiff', '.tif']:
            data_int = (np.clip(data, 0, 1) * 65535).astype(np.uint16)
            cv2.imwrite(str(filepath), data_int)
        else:
            # Float for EXR
            cv2.imwrite(str(filepath), data.astype(np.float32))


class DepthTab(QWidget):
    """
    Tab: Depth Anything V3

    Complete depth estimation with all DA3 features:
    - Monocular depth estimation
    - Multi-view depth + pose estimation
    - 3D Gaussian splatting
    - Sky segmentation
    - Metric depth (real-world scale)
    - Normal maps
    - Point cloud export
    """

    depth_generated = Signal(object)  # DepthResult

    def __init__(self, canvas, parent=None):
        super().__init__(parent)

        self.canvas = canvas
        self.current_image = None
        self.current_depth_result = None
        self.depth_model = None
        self.multi_view_images = []
        self.depth_layers = {}

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("<h2>🌊 Depth Anything V3</h2>")
        layout.addWidget(header)

        # Create tab widget for different modes
        self.mode_tabs = QTabWidget()

        # Tab 1: Monocular Depth
        self.monocular_tab = self._create_monocular_tab()
        self.mode_tabs.addTab(self.monocular_tab, "📷 Single Image")

        # Tab 2: Multi-View Depth
        self.multiview_tab = self._create_multiview_tab()
        self.mode_tabs.addTab(self.multiview_tab, "🎬 Multi-View")

        # Tab 3: 3D Gaussian Splatting
        self.gaussian_tab = self._create_gaussian_tab()
        self.mode_tabs.addTab(self.gaussian_tab, "✨ Gaussian Splat")

        # Tab 4: Export
        self.export_tab = self._create_export_tab()
        self.mode_tabs.addTab(self.export_tab, "💾 Export")

        layout.addWidget(self.mode_tabs)

        layout.addStretch()

    def _create_monocular_tab(self):
        """Create monocular depth estimation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Model Size:"))
        self.combo_model_size = QComboBox()
        self.combo_model_size.addItems(["Small (25M)", "Base (99M)", "Large (335M)", "Giant (1.3B)"])
        self.combo_model_size.setCurrentText("Large (335M)")
        size_layout.addWidget(self.combo_model_size)
        model_layout.addLayout(size_layout)

        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Model Type:"))
        self.combo_model_type = QComboBox()
        self.combo_model_type.addItems(["Main (Relative)", "Metric (Real Scale)", "Monocular", "Nested"])
        type_layout.addWidget(self.combo_model_type)
        model_layout.addLayout(type_layout)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()

        self.check_depth_ray = QCheckBox("Generate Depth-Ray Representation (DA3)")
        self.check_depth_ray.setChecked(True)
        output_layout.addWidget(self.check_depth_ray)

        self.check_normals = QCheckBox("Generate Normal Maps")
        self.check_normals.setChecked(True)
        output_layout.addWidget(self.check_normals)

        self.check_point_cloud = QCheckBox("Generate Point Cloud")
        self.check_point_cloud.setChecked(False)
        output_layout.addWidget(self.check_point_cloud)

        self.check_sky_seg = QCheckBox("Sky Segmentation (DA3)")
        self.check_sky_seg.setChecked(True)
        output_layout.addWidget(self.check_sky_seg)

        self.check_estimate_intrinsics = QCheckBox("Auto-Estimate Camera Intrinsics (DA3)")
        self.check_estimate_intrinsics.setChecked(True)
        output_layout.addWidget(self.check_estimate_intrinsics)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Processing options
        processing_group = QGroupBox("Processing")
        processing_layout = QVBoxLayout()

        self.check_temporal = QCheckBox("Temporal Smoothing (Video)")
        self.check_temporal.setChecked(True)
        processing_layout.addWidget(self.check_temporal)

        temporal_layout = QHBoxLayout()
        temporal_layout.addWidget(QLabel("Temporal Alpha:"))
        self.spin_temporal_alpha = QDoubleSpinBox()
        self.spin_temporal_alpha.setRange(0.0, 1.0)
        self.spin_temporal_alpha.setSingleStep(0.05)
        self.spin_temporal_alpha.setValue(0.85)
        temporal_layout.addWidget(self.spin_temporal_alpha)
        processing_layout.addLayout(temporal_layout)

        self.check_edge_smooth = QCheckBox("Edge-Aware Smoothing")
        self.check_edge_smooth.setChecked(True)
        processing_layout.addWidget(self.check_edge_smooth)

        processing_group.setLayout(processing_layout)
        layout.addWidget(processing_group)

        # Metric depth settings
        metric_group = QGroupBox("Metric Depth Settings")
        metric_layout = QVBoxLayout()

        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min Depth (m):"))
        self.spin_min_depth = QDoubleSpinBox()
        self.spin_min_depth.setRange(0.0, 100.0)
        self.spin_min_depth.setValue(0.01)
        min_layout.addWidget(self.spin_min_depth)
        metric_layout.addLayout(min_layout)

        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max Depth (m):"))
        self.spin_max_depth = QDoubleSpinBox()
        self.spin_max_depth.setRange(1.0, 1000.0)
        self.spin_max_depth.setValue(100.0)
        max_layout.addWidget(self.spin_max_depth)
        metric_layout.addLayout(max_layout)

        metric_group.setLayout(metric_layout)
        layout.addWidget(metric_group)

        # Generate button
        self.btn_generate_depth = QPushButton("🌊 Estimate Depth")
        self.btn_generate_depth.setMinimumHeight(50)
        self.btn_generate_depth.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_generate_depth.clicked.connect(self._estimate_depth)
        layout.addWidget(self.btn_generate_depth)

        # Results info
        self.depth_info = QTextEdit()
        self.depth_info.setReadOnly(True)
        self.depth_info.setMaximumHeight(150)
        self.depth_info.setPlaceholderText("Depth estimation results will appear here...")
        layout.addWidget(self.depth_info)

        return widget

    def _create_multiview_tab(self):
        """Create multi-view depth tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("<h3>Multi-View Depth & Pose Estimation</h3>"))

        info = QLabel(
            "Load multiple images from different viewpoints.\n"
            "DA3 will estimate:\n"
            "• Consistent depth maps for all views\n"
            "• Camera poses (rotation + translation)\n"
            "• Fused 3D point cloud"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Image list
        images_group = QGroupBox("Multi-View Images")
        images_layout = QVBoxLayout()

        self.multiview_list = QListWidget()
        images_layout.addWidget(self.multiview_list)

        btn_layout = QHBoxLayout()
        btn_add = QPushButton("➕ Add Images")
        btn_add.clicked.connect(self._add_multiview_images)
        btn_layout.addWidget(btn_add)

        btn_clear = QPushButton("🗑️ Clear")
        btn_clear.clicked.connect(self._clear_multiview)
        btn_layout.addWidget(btn_clear)

        images_layout.addLayout(btn_layout)
        images_group.setLayout(images_layout)
        layout.addWidget(images_group)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()

        self.check_estimate_poses = QCheckBox("Estimate Camera Poses (DA3)")
        self.check_estimate_poses.setChecked(True)
        options_layout.addWidget(self.check_estimate_poses)

        self.check_fuse_clouds = QCheckBox("Fuse Point Clouds")
        self.check_fuse_clouds.setChecked(True)
        options_layout.addWidget(self.check_fuse_clouds)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Process button
        self.btn_multiview = QPushButton("🎬 Process Multi-View")
        self.btn_multiview.setMinimumHeight(50)
        self.btn_multiview.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_multiview.clicked.connect(self._process_multiview)
        layout.addWidget(self.btn_multiview)

        # Results
        self.multiview_info = QTextEdit()
        self.multiview_info.setReadOnly(True)
        self.multiview_info.setMaximumHeight(150)
        layout.addWidget(self.multiview_info)

        layout.addStretch()
        return widget

    def _create_gaussian_tab(self):
        """Create 3D Gaussian splatting tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("<h3>3D Gaussian Splatting (DA3)</h3>"))

        info = QLabel(
            "Create 3D Gaussian representation for novel view synthesis.\n"
            "Requires multi-view images with depth + poses."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Settings
        settings_group = QGroupBox("Gaussian Splatting Settings")
        settings_layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.combo_gaussian_mode = QComboBox()
        self.combo_gaussian_mode.addItems(["Fast", "Quality", "Realtime"])
        self.combo_gaussian_mode.setCurrentText("Quality")
        mode_layout.addWidget(self.combo_gaussian_mode)
        settings_layout.addLayout(mode_layout)

        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Optimization Iterations:"))
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(100, 10000)
        self.spin_iterations.setValue(1000)
        self.spin_iterations.setSingleStep(100)
        iter_layout.addWidget(self.spin_iterations)
        settings_layout.addLayout(iter_layout)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Process button
        self.btn_gaussian = QPushButton("✨ Create Gaussian Splat")
        self.btn_gaussian.setMinimumHeight(50)
        self.btn_gaussian.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        self.btn_gaussian.clicked.connect(self._create_gaussian_splat)
        layout.addWidget(self.btn_gaussian)

        # Novel view rendering
        render_group = QGroupBox("Novel View Rendering")
        render_layout = QVBoxLayout()

        render_layout.addWidget(QLabel("Render new viewpoints from Gaussian representation"))

        self.btn_render_view = QPushButton("🎥 Render Novel View")
        self.btn_render_view.clicked.connect(self._render_novel_view)
        render_layout.addWidget(self.btn_render_view)

        render_group.setLayout(render_layout)
        layout.addWidget(render_group)

        # Results
        self.gaussian_info = QTextEdit()
        self.gaussian_info.setReadOnly(True)
        self.gaussian_info.setMaximumHeight(150)
        layout.addWidget(self.gaussian_info)

        layout.addStretch()
        return widget

    def _create_export_tab(self):
        """Create depth export tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("<h3>Export Depth Data</h3>"))

        # Layer selection
        layers_group = QGroupBox("Layers to Export")
        layers_layout = QVBoxLayout()

        self.check_depth_map = QCheckBox("Depth Map (Relative 0-1)")
        self.check_depth_map.setChecked(True)
        layers_layout.addWidget(self.check_depth_map)

        self.check_metric_depth = QCheckBox("Metric Depth (Meters)")
        self.check_metric_depth.setChecked(True)
        layers_layout.addWidget(self.check_metric_depth)

        self.check_normal_map = QCheckBox("Normal Map")
        self.check_normal_map.setChecked(True)
        layers_layout.addWidget(self.check_normal_map)

        self.check_disparity = QCheckBox("Disparity Map")
        self.check_disparity.setChecked(False)
        layers_layout.addWidget(self.check_disparity)

        self.check_depth_edges = QCheckBox("Depth Edges")
        self.check_depth_edges.setChecked(False)
        layers_layout.addWidget(self.check_depth_edges)

        self.check_sky_mask = QCheckBox("Sky Mask (DA3)")
        self.check_sky_mask.setChecked(True)
        layers_layout.addWidget(self.check_sky_mask)

        self.check_confidence = QCheckBox("Confidence Map")
        self.check_confidence.setChecked(False)
        layers_layout.addWidget(self.check_confidence)

        layers_group.setLayout(layers_layout)
        layout.addWidget(layers_group)

        # Point cloud export
        pc_group = QGroupBox("Point Cloud Export")
        pc_layout = QVBoxLayout()

        pc_format_layout = QHBoxLayout()
        pc_format_layout.addWidget(QLabel("Format:"))
        self.combo_pc_format = QComboBox()
        self.combo_pc_format.addItems(["PLY", "OBJ", "XYZ"])
        pc_format_layout.addWidget(self.combo_pc_format)
        pc_layout.addLayout(pc_format_layout)

        self.btn_export_pc = QPushButton("💾 Export Point Cloud")
        self.btn_export_pc.clicked.connect(self._export_point_cloud)
        pc_layout.addWidget(self.btn_export_pc)

        pc_group.setLayout(pc_layout)
        layout.addWidget(pc_group)

        # Image export
        export_group = QGroupBox("Image Export")
        export_layout = QVBoxLayout()

        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.combo_export_format = QComboBox()
        self.combo_export_format.addItems(["EXR (32-bit)", "PNG (16-bit)", "TIFF (16-bit)"])
        format_layout.addWidget(self.combo_export_format)
        export_layout.addLayout(format_layout)

        self.btn_export_depth = QPushButton("💾 Export Depth Layers")
        self.btn_export_depth.setMinimumHeight(40)
        self.btn_export_depth.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_export_depth.clicked.connect(self._export_depth_layers)
        export_layout.addWidget(self.btn_export_depth)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        layout.addStretch()
        return widget

    def set_image(self, image: np.ndarray):
        """Set current image."""
        self.current_image = image

    def set_depth_model(self, model):
        """Set depth model instance."""
        self.depth_model = model

    def _estimate_depth(self):
        """Estimate depth from current image."""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return

        if self.depth_model is None:
            QMessageBox.critical(self, "Error", "Depth model not loaded")
            return

        try:
            self.btn_generate_depth.setEnabled(False)
            self.btn_generate_depth.setText("Processing...")
            QApplication.processEvents()

            # Configure model
            self.depth_model.depth_config.generate_normals = self.check_normals.isChecked()
            self.depth_model.depth_config.use_depth_ray = self.check_depth_ray.isChecked()
            self.depth_model.depth_config.sky_segmentation = self.check_sky_seg.isChecked()
            self.depth_model.depth_config.estimate_intrinsics = self.check_estimate_intrinsics.isChecked()
            self.depth_model.depth_config.temporal_smoothing = self.check_temporal.isChecked()
            self.depth_model.depth_config.temporal_alpha = self.spin_temporal_alpha.value()
            self.depth_model.depth_config.edge_aware_smoothing = self.check_edge_smooth.isChecked()
            self.depth_model.depth_config.min_depth = self.spin_min_depth.value()
            self.depth_model.depth_config.max_depth = self.spin_max_depth.value()

            # Estimate depth
            result = self.depth_model.estimate_depth(
                self.current_image,
                generate_normals=self.check_normals.isChecked(),
                generate_point_cloud=self.check_point_cloud.isChecked(),
            )

            self.current_depth_result = result

            # Store layers
            self.depth_layers = {
                'depth_map': result.depth_normalized,
                'metric_depth': result.metric_depth,
                'normal_map': result.normals,
                'disparity': result.disparity,
                'depth_edges': result.depth_edges,
                'sky_mask': result.sky_mask,
                'confidence': result.confidence,
                'depth_ray': result.depth_ray,
            }

            # Show depth on canvas
            self.canvas.set_overlay(result.depth_normalized, opacity=0.7)

            # Emit signal
            self.depth_generated.emit(result)

            # Display info
            info_text = f"""<b>Depth Estimation Complete</b>

<b>Model:</b> Depth Anything V{result.metadata.get('da_version', '3')}
<b>Size:</b> {result.metadata.get('model_size', 'N/A')}
<b>Resolution:</b> {result.depth_map.shape[1]}x{result.depth_map.shape[0]}
<b>Processing Time:</b> {result.metadata.get('processing_time_ms', 0):.1f} ms

<b>Outputs Generated:</b>
• Depth Map: {result.depth_map.shape}
• Normalized: [0-1] range
"""

            if result.metric_depth is not None:
                min_m = result.metric_depth.min()
                max_m = result.metric_depth.max()
                info_text += f"• Metric Depth: {min_m:.2f}m - {max_m:.2f}m\n"

            if result.normals is not None:
                info_text += f"• Normal Map: {result.normals.shape}\n"

            if result.point_cloud is not None:
                info_text += f"• Point Cloud: {len(result.point_cloud)} points\n"

            if result.sky_mask is not None:
                sky_percent = (result.sky_mask.sum() / result.sky_mask.size) * 100
                info_text += f"• Sky Segmentation: {sky_percent:.1f}% sky\n"

            if result.intrinsics is not None:
                info_text += f"• Camera Intrinsics: fx={result.intrinsics.fx:.1f}, fy={result.intrinsics.fy:.1f}\n"

            self.depth_info.setHtml(info_text)

            QMessageBox.information(self, "Success", "Depth estimation complete!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Depth estimation failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_generate_depth.setEnabled(True)
            self.btn_generate_depth.setText("🌊 Estimate Depth")

    def _add_multiview_images(self):
        """Add images for multi-view processing."""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Multi-View Images",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.exr);;All Files (*)"
        )

        if not filepaths:
            return

        import cv2
        for filepath in filepaths:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0

                self.multi_view_images.append(img)
                self.multiview_list.addItem(f"Image {len(self.multi_view_images)}: {filepath}")

    def _clear_multiview(self):
        """Clear multi-view images."""
        self.multi_view_images.clear()
        self.multiview_list.clear()

    def _process_multiview(self):
        """Process multi-view depth estimation."""
        if len(self.multi_view_images) < 2:
            QMessageBox.warning(self, "Warning", "Add at least 2 images for multi-view processing")
            return

        if self.depth_model is None:
            QMessageBox.critical(self, "Error", "Depth model not loaded")
            return

        try:
            self.btn_multiview.setEnabled(False)
            self.btn_multiview.setText("Processing Multi-View...")
            QApplication.processEvents()

            # Estimate multi-view depth
            result = self.depth_model.estimate_multiview_depth(
                self.multi_view_images,
                known_poses=None if self.check_estimate_poses.isChecked() else []
            )

            # Display info
            info_text = f"""<b>Multi-View Processing Complete</b>

<b>Views Processed:</b> {result.metadata.get('num_views', 0)}
<b>Fused Points:</b> {result.metadata.get('fused_points', 0):,}

<b>Camera Poses:</b>
"""
            for i, pose in enumerate(result.poses):
                info_text += f"View {i+1}: R={pose.rotation.shape}, t={pose.translation}\n"

            self.multiview_info.setHtml(info_text)

            QMessageBox.information(self, "Success", "Multi-view processing complete!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Multi-view processing failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_multiview.setEnabled(True)
            self.btn_multiview.setText("🎬 Process Multi-View")

    def _create_gaussian_splat(self):
        """Create 3D Gaussian splatting."""
        if len(self.multi_view_images) < 2:
            QMessageBox.warning(self, "Warning", "Add multi-view images first")
            return

        if self.depth_model is None:
            QMessageBox.critical(self, "Error", "Depth model not loaded")
            return

        try:
            self.btn_gaussian.setEnabled(False)
            self.btn_gaussian.setText("Creating Gaussian Splat...")
            QApplication.processEvents()

            # Create Gaussian splat
            result = self.depth_model.create_gaussian_splat(
                self.multi_view_images,
                num_iterations=self.spin_iterations.value()
            )

            info_text = f"""<b>Gaussian Splatting Complete</b>

<b>Gaussians:</b> {result.metadata.get('num_gaussians', 0):,}
<b>Views:</b> {result.metadata.get('num_views', 0)}
<b>Mode:</b> {self.combo_gaussian_mode.currentText()}

Ready for novel view synthesis!
"""
            self.gaussian_info.setHtml(info_text)

            QMessageBox.information(self, "Success", "Gaussian splatting complete!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gaussian splatting failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_gaussian.setEnabled(True)
            self.btn_gaussian.setText("✨ Create Gaussian Splat")

    def _render_novel_view(self):
        """Render a novel view."""
        QMessageBox.information(self, "Coming Soon", "Novel view rendering will be available soon!")

    def _export_point_cloud(self):
        """Export point cloud."""
        if self.current_depth_result is None or self.current_depth_result.point_cloud is None:
            QMessageBox.warning(self, "Warning", "No point cloud available. Enable 'Generate Point Cloud' first.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Point Cloud",
            "",
            "PLY Files (*.ply);;OBJ Files (*.obj);;XYZ Files (*.xyz)"
        )

        if not filepath:
            return

        try:
            from pathlib import Path
            format_ext = Path(filepath).suffix[1:].lower()

            self.depth_model.export_point_cloud(
                self.current_depth_result.point_cloud,
                colors=None,
                output_path=Path(filepath),
                format=format_ext
            )

            QMessageBox.information(self, "Success", f"Point cloud exported to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def _export_depth_layers(self):
        """Export depth layers."""
        if not self.depth_layers:
            QMessageBox.warning(self, "Warning", "No depth data available. Estimate depth first.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Depth Layers",
            "",
            "EXR Files (*.exr);;PNG Files (*.png);;TIFF Files (*.tiff)"
        )

        if not filepath:
            return

        try:
            import cv2
            from pathlib import Path

            output_path = Path(filepath)

            # Export selected layers
            if self.combo_export_format.currentText().startswith("EXR"):
                # Multi-layer EXR
                self._export_exr_multilayer(output_path)
            else:
                # Separate files
                self._export_separate_depth_files(output_path)

            QMessageBox.information(self, "Success", f"Depth layers exported to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def _export_exr_multilayer(self, output_path):
        """Export multi-layer EXR."""
        import OpenEXR
        import Imath

        header = OpenEXR.Header(
            self.depth_layers['depth_map'].shape[1],
            self.depth_layers['depth_map'].shape[0]
        )

        channels = {}

        if self.check_depth_map.isChecked() and 'depth_map' in self.depth_layers:
            channels['depth.Z'] = self.depth_layers['depth_map'].astype(np.float32).tobytes()

        if self.check_metric_depth.isChecked() and 'metric_depth' in self.depth_layers:
            if self.depth_layers['metric_depth'] is not None:
                channels['metric_depth.Z'] = self.depth_layers['metric_depth'].astype(np.float32).tobytes()

        if self.check_normal_map.isChecked() and 'normal_map' in self.depth_layers:
            if self.depth_layers['normal_map'] is not None:
                normals = self.depth_layers['normal_map']
                channels['normal.R'] = normals[:, :, 0].astype(np.float32).tobytes()
                channels['normal.G'] = normals[:, :, 1].astype(np.float32).tobytes()
                channels['normal.B'] = normals[:, :, 2].astype(np.float32).tobytes()

        exr = OpenEXR.OutputFile(str(output_path), header)
        exr.writePixels(channels)
        exr.close()

    def _export_separate_depth_files(self, output_path):
        """Export separate depth files."""
        import cv2
        from pathlib import Path

        output_dir = output_path.parent
        stem = output_path.stem
        ext = output_path.suffix

        if self.check_depth_map.isChecked():
            data = (self.depth_layers['depth_map'] * 65535).astype(np.uint16)
            cv2.imwrite(str(output_dir / f"{stem}_depth{ext}"), data)

        if self.check_metric_depth.isChecked() and self.depth_layers['metric_depth'] is not None:
            data = self.depth_layers['metric_depth'].astype(np.float32)
            cv2.imwrite(str(output_dir / f"{stem}_metric{ext}"), data)

        if self.check_normal_map.isChecked() and self.depth_layers['normal_map'] is not None:
            # Convert normals from [-1, 1] to [0, 1] for visualization
            normals = (self.depth_layers['normal_map'] + 1.0) / 2.0
            data = (normals * 65535).astype(np.uint16)
            cv2.imwrite(str(output_dir / f"{stem}_normals{ext}"), data)
