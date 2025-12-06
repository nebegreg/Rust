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
