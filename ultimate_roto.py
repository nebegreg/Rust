#!/usr/bin/env python3
"""
Ultimate Rotoscopy Application (fresh build)
===========================================

From a clean slate, this application aligns with the `roadmap` goals:
- SAM3 segmentation with prompt points and device-aware execution
- Depth Anything V3/V2/AnyMoRe depth estimation for Z guidance
- Matte Anything alpha generation for compositing-ready mattes
- Export of mask, depth, alpha, EXR AOVs, and FBX meshes for Flame/VFX workflows

The GUI is intentionally compact and responsive, built with PySide6 and
non-blocking workers so artists can iterate quickly. Heavy model imports
are lazy-loaded inside the worker thread with explicit error messages if
optional dependencies are missing.

Usage
-----
    python ultimate_roto.py                # Launch GUI

Runtime Dependencies
--------------------
- PySide6, OpenCV (cv2), NumPy
- torch (with CUDA for GPU acceleration)
- sam3, depth_anything_v3, depth_anything, transformers, matte_anything (community packages)

The app will fall back to CPU automatically when CUDA is unavailable.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from flame_export import FlameExporter
from fbx_depth_export import DepthToFBX
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


class TaskKind(Enum):
    SEGMENT = auto()
    DEPTH = auto()
    MATTE = auto()


class DepthEngine(str, Enum):
    """Supported depth backbones with fallbacks."""

    V3 = "Depth Anything V3"
    V2 = "Depth Anything V2"
    ANYMORE = "Depth AnyMoRe"


@dataclass
class TaskRequest:
    kind: TaskKind
    image_path: Path
    device: str
    points: Sequence[Tuple[int, int]]
    depth_engine: DepthEngine = DepthEngine.V3


@dataclass
class TaskResult:
    mask: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    score: Optional[float] = None
    message: str = ""


class ModelHub:
    """Lazily loads third-party models with graceful failures."""

    def __init__(self) -> None:
        self._sam3_predictor = None
        self._sam3_device: Optional[str] = None
        self._depth_models: dict[DepthEngine, object] = {}
        self._matte_model = None

    def resolve_device(self, preferred: str) -> str:
        if preferred == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if preferred == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return preferred

    def predictor(self, device: str):
        target = self.resolve_device(device)
        if self._sam3_predictor is not None and self._sam3_device == target:
            return self._sam3_predictor

        sam3 = importlib.import_module("sam3")
        predictor_cls = getattr(sam3, "Sam3ImagePredictor")
        registry = getattr(sam3, "sam3_model_registry")
        checkpoint = registry["sam3_hiera_small"]()
        predictor = predictor_cls(checkpoint)
        predictor.model.to(device=target)
        predictor.model.eval()

        self._sam3_predictor = predictor
        self._sam3_device = target
        return predictor

    def depth_model(self, device: str, engine: DepthEngine):
        resolved = self.resolve_device(device)
        if engine in self._depth_models:
            return self._depth_models[engine]

        if engine == DepthEngine.V3:
            depth_anything = importlib.import_module("depth_anything_v3.dpt")
            depth_cls = getattr(depth_anything, "DepthAnythingV3")
            model = depth_cls(model_type="vitl", device=resolved)
        elif engine == DepthEngine.V2:
            try:
                depth_anything = importlib.import_module("depth_anything")
                depth_cls = getattr(depth_anything, "DepthAnything")
                model = depth_cls(model_type="vits", device=resolved)
            except ModuleNotFoundError:
                transformers = importlib.import_module("transformers")
                pipeline = getattr(transformers, "pipeline")
                model = pipeline(
                    "depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Base-hf",
                    device=0 if resolved == "cuda" else -1,
                )
        else:
            try:
                depth_anymore = importlib.import_module("depth_anymore")
                builder = getattr(depth_anymore, "build_model", None)
                if builder is not None:
                    model = builder(device=resolved)
                else:
                    model_cls = getattr(depth_anymore, "DepthAnyMoRe")
                    model = model_cls(device=resolved)
            except ModuleNotFoundError:
                transformers = importlib.import_module("transformers")
                pipeline = getattr(transformers, "pipeline")
                model = pipeline(
                    "depth-estimation",
                    model="depth-anything/Depth-AnyMoRe-Base",
                    device=0 if resolved == "cuda" else -1,
                )

        self._depth_models[engine] = model
        return model

    def matte_model(self, device: str):
        if self._matte_model is not None:
            return self._matte_model
        matte_mod = importlib.import_module("matte_anything")
        matte_cls = getattr(matte_mod, "MatteAnything")
        model = matte_cls(device=self.resolve_device(device))
        self._matte_model = model
        return model


class ProcessingWorker(QThread):
    finished_with_result = Signal(TaskResult)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, hub: ModelHub, request: TaskRequest):
        super().__init__()
        self.hub = hub
        self.request = request

    def run(self) -> None:
        try:
            image = cv2.imread(str(self.request.image_path))
            if image is None:
                raise FileNotFoundError(f"Image introuvable: {self.request.image_path}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.request.kind == TaskKind.SEGMENT:
                result = self._run_segmentation(rgb)
            elif self.request.kind == TaskKind.DEPTH:
                result = self._run_depth(rgb)
            else:
                result = self._run_matte(rgb)
            self.finished_with_result.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _run_segmentation(self, rgb: np.ndarray) -> TaskResult:
        self.status.emit("Chargement du modèle SAM3…")
        predictor = self.hub.predictor(self.request.device)
        points = np.array(self.request.points, dtype=np.float32)
        labels = np.ones(len(points), dtype=np.int32)

        self.status.emit("Encodage de l'image…")
        predictor.set_image(rgb)

        self.status.emit("Prédiction du masque…")
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        idx = int(scores.argmax())
        return TaskResult(mask=masks[idx].astype(np.uint8), score=float(scores[idx]))

    def _run_depth(self, rgb: np.ndarray) -> TaskResult:
        self.status.emit(f"Chargement du modèle {self.request.depth_engine.value}…")
        model = self.hub.depth_model(self.request.device, self.request.depth_engine)

        depth = model(rgb)
        if hasattr(depth, "cpu"):
            depth = depth.cpu().numpy()
        elif hasattr(depth, "numpy"):
            depth = depth.numpy()
        return TaskResult(depth=depth.astype(np.float32))

    def _run_matte(self, rgb: np.ndarray) -> TaskResult:
        self.status.emit("Chargement du modèle Matte Anything…")
        model = self.hub.matte_model(self.request.device)
        alpha = model.process_image(rgb)
        return TaskResult(alpha=alpha.astype(np.float32))


class ImageViewport(QLabel):
    point_added = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(960, 540)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background: #1e1e1e; border: 1px solid #444; }")
        self._image: Optional[np.ndarray] = None
        self._overlay: Optional[np.ndarray] = None
        self._points: List[Tuple[int, int]] = []
        self._alpha = 0.5

    def set_image(self, image: np.ndarray) -> None:
        self._image = image
        self._overlay = None
        self.update_view()

    def set_overlay(self, overlay: Optional[np.ndarray], alpha: float = 0.5) -> None:
        self._overlay = overlay
        self._alpha = alpha
        self.update_view()

    def set_points(self, points: Iterable[Tuple[int, int]]) -> None:
        self._points = list(points)
        self.update_view()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._image is None:
            return
        pos = event.position()
        pixmap = self.pixmap()
        if pixmap is None:
            return
        label_w, label_h = self.width(), self.height()
        pm_w, pm_h = pixmap.width(), pixmap.height()
        offset_x = (label_w - pm_w) // 2
        offset_y = (label_h - pm_h) // 2
        x = pos.x() - offset_x
        y = pos.y() - offset_y
        if not (0 <= x < pm_w and 0 <= y < pm_h):
            return
        scale_x = self._image.shape[1] / pm_w
        scale_y = self._image.shape[0] / pm_h
        self.point_added.emit(int(x * scale_x), int(y * scale_y))

    def update_view(self) -> None:
        if self._image is None:
            self.setText("Aucune image")
            return
        display = self._image.copy()
        if self._overlay is not None and self._overlay.shape[:2] == display.shape[:2]:
            overlay = self._overlay
            if overlay.ndim == 2:
                overlay = np.stack([overlay] * 3, axis=-1)
            overlay = overlay.astype(np.float32)
            overlay = overlay / (overlay.max() or 1)
            overlay = (overlay * 255).astype(np.uint8)
            display = cv2.addWeighted(display, 1 - self._alpha, overlay, self._alpha, 0)

        painter_image = self._to_qimage(display)
        painter = QPainter()
        painter.begin(painter_image)
        pen = QPen(QColor("#2dd4bf"))
        pen.setWidth(6)
        painter.setPen(pen)
        for x, y in self._points:
            painter.drawPoint(int(x), int(y))
        painter.end()

        self.setPixmap(QPixmap.fromImage(painter_image))

    @staticmethod
    def _to_qimage(image: np.ndarray) -> QImage:
        h, w, _ = image.shape
        return QImage(image.data, w, h, w * 3, QImage.Format.Format_RGB888)


class RotoWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Ultimate Rotoscopy")
        self.hub = ModelHub()
        self.viewport = ImageViewport()
        self.points: List[Tuple[int, int]] = []
        self.current_image_path: Optional[Path] = None
        self._last_mask: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._last_alpha: Optional[np.ndarray] = None
        self._last_rgb: Optional[np.ndarray] = None

        self._worker: Optional[ProcessingWorker] = None

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self._build_ui()
        self.viewport.point_added.connect(self._on_point_added)

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        splitter = QSplitter()
        layout.addWidget(splitter)

        splitter.addWidget(self.viewport)
        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout()
        panel.setLayout(panel_layout)

        file_btn = QPushButton("Ouvrir une image…")
        file_btn.clicked.connect(self._open_image)
        panel_layout.addWidget(file_btn)

        device_box = QGroupBox("Périphérique")
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device"))
        device_layout.addWidget(self.device_combo)
        device_box.setLayout(device_layout)
        panel_layout.addWidget(device_box)

        depth_engine_box = QGroupBox("Moteur de profondeur")
        depth_engine_layout = QVBoxLayout()
        self.depth_engine_combo = QComboBox()
        self.depth_engine_combo.addItems([
            DepthEngine.V3.value,
            DepthEngine.V2.value,
            DepthEngine.ANYMORE.value,
        ])
        depth_engine_layout.addWidget(QLabel("Choisir la version"))
        depth_engine_layout.addWidget(self.depth_engine_combo)
        depth_engine_box.setLayout(depth_engine_layout)
        panel_layout.addWidget(depth_engine_box)

        points_box = QGroupBox("Points de segmentation")
        points_layout = QVBoxLayout()
        self.points_list = QListWidget()
        clear_btn = QPushButton("Réinitialiser les points")
        clear_btn.clicked.connect(self._clear_points)
        points_layout.addWidget(self.points_list)
        points_layout.addWidget(clear_btn)
        points_box.setLayout(points_layout)
        panel_layout.addWidget(points_box)

        overlay_box = QGroupBox("Opacity overlay")
        overlay_layout = QHBoxLayout()
        self.overlay_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_slider.setRange(5, 95)
        self.overlay_slider.setValue(50)
        self.overlay_slider.valueChanged.connect(self._update_overlay_alpha)
        overlay_layout.addWidget(QLabel("Masque"))
        overlay_layout.addWidget(self.overlay_slider)
        overlay_box.setLayout(overlay_layout)
        panel_layout.addWidget(overlay_box)

        actions_box = QGroupBox("Actions")
        actions_layout = QGridLayout()
        seg_btn = QPushButton("SAM3")
        seg_btn.clicked.connect(lambda: self._start_task(TaskKind.SEGMENT))
        depth_btn = QPushButton("Depth Anything V3")
        depth_btn.clicked.connect(lambda: self._start_task(TaskKind.DEPTH))
        matte_btn = QPushButton("Matte Anything")
        matte_btn.clicked.connect(lambda: self._start_task(TaskKind.MATTE))
        save_btn = QPushButton("Exporter")
        save_btn.clicked.connect(self._export_assets)
        actions_layout.addWidget(seg_btn, 0, 0)
        actions_layout.addWidget(depth_btn, 0, 1)
        actions_layout.addWidget(matte_btn, 1, 0)
        actions_layout.addWidget(save_btn, 1, 1)
        actions_box.setLayout(actions_layout)
        panel_layout.addWidget(actions_box)

        panel_layout.addStretch(1)
        return panel

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choisir une image", "", "Images (*.png *.jpg *.jpeg *.tif)")
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            QMessageBox.critical(self, "Erreur", f"Impossible de lire {path}")
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.viewport.set_image(rgb)
        self._last_rgb = rgb
        self.points.clear()
        self.points_list.clear()
        self._last_mask = None
        self._last_depth = None
        self._last_alpha = None
        self.current_image_path = Path(path)
        self.status_bar.showMessage(f"Image chargée: {path}")

    def _on_point_added(self, x: int, y: int) -> None:
        self.points.append((x, y))
        item = QListWidgetItem(f"({x}, {y})")
        self.points_list.addItem(item)
        self.viewport.set_points(self.points)

    def _clear_points(self) -> None:
        self.points.clear()
        self.points_list.clear()
        self.viewport.set_points([])

    def _update_overlay_alpha(self, value: int) -> None:
        self.viewport.set_overlay(self.viewport._overlay, alpha=value / 100)

    def _start_task(self, kind: TaskKind) -> None:
        if self.current_image_path is None:
            QMessageBox.warning(self, "Alerte", "Chargez d'abord une image.")
            return
        if kind == TaskKind.SEGMENT and not self.points:
            QMessageBox.warning(self, "Alerte", "Ajoutez au moins un point.")
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Info", "Un traitement est déjà en cours.")
            return

        request = TaskRequest(
            kind=kind,
            image_path=self.current_image_path,
            device=self.device_combo.currentText(),
            points=tuple(self.points),
            depth_engine=DepthEngine(self.depth_engine_combo.currentText()),
        )
        self._worker = ProcessingWorker(self.hub, request)
        self._worker.finished_with_result.connect(self._handle_result)
        self._worker.failed.connect(self._handle_error)
        self._worker.status.connect(self.status_bar.showMessage)
        self._worker.start()
        self.status_bar.showMessage("Traitement en cours…")

    def _handle_result(self, result: TaskResult) -> None:
        if result.mask is not None:
            self.viewport.set_overlay(result.mask * 255, alpha=self.overlay_slider.value() / 100)
            self.status_bar.showMessage(f"Masque SAM3 prêt (score={result.score:.3f})")
            self._last_mask = result.mask
        if result.depth is not None:
            depth_norm = result.depth.astype(np.float32)
            depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-5)
            depth_rgb = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            depth_rgb = cv2.cvtColor(depth_rgb, cv2.COLOR_BGR2RGB)
            self.viewport.set_overlay(depth_rgb, alpha=self.overlay_slider.value() / 100)
            self._last_depth = depth_norm
            self.status_bar.showMessage("Carte de profondeur prête")
        if result.alpha is not None:
            alpha_rgb = np.stack([result.alpha] * 3, axis=-1)
            alpha_rgb = (alpha_rgb / (alpha_rgb.max() or 1) * 255).astype(np.uint8)
            self.viewport.set_overlay(alpha_rgb, alpha=self.overlay_slider.value() / 100)
            self._last_alpha = result.alpha
            self.status_bar.showMessage("Matte généré")

    def _handle_error(self, message: str) -> None:
        QMessageBox.critical(self, "Erreur", message)
        self.status_bar.showMessage(message)

    def _export_assets(self) -> None:
        if self.current_image_path is None:
            QMessageBox.warning(self, "Alerte", "Chargez d'abord une image.")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Dossier de sortie")
        if not output_dir:
            return
        base = Path(output_dir) / self.current_image_path.stem
        if getattr(self, "_last_mask", None) is not None:
            mask_path = base.parent / f"{base.name}_mask.png"
            cv2.imwrite(str(mask_path), (self._last_mask * 255).astype(np.uint8))
        if getattr(self, "_last_depth", None) is not None:
            depth_path = base.parent / f"{base.name}_depth.exr"
            cv2.imwrite(str(depth_path), self._last_depth.astype(np.float32))
        if getattr(self, "_last_alpha", None) is not None:
            alpha_path = base.parent / f"{base.name}_alpha.png"
            cv2.imwrite(str(alpha_path), (self._last_alpha * 255).astype(np.uint8))

        # Flame-friendly AOV pack and FBX mesh
        try:
            exporter = FlameExporter(output_dir, clip_name=base.name)
            if self.viewport._image is not None:
                exporter.add_rgba(self.viewport._image, getattr(self, "_last_alpha", None))
            if getattr(self, "_last_depth", None) is not None:
                exporter.add_depth(self._last_depth, normalize=False)
            if getattr(self, "_last_mask", None) is not None:
                exporter.add_alpha(self._last_mask)
            exporter.export_frame(1)
            exporter.generate_clip_xml()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Flame", f"Export Flame partiel: {exc}")

        if getattr(self, "_last_depth", None) is not None:
            try:
                mesher = DepthToFBX()
                fbx_path = mesher.export_mesh(
                    self._last_depth,
                    mask=getattr(self, "_last_mask", None),
                    output_path=base.parent / f"{base.name}_depth_mesh.fbx",
                )
                self.status_bar.showMessage(f"Exports EXR/FBX terminés vers {fbx_path.parent}")
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "FBX", f"Export FBX échoué: {exc}")
        else:
            self.status_bar.showMessage(f"Exports terminés vers {output_dir}")


def main() -> None:
    app = QApplication(sys.argv)
    window = RotoWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
