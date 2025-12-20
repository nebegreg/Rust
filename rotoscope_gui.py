#!/usr/bin/env python3
"""
PySide6 Rotoscopy Application
=============================

High-performance GUI for interactive SAM3 segmentation.

Features
--------
- Fast PySide6 interface with responsive viewport
- GPU/CPU auto-selection with graceful CUDA fallback
- Interactive point selection and live overlay preview
- Background worker thread for non-blocking segmentation
- One-click mask export

Usage
-----
    python rotoscope_gui.py

Requirements
------------
    pip install PySide6 opencv-python numpy torch
    pip install git+https://github.com/facebookresearch/sam3.git
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
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
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from roto import ensure_points_in_frame, validate_points


@dataclass
class SegmentJob:
    image_path: Path
    points: List[Tuple[int, int]]
    device: str


class Sam3Session:
    """Lazy SAM3 loader with automatic device fallback."""

    def __init__(self) -> None:
        self._predictor = None
        self._device: Optional[str] = None

    def predictor(self, preferred_device: str):
        target = preferred_device
        if preferred_device == "cuda" and not torch.cuda.is_available():
            target = "cpu"

        if self._predictor is not None and self._device == target:
            return self._predictor

        from sam3 import Sam3ImagePredictor, sam3_model_registry

        model_type = "sam3_hiera_small"
        checkpoint = sam3_model_registry[model_type]()

        predictor = Sam3ImagePredictor(checkpoint)
        predictor.model.to(device=target)
        predictor.model.eval()

        self._predictor = predictor
        self._device = target
        return predictor

    @property
    def device(self) -> Optional[str]:
        return self._device


class SegmentationWorker(QThread):
    mask_ready = Signal(np.ndarray, float)
    error = Signal(str)
    status = Signal(str)

    def __init__(self, session: Sam3Session, job: SegmentJob):
        super().__init__()
        self.session = session
        self.job = job

    def run(self):
        try:
            self.status.emit("Chargement de l'image...")
            image = cv2.imread(str(self.job.image_path))
            if image is None:
                raise ValueError(f"Image introuvable: {self.job.image_path}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ensure_points_in_frame(self.job.points, image_rgb.shape[1], image_rgb.shape[0])

            predictor = self.session.predictor(self.job.device)
            active_device = self.session.device or self.job.device
            self.status.emit(f"Encodage sur {active_device}...")
            predictor.set_image(image_rgb)

            point_coords = np.array(self.job.points, dtype=np.float32)
            point_labels = np.ones(len(self.job.points), dtype=np.int32)

            self.status.emit("Prédiction du masque...")
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

            best_idx = scores.argmax()
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])

            self.mask_ready.emit(best_mask.astype(np.uint8), best_score)
            self.status.emit("Segmentation terminée")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class ImageViewport(QLabel):
    point_selected = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(720, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self._image: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._points: List[Tuple[int, int]] = []
        self._overlay_alpha = 0.45
        self._pixmap: Optional[QPixmap] = None
        self.setText("Aucune image")

    def set_image(self, image: np.ndarray):
        self._image = image.copy()
        self._mask = None
        self._pixmap = None
        self.update_display()

    def set_mask(self, mask: Optional[np.ndarray]):
        self._mask = mask.copy() if mask is not None else None
        self.update_display()

    def set_points(self, points: Sequence[Tuple[int, int]]):
        self._points = list(points)
        self.update_display()

    def set_overlay_alpha(self, value: float):
        self._overlay_alpha = value
        self.update_display()

    def clear(self):
        self._image = None
        self._mask = None
        self._pixmap = None
        self._points.clear()
        self.update_display()

    def mousePressEvent(self, event):
        if self._image is None or self._pixmap is None:
            return

        label_size = self.size()
        pixmap_size = self._pixmap.size()
        offset_x = (label_size.width() - pixmap_size.width()) // 2
        offset_y = (label_size.height() - pixmap_size.height()) // 2

        click_x = event.position().x() - offset_x
        click_y = event.position().y() - offset_y

        if 0 <= click_x < pixmap_size.width() and 0 <= click_y < pixmap_size.height():
            scale_x = self._image.shape[1] / pixmap_size.width()
            scale_y = self._image.shape[0] / pixmap_size.height()
            self.point_selected.emit(int(click_x * scale_x), int(click_y * scale_y))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._image is not None:
            self.update_display()

    def update_display(self):
        if self._image is None:
            self.setText("Aucune image")
            return

        display = self._image.copy()

        if self._mask is not None and self._mask.shape[:2] == display.shape[:2]:
            overlay = np.zeros_like(display)
            overlay[..., 1] = (self._mask * 255).astype(np.uint8)
            display = cv2.addWeighted(display, 1 - self._overlay_alpha, overlay, self._overlay_alpha, 0)

        for idx, (x, y) in enumerate(self._points, start=1):
            cv2.circle(display, (x, y), 6, (255, 90, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(
                display,
                str(idx),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        h, w, _ = display.shape
        bytes_per_line = 3 * w
        q_image = QImage(display.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(q_image)

        scaled = self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultimate Rotoscopy GUI (PySide6)")
        self.resize(1280, 800)

        self.session = Sam3Session()
        self.image_path: Optional[Path] = None
        self.image_rgb: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.points: List[Tuple[int, int]] = []
        self.current_device = "auto"

        self.viewport = ImageViewport()
        self.viewport.point_selected.connect(self.add_point)

        self.point_list = QListWidget()
        self.point_list.setUniformItemSizes(True)

        self.opacity_slider = self._build_opacity_slider()
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.run_button = QPushButton("Segmenter")
        self.run_button.clicked.connect(self.run_segmentation)

        self.save_button = QPushButton("Exporter le masque")
        self.save_button.clicked.connect(self.save_mask)
        self.save_button.setEnabled(False)

        layout = QHBoxLayout()
        layout.addWidget(self.viewport, stretch=3)
        layout.addWidget(self._build_side_panel(), stretch=1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self._build_menu()
        self.status.showMessage("Prêt")

    def _build_menu(self):
        open_action = QAction("Ouvrir une image", self)
        open_action.triggered.connect(self.open_image_dialog)

        clear_action = QAction("Réinitialiser", self)
        clear_action.triggered.connect(self.reset_state)

        toolbar = self.addToolBar("Main")
        toolbar.addAction(open_action)
        toolbar.addAction(clear_action)

    def _build_opacity_slider(self):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(45)
        slider.valueChanged.connect(self._on_opacity_changed)
        return slider

    def _build_side_panel(self):
        panel = QVBoxLayout()
        panel.addWidget(self._group_file())
        panel.addWidget(self._group_points())
        panel.addWidget(self._group_settings())
        panel.addStretch(1)
        panel.addWidget(self.run_button)
        panel.addWidget(self.save_button)

        widget = QWidget()
        widget.setLayout(panel)
        return widget

    def _group_file(self):
        box = QGroupBox("Fichier")
        layout = QVBoxLayout()

        btn = QPushButton("Ouvrir...")
        btn.clicked.connect(self.open_image_dialog)
        self.file_label = QLabel("Aucune image")
        self.file_label.setWordWrap(True)

        layout.addWidget(btn)
        layout.addWidget(self.file_label)
        box.setLayout(layout)
        return box

    def _group_points(self):
        box = QGroupBox("Points de segmentation")
        layout = QVBoxLayout()

        info = QLabel("Cliquez sur l'image pour ajouter des points foreground.")
        info.setWordWrap(True)

        clear_btn = QPushButton("Effacer les points")
        clear_btn.clicked.connect(self.clear_points)

        layout.addWidget(info)
        layout.addWidget(self.point_list)
        layout.addWidget(clear_btn)
        box.setLayout(layout)
        return box

    def _group_settings(self):
        box = QGroupBox("Paramètres")
        layout = QGridLayout()

        self.device_label = QLabel("Périphérique : Auto (CUDA→CPU)")
        self.device_label.setWordWrap(True)

        self.device_choice = self._create_device_buttons()

        layout.addWidget(self.device_label, 0, 0, 1, 2)
        layout.addWidget(self.device_choice, 1, 0, 1, 2)

        layout.addWidget(QLabel("Opacité masque"), 2, 0)
        layout.addWidget(self.opacity_slider, 2, 1)

        box.setLayout(layout)
        return box

    def _create_device_buttons(self):
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.device_buttons = []
        for label, value in [("Auto", "auto"), ("CUDA", "cuda"), ("CPU", "cpu")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, v=value, b=btn: self._select_device(v, b))
            layout.addWidget(btn)
            self.device_buttons.append(btn)

        self.device_buttons[0].setChecked(True)
        container.setLayout(layout)
        return container

    def _select_device(self, device: str, button: QPushButton):
        for btn in self.device_buttons:
            btn.setChecked(btn is button)

        self.current_device = device
        if device == "auto":
            label = "Auto (CUDA→CPU)"
        elif device == "cuda" and not torch.cuda.is_available():
            label = "CPU (CUDA indisponible)"
        else:
            label = device.upper()

        self.device_label.setText(f"Périphérique : {label}")

    def open_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choisir une image", "", "Images (*.png *.jpg *.jpeg *.tif *.bmp)")
        if file_path:
            self.load_image(Path(file_path))

    def load_image(self, path: Path):
        image = cv2.imread(str(path))
        if image is None:
            QMessageBox.critical(self, "Erreur", f"Impossible de charger: {path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_path = path
        self.image_rgb = image_rgb
        self.viewport.set_image(image_rgb)
        self.file_label.setText(str(path))
        self.status.showMessage(f"Image chargée: {path}")
        self.clear_points()
        self.mask = None
        self.save_button.setEnabled(False)

    def add_point(self, x: int, y: int):
        self.points.append((x, y))
        self._refresh_points()

    def clear_points(self):
        self.points.clear()
        self._refresh_points()

    def _refresh_points(self):
        self.point_list.clear()
        for idx, (x, y) in enumerate(self.points, start=1):
            item = QListWidgetItem(f"{idx}. ({x}, {y})")
            self.point_list.addItem(item)
        self.viewport.set_points(self.points)

    def _on_opacity_changed(self, value: int):
        self.viewport.set_overlay_alpha(value / 100.0)

    def run_segmentation(self):
        if self.image_path is None or self.image_rgb is None:
            QMessageBox.warning(self, "Avertissement", "Chargez une image avant de segmenter.")
            return
        if not self.points:
            QMessageBox.warning(self, "Avertissement", "Ajoutez au moins un point.")
            return

        try:
            validate_points(self.points)
        except ValueError as exc:
            QMessageBox.critical(self, "Points invalides", str(exc))
            return

        device = self.current_device if self.current_device != "auto" else "cuda"

        job = SegmentJob(image_path=self.image_path, points=list(self.points), device=device)
        self.run_button.setEnabled(False)
        self.status.showMessage("Préparation de la segmentation...")

        worker = SegmentationWorker(self.session, job)
        worker.mask_ready.connect(self._on_mask_ready)
        worker.error.connect(self._on_error)
        worker.status.connect(self.status.showMessage)
        worker.finished.connect(lambda: self.run_button.setEnabled(True))
        worker.start()
        self.worker = worker

    def _on_mask_ready(self, mask: np.ndarray, score: float):
        self.mask = mask
        self.viewport.set_mask(mask)
        self.save_button.setEnabled(True)
        self.status.showMessage(f"Masque prêt (score {score:.3f})")

    def _on_error(self, message: str):
        QMessageBox.critical(self, "Erreur", message)
        self.status.showMessage("Erreur: " + message)

    def save_mask(self):
        if self.mask is None:
            QMessageBox.warning(self, "Avertissement", "Aucun masque à enregistrer.")
            return

        default_name = self.image_path.stem + "_mask.png" if self.image_path else "mask.png"
        file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le masque", default_name, "PNG (*.png)")
        if not file_path:
            return

        mask_uint8 = (self.mask * 255).astype(np.uint8)
        cv2.imwrite(file_path, mask_uint8)
        self.status.showMessage(f"Masque sauvegardé: {file_path}")

    def reset_state(self):
        self.image_path = None
        self.image_rgb = None
        self.mask = None
        self.points.clear()
        self.file_label.setText("Aucune image")
        self.point_list.clear()
        self.viewport.clear()
        self.save_button.setEnabled(False)
        self.status.showMessage("Réinitialisé")


def main():
    app = QApplication(sys.argv)
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
