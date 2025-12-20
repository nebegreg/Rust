#!/usr/bin/env python3
"""
Ultimate Rotoscopy - Version Minimale qui MARCHE
================================================

UN SEUL OBJECTIF: Segmenter une image avec SAM3 et sauvegarder le masque.
C'est TOUT. Rien d'autre.

Usage:
    python roto.py image.jpg 100,100 200,200 --output mask.png

Requirements:
    pip install git+https://github.com/facebookresearch/sam3.git
    pip install opencv-python numpy pillow torch
"""

import sys
import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch


def load_sam3(device: str = "cuda"):
    """Charge SAM3 - version la plus simple possible."""
    print("Loading SAM3...")

    try:
        from sam3 import Sam3ImagePredictor, sam3_model_registry

        # Modèle le plus petit pour commencer
        model_type = "sam3_hiera_small"

        requested_device = device
        if device == "cuda" and not torch.cuda.is_available():
            print("  ⚠️ CUDA non disponible, bascule sur CPU")
            requested_device = "cpu"

        print(f"  Model: {model_type}")
        sam_checkpoint = sam3_model_registry[model_type]()
        predictor = Sam3ImagePredictor(sam_checkpoint)

        # Move to device
        predictor.model.to(device=requested_device)
        predictor.model.eval()

        print(f"✓ SAM3 loaded on {requested_device}")
        return predictor

    except ImportError as e:
        print("✗ SAM3 not installed!")
        print("  Install: pip install git+https://github.com/facebookresearch/sam3.git")
        print("  Auth: huggingface-cli login")
        print(f"  Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ SAM3 loading failed: {e}")
        sys.exit(1)


def validate_points(points: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """S'assure que les points sont des tuples d'entiers positifs."""

    validated: List[Tuple[int, int]] = []
    for idx, (x, y) in enumerate(points, start=1):
        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError(f"Point {idx} doit contenir des entiers (reçu: {x},{y})")
        if x < 0 or y < 0:
            raise ValueError(f"Point {idx} doit être positif (reçu: {x},{y})")
        validated.append((x, y))
    return validated


def ensure_points_in_frame(points: Sequence[Tuple[int, int]], width: int, height: int) -> None:
    """Valide que tous les points sont dans l'image."""

    for idx, (x, y) in enumerate(points, start=1):
        if not (0 <= x < width) or not (0 <= y < height):
            raise ValueError(
                f"Point {idx} ({x},{y}) est hors de l'image ({width}x{height})."
            )


def segment_image(predictor, image_path, points: Sequence[Tuple[int, int]]):
    """
    Segmente une image avec des points.

    Args:
        predictor: SAM3 predictor
        image_path: Chemin vers l'image
        points: Liste de tuples (x, y)

    Returns:
        mask: Numpy array (H, W) avec le masque
    """
    print(f"\nProcessing: {image_path}")

    # Charger image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"✗ Cannot load image: {image_path}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"  Image size: {image_rgb.shape[1]}x{image_rgb.shape[0]}")

    ensure_points_in_frame(points, image_rgb.shape[1], image_rgb.shape[0])

    # Préparer points
    point_coords = np.array(points, dtype=np.float32)
    point_labels = np.ones(len(points), dtype=np.int32)  # Tous foreground

    print(f"  Points: {len(points)} foreground points")
    for i, (x, y) in enumerate(points):
        print(f"    Point {i+1}: ({x}, {y})")

    # Encoder image
    print("  Encoding image...")
    predictor.set_image(image_rgb)

    # Prédire masque
    print("  Predicting mask...")
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    # Prendre le meilleur masque
    best_idx = scores.argmax()
    best_mask = masks[best_idx]
    best_score = scores[best_idx]

    print(f"✓ Segmentation done (score: {best_score:.3f})")

    return best_mask


def save_mask(mask, output_path: Path):
    """Sauvegarde le masque en PNG."""
    print(f"\nSaving mask: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convertir en uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Sauvegarder
    cv2.imwrite(str(output_path), mask_uint8)

    print(f"✓ Mask saved ({mask.shape[1]}x{mask.shape[0]})")


def main():
    parser = argparse.ArgumentParser(description="Segment image with SAM3")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("points", nargs="+", help="Points as x,y (e.g., 100,100 200,200)")
    parser.add_argument("--output", "-o", default="mask.png", help="Output mask path")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")

    args = parser.parse_args()

    # Parser les points
    try:
        parsed_points = [tuple(map(int, pt_str.split(","))) for pt_str in args.points]
        points = validate_points(parsed_points)
    except ValueError as exc:
        print(f"✗ Invalid points format. Use: x,y (e.g., 100,100). {exc}")
        sys.exit(1)

    if not points:
        print("✗ Need at least 1 point")
        sys.exit(1)

    # Vérifier image existe
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"✗ Image not found: {image_path}")
        sys.exit(1)

    print("=" * 60)
    print("  Ultimate Rotoscopy - Minimal Version")
    print("=" * 60)

    # 1. Charger SAM3
    predictor = load_sam3(args.device)

    # 2. Segmenter
    mask = segment_image(predictor, image_path, points)

    # 3. Sauvegarder
    save_mask(mask, Path(args.output))

    print("\n" + "=" * 60)
    print("  ✓ SUCCESS")
    print("=" * 60)
    print(f"\nMask saved to: {args.output}")
    print("\nNext steps:")
    print("  - View mask: display mask.png")
    print("  - Try different points if mask is not good")
    print("  - Use --device cpu if no GPU")


if __name__ == "__main__":
    main()
