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
import numpy as np
import cv2
from pathlib import Path


def load_sam3(device="cuda"):
    """Charge SAM3 - version la plus simple possible."""
    print("Loading SAM3...")

    try:
        from sam3 import sam3_model_registry, Sam3ImagePredictor

        # Modèle le plus petit pour commencer
        model_type = "sam3_hiera_small"

        print(f"  Model: {model_type}")
        sam_checkpoint = sam3_model_registry[model_type]()
        predictor = Sam3ImagePredictor(sam_checkpoint)

        # Move to device
        predictor.model.to(device=device)
        predictor.model.eval()

        print(f"✓ SAM3 loaded on {device}")
        return predictor

    except ImportError as e:
        print(f"✗ SAM3 not installed!")
        print(f"  Install: pip install git+https://github.com/facebookresearch/sam3.git")
        print(f"  Auth: huggingface-cli login")
        print(f"  Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ SAM3 loading failed: {e}")
        sys.exit(1)


def segment_image(predictor, image_path, points):
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


def save_mask(mask, output_path):
    """Sauvegarde le masque en PNG."""
    print(f"\nSaving mask: {output_path}")

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
        points = []
        for pt_str in args.points:
            x, y = map(int, pt_str.split(","))
            points.append((x, y))
    except:
        print("✗ Invalid points format. Use: x,y (e.g., 100,100)")
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
    save_mask(mask, args.output)

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
