#!/usr/bin/env python3
"""
Depth Anything V3 - Complete Wrapper
Simple, clean wrapper for depth estimation.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import torch
    from PIL import Image
    import cv2
except ImportError as e:
    print("Missing dependencies!")
    print("Install: pip install torch torchvision Pillow opencv-python numpy xformers")
    sys.exit(1)


@dataclass
class DepthResult:
    """Result from depth estimation."""
    depth: np.ndarray  # (H, W) - Depth map
    confidence: Optional[np.ndarray] = None  # (H, W) - Confidence map
    normals: Optional[np.ndarray] = None  # (H, W, 3) - Normal map

    def get_depth_colormap(self) -> np.ndarray:
        """Get colorized depth map for visualization."""
        # Normalize depth to 0-255
        depth_norm = (self.depth - self.depth.min()) / (self.depth.max() - self.depth.min())
        depth_uint8 = (depth_norm * 255).astype(np.uint8)

        # Apply colormap (turbo is good for depth)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        return depth_colored


class DepthAnything3:
    """
    Depth Anything V3 Processor.

    Metric depth estimation with high accuracy.
    """

    def __init__(self, device: str = "cuda", model_size: str = "base"):
        """
        Initialize Depth Anything V3.

        Args:
            device: 'cuda' or 'cpu'
            model_size: 'base' or 'large'
        """
        self.device = device
        self.model_size = model_size
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load DA3 model."""
        print(f"Loading Depth Anything V3 ({self.model_size})...")

        try:
            from depth_anything_3.api import DepthAnything3 as DA3API

            # Model names
            model_names = {
                "base": "depth-anything/DA3-BASE",
                "large": "depth-anything/DA3-LARGE",
            }

            model_name = model_names.get(self.model_size, model_names["base"])

            # Load model
            self.model = DA3API.from_pretrained(model_name)

            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(device=self.device)
                print(f"✓ Depth Anything V3 loaded on {self.device}")
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                self.model = self.model.to(device=self.device)
                print(f"✓ Depth Anything V3 loaded on {self.device}")

        except ImportError as e:
            print(f"✗ Depth Anything V3 not installed!")
            print(f"  Install: pip install xformers torch>=2 torchvision")
            print(f"  Then: git clone https://github.com/ByteDance-Seed/Depth-Anything-3")
            print(f"        cd Depth-Anything-3 && pip install -e .")
            print(f"  Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ DA3 loading failed: {e}")
            sys.exit(1)

    def estimate_depth(
        self,
        image_path: Path,
        use_ray_pose: bool = False
    ) -> DepthResult:
        """
        Estimate depth from image.

        Args:
            image_path: Path to input image
            use_ray_pose: Use ray-based pose estimation (slower but more accurate)

        Returns:
            DepthResult with depth map and confidence
        """
        print(f"\n[DEPTH ESTIMATION] Processing: {image_path}")

        # Load image
        image = Image.open(str(image_path))

        # Run inference
        prediction = self.model.inference(
            [str(image_path)],
            use_ray_pose=use_ray_pose
        )

        # Extract results
        depth = prediction.depth[0]  # First image

        # Get confidence if available
        confidence = None
        if hasattr(prediction, 'conf') and prediction.conf is not None:
            confidence = prediction.conf[0]

        print(f"✓ Depth map generated: {depth.shape}")
        print(f"  Depth range: {depth.min():.3f} - {depth.max():.3f}")

        if confidence is not None:
            print(f"  Mean confidence: {confidence.mean():.3f}")

        return DepthResult(
            depth=depth,
            confidence=confidence
        )


def main():
    """Test Depth Anything V3."""
    import argparse

    parser = argparse.ArgumentParser(description="Depth Anything V3 - Depth Estimation")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("-o", "--output", type=str, default="depth_output", help="Output directory")
    parser.add_argument("--model", type=str, default="base", choices=["base", "large"], help="Model size")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--ray-pose", action="store_true", help="Use ray-based pose (slower, more accurate)")

    args = parser.parse_args()

    # Check input
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize DA3
    da3 = DepthAnything3(device=args.device, model_size=args.model)

    # Estimate depth
    result = da3.estimate_depth(image_path, use_ray_pose=args.ray_pose)

    # Save depth map (raw)
    depth_path = output_dir / f"{image_path.stem}_depth.npy"
    np.save(depth_path, result.depth)
    print(f"\n✓ Saved depth map: {depth_path}")

    # Save depth visualization
    depth_colored = result.get_depth_colormap()
    depth_vis_path = output_dir / f"{image_path.stem}_depth_vis.png"
    cv2.imwrite(str(depth_vis_path), cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved depth visualization: {depth_vis_path}")

    # Save confidence map if available
    if result.confidence is not None:
        conf_path = output_dir / f"{image_path.stem}_confidence.npy"
        np.save(conf_path, result.confidence)

        # Visualize confidence
        conf_norm = (result.confidence * 255).astype(np.uint8)
        conf_vis_path = output_dir / f"{image_path.stem}_confidence_vis.png"
        cv2.imwrite(str(conf_vis_path), conf_norm)
        print(f"✓ Saved confidence map: {conf_path}")

    print(f"\n🎉 Depth estimation complete!")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
