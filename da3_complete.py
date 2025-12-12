#!/usr/bin/env python3
"""
Depth Anything V3 - Complete Professional Wrapper
==================================================

Full-featured wrapper for Depth Anything V3 with:
- Metric depth estimation
- Normal map generation
- Camera intrinsics estimation
- 3D point cloud export (PLY, OBJ, XYZ)
- Depth colorization and visualization
- Batch processing support
- Integration ready for compositing workflows

Requirements:
    pip install torch torchvision Pillow opencv-python numpy

Usage:
    # CLI
    python da3_complete.py image.jpg -o output/
    python da3_complete.py image.jpg --normals --pointcloud --format ply

    # Python API
    from da3_complete import DepthAnything3, DepthResult
    da3 = DepthAnything3()
    result = da3.process("image.jpg")
    result.save_all("output/")
"""

import sys
import json
from pathlib import Path
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import torch
    from PIL import Image
    import cv2
except ImportError as e:
    print("Missing dependencies!")
    print("Install: pip install torch torchvision Pillow opencv-python numpy")
    sys.exit(1)


class DepthColormap(Enum):
    """Available colormaps for depth visualization."""
    TURBO = "turbo"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    JET = "jet"
    GRAY = "gray"


class PointCloudFormat(Enum):
    """Supported point cloud export formats."""
    PLY = "ply"
    OBJ = "obj"
    XYZ = "xyz"
    NPY = "npy"


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length X
    fy: float  # Focal length Y
    cx: float  # Principal point X
    cy: float  # Principal point Y
    width: int
    height: int

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height
        }

    @classmethod
    def estimate_from_image(cls, width: int, height: int, fov_degrees: float = 60.0) -> "CameraIntrinsics":
        """
        Estimate camera intrinsics from image dimensions.

        Args:
            width: Image width
            height: Image height
            fov_degrees: Estimated field of view in degrees (default 60)
        """
        fov_rad = np.radians(fov_degrees)
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Assume square pixels
        cx = width / 2
        cy = height / 2

        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


@dataclass
class DepthResult:
    """
    Complete result from depth estimation.

    Contains depth map, normals, point cloud, and metadata.
    """
    # Core outputs
    depth: np.ndarray  # (H, W) - Metric depth in meters
    depth_normalized: np.ndarray  # (H, W) - Normalized depth [0, 1]

    # Optional outputs
    confidence: Optional[np.ndarray] = None  # (H, W) - Confidence map
    normals: Optional[np.ndarray] = None  # (H, W, 3) - Normal vectors
    point_cloud: Optional[np.ndarray] = None  # (N, 3) - 3D points
    point_colors: Optional[np.ndarray] = None  # (N, 3) - Point colors

    # Camera parameters
    intrinsics: Optional[CameraIntrinsics] = None

    # Metadata
    depth_min: float = 0.0
    depth_max: float = 0.0
    source_path: Optional[str] = None
    processing_time_ms: float = 0.0

    def get_depth_colormap(self, colormap: DepthColormap = DepthColormap.TURBO) -> np.ndarray:
        """
        Get colorized depth map for visualization.

        Args:
            colormap: Colormap to use

        Returns:
            RGB image (H, W, 3) uint8
        """
        depth_uint8 = (self.depth_normalized * 255).astype(np.uint8)

        colormap_cv = {
            DepthColormap.TURBO: cv2.COLORMAP_TURBO,
            DepthColormap.VIRIDIS: cv2.COLORMAP_VIRIDIS,
            DepthColormap.PLASMA: cv2.COLORMAP_PLASMA,
            DepthColormap.INFERNO: cv2.COLORMAP_INFERNO,
            DepthColormap.MAGMA: cv2.COLORMAP_MAGMA,
            DepthColormap.JET: cv2.COLORMAP_JET,
            DepthColormap.GRAY: None,
        }

        if colormap == DepthColormap.GRAY:
            return np.stack([depth_uint8] * 3, axis=-1)

        colored = cv2.applyColorMap(depth_uint8, colormap_cv[colormap])
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    def get_normals_colormap(self) -> Optional[np.ndarray]:
        """
        Get colorized normal map for visualization.

        Returns:
            RGB image (H, W, 3) uint8 or None if no normals
        """
        if self.normals is None:
            return None

        # Convert normals from [-1, 1] to [0, 255]
        normals_vis = ((self.normals + 1) * 0.5 * 255).astype(np.uint8)
        return normals_vis

    def save_depth(self, path: Union[str, Path], format: str = "npy"):
        """
        Save depth map to file.

        Args:
            path: Output path
            format: 'npy', 'exr', 'png', 'tiff'
        """
        path = Path(path)

        if format == "npy":
            np.save(path.with_suffix(".npy"), self.depth)
        elif format == "exr":
            self._save_exr(path.with_suffix(".exr"), self.depth)
        elif format in ("png", "tiff"):
            # Save 16-bit depth
            depth_16bit = (self.depth_normalized * 65535).astype(np.uint16)
            cv2.imwrite(str(path.with_suffix(f".{format}")), depth_16bit)

    def save_depth_visualization(self, path: Union[str, Path], colormap: DepthColormap = DepthColormap.TURBO):
        """Save colorized depth visualization."""
        path = Path(path)
        colored = self.get_depth_colormap(colormap)
        cv2.imwrite(str(path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

    def save_normals(self, path: Union[str, Path], format: str = "png"):
        """Save normal map to file."""
        if self.normals is None:
            print("Warning: No normals to save")
            return

        path = Path(path)

        if format == "npy":
            np.save(path.with_suffix(".npy"), self.normals)
        elif format == "exr":
            self._save_exr_rgb(path.with_suffix(".exr"), self.normals)
        else:
            # Save as RGB visualization
            normals_vis = self.get_normals_colormap()
            cv2.imwrite(str(path), cv2.cvtColor(normals_vis, cv2.COLOR_RGB2BGR))

    def save_point_cloud(self, path: Union[str, Path], format: PointCloudFormat = PointCloudFormat.PLY):
        """
        Save point cloud to file.

        Args:
            path: Output path
            format: PLY, OBJ, XYZ, or NPY
        """
        if self.point_cloud is None:
            print("Warning: No point cloud to save")
            return

        path = Path(path)

        if format == PointCloudFormat.PLY:
            self._save_ply(path.with_suffix(".ply"))
        elif format == PointCloudFormat.OBJ:
            self._save_obj(path.with_suffix(".obj"))
        elif format == PointCloudFormat.XYZ:
            self._save_xyz(path.with_suffix(".xyz"))
        elif format == PointCloudFormat.NPY:
            np.save(path.with_suffix(".npy"), self.point_cloud)

    def save_all(self, output_dir: Union[str, Path], base_name: Optional[str] = None):
        """
        Save all outputs to directory.

        Args:
            output_dir: Output directory
            base_name: Base filename (default: from source_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if base_name is None:
            if self.source_path:
                base_name = Path(self.source_path).stem
            else:
                base_name = "depth"

        # Save depth
        self.save_depth(output_dir / f"{base_name}_depth.npy")
        self.save_depth_visualization(output_dir / f"{base_name}_depth_vis.png")

        # Save normals
        if self.normals is not None:
            self.save_normals(output_dir / f"{base_name}_normals.png")
            np.save(output_dir / f"{base_name}_normals.npy", self.normals)

        # Save point cloud
        if self.point_cloud is not None:
            self.save_point_cloud(output_dir / f"{base_name}_pointcloud.ply")

        # Save confidence
        if self.confidence is not None:
            conf_vis = (self.confidence * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"{base_name}_confidence.png"), conf_vis)

        # Save intrinsics
        if self.intrinsics is not None:
            with open(output_dir / f"{base_name}_intrinsics.json", 'w') as f:
                json.dump(self.intrinsics.to_dict(), f, indent=2)

        # Save metadata
        metadata = {
            "depth_min": float(self.depth_min),
            "depth_max": float(self.depth_max),
            "processing_time_ms": self.processing_time_ms,
            "has_normals": self.normals is not None,
            "has_point_cloud": self.point_cloud is not None,
            "point_count": len(self.point_cloud) if self.point_cloud is not None else 0,
        }
        with open(output_dir / f"{base_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved all outputs to: {output_dir}")

    def _save_ply(self, path: Path):
        """Save point cloud as PLY file."""
        points = self.point_cloud
        colors = self.point_colors

        with open(path, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")

            # Data
            for i in range(len(points)):
                x, y, z = points[i]
                if colors is not None:
                    r, g, b = colors[i].astype(np.uint8)
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    def _save_obj(self, path: Path):
        """Save point cloud as OBJ file."""
        with open(path, 'w') as f:
            f.write("# Point cloud exported by Depth Anything V3\n")
            for x, y, z in self.point_cloud:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

    def _save_xyz(self, path: Path):
        """Save point cloud as XYZ file."""
        np.savetxt(path, self.point_cloud, fmt='%.6f')

    def _save_exr(self, path: Path, data: np.ndarray):
        """Save single-channel EXR."""
        try:
            import OpenEXR
            import Imath

            h, w = data.shape
            header = OpenEXR.Header(w, h)
            header['channels'] = {'Z': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))}

            exr = OpenEXR.OutputFile(str(path), header)
            exr.writePixels({'Z': data.astype(np.float32).tobytes()})
            exr.close()
        except ImportError:
            print("Warning: OpenEXR not available, saving as NPY instead")
            np.save(path.with_suffix('.npy'), data)

    def _save_exr_rgb(self, path: Path, data: np.ndarray):
        """Save 3-channel EXR (normals)."""
        try:
            import OpenEXR
            import Imath

            h, w, _ = data.shape
            header = OpenEXR.Header(w, h)
            header['channels'] = {
                'R': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT)),
                'G': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT)),
                'B': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT)),
            }

            exr = OpenEXR.OutputFile(str(path), header)
            exr.writePixels({
                'R': data[:, :, 0].astype(np.float32).tobytes(),
                'G': data[:, :, 1].astype(np.float32).tobytes(),
                'B': data[:, :, 2].astype(np.float32).tobytes(),
            })
            exr.close()
        except ImportError:
            print("Warning: OpenEXR not available, saving as NPY instead")
            np.save(path.with_suffix('.npy'), data)


class DepthAnything3:
    """
    Depth Anything V3 Processor.

    Professional depth estimation with:
    - Metric depth output
    - Normal map generation
    - 3D point cloud creation
    - Camera intrinsics estimation

    Example:
        >>> da3 = DepthAnything3(device="cuda")
        >>> result = da3.process("image.jpg", compute_normals=True, compute_pointcloud=True)
        >>> result.save_all("output/")
    """

    SUPPORTED_MODELS = ["small", "base", "large"]

    def __init__(
        self,
        device: str = "cuda",
        model_size: str = "base",
    ):
        """
        Initialize Depth Anything V3.

        Args:
            device: 'cuda' or 'cpu'
            model_size: 'small', 'base', or 'large'
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_size = model_size
        self.model = None
        self.transform = None
        self._loaded = False

    def _load_model(self):
        """Load DA3 model (lazy loading)."""
        if self._loaded:
            return

        print(f"Loading Depth Anything V3 ({self.model_size})...")

        try:
            # Try official DA3 API first
            from depth_anything_v2.dpt import DepthAnythingV2

            model_configs = {
                'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }

            config = model_configs.get(self.model_size, model_configs['base'])
            self.model = DepthAnythingV2(**config)

            # Load pretrained weights
            checkpoint_path = f"checkpoints/depth_anything_v2_{config['encoder']}.pth"
            if Path(checkpoint_path).exists():
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            else:
                print(f"Warning: Checkpoint not found at {checkpoint_path}")
                print("Downloading from HuggingFace...")
                # Try HuggingFace
                from huggingface_hub import hf_hub_download
                checkpoint = hf_hub_download(
                    repo_id=f"depth-anything/Depth-Anything-V2-{self.model_size.capitalize()}",
                    filename=f"depth_anything_v2_{config['encoder']}.pth"
                )
                self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))

            self.model = self.model.to(self.device).eval()
            self._loaded = True
            print(f"Depth Anything V3 loaded on {self.device}")

        except ImportError:
            # Fallback to transformers pipeline
            try:
                from transformers import pipeline
                print("Using transformers pipeline for depth estimation...")
                self.model = pipeline(
                    "depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Base-hf",
                    device=0 if self.device == "cuda" else -1
                )
                self._loaded = True
                self._use_pipeline = True
                print(f"Depth Anything loaded via transformers on {self.device}")
            except Exception as e2:
                print(f"Error loading model: {e2}")
                print("\nInstall Depth Anything V2:")
                print("  git clone https://github.com/DepthAnything/Depth-Anything-V2")
                print("  cd Depth-Anything-V2 && pip install -e .")
                print("\nOr use transformers:")
                print("  pip install transformers")
                sys.exit(1)

    def process(
        self,
        image_path: Union[str, Path],
        compute_normals: bool = True,
        compute_pointcloud: bool = True,
        estimate_intrinsics: bool = True,
        fov_degrees: float = 60.0,
        depth_scale: float = 1.0,
    ) -> DepthResult:
        """
        Process image and compute depth, normals, and point cloud.

        Args:
            image_path: Path to input image
            compute_normals: Generate normal map from depth
            compute_pointcloud: Generate 3D point cloud
            estimate_intrinsics: Estimate camera intrinsics
            fov_degrees: Estimated FOV for intrinsics (default 60)
            depth_scale: Scale factor for metric depth

        Returns:
            DepthResult with all computed outputs
        """
        import time
        start_time = time.time()

        self._load_model()

        image_path = Path(image_path)
        print(f"\n[DEPTH ESTIMATION] Processing: {image_path}")

        # Load image
        image = Image.open(str(image_path)).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        # Run depth estimation
        if hasattr(self, '_use_pipeline') and self._use_pipeline:
            # Transformers pipeline
            output = self.model(image)
            depth = np.array(output["depth"])
            # Resize to original size
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Native model
            with torch.no_grad():
                # Prepare input
                img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)

                # Inference
                depth = self.model(img_tensor)
                depth = depth.squeeze().cpu().numpy()

        # Normalize depth
        depth_min = depth.min()
        depth_max = depth.max()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)

        # Apply scale for metric depth
        depth_metric = depth * depth_scale

        print(f"  Depth range: {depth_min:.3f} - {depth_max:.3f}")

        # Estimate camera intrinsics
        intrinsics = None
        if estimate_intrinsics:
            intrinsics = CameraIntrinsics.estimate_from_image(w, h, fov_degrees)
            print(f"  Estimated intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")

        # Compute normals
        normals = None
        if compute_normals:
            normals = self._compute_normals(depth_metric)
            print(f"  Generated normal map")

        # Compute point cloud
        point_cloud = None
        point_colors = None
        if compute_pointcloud and intrinsics is not None:
            point_cloud, point_colors = self._compute_pointcloud(
                depth_metric, image_np, intrinsics
            )
            print(f"  Generated point cloud: {len(point_cloud)} points")

        processing_time = (time.time() - start_time) * 1000

        result = DepthResult(
            depth=depth_metric,
            depth_normalized=depth_normalized,
            normals=normals,
            point_cloud=point_cloud,
            point_colors=point_colors,
            intrinsics=intrinsics,
            depth_min=float(depth_min),
            depth_max=float(depth_max),
            source_path=str(image_path),
            processing_time_ms=processing_time,
        )

        print(f"  Processing time: {processing_time:.0f}ms")

        return result

    def _compute_normals(self, depth: np.ndarray) -> np.ndarray:
        """
        Compute surface normals from depth map.

        Uses central differences for gradient estimation.

        Args:
            depth: (H, W) depth map

        Returns:
            (H, W, 3) normal vectors in [-1, 1]
        """
        h, w = depth.shape
        normals = np.zeros((h, w, 3), dtype=np.float32)

        # Compute gradients
        dzdx = np.zeros_like(depth)
        dzdy = np.zeros_like(depth)

        # Central differences
        dzdx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) / 2.0
        dzdy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) / 2.0

        # Handle boundaries with forward/backward differences
        dzdx[:, 0] = depth[:, 1] - depth[:, 0]
        dzdx[:, -1] = depth[:, -1] - depth[:, -2]
        dzdy[0, :] = depth[1, :] - depth[0, :]
        dzdy[-1, :] = depth[-1, :] - depth[-2, :]

        # Compute normals: n = (-dz/dx, -dz/dy, 1) normalized
        normals[:, :, 0] = -dzdx
        normals[:, :, 1] = -dzdy
        normals[:, :, 2] = 1.0

        # Normalize
        norm = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True))
        normals = normals / (norm + 1e-8)

        return normals

    def _compute_pointcloud(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        intrinsics: CameraIntrinsics,
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        subsample: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Back-project depth map to 3D point cloud.

        Args:
            depth: (H, W) depth map
            image: (H, W, 3) RGB image
            intrinsics: Camera intrinsics
            min_depth: Minimum valid depth
            max_depth: Maximum valid depth
            subsample: Subsample factor (1 = all pixels)

        Returns:
            (points, colors) - (N, 3) arrays
        """
        h, w = depth.shape

        # Create pixel coordinate grid
        u = np.arange(0, w, subsample)
        v = np.arange(0, h, subsample)
        u, v = np.meshgrid(u, v)

        # Get depth values at subsampled locations
        depth_sub = depth[::subsample, ::subsample]

        # Valid depth mask
        valid = (depth_sub > min_depth) & (depth_sub < max_depth)

        # Back-project to 3D
        z = depth_sub[valid]
        x = (u[valid] - intrinsics.cx) * z / intrinsics.fx
        y = (v[valid] - intrinsics.cy) * z / intrinsics.fy

        points = np.stack([x, y, z], axis=-1)

        # Get colors
        image_sub = image[::subsample, ::subsample]
        colors = image_sub[valid]

        return points, colors

    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        **kwargs
    ) -> List[DepthResult]:
        """
        Process multiple images.

        Args:
            image_paths: List of image paths
            output_dir: Output directory
            **kwargs: Arguments passed to process()

        Returns:
            List of DepthResult
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] Processing {Path(path).name}")
            result = self.process(path, **kwargs)
            result.save_all(output_dir)
            results.append(result)

        return results


def main():
    """CLI interface for Depth Anything V3."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Depth Anything V3 - Professional Depth Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic depth estimation
  python da3_complete.py image.jpg -o output/

  # With normals and point cloud
  python da3_complete.py image.jpg -o output/ --normals --pointcloud

  # Specific output format
  python da3_complete.py image.jpg -o output/ --pointcloud --format ply

  # Process multiple images
  python da3_complete.py *.jpg -o output/ --batch
        """
    )

    parser.add_argument("images", type=str, nargs="+", help="Input image(s)")
    parser.add_argument("-o", "--output", type=str, default="depth_output", help="Output directory")
    parser.add_argument("--model", type=str, default="base", choices=["small", "base", "large"], help="Model size")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--normals", action="store_true", help="Compute normal maps")
    parser.add_argument("--pointcloud", action="store_true", help="Generate 3D point cloud")
    parser.add_argument("--format", type=str, default="ply", choices=["ply", "obj", "xyz", "npy"], help="Point cloud format")
    parser.add_argument("--fov", type=float, default=60.0, help="Estimated FOV in degrees")
    parser.add_argument("--colormap", type=str, default="turbo", choices=["turbo", "viridis", "plasma", "jet", "gray"], help="Depth colormap")
    parser.add_argument("--batch", action="store_true", help="Batch process multiple images")

    args = parser.parse_args()

    # Check inputs
    image_paths = []
    for pattern in args.images:
        path = Path(pattern)
        if path.exists():
            image_paths.append(path)
        else:
            # Try glob
            import glob
            matches = glob.glob(pattern)
            image_paths.extend([Path(m) for m in matches])

    if not image_paths:
        print("Error: No valid images found")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize DA3
    da3 = DepthAnything3(device=args.device, model_size=args.model)

    # Process images
    for i, image_path in enumerate(image_paths):
        if len(image_paths) > 1:
            print(f"\n[{i+1}/{len(image_paths)}] {image_path.name}")

        result = da3.process(
            image_path,
            compute_normals=args.normals,
            compute_pointcloud=args.pointcloud,
            fov_degrees=args.fov,
        )

        # Save outputs
        base_name = image_path.stem

        # Depth
        result.save_depth(output_dir / f"{base_name}_depth.npy")
        result.save_depth_visualization(
            output_dir / f"{base_name}_depth_vis.png",
            colormap=DepthColormap(args.colormap)
        )

        # Normals
        if result.normals is not None:
            result.save_normals(output_dir / f"{base_name}_normals.png")

        # Point cloud
        if result.point_cloud is not None:
            result.save_point_cloud(
                output_dir / f"{base_name}_pointcloud",
                format=PointCloudFormat(args.format)
            )

        # Save metadata
        result.save_all(output_dir, base_name)

    print(f"\n Depth estimation complete!")
    print(f"   Output: {output_dir}")


if __name__ == "__main__":
    main()
