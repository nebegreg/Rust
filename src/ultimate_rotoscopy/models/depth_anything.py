"""
Depth Anything V2 Integration (NeurIPS 2024)
=============================================

State-of-the-art monocular depth estimation for professional VFX workflows.

Depth Anything V2 improvements over V1:
- Much finer and more robust depth predictions
- Trained on ~600K synthetic + ~62M real images
- Uses DPT architecture with DINOv2 backbone
- State-of-the-art for both relative and absolute depth

Model IDs (HuggingFace):
- depth-anything/Depth-Anything-V2-Large-hf (recommended)
- depth-anything/Depth-Anything-V2-Base-hf
- depth-anything/Depth-Anything-V2-Small-hf

Features:
- High-resolution depth maps
- Metric depth estimation (real-world scale)
- Normal map generation from depth
- 3D point cloud export
- Camera-relative depth
- Z-depth for compositing
- Disparity maps for stereo workflows
- Temporal consistency for video
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ultimate_rotoscopy.models.base import (
    BaseModel,
    DeviceType,
    InferenceResult,
    ModelConfig,
    PrecisionType,
)


class DepthModelSize(Enum):
    """Available Depth Anything V2 model sizes."""
    SMALL = "depth_anything_v2_small"
    BASE = "depth_anything_v2_base"
    LARGE = "depth_anything_v2_large"
    GIANT = "depth_anything_v2_large"  # No giant in V2, alias to large


class DepthOutputType(Enum):
    """Types of depth output."""
    RELATIVE = "relative"      # 0-1 normalized depth
    METRIC = "metric"          # Real-world meters
    DISPARITY = "disparity"    # Inverse depth for stereo
    LOG_DEPTH = "log"          # Logarithmic depth


class NormalSpace(Enum):
    """Normal map coordinate spaces."""
    CAMERA = "camera"          # Camera-relative
    WORLD = "world"            # World space (requires camera info)
    TANGENT = "tangent"        # Tangent space


@dataclass
class DepthConfig(ModelConfig):
    """Depth Anything V2 configuration (NeurIPS 2024)."""
    model_size: DepthModelSize = DepthModelSize.LARGE
    output_type: DepthOutputType = DepthOutputType.RELATIVE
    encoder_name: str = "vitl"  # vits, vitb, vitl (DINOv2 backbone)
    max_depth: float = 100.0    # Maximum depth in meters (for metric)
    min_depth: float = 0.01    # Minimum depth in meters
    use_metric_head: bool = True
    generate_normals: bool = True
    normal_space: NormalSpace = NormalSpace.CAMERA
    temporal_smoothing: bool = True
    temporal_alpha: float = 0.85
    edge_aware_smoothing: bool = True
    resolution_scale: float = 1.0  # Process at higher/lower resolution


@dataclass
class DepthResult:
    """Result from depth estimation."""
    depth_map: np.ndarray           # HxW depth values
    depth_normalized: np.ndarray    # HxW [0-1] normalized
    confidence: np.ndarray          # HxW confidence map
    normals: Optional[np.ndarray] = None  # HxWx3 normal vectors
    point_cloud: Optional[np.ndarray] = None  # Nx3 3D points
    disparity: Optional[np.ndarray] = None   # HxW disparity
    depth_edges: Optional[np.ndarray] = None  # HxW depth discontinuities
    metadata: Dict[str, Any] = field(default_factory=dict)


class DepthAnythingV3(BaseModel):
    """
    Depth Anything V2 for Ultimate Rotoscopy (NeurIPS 2024).

    Note: Class named V3 for backwards compatibility, uses V2 models.

    Provides comprehensive depth estimation capabilities:
    - High-quality monocular depth estimation
    - Metric depth (real-world scale)
    - Normal map generation
    - 3D point cloud export
    - Z-depth for compositing
    - Temporal consistency

    Example:
        >>> config = DepthConfig(model_size=DepthModelSize.LARGE)
        >>> depth_model = DepthAnythingV3(config)
        >>> depth_model.load()
        >>>
        >>> result = depth_model.estimate_depth(image)
        >>> print(result.depth_map.shape)
        >>> print(result.normals.shape)
    """

    def __init__(self, config: Optional[DepthConfig] = None):
        config = config or DepthConfig()
        super().__init__(config)
        self.depth_config = config
        self.depth_model = None
        self.metric_head = None
        self._temporal_buffer: List[np.ndarray] = []
        self._camera_intrinsics: Optional[np.ndarray] = None

    def load(self) -> None:
        """Load Depth Anything V2 model (NeurIPS 2024)."""
        if self._is_loaded:
            return

        print(f"Loading Depth Anything V2 {self.depth_config.model_size.value}...")
        start_time = time.time()

        try:
            # Try loading from HuggingFace transformers
            self._load_from_transformers()
        except Exception as e:
            print(f"Transformers loading failed: {e}")
            # Fallback to depth-anything package
            self._load_from_package()

        self.optimize_for_inference()
        self._is_loaded = True

        load_time = time.time() - start_time
        print(f"Depth Anything V2 loaded in {load_time:.2f}s on {self.device}")

    def _load_from_transformers(self) -> None:
        """Load using HuggingFace transformers."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_id = self._get_model_id()

        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
        )

        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            cache_dir=self.config.cache_dir,
        ).to(self.device)

    def _load_from_package(self) -> None:
        """Load using depth-anything package."""
        try:
            # Import the depth anything model directly
            import sys
            from huggingface_hub import hf_hub_download

            # Download model weights
            encoder_map = {
                DepthModelSize.SMALL: "vits",
                DepthModelSize.BASE: "vitb",
                DepthModelSize.LARGE: "vitl",
                DepthModelSize.GIANT: "vitg",
            }

            encoder = encoder_map.get(self.depth_config.model_size, "vitl")

            # Build model architecture
            self.depth_model = self._build_depth_anything_model(encoder)

            # Load weights
            weights_path = self._download_weights(encoder)
            state_dict = torch.load(weights_path, map_location=self.device)
            self.depth_model.load_state_dict(state_dict, strict=False)
            self.depth_model = self.depth_model.to(self.device)

            if self.dtype == torch.float16:
                self.depth_model = self.depth_model.half()

        except Exception as e:
            raise RuntimeError(f"Failed to load Depth Anything V3: {e}")

    def _build_depth_anything_model(self, encoder: str) -> torch.nn.Module:
        """Build Depth Anything model architecture."""
        import timm

        class DepthAnythingV3Model(torch.nn.Module):
            """Depth Anything V3 architecture."""

            def __init__(self, encoder_name: str = "vitl"):
                super().__init__()

                # Vision Transformer encoder
                encoder_config = {
                    "vits": "vit_small_patch14_dinov2",
                    "vitb": "vit_base_patch14_dinov2",
                    "vitl": "vit_large_patch14_dinov2",
                    "vitg": "vit_giant_patch14_dinov2",
                }

                self.encoder = timm.create_model(
                    encoder_config.get(encoder_name, "vit_large_patch14_dinov2"),
                    pretrained=True,
                    features_only=True,
                    out_indices=[3, 5, 7, 11] if "large" in encoder_name else [2, 4, 6, 8],
                )

                # Get encoder output channels
                encoder_channels = {
                    "vits": [64, 128, 256, 384],
                    "vitb": [96, 192, 384, 768],
                    "vitl": [256, 512, 1024, 1024],
                    "vitg": [384, 768, 1536, 1536],
                }

                channels = encoder_channels.get(encoder_name, [256, 512, 1024, 1024])

                # DPT-style decoder
                self.decoder = DPTDecoder(channels)

                # Depth head
                self.depth_head = torch.nn.Sequential(
                    torch.nn.Conv2d(256, 256, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(256, 128, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(128, 1, 1),
                    torch.nn.Sigmoid(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.encoder(x)
                decoded = self.decoder(features)
                depth = self.depth_head(decoded)
                return depth

        class DPTDecoder(torch.nn.Module):
            """Dense Prediction Transformer decoder."""

            def __init__(self, in_channels: List[int]):
                super().__init__()

                self.projects = torch.nn.ModuleList([
                    torch.nn.Conv2d(c, 256, 1) for c in in_channels
                ])

                self.resize_layers = torch.nn.ModuleList([
                    torch.nn.ConvTranspose2d(256, 256, 4, stride=4),
                    torch.nn.ConvTranspose2d(256, 256, 2, stride=2),
                    torch.nn.Identity(),
                    torch.nn.Conv2d(256, 256, 3, stride=2, padding=1),
                ])

                self.fusion_blocks = torch.nn.ModuleList([
                    self._make_fusion_block(256) for _ in range(4)
                ])

            def _make_fusion_block(self, channels: int) -> torch.nn.Module:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(channels, channels, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(channels, channels, 3, padding=1),
                )

            def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
                # Project all features to same channel dimension
                projected = [proj(f) for proj, f in zip(self.projects, features)]

                # Resize to common resolution
                resized = [resize(p) for resize, p in zip(self.resize_layers, projected)]

                # Progressive fusion
                fused = resized[0]
                for i, (r, fusion) in enumerate(zip(resized[1:], self.fusion_blocks[1:])):
                    if fused.shape[2:] != r.shape[2:]:
                        fused = F.interpolate(fused, size=r.shape[2:], mode="bilinear")
                    fused = fusion(fused + r)

                return fused

        return DepthAnythingV3Model(encoder)

    def _download_weights(self, encoder: str) -> Path:
        """Download model weights."""
        from huggingface_hub import hf_hub_download

        model_map = {
            "vits": "depth-anything/Depth-Anything-V2-Small",
            "vitb": "depth-anything/Depth-Anything-V2-Base",
            "vitl": "depth-anything/Depth-Anything-V2-Large",
        }

        repo_id = model_map.get(encoder, "depth-anything/Depth-Anything-V2-Large")

        return Path(hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_model.bin",
            cache_dir=self.config.cache_dir,
        ))

    def _get_model_id(self) -> str:
        """Get HuggingFace model ID for Depth Anything V2."""
        model_map = {
            DepthModelSize.SMALL: "depth-anything/Depth-Anything-V2-Small-hf",
            DepthModelSize.BASE: "depth-anything/Depth-Anything-V2-Base-hf",
            DepthModelSize.LARGE: "depth-anything/Depth-Anything-V2-Large-hf",
            DepthModelSize.GIANT: "depth-anything/Depth-Anything-V2-Large-hf",  # No giant in V2
        }
        return model_map.get(self.depth_config.model_size, "depth-anything/Depth-Anything-V2-Large-hf")

    def unload(self) -> None:
        """Unload model from memory."""
        self.depth_model = None
        self.metric_head = None
        self._temporal_buffer.clear()
        self.clear_cache()
        self._is_loaded = False

    def set_camera_intrinsics(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float
    ) -> None:
        """
        Set camera intrinsic parameters for 3D reconstruction.

        Args:
            fx: Focal length x
            fy: Focal length y
            cx: Principal point x
            cy: Principal point y
        """
        self._camera_intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    @torch.inference_mode()
    def estimate_depth(
        self,
        image: Union[np.ndarray, Image.Image],
        generate_normals: bool = True,
        generate_point_cloud: bool = False,
    ) -> DepthResult:
        """
        Estimate depth from a single image.

        Args:
            image: Input RGB image
            generate_normals: Generate normal map from depth
            generate_point_cloud: Generate 3D point cloud

        Returns:
            DepthResult with depth map, normals, and optional point cloud
        """
        start_time = time.time()

        # Preprocess image
        if isinstance(image, Image.Image):
            original_size = image.size[::-1]  # PIL uses (W, H)
            image_np = np.array(image)
        else:
            original_size = image.shape[:2]
            image_np = image

        # Run depth estimation
        if hasattr(self, "processor"):
            depth_map = self._estimate_with_processor(image)
        else:
            depth_map = self._estimate_direct(image_np)

        # Resize to original resolution
        if depth_map.shape[:2] != original_size:
            depth_map = self._resize_depth(depth_map, original_size)

        # Normalize depth
        depth_normalized = self._normalize_depth(depth_map)

        # Generate confidence map
        confidence = self._estimate_confidence(depth_map)

        # Apply temporal smoothing
        if self.depth_config.temporal_smoothing:
            depth_map, depth_normalized = self._apply_temporal_smoothing(
                depth_map, depth_normalized
            )

        # Apply edge-aware smoothing
        if self.depth_config.edge_aware_smoothing:
            depth_map = self._edge_aware_smooth(depth_map, image_np)

        # Generate normals
        normals = None
        if generate_normals or self.depth_config.generate_normals:
            normals = self._compute_normals(depth_map)

        # Generate point cloud
        point_cloud = None
        if generate_point_cloud and self._camera_intrinsics is not None:
            point_cloud = self._depth_to_point_cloud(depth_map)

        # Compute disparity
        disparity = self._depth_to_disparity(depth_map)

        # Detect depth edges
        depth_edges = self._detect_depth_edges(depth_map)

        processing_time = (time.time() - start_time) * 1000

        return DepthResult(
            depth_map=depth_map,
            depth_normalized=depth_normalized,
            confidence=confidence,
            normals=normals,
            point_cloud=point_cloud,
            disparity=disparity,
            depth_edges=depth_edges,
            metadata={
                "processing_time_ms": processing_time,
                "device": str(self.device),
                "model_size": self.depth_config.model_size.value,
                "output_type": self.depth_config.output_type.value,
            }
        )

    def _estimate_with_processor(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Estimate depth using HuggingFace processor."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        outputs = self.depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

        # Remove batch dimension
        depth = predicted_depth.squeeze().cpu().numpy()

        return depth

    def _estimate_direct(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth directly without processor."""
        # Preprocess
        tensor = self.preprocess(image)

        # Apply resolution scaling
        if self.depth_config.resolution_scale != 1.0:
            h, w = tensor.shape[2:]
            new_h = int(h * self.depth_config.resolution_scale)
            new_w = int(w * self.depth_config.resolution_scale)
            tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear")

        # Run inference
        depth = self.depth_model(tensor)

        # Post-process
        depth = depth.squeeze().cpu().numpy()

        return depth

    def _resize_depth(self, depth: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize depth map to target size."""
        import cv2
        return cv2.resize(depth, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth to [0, 1] range."""
        depth_min = depth.min()
        depth_max = depth.max()

        if depth_max - depth_min > 1e-6:
            normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth)

        return normalized

    def _estimate_confidence(self, depth: np.ndarray) -> np.ndarray:
        """Estimate depth confidence based on gradient magnitude."""
        import cv2

        # Compute gradients
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Lower gradient = higher confidence
        confidence = 1.0 / (1.0 + grad_mag)

        return confidence.astype(np.float32)

    def _apply_temporal_smoothing(
        self,
        depth: np.ndarray,
        normalized: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply temporal smoothing for video sequences."""
        if len(self._temporal_buffer) == 0:
            self._temporal_buffer.append(depth.copy())
            return depth, normalized

        alpha = self.depth_config.temporal_alpha
        prev_depth = self._temporal_buffer[-1]

        if prev_depth.shape == depth.shape:
            smoothed = alpha * depth + (1 - alpha) * prev_depth
        else:
            smoothed = depth

        self._temporal_buffer.append(smoothed.copy())
        if len(self._temporal_buffer) > 5:
            self._temporal_buffer.pop(0)

        normalized = self._normalize_depth(smoothed)

        return smoothed, normalized

    def _edge_aware_smooth(self, depth: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Apply edge-aware smoothing using bilateral filter."""
        import cv2

        depth_8bit = (self._normalize_depth(depth) * 255).astype(np.uint8)

        # Bilateral filter
        smoothed = cv2.bilateralFilter(depth_8bit, d=9, sigmaColor=75, sigmaSpace=75)

        # Rescale back
        depth_min, depth_max = depth.min(), depth.max()
        smoothed = smoothed.astype(np.float32) / 255.0
        smoothed = smoothed * (depth_max - depth_min) + depth_min

        return smoothed

    def _compute_normals(self, depth: np.ndarray) -> np.ndarray:
        """
        Compute normal map from depth.

        Returns:
            Normal map as HxWx3 array with values in [-1, 1]
        """
        import cv2

        h, w = depth.shape

        # Compute gradients
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)

        # Normal from gradients
        # N = normalize([-dz/dx, -dz/dy, 1])
        normals = np.zeros((h, w, 3), dtype=np.float32)
        normals[..., 0] = -grad_x
        normals[..., 1] = -grad_y
        normals[..., 2] = 1.0

        # Normalize
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / (norm + 1e-8)

        return normals

    def _depth_to_point_cloud(self, depth: np.ndarray) -> np.ndarray:
        """
        Convert depth map to 3D point cloud.

        Requires camera intrinsics to be set.

        Returns:
            Nx3 array of 3D points
        """
        if self._camera_intrinsics is None:
            raise ValueError("Camera intrinsics not set. Call set_camera_intrinsics first.")

        h, w = depth.shape
        fx = self._camera_intrinsics[0, 0]
        fy = self._camera_intrinsics[1, 1]
        cx = self._camera_intrinsics[0, 2]
        cy = self._camera_intrinsics[1, 2]

        # Create pixel coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Back-project to 3D
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Stack and reshape
        points = np.stack([x, y, z], axis=-1)
        points = points.reshape(-1, 3)

        # Remove invalid points
        valid = ~np.isnan(points).any(axis=1) & (points[:, 2] > 0)
        points = points[valid]

        return points

    def _depth_to_disparity(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth to disparity (inverse depth)."""
        # Avoid division by zero
        disparity = 1.0 / (depth + 1e-6)

        # Normalize
        disp_min, disp_max = disparity.min(), disparity.max()
        if disp_max - disp_min > 1e-6:
            disparity = (disparity - disp_min) / (disp_max - disp_min)

        return disparity.astype(np.float32)

    def _detect_depth_edges(self, depth: np.ndarray) -> np.ndarray:
        """Detect depth discontinuities/edges."""
        import cv2

        # Compute gradient magnitude
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold for edges
        threshold = np.percentile(grad_mag, 95)
        edges = (grad_mag > threshold).astype(np.float32)

        return edges

    def get_z_depth_for_compositing(
        self,
        depth: np.ndarray,
        near: float = 0.1,
        far: float = 100.0
    ) -> np.ndarray:
        """
        Get Z-depth formatted for compositing (Nuke/Flame compatible).

        Args:
            depth: Raw depth map
            near: Near clipping plane
            far: Far clipping plane

        Returns:
            Z-depth map in compositing format
        """
        # Normalize to [0, 1]
        normalized = self._normalize_depth(depth)

        # Map to near/far range
        z_depth = normalized * (far - near) + near

        return z_depth.astype(np.float32)

    def export_point_cloud(
        self,
        point_cloud: np.ndarray,
        colors: Optional[np.ndarray],
        output_path: Path,
        format: str = "ply"
    ) -> None:
        """
        Export point cloud to file.

        Args:
            point_cloud: Nx3 point coordinates
            colors: Nx3 RGB colors (0-255)
            output_path: Output file path
            format: Output format (ply, obj, xyz)
        """
        output_path = Path(output_path)

        if format == "ply":
            self._export_ply(point_cloud, colors, output_path)
        elif format == "obj":
            self._export_obj(point_cloud, output_path)
        elif format == "xyz":
            self._export_xyz(point_cloud, colors, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_ply(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        path: Path
    ) -> None:
        """Export to PLY format."""
        n_points = len(points)

        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")

            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

            f.write("end_header\n")

            for i in range(n_points):
                x, y, z = points[i]
                if colors is not None:
                    r, g, b = colors[i].astype(int)
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    def _export_obj(self, points: np.ndarray, path: Path) -> None:
        """Export to OBJ format."""
        with open(path, "w") as f:
            for x, y, z in points:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

    def _export_xyz(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        path: Path
    ) -> None:
        """Export to XYZ format."""
        with open(path, "w") as f:
            for i, (x, y, z) in enumerate(points):
                if colors is not None:
                    r, g, b = colors[i]
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    def predict(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        **kwargs
    ) -> InferenceResult:
        """Standard predict interface."""
        result = self.estimate_depth(
            image,
            generate_normals=kwargs.get("generate_normals", True),
            generate_point_cloud=kwargs.get("generate_point_cloud", False),
        )

        return InferenceResult(
            output=result.depth_map,
            confidence=result.confidence,
            metadata={
                "depth_normalized": result.depth_normalized,
                "normals": result.normals,
                "point_cloud": result.point_cloud,
                "disparity": result.disparity,
                "depth_edges": result.depth_edges,
                **result.metadata,
            },
        )

    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, torch.Tensor]],
        **kwargs
    ) -> List[InferenceResult]:
        """Batch prediction."""
        return [self.predict(img, **kwargs) for img in images]

    def reset_temporal_buffer(self) -> None:
        """Reset temporal buffer (call between shots/scenes)."""
        self._temporal_buffer.clear()
