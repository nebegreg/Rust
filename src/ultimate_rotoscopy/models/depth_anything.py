"""
Depth Anything 3 Integration
============================

ByteDance's latest Depth Anything 3 for professional VFX workflows.
GitHub: https://github.com/ByteDance-Seed/Depth-Anything-3

Depth Anything 3 Key Features (over V2):
- Simplified backbone with vanilla DINO encoder
- Unified depth-ray representation
- Multi-view depth and camera pose estimation (NEW)
- 3D Gaussian splatting for novel view synthesis (NEW)
- Camera intrinsics estimation (NEW)
- Sky segmentation (NEW)
- Metric depth recovery with scale alignment

Model Series:
- Main: Giant (1.3B), Large (335M), Base (99M), Small (25M)
- Metric: Real-world scale depth
- Monocular: Single image depth
- Nested: Hierarchical depth representation

Requirements:
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

Installation:
    pip install depth-anything-3
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
    """Available Depth Anything 3 model sizes."""
    SMALL = "depth_anything_3_small"    # 25M params, fastest
    BASE = "depth_anything_3_base"      # 99M params, balanced
    LARGE = "depth_anything_3_large"    # 335M params, high quality
    GIANT = "depth_anything_3_giant"    # 1.3B params, best quality


class DepthModelType(Enum):
    """Depth Anything 3 model types."""
    MAIN = "main"           # Standard relative depth
    METRIC = "metric"       # Real-world metric depth
    MONOCULAR = "monocular" # Optimized for single images
    NESTED = "nested"       # Hierarchical depth representation


class DepthOutputType(Enum):
    """Types of depth output."""
    RELATIVE = "relative"      # 0-1 normalized depth
    METRIC = "metric"          # Real-world meters
    DISPARITY = "disparity"    # Inverse depth for stereo
    LOG_DEPTH = "log"          # Logarithmic depth
    DEPTH_RAY = "depth_ray"    # Unified depth-ray representation (DA3)


class NormalSpace(Enum):
    """Normal map coordinate spaces."""
    CAMERA = "camera"          # Camera-relative
    WORLD = "world"            # World space
    TANGENT = "tangent"        # Tangent space


class GaussianSplatMode(Enum):
    """3D Gaussian splatting modes."""
    FAST = "fast"              # Quick preview
    QUALITY = "quality"        # High quality
    REALTIME = "realtime"      # Real-time rendering


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float                  # Focal length x
    fy: float                  # Focal length y
    cx: float                  # Principal point x
    cy: float                  # Principal point y
    width: int                 # Image width
    height: int                # Image height
    distortion: Optional[np.ndarray] = None  # Distortion coefficients


@dataclass
class CameraPose:
    """Camera extrinsic parameters (pose)."""
    rotation: np.ndarray       # 3x3 rotation matrix
    translation: np.ndarray    # 3x1 translation vector
    timestamp: Optional[float] = None
    frame_id: Optional[int] = None


@dataclass
class MultiViewResult:
    """Result from multi-view depth estimation."""
    depths: List[np.ndarray]           # Per-view depth maps
    poses: List[CameraPose]            # Estimated camera poses
    point_cloud: Optional[np.ndarray] = None  # Fused 3D point cloud
    mesh: Optional[Any] = None         # Reconstructed mesh
    confidence: Optional[np.ndarray] = None  # Per-point confidence
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GaussianSplatResult:
    """Result from 3D Gaussian splatting."""
    gaussians: np.ndarray              # Gaussian parameters (pos, scale, rot, opacity, sh)
    rendered_views: List[np.ndarray]   # Rendered novel views
    point_cloud: np.ndarray            # Underlying point cloud
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthConfig(ModelConfig):
    """Depth Anything 3 configuration."""
    model_size: DepthModelSize = DepthModelSize.LARGE
    model_type: DepthModelType = DepthModelType.MAIN
    output_type: DepthOutputType = DepthOutputType.RELATIVE
    encoder_name: str = "dinov2_vitl"  # DINOv2 backbone
    max_depth: float = 100.0           # Maximum depth in meters
    min_depth: float = 0.01            # Minimum depth in meters
    use_metric_head: bool = True
    generate_normals: bool = True
    normal_space: NormalSpace = NormalSpace.CAMERA
    temporal_smoothing: bool = True
    temporal_alpha: float = 0.85
    edge_aware_smoothing: bool = True
    resolution_scale: float = 1.0
    # DA3 specific options
    use_depth_ray: bool = True         # Unified depth-ray representation
    estimate_intrinsics: bool = True   # Auto-estimate camera intrinsics
    sky_segmentation: bool = True      # Detect and mask sky regions
    # Multi-view options
    multiview_fusion: bool = False     # Enable multi-view fusion
    gaussian_splatting: bool = False   # Enable 3D Gaussian splatting
    gaussian_mode: GaussianSplatMode = GaussianSplatMode.QUALITY


@dataclass
class DepthResult:
    """Result from depth estimation."""
    depth_map: np.ndarray              # HxW depth values
    depth_normalized: np.ndarray       # HxW [0-1] normalized
    confidence: np.ndarray             # HxW confidence map
    normals: Optional[np.ndarray] = None       # HxWx3 normal vectors
    point_cloud: Optional[np.ndarray] = None   # Nx3 3D points
    disparity: Optional[np.ndarray] = None     # HxW disparity
    depth_edges: Optional[np.ndarray] = None   # HxW depth discontinuities
    # DA3 specific outputs
    depth_ray: Optional[np.ndarray] = None     # HxWx4 depth-ray representation
    sky_mask: Optional[np.ndarray] = None      # HxW sky segmentation mask
    intrinsics: Optional[CameraIntrinsics] = None  # Estimated intrinsics
    metric_depth: Optional[np.ndarray] = None  # Real-world scale depth
    metadata: Dict[str, Any] = field(default_factory=dict)


class DepthAnythingV3(BaseModel):
    """
    Depth Anything 3 for Ultimate Rotoscopy.

    Provides comprehensive depth estimation capabilities:
    - High-quality monocular depth estimation
    - Multi-view depth and pose estimation (NEW)
    - 3D Gaussian splatting for novel views (NEW)
    - Camera intrinsics estimation (NEW)
    - Sky segmentation (NEW)
    - Unified depth-ray representation (NEW)
    - Metric depth recovery
    - Normal map generation
    - 3D point cloud export

    Example:
        >>> config = DepthConfig(model_size=DepthModelSize.LARGE)
        >>> depth_model = DepthAnythingV3(config)
        >>> depth_model.load()
        >>>
        >>> # Single image depth
        >>> result = depth_model.estimate_depth(image)
        >>>
        >>> # Multi-view depth with pose estimation
        >>> mv_result = depth_model.estimate_multiview_depth(images)
        >>>
        >>> # 3D Gaussian splatting
        >>> gs_result = depth_model.create_gaussian_splat(images, depths)
    """

    def __init__(self, config: Optional[DepthConfig] = None):
        config = config or DepthConfig()
        super().__init__(config)
        self.depth_config = config
        self.depth_model = None
        self.metric_head = None
        self.intrinsics_head = None
        self.sky_segmentor = None
        self.pose_estimator = None
        self.gaussian_renderer = None
        self.processor = None  # HuggingFace processor for DA2/DA3
        self._temporal_buffer: List[np.ndarray] = []
        self._camera_intrinsics: Optional[CameraIntrinsics] = None
        self._da_version = 3

    def load(self) -> None:
        """Load Depth Anything 3 model."""
        if self._is_loaded:
            return

        print(f"Loading Depth Anything 3 {self.depth_config.model_size.value}...")
        start_time = time.time()

        try:
            # Try loading DA3 from official package
            self._load_depth_anything_3()
            self._da_version = 3
        except ImportError as e:
            print(f"Depth Anything 3 not available ({e}), trying V2...")
            try:
                self._load_from_transformers()
                self._da_version = 2
            except Exception:
                self._load_from_package()
                self._da_version = 2

        # Load additional heads for DA3
        if self._da_version == 3:
            self._load_auxiliary_models()

        self.optimize_for_inference()
        self._is_loaded = True

        load_time = time.time() - start_time
        print(f"Depth Anything V{self._da_version} loaded in {load_time:.2f}s on {self.device}")

    def _load_depth_anything_3(self) -> None:
        """Load Depth Anything 3 from official package."""
        try:
            from depth_anything_3.api import DepthAnything3
            from depth_anything_3.models import (
                build_depth_anything_3,
                build_metric_head,
                build_intrinsics_head,
            )

            # Map enum to model config
            model_map = {
                DepthModelSize.SMALL: "depth_anything_3_vits",
                DepthModelSize.BASE: "depth_anything_3_vitb",
                DepthModelSize.LARGE: "depth_anything_3_vitl",
                DepthModelSize.GIANT: "depth_anything_3_vitg",
            }

            model_name = model_map.get(
                self.depth_config.model_size,
                "depth_anything_3_vitl"
            )

            # Build main model
            self.depth_model = build_depth_anything_3(model_name)
            self.depth_model = self.depth_model.to(device=self.device)

            if self.dtype == torch.float16:
                self.depth_model = self.depth_model.half()

            # Build metric head if needed
            if self.depth_config.use_metric_head:
                self.metric_head = build_metric_head(model_name)
                self.metric_head = self.metric_head.to(device=self.device)

            # Build intrinsics estimation head
            if self.depth_config.estimate_intrinsics:
                self.intrinsics_head = build_intrinsics_head(model_name)
                self.intrinsics_head = self.intrinsics_head.to(device=self.device)

            print(f"Loaded Depth Anything 3 from depth_anything_3 package")

        except ImportError:
            # Try HuggingFace (when available)
            try:
                from transformers import (
                    DepthAnything3Model,
                    DepthAnything3Processor
                )

                model_id = self._get_da3_model_id()

                self.processor = DepthAnything3Processor.from_pretrained(
                    model_id,
                    cache_dir=self.config.cache_dir,
                )

                self.depth_model = DepthAnything3Model.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    cache_dir=self.config.cache_dir,
                ).to(self.device)

                print(f"Loaded Depth Anything 3 from {model_id}")

            except ImportError:
                raise ImportError(
                    "Depth Anything 3 not found. Install with: "
                    "pip install depth-anything-3 or "
                    "pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
                )

    def _load_from_transformers(self) -> None:
        """Load using HuggingFace transformers (DA V2)."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_id = self._get_da2_model_id()

        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
        )

        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            cache_dir=self.config.cache_dir,
        ).to(self.device)

        print(f"Loaded Depth Anything V2 from {model_id}")

    def _load_from_package(self) -> None:
        """Load using depth-anything package."""
        try:
            import timm

            # Build model architecture
            self.depth_model = self._build_depth_anything_model()

            # Load weights
            weights_path = self._download_weights()
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location=self.device)
                self.depth_model.load_state_dict(state_dict, strict=False)

            self.depth_model = self.depth_model.to(self.device)

            if self.dtype == torch.float16:
                self.depth_model = self.depth_model.half()

        except Exception as e:
            raise RuntimeError(f"Failed to load Depth Anything: {e}")

    def _load_auxiliary_models(self) -> None:
        """Load auxiliary models for DA3 features."""
        # Sky segmentation
        if self.depth_config.sky_segmentation:
            try:
                from depth_anything_3.sky import SkySegmentor
                self.sky_segmentor = SkySegmentor().to(self.device)
            except ImportError:
                # Fallback to simple sky detection
                self.sky_segmentor = None
                print("Sky segmentor not available, using fallback")

        # Pose estimator for multi-view
        if self.depth_config.multiview_fusion:
            try:
                from depth_anything_3.pose import PoseEstimator
                self.pose_estimator = PoseEstimator().to(self.device)
            except ImportError:
                self.pose_estimator = None

        # Gaussian splatting renderer
        if self.depth_config.gaussian_splatting:
            try:
                from depth_anything_3.gaussian import GaussianRenderer
                self.gaussian_renderer = GaussianRenderer(
                    mode=self.depth_config.gaussian_mode.value
                )
            except ImportError:
                self.gaussian_renderer = None

    def _get_da3_model_id(self) -> str:
        """Get HuggingFace Depth Anything 3 model ID."""
        model_map = {
            DepthModelSize.SMALL: "ByteDance/Depth-Anything-3-Small",
            DepthModelSize.BASE: "ByteDance/Depth-Anything-3-Base",
            DepthModelSize.LARGE: "ByteDance/Depth-Anything-3-Large",
            DepthModelSize.GIANT: "ByteDance/Depth-Anything-3-Giant",
        }
        return model_map.get(
            self.depth_config.model_size,
            "ByteDance/Depth-Anything-3-Large"
        )

    def _get_da2_model_id(self) -> str:
        """Get HuggingFace Depth Anything V2 model ID."""
        model_map = {
            DepthModelSize.SMALL: "depth-anything/Depth-Anything-V2-Small-hf",
            DepthModelSize.BASE: "depth-anything/Depth-Anything-V2-Base-hf",
            DepthModelSize.LARGE: "depth-anything/Depth-Anything-V2-Large-hf",
            DepthModelSize.GIANT: "depth-anything/Depth-Anything-V2-Large-hf",
        }
        return model_map.get(
            self.depth_config.model_size,
            "depth-anything/Depth-Anything-V2-Large-hf"
        )

    def _build_depth_anything_model(self) -> torch.nn.Module:
        """Build Depth Anything model architecture."""
        import timm

        class DepthAnything3Model(torch.nn.Module):
            """Depth Anything 3 architecture with DINOv2 backbone."""

            def __init__(self, encoder_name: str = "dinov2_vitl"):
                super().__init__()

                # DINOv2 encoder (simplified vanilla DINO in DA3)
                encoder_config = {
                    "dinov2_vits": "vit_small_patch14_dinov2.lvd142m",
                    "dinov2_vitb": "vit_base_patch14_dinov2.lvd142m",
                    "dinov2_vitl": "vit_large_patch14_dinov2.lvd142m",
                    "dinov2_vitg": "vit_giant_patch14_dinov2.lvd142m",
                }

                self.encoder = timm.create_model(
                    encoder_config.get(encoder_name, "vit_large_patch14_dinov2.lvd142m"),
                    pretrained=True,
                    features_only=True,
                    out_indices=[3, 5, 7, 11],
                )

                # Get encoder output channels
                encoder_channels = {
                    "dinov2_vits": [64, 128, 256, 384],
                    "dinov2_vitb": [96, 192, 384, 768],
                    "dinov2_vitl": [256, 512, 1024, 1024],
                    "dinov2_vitg": [384, 768, 1536, 1536],
                }

                channels = encoder_channels.get(encoder_name, [256, 512, 1024, 1024])

                # DPT-style decoder
                self.decoder = self._build_decoder(channels)

                # Depth head with depth-ray output
                self.depth_head = torch.nn.Sequential(
                    torch.nn.Conv2d(256, 256, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(256, 128, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(128, 1, 1),
                    torch.nn.Sigmoid(),
                )

                # Depth-ray head (DA3 feature)
                self.depth_ray_head = torch.nn.Sequential(
                    torch.nn.Conv2d(256, 128, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(128, 4, 1),  # depth + 3D ray direction
                )

            def _build_decoder(self, in_channels: List[int]) -> torch.nn.Module:
                """Build DPT decoder."""
                class DPTDecoder(torch.nn.Module):
                    def __init__(self, channels: List[int]):
                        super().__init__()
                        self.projects = torch.nn.ModuleList([
                            torch.nn.Conv2d(c, 256, 1) for c in channels
                        ])
                        self.resize_layers = torch.nn.ModuleList([
                            torch.nn.ConvTranspose2d(256, 256, 4, stride=4),
                            torch.nn.ConvTranspose2d(256, 256, 2, stride=2),
                            torch.nn.Identity(),
                            torch.nn.Conv2d(256, 256, 3, stride=2, padding=1),
                        ])
                        self.fusion_blocks = torch.nn.ModuleList([
                            torch.nn.Sequential(
                                torch.nn.Conv2d(256, 256, 3, padding=1),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Conv2d(256, 256, 3, padding=1),
                            ) for _ in range(4)
                        ])

                    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
                        projected = [proj(f) for proj, f in zip(self.projects, features)]
                        resized = [resize(p) for resize, p in zip(self.resize_layers, projected)]

                        fused = resized[0]
                        for i, (r, fusion) in enumerate(zip(resized[1:], self.fusion_blocks[1:])):
                            if fused.shape[2:] != r.shape[2:]:
                                fused = F.interpolate(fused, size=r.shape[2:], mode="bilinear")
                            fused = fusion(fused + r)
                        return fused

                return DPTDecoder(in_channels)

            def forward(
                self,
                x: torch.Tensor,
                return_depth_ray: bool = False
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                features = self.encoder(x)
                decoded = self.decoder(features)
                depth = self.depth_head(decoded)

                if return_depth_ray:
                    depth_ray = self.depth_ray_head(decoded)
                    return depth, depth_ray
                return depth

        encoder_map = {
            DepthModelSize.SMALL: "dinov2_vits",
            DepthModelSize.BASE: "dinov2_vitb",
            DepthModelSize.LARGE: "dinov2_vitl",
            DepthModelSize.GIANT: "dinov2_vitg",
        }

        encoder = encoder_map.get(self.depth_config.model_size, "dinov2_vitl")
        return DepthAnything3Model(encoder)

    def _download_weights(self) -> Path:
        """Download model weights."""
        from huggingface_hub import hf_hub_download

        model_map = {
            DepthModelSize.SMALL: "depth-anything/Depth-Anything-V2-Small",
            DepthModelSize.BASE: "depth-anything/Depth-Anything-V2-Base",
            DepthModelSize.LARGE: "depth-anything/Depth-Anything-V2-Large",
        }

        repo_id = model_map.get(
            self.depth_config.model_size,
            "depth-anything/Depth-Anything-V2-Large"
        )

        try:
            return Path(hf_hub_download(
                repo_id=repo_id,
                filename="pytorch_model.bin",
                cache_dir=self.config.cache_dir,
            ))
        except Exception:
            return Path("")

    def unload(self) -> None:
        """Unload model from memory."""
        self.depth_model = None
        self.metric_head = None
        self.intrinsics_head = None
        self.sky_segmentor = None
        self.pose_estimator = None
        self.gaussian_renderer = None
        self._temporal_buffer.clear()
        self.clear_cache()
        self._is_loaded = False

    def set_camera_intrinsics(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int = 0,
        height: int = 0,
    ) -> None:
        """Set camera intrinsic parameters."""
        self._camera_intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height
        )

    @torch.inference_mode()
    def estimate_depth(
        self,
        image: Union[np.ndarray, Image.Image],
        generate_normals: bool = True,
        generate_point_cloud: bool = False,
        estimate_intrinsics: bool = None,
    ) -> DepthResult:
        """
        Estimate depth from a single image.

        Args:
            image: Input RGB image
            generate_normals: Generate normal map from depth
            generate_point_cloud: Generate 3D point cloud
            estimate_intrinsics: Auto-estimate camera intrinsics (DA3)

        Returns:
            DepthResult with depth map, normals, and DA3-specific outputs
        """
        start_time = time.time()

        # Preprocess image
        if isinstance(image, Image.Image):
            original_size = image.size[::-1]
            image_np = np.array(image)
        else:
            original_size = image.shape[:2]
            image_np = image

        # Run depth estimation
        if hasattr(self, "processor") and self.processor is not None:
            depth_map, depth_ray = self._estimate_with_processor(image, return_ray=True)
        else:
            depth_map, depth_ray = self._estimate_direct(image_np, return_ray=True)

        # Resize to original resolution
        if depth_map.shape[:2] != original_size:
            depth_map = self._resize_depth(depth_map, original_size)
            if depth_ray is not None:
                depth_ray = self._resize_depth_ray(depth_ray, original_size)

        # Normalize depth
        depth_normalized = self._normalize_depth(depth_map)

        # Generate confidence map
        confidence = self._estimate_confidence(depth_map)

        # Estimate camera intrinsics (DA3 feature)
        intrinsics = None
        if estimate_intrinsics or (estimate_intrinsics is None and self.depth_config.estimate_intrinsics):
            intrinsics = self._estimate_intrinsics(image_np, depth_map)
            if intrinsics is not None:
                self._camera_intrinsics = intrinsics

        # Sky segmentation (DA3 feature)
        sky_mask = None
        if self.depth_config.sky_segmentation:
            sky_mask = self._segment_sky(image_np, depth_map)

        # Metric depth estimation (DA3 feature)
        metric_depth = None
        if self.depth_config.use_metric_head and self.metric_head is not None:
            metric_depth = self._estimate_metric_depth(image_np, depth_map)
        elif self.depth_config.output_type == DepthOutputType.METRIC:
            metric_depth = self._scale_to_metric(depth_map)

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
            depth_ray=depth_ray,
            sky_mask=sky_mask,
            intrinsics=intrinsics,
            metric_depth=metric_depth,
            metadata={
                "processing_time_ms": processing_time,
                "device": str(self.device),
                "model_size": self.depth_config.model_size.value,
                "output_type": self.depth_config.output_type.value,
                "da_version": self._da_version,
            }
        )

    def _estimate_with_processor(
        self,
        image: Union[np.ndarray, Image.Image],
        return_ray: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Estimate depth using HuggingFace processor."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.depth_model(**inputs)

        depth = outputs.predicted_depth.squeeze().cpu().numpy()

        depth_ray = None
        if return_ray and hasattr(outputs, 'depth_ray'):
            depth_ray = outputs.depth_ray.squeeze().cpu().numpy()

        return depth, depth_ray

    def _estimate_direct(
        self,
        image: np.ndarray,
        return_ray: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Estimate depth directly without processor."""
        tensor = self.preprocess(image)

        if self.depth_config.resolution_scale != 1.0:
            h, w = tensor.shape[2:]
            new_h = int(h * self.depth_config.resolution_scale)
            new_w = int(w * self.depth_config.resolution_scale)
            tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear")

        if return_ray and self.depth_config.use_depth_ray:
            depth, depth_ray = self.depth_model(tensor, return_depth_ray=True)
            return depth.squeeze().cpu().numpy(), depth_ray.squeeze().cpu().numpy()
        else:
            depth = self.depth_model(tensor)
            return depth.squeeze().cpu().numpy(), None

    def _estimate_intrinsics(
        self,
        image: np.ndarray,
        depth: np.ndarray
    ) -> Optional[CameraIntrinsics]:
        """
        Estimate camera intrinsics from image and depth (DA3 feature).

        Uses learned intrinsics estimation head or geometric methods.
        """
        h, w = image.shape[:2]

        if self.intrinsics_head is not None:
            # Use learned intrinsics head
            tensor = self.preprocess(image)
            with torch.inference_mode():
                intrinsics_pred = self.intrinsics_head(tensor)
                fx, fy, cx, cy = intrinsics_pred[0].cpu().numpy()
        else:
            # Fallback: estimate from image size (common assumptions)
            # Assume standard field of view
            fov = 60  # degrees
            fx = fy = w / (2 * np.tan(np.radians(fov / 2)))
            cx = w / 2
            cy = h / 2

        return CameraIntrinsics(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            width=w,
            height=h,
        )

    def _segment_sky(
        self,
        image: np.ndarray,
        depth: np.ndarray
    ) -> np.ndarray:
        """
        Segment sky regions (DA3 feature).

        Sky regions typically have maximum/infinite depth.
        """
        if self.sky_segmentor is not None:
            tensor = self.preprocess(image)
            with torch.inference_mode():
                sky_mask = self.sky_segmentor(tensor)
                return sky_mask.squeeze().cpu().numpy() > 0.5
        else:
            # Fallback: use depth-based sky detection
            # Sky typically has very large/max depth values
            depth_normalized = self._normalize_depth(depth)

            # High depth values likely indicate sky
            sky_threshold = 0.95
            sky_mask = depth_normalized > sky_threshold

            # Additional heuristic: sky is usually in upper portion
            h = depth.shape[0]
            upper_weight = np.linspace(1.0, 0.3, h)[:, np.newaxis]
            weighted_mask = sky_mask * (upper_weight > 0.7)

            return weighted_mask.astype(bool)

    def _estimate_metric_depth(
        self,
        image: np.ndarray,
        relative_depth: np.ndarray
    ) -> np.ndarray:
        """Convert relative depth to metric depth using learned scale."""
        if self.metric_head is not None:
            tensor = self.preprocess(image)
            depth_tensor = torch.from_numpy(relative_depth).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                scale, shift = self.metric_head(tensor, depth_tensor)
                metric_depth = relative_depth * scale.item() + shift.item()
        else:
            metric_depth = self._scale_to_metric(relative_depth)

        return metric_depth

    def _scale_to_metric(self, depth: np.ndarray) -> np.ndarray:
        """Simple scaling to metric depth."""
        normalized = self._normalize_depth(depth)
        return normalized * (self.depth_config.max_depth - self.depth_config.min_depth) + self.depth_config.min_depth

    # =========================================================================
    # Multi-View Depth Estimation (DA3 Feature)
    # =========================================================================

    @torch.inference_mode()
    def estimate_multiview_depth(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        known_poses: Optional[List[CameraPose]] = None,
    ) -> MultiViewResult:
        """
        Estimate depth from multiple views with pose estimation (DA3 feature).

        Args:
            images: List of input images from different viewpoints
            known_poses: Optional known camera poses

        Returns:
            MultiViewResult with per-view depths and estimated poses
        """
        depths = []
        poses = []

        # Process each view
        for i, image in enumerate(images):
            result = self.estimate_depth(image, generate_point_cloud=True)
            depths.append(result.depth_map)

        # Estimate poses if not provided
        if known_poses is None:
            poses = self._estimate_poses(images, depths)
        else:
            poses = known_poses

        # Fuse point clouds from all views
        point_cloud = self._fuse_point_clouds(images, depths, poses)

        return MultiViewResult(
            depths=depths,
            poses=poses,
            point_cloud=point_cloud,
            metadata={
                "num_views": len(images),
                "fused_points": len(point_cloud) if point_cloud is not None else 0,
            }
        )

    def _estimate_poses(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        depths: List[np.ndarray]
    ) -> List[CameraPose]:
        """Estimate camera poses from images and depths."""
        poses = []

        if self.pose_estimator is not None:
            # Use learned pose estimator
            for i, (img, depth) in enumerate(zip(images, depths)):
                if isinstance(img, Image.Image):
                    img = np.array(img)

                tensor = self.preprocess(img)
                depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(self.device)

                pose_pred = self.pose_estimator(tensor, depth_tensor)
                rotation = pose_pred[:, :9].reshape(3, 3).cpu().numpy()
                translation = pose_pred[:, 9:12].cpu().numpy().flatten()

                poses.append(CameraPose(
                    rotation=rotation,
                    translation=translation,
                    frame_id=i,
                ))
        else:
            # Fallback: use feature matching for relative poses
            poses = self._estimate_poses_from_features(images)

        return poses

    def _estimate_poses_from_features(
        self,
        images: List[Union[np.ndarray, Image.Image]]
    ) -> List[CameraPose]:
        """Estimate poses using feature matching."""
        import cv2

        poses = [CameraPose(
            rotation=np.eye(3),
            translation=np.zeros(3),
            frame_id=0,
        )]

        for i in range(1, len(images)):
            img1 = np.array(images[i-1]) if isinstance(images[i-1], Image.Image) else images[i-1]
            img2 = np.array(images[i]) if isinstance(images[i], Image.Image) else images[i]

            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

            # Feature detection and matching
            orb = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)

            if des1 is None or des2 is None:
                poses.append(CameraPose(
                    rotation=np.eye(3),
                    translation=np.zeros(3),
                    frame_id=i,
                ))
                continue

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:100]

            if len(matches) < 8:
                poses.append(CameraPose(
                    rotation=np.eye(3),
                    translation=np.zeros(3),
                    frame_id=i,
                ))
                continue

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            # Estimate essential matrix
            if self._camera_intrinsics is not None:
                K = np.array([
                    [self._camera_intrinsics.fx, 0, self._camera_intrinsics.cx],
                    [0, self._camera_intrinsics.fy, self._camera_intrinsics.cy],
                    [0, 0, 1]
                ])
            else:
                h, w = img1.shape[:2]
                K = np.array([
                    [w, 0, w/2],
                    [0, w, h/2],
                    [0, 0, 1]
                ])

            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
            if E is None:
                poses.append(CameraPose(
                    rotation=np.eye(3),
                    translation=np.zeros(3),
                    frame_id=i,
                ))
                continue

            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

            poses.append(CameraPose(
                rotation=R,
                translation=t.flatten(),
                frame_id=i,
            ))

        return poses

    def _fuse_point_clouds(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        depths: List[np.ndarray],
        poses: List[CameraPose],
    ) -> Optional[np.ndarray]:
        """Fuse point clouds from multiple views."""
        if self._camera_intrinsics is None:
            return None

        all_points = []
        all_colors = []

        for img, depth, pose in zip(images, depths, poses):
            if isinstance(img, Image.Image):
                img = np.array(img)

            # Back-project to 3D
            points = self._depth_to_point_cloud(depth)
            if points is None:
                continue

            # Transform to world coordinates
            points_world = (pose.rotation @ points.T).T + pose.translation

            # Get colors
            h, w = depth.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            colors = img[v.flatten(), u.flatten()]

            all_points.append(points_world)
            all_colors.append(colors)

        if not all_points:
            return None

        return np.vstack(all_points)

    # =========================================================================
    # 3D Gaussian Splatting (DA3 Feature)
    # =========================================================================

    @torch.inference_mode()
    def create_gaussian_splat(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        depths: Optional[List[np.ndarray]] = None,
        poses: Optional[List[CameraPose]] = None,
        num_iterations: int = 1000,
    ) -> GaussianSplatResult:
        """
        Create 3D Gaussian splatting representation (DA3 feature).

        Args:
            images: Input images
            depths: Optional pre-computed depth maps
            poses: Optional camera poses
            num_iterations: Optimization iterations

        Returns:
            GaussianSplatResult with Gaussian parameters and rendered views
        """
        # Estimate depths if not provided
        if depths is None:
            depths = [self.estimate_depth(img).depth_map for img in images]

        # Estimate poses if not provided
        if poses is None:
            poses = self._estimate_poses(images, depths)

        # Get fused point cloud
        point_cloud = self._fuse_point_clouds(images, depths, poses)
        if point_cloud is None:
            raise ValueError("Could not create point cloud for Gaussian splatting")

        # Initialize Gaussians from point cloud
        if self.gaussian_renderer is not None:
            gaussians = self.gaussian_renderer.initialize_from_points(point_cloud)

            # Optimize Gaussians
            for i in range(num_iterations):
                loss = self.gaussian_renderer.optimize_step(
                    gaussians, images, poses
                )

            # Render novel views
            rendered_views = []
            for pose in poses:
                view = self.gaussian_renderer.render(gaussians, pose)
                rendered_views.append(view)

        else:
            # Fallback: simple Gaussian initialization without optimization
            gaussians = self._initialize_simple_gaussians(point_cloud)
            rendered_views = []

        return GaussianSplatResult(
            gaussians=gaussians,
            rendered_views=rendered_views,
            point_cloud=point_cloud,
            metadata={
                "num_gaussians": len(gaussians),
                "num_views": len(images),
            }
        )

    def _initialize_simple_gaussians(self, points: np.ndarray) -> np.ndarray:
        """Initialize Gaussians from points (simple fallback)."""
        num_points = len(points)

        # Each Gaussian: position (3), scale (3), rotation (4 quaternion),
        # opacity (1), spherical harmonics (48 for RGB)
        gaussians = np.zeros((num_points, 59), dtype=np.float32)

        # Positions
        gaussians[:, :3] = points

        # Initial scales (small)
        gaussians[:, 3:6] = 0.01

        # Identity rotation (quaternion w, x, y, z)
        gaussians[:, 6] = 1.0  # w

        # Opacity
        gaussians[:, 10] = 0.5

        return gaussians

    def render_novel_view(
        self,
        gaussians: np.ndarray,
        camera_pose: CameraPose,
        image_size: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """
        Render a novel view from Gaussian representation.

        Args:
            gaussians: Gaussian parameters
            camera_pose: Target camera pose
            image_size: Output image size (H, W)

        Returns:
            Rendered RGB image
        """
        if self.gaussian_renderer is not None:
            return self.gaussian_renderer.render(
                gaussians, camera_pose, image_size
            )
        else:
            # Simple point rendering fallback
            return self._simple_point_render(gaussians[:, :3], camera_pose, image_size)

    def _simple_point_render(
        self,
        points: np.ndarray,
        pose: CameraPose,
        image_size: Tuple[int, int],
    ) -> np.ndarray:
        """Simple point-based rendering fallback."""
        h, w = image_size
        image = np.zeros((h, w, 3), dtype=np.uint8)

        if self._camera_intrinsics is None:
            return image

        # Transform points to camera space
        points_cam = (pose.rotation.T @ (points - pose.translation).T).T

        # Project to image
        fx, fy = self._camera_intrinsics.fx, self._camera_intrinsics.fy
        cx, cy = self._camera_intrinsics.cx, self._camera_intrinsics.cy

        z = points_cam[:, 2]
        valid = z > 0

        u = (fx * points_cam[valid, 0] / z[valid] + cx).astype(int)
        v = (fy * points_cam[valid, 1] / z[valid] + cy).astype(int)

        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        image[v[in_bounds], u[in_bounds]] = 255

        return image

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _resize_depth(self, depth: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize depth map to target size."""
        import cv2
        return cv2.resize(depth, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    def _resize_depth_ray(self, depth_ray: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize depth-ray representation."""
        import cv2
        h, w = target_size
        if depth_ray.ndim == 3:
            resized = np.zeros((h, w, depth_ray.shape[2]), dtype=depth_ray.dtype)
            for i in range(depth_ray.shape[2]):
                resized[:, :, i] = cv2.resize(
                    depth_ray[:, :, i], (w, h), interpolation=cv2.INTER_LINEAR
                )
            return resized
        return cv2.resize(depth_ray, (w, h), interpolation=cv2.INTER_LINEAR)

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

        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
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
        smoothed = cv2.bilateralFilter(depth_8bit, d=9, sigmaColor=75, sigmaSpace=75)

        depth_min, depth_max = depth.min(), depth.max()
        smoothed = smoothed.astype(np.float32) / 255.0
        smoothed = smoothed * (depth_max - depth_min) + depth_min

        return smoothed

    def _compute_normals(self, depth: np.ndarray) -> np.ndarray:
        """Compute normal map from depth."""
        import cv2

        h, w = depth.shape
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)

        normals = np.zeros((h, w, 3), dtype=np.float32)
        normals[..., 0] = -grad_x
        normals[..., 1] = -grad_y
        normals[..., 2] = 1.0

        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / (norm + 1e-8)

        return normals

    def _depth_to_point_cloud(self, depth: np.ndarray) -> Optional[np.ndarray]:
        """Convert depth map to 3D point cloud."""
        if self._camera_intrinsics is None:
            return None

        h, w = depth.shape
        fx = self._camera_intrinsics.fx
        fy = self._camera_intrinsics.fy
        cx = self._camera_intrinsics.cx
        cy = self._camera_intrinsics.cy

        u, v = np.meshgrid(np.arange(w), np.arange(h))

        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack([x, y, z], axis=-1)
        points = points.reshape(-1, 3)

        valid = ~np.isnan(points).any(axis=1) & (points[:, 2] > 0)
        points = points[valid]

        return points

    def _depth_to_disparity(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth to disparity (inverse depth)."""
        disparity = 1.0 / (depth + 1e-6)

        disp_min, disp_max = disparity.min(), disparity.max()
        if disp_max - disp_min > 1e-6:
            disparity = (disparity - disp_min) / (disp_max - disp_min)

        return disparity.astype(np.float32)

    def _detect_depth_edges(self, depth: np.ndarray) -> np.ndarray:
        """Detect depth discontinuities/edges."""
        import cv2

        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        threshold = np.percentile(grad_mag, 95)
        edges = (grad_mag > threshold).astype(np.float32)

        return edges

    def get_z_depth_for_compositing(
        self,
        depth: np.ndarray,
        near: float = 0.1,
        far: float = 100.0
    ) -> np.ndarray:
        """Get Z-depth formatted for compositing (Nuke/Flame compatible)."""
        normalized = self._normalize_depth(depth)
        z_depth = normalized * (far - near) + near
        return z_depth.astype(np.float32)

    def export_point_cloud(
        self,
        point_cloud: np.ndarray,
        colors: Optional[np.ndarray],
        output_path: Path,
        format: str = "ply"
    ) -> None:
        """Export point cloud to file."""
        output_path = Path(output_path)

        if format == "ply":
            self._export_ply(point_cloud, colors, output_path)
        elif format == "obj":
            self._export_obj(point_cloud, output_path)
        elif format == "xyz":
            self._export_xyz(point_cloud, colors, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_ply(self, points: np.ndarray, colors: Optional[np.ndarray], path: Path) -> None:
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

    def _export_xyz(self, points: np.ndarray, colors: Optional[np.ndarray], path: Path) -> None:
        """Export to XYZ format."""
        with open(path, "w") as f:
            for i, (x, y, z) in enumerate(points):
                if colors is not None:
                    r, g, b = colors[i]
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    # =========================================================================
    # Standard Interface
    # =========================================================================

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
                "depth_ray": result.depth_ray,
                "sky_mask": result.sky_mask,
                "intrinsics": result.intrinsics,
                "metric_depth": result.metric_depth,
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
