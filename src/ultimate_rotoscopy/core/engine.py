"""
Rotoscopy Engine - Core Orchestration Layer
============================================

The main engine that coordinates all AI models and processing pipelines
for professional rotoscopy workflows.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

from ultimate_rotoscopy.models.sam3 import (
    SAM3Segmentor,
    SAM3Config,
    SAM3ModelSize,
    SegmentationPrompt,
    PromptType,
)
from ultimate_rotoscopy.models.depth_anything import (
    DepthAnythingV3,
    DepthConfig,
    DepthModelSize,
)
from ultimate_rotoscopy.models.matte_anything import (
    MatteAnything,
    MatteConfig,
    MatteQuality,
    EdgeMode,
)


class ProcessingMode(Enum):
    """Processing mode for the engine."""
    FAST = "fast"           # Speed priority
    BALANCED = "balanced"   # Balance speed/quality
    QUALITY = "quality"     # Quality priority
    MAXIMUM = "maximum"     # Maximum quality


class OutputFormat(Enum):
    """Output format options."""
    NUMPY = "numpy"
    PIL = "pil"
    TORCH = "torch"


@dataclass
class EngineConfig:
    """Configuration for the rotoscopy engine."""
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    device: str = "cuda"
    enable_sam3: bool = True
    enable_depth: bool = True
    enable_matte: bool = True
    sam3_size: SAM3ModelSize = SAM3ModelSize.LARGE
    depth_size: DepthModelSize = DepthModelSize.LARGE
    matte_quality: MatteQuality = MatteQuality.HIGH
    parallel_processing: bool = True
    max_workers: int = 4
    cache_embeddings: bool = True
    output_format: OutputFormat = OutputFormat.NUMPY
    auto_load_models: bool = True


@dataclass
class ProcessingResult:
    """Complete result from rotoscopy processing."""
    # Segmentation
    masks: Optional[np.ndarray] = None
    mask_scores: Optional[np.ndarray] = None
    refined_mask: Optional[np.ndarray] = None

    # Depth
    depth_map: Optional[np.ndarray] = None
    depth_normalized: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    point_cloud: Optional[np.ndarray] = None
    disparity: Optional[np.ndarray] = None
    z_depth: Optional[np.ndarray] = None

    # Matte
    alpha: Optional[np.ndarray] = None
    foreground: Optional[np.ndarray] = None
    edge_mask: Optional[np.ndarray] = None
    hair_mask: Optional[np.ndarray] = None
    motion_mask: Optional[np.ndarray] = None

    # Combined outputs
    composite: Optional[np.ndarray] = None
    aov_package: Optional[Dict[str, np.ndarray]] = None

    # Metadata
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RotoscopyEngine:
    """
    Main Rotoscopy Engine for Ultimate Rotoscopy.

    Orchestrates all AI models (SAM3, Depth Anything V3, Matte Anything)
    for comprehensive rotoscopy processing.

    Example:
        >>> engine = RotoscopyEngine()
        >>> engine.load_models()
        >>>
        >>> # Full pipeline processing
        >>> result = engine.process(
        ...     image,
        ...     prompt=SegmentationPrompt(
        ...         prompt_type=PromptType.POINT,
        ...         points=np.array([[100, 200]]),
        ...         point_labels=np.array([1])
        ...     )
        ... )
        >>>
        >>> # Access all outputs
        >>> alpha = result.alpha
        >>> depth = result.depth_map
        >>> normals = result.normals
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.sam3: Optional[SAM3Segmentor] = None
        self.depth: Optional[DepthAnythingV3] = None
        self.matte: Optional[MatteAnything] = None
        self._models_loaded = False
        self._executor: Optional[ThreadPoolExecutor] = None

        if self.config.auto_load_models:
            self.load_models()

    def load_models(self) -> None:
        """Load all enabled AI models."""
        if self._models_loaded:
            return

        print("Loading Ultimate Rotoscopy Engine...")
        start_time = time.time()

        # Configure models based on processing mode
        sam3_config, depth_config, matte_config = self._get_model_configs()

        # Load models (can be parallelized)
        if self.config.parallel_processing:
            self._load_models_parallel(sam3_config, depth_config, matte_config)
        else:
            self._load_models_sequential(sam3_config, depth_config, matte_config)

        # Initialize thread pool
        if self.config.parallel_processing:
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        self._models_loaded = True
        load_time = time.time() - start_time
        print(f"Engine loaded in {load_time:.2f}s")

    def _get_model_configs(self) -> Tuple[SAM3Config, DepthConfig, MatteConfig]:
        """Get model configurations based on processing mode."""
        mode = self.config.processing_mode

        if mode == ProcessingMode.FAST:
            sam3_config = SAM3Config(
                model_size=SAM3ModelSize.BASE,
                edge_refinement=False,
                use_temporal_consistency=False,
            )
            depth_config = DepthConfig(
                model_size=DepthModelSize.SMALL,
                generate_normals=False,
                temporal_smoothing=False,
            )
            matte_config = MatteConfig(
                quality=MatteQuality.DRAFT,
                edge_mode=EdgeMode.NONE,
                temporal_consistency=False,
            )
        elif mode == ProcessingMode.BALANCED:
            sam3_config = SAM3Config(
                model_size=self.config.sam3_size,
                edge_refinement=True,
            )
            depth_config = DepthConfig(
                model_size=self.config.depth_size,
                generate_normals=True,
            )
            matte_config = MatteConfig(
                quality=self.config.matte_quality,
                edge_mode=EdgeMode.SOFT,
            )
        elif mode == ProcessingMode.QUALITY:
            sam3_config = SAM3Config(
                model_size=SAM3ModelSize.LARGE,
                edge_refinement=True,
                hq_token=True,
            )
            depth_config = DepthConfig(
                model_size=DepthModelSize.LARGE,
                generate_normals=True,
                edge_aware_smoothing=True,
            )
            matte_config = MatteConfig(
                quality=MatteQuality.HIGH,
                edge_mode=EdgeMode.HAIR,
                refine_foreground=True,
            )
        else:  # MAXIMUM
            sam3_config = SAM3Config(
                model_size=SAM3ModelSize.HUGE,
                edge_refinement=True,
                hq_token=True,
                use_temporal_consistency=True,
            )
            depth_config = DepthConfig(
                model_size=DepthModelSize.GIANT,
                generate_normals=True,
                edge_aware_smoothing=True,
                temporal_smoothing=True,
            )
            matte_config = MatteConfig(
                quality=MatteQuality.ULTRA,
                edge_mode=EdgeMode.HAIR,
                handle_motion_blur=True,
                refine_foreground=True,
                color_decontamination=True,
                spill_suppression=True,
            )

        return sam3_config, depth_config, matte_config

    def _load_models_parallel(
        self,
        sam3_config: SAM3Config,
        depth_config: DepthConfig,
        matte_config: MatteConfig
    ) -> None:
        """Load models in parallel."""
        import concurrent.futures

        def load_sam3():
            if self.config.enable_sam3:
                model = SAM3Segmentor(sam3_config)
                model.load()
                return model
            return None

        def load_depth():
            if self.config.enable_depth:
                model = DepthAnythingV3(depth_config)
                model.load()
                return model
            return None

        def load_matte():
            if self.config.enable_matte:
                model = MatteAnything(matte_config)
                model.load()
                return model
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(load_sam3): "sam3",
                executor.submit(load_depth): "depth",
                executor.submit(load_matte): "matte",
            }

            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                model = future.result()
                if name == "sam3":
                    self.sam3 = model
                elif name == "depth":
                    self.depth = model
                elif name == "matte":
                    self.matte = model

    def _load_models_sequential(
        self,
        sam3_config: SAM3Config,
        depth_config: DepthConfig,
        matte_config: MatteConfig
    ) -> None:
        """Load models sequentially."""
        if self.config.enable_sam3:
            self.sam3 = SAM3Segmentor(sam3_config)
            self.sam3.load()

        if self.config.enable_depth:
            self.depth = DepthAnythingV3(depth_config)
            self.depth.load()

        if self.config.enable_matte:
            self.matte = MatteAnything(matte_config)
            self.matte.load()

    def unload_models(self) -> None:
        """Unload all models from memory."""
        if self.sam3:
            self.sam3.unload()
        if self.depth:
            self.depth.unload()
        if self.matte:
            self.matte.unload()

        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        self._models_loaded = False

    def process(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        prompt: Optional[SegmentationPrompt] = None,
        previous_frame: Optional[np.ndarray] = None,
        generate_depth: bool = True,
        generate_normals: bool = True,
        generate_matte: bool = True,
        generate_point_cloud: bool = False,
        camera_intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ) -> ProcessingResult:
        """
        Process an image through the full rotoscopy pipeline.

        Args:
            image: Input image (RGB)
            prompt: Optional segmentation prompt (points, boxes, text)
            previous_frame: Previous frame for temporal consistency
            generate_depth: Generate depth map
            generate_normals: Generate normal map
            generate_matte: Generate alpha matte
            generate_point_cloud: Generate 3D point cloud
            camera_intrinsics: (fx, fy, cx, cy) for 3D reconstruction

        Returns:
            ProcessingResult with all outputs
        """
        start_time = time.time()

        # Load image if path
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image))
        elif isinstance(image, Image.Image):
            image = np.array(image)

        result = ProcessingResult()

        # Set camera intrinsics if provided
        if camera_intrinsics and self.depth:
            self.depth.set_camera_intrinsics(*camera_intrinsics)

        # Stage 1: Segmentation
        if self.sam3 and prompt is not None:
            seg_result = self.sam3.segment(image, prompt)
            result.masks = seg_result.masks
            result.mask_scores = seg_result.scores
            result.refined_mask = seg_result.refined_mask

        # Run depth and matte in parallel if enabled
        if self.config.parallel_processing and self._executor:
            self._process_parallel(
                image, result, previous_frame,
                generate_depth, generate_normals,
                generate_matte, generate_point_cloud
            )
        else:
            self._process_sequential(
                image, result, previous_frame,
                generate_depth, generate_normals,
                generate_matte, generate_point_cloud
            )

        # Generate AOV package
        result.aov_package = self._create_aov_package(result)

        result.processing_time_ms = (time.time() - start_time) * 1000
        result.metadata = {
            "processing_mode": self.config.processing_mode.value,
            "models_used": {
                "sam3": self.sam3 is not None and prompt is not None,
                "depth": self.depth is not None and generate_depth,
                "matte": self.matte is not None and generate_matte,
            }
        }

        return result

    def _process_parallel(
        self,
        image: np.ndarray,
        result: ProcessingResult,
        previous_frame: Optional[np.ndarray],
        generate_depth: bool,
        generate_normals: bool,
        generate_matte: bool,
        generate_point_cloud: bool,
    ) -> None:
        """Process depth and matte in parallel."""
        futures = {}

        if self.depth and generate_depth:
            futures[self._executor.submit(
                self.depth.estimate_depth,
                image,
                generate_normals=generate_normals,
                generate_point_cloud=generate_point_cloud,
            )] = "depth"

        if self.matte and generate_matte:
            # Use segmentation mask if available
            mask = result.refined_mask if result.refined_mask is not None else (
                result.masks[0] if result.masks is not None and len(result.masks) > 0 else None
            )
            futures[self._executor.submit(
                self.matte.generate_matte,
                image,
                mask=mask,
                previous_frame=previous_frame,
            )] = "matte"

        for future in as_completed(futures):
            name = futures[future]
            output = future.result()

            if name == "depth":
                result.depth_map = output.depth_map
                result.depth_normalized = output.depth_normalized
                result.normals = output.normals
                result.point_cloud = output.point_cloud
                result.disparity = output.disparity
                result.z_depth = self.depth.get_z_depth_for_compositing(output.depth_map)

            elif name == "matte":
                result.alpha = output.alpha
                result.foreground = output.foreground
                result.edge_mask = output.edge_mask
                result.hair_mask = output.hair_mask
                result.motion_mask = output.motion_mask

    def _process_sequential(
        self,
        image: np.ndarray,
        result: ProcessingResult,
        previous_frame: Optional[np.ndarray],
        generate_depth: bool,
        generate_normals: bool,
        generate_matte: bool,
        generate_point_cloud: bool,
    ) -> None:
        """Process depth and matte sequentially."""
        # Stage 2: Depth estimation
        if self.depth and generate_depth:
            depth_result = self.depth.estimate_depth(
                image,
                generate_normals=generate_normals,
                generate_point_cloud=generate_point_cloud,
            )
            result.depth_map = depth_result.depth_map
            result.depth_normalized = depth_result.depth_normalized
            result.normals = depth_result.normals
            result.point_cloud = depth_result.point_cloud
            result.disparity = depth_result.disparity
            result.z_depth = self.depth.get_z_depth_for_compositing(depth_result.depth_map)

        # Stage 3: Alpha matting
        if self.matte and generate_matte:
            # Use segmentation mask if available
            mask = result.refined_mask if result.refined_mask is not None else (
                result.masks[0] if result.masks is not None and len(result.masks) > 0 else None
            )
            matte_result = self.matte.generate_matte(
                image,
                mask=mask,
                previous_frame=previous_frame,
            )
            result.alpha = matte_result.alpha
            result.foreground = matte_result.foreground
            result.edge_mask = matte_result.edge_mask
            result.hair_mask = matte_result.hair_mask
            result.motion_mask = matte_result.motion_mask

    def _create_aov_package(self, result: ProcessingResult) -> Dict[str, np.ndarray]:
        """Create AOV (Arbitrary Output Variable) package for compositing."""
        aovs = {}

        # Alpha/Matte
        if result.alpha is not None:
            aovs["alpha"] = result.alpha
            aovs["matte"] = result.alpha

        # Depth passes
        if result.depth_normalized is not None:
            aovs["depth"] = result.depth_normalized
        if result.z_depth is not None:
            aovs["z_depth"] = result.z_depth
        if result.disparity is not None:
            aovs["disparity"] = result.disparity

        # Normals (camera space)
        if result.normals is not None:
            # Convert from [-1,1] to [0,1] for compositing
            normals_viz = (result.normals + 1) / 2
            aovs["normals"] = normals_viz
            aovs["normal_x"] = normals_viz[..., 0]
            aovs["normal_y"] = normals_viz[..., 1]
            aovs["normal_z"] = normals_viz[..., 2]

        # Masks
        if result.masks is not None and len(result.masks) > 0:
            aovs["segmentation"] = result.masks[0].astype(np.float32)
        if result.edge_mask is not None:
            aovs["edge_mask"] = result.edge_mask
        if result.hair_mask is not None:
            aovs["hair_mask"] = result.hair_mask
        if result.motion_mask is not None:
            aovs["motion_mask"] = result.motion_mask

        # Foreground
        if result.foreground is not None:
            aovs["foreground"] = result.foreground

        return aovs

    def segment(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: SegmentationPrompt,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quick segmentation using SAM3.

        Returns:
            Tuple of (masks, scores)
        """
        if not self.sam3:
            raise RuntimeError("SAM3 not loaded. Enable enable_sam3 in config.")

        result = self.sam3.segment(image, prompt)
        return result.masks, result.scores

    def estimate_depth(
        self,
        image: Union[np.ndarray, Image.Image],
        generate_normals: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Quick depth estimation.

        Returns:
            Tuple of (depth_map, normals)
        """
        if not self.depth:
            raise RuntimeError("Depth model not loaded. Enable enable_depth in config.")

        result = self.depth.estimate_depth(image, generate_normals=generate_normals)
        return result.depth_map, result.normals

    def generate_matte(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quick matte generation.

        Returns:
            Tuple of (alpha, foreground)
        """
        if not self.matte:
            raise RuntimeError("Matte model not loaded. Enable enable_matte in config.")

        result = self.matte.generate_matte(image, mask=mask)
        return result.alpha, result.foreground

    def auto_segment_and_matte(
        self,
        image: Union[np.ndarray, Image.Image],
        points: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
    ) -> ProcessingResult:
        """
        Automatic segmentation and matting in one call.

        Args:
            image: Input image
            points: Nx2 array of (x, y) foreground points
            boxes: Nx4 array of (x1, y1, x2, y2) bounding boxes

        Returns:
            Complete ProcessingResult
        """
        prompt = None
        if points is not None:
            prompt = SegmentationPrompt(
                prompt_type=PromptType.POINT,
                points=points,
                point_labels=np.ones(len(points), dtype=np.int32),
            )
        elif boxes is not None:
            prompt = SegmentationPrompt(
                prompt_type=PromptType.BOX,
                boxes=boxes,
            )

        return self.process(
            image,
            prompt=prompt,
            generate_depth=True,
            generate_normals=True,
            generate_matte=True,
        )

    def process_video_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        prompt: Optional[SegmentationPrompt] = None,
        previous_frame: Optional[np.ndarray] = None,
    ) -> ProcessingResult:
        """
        Process a single video frame with temporal consistency.

        Args:
            frame: Current frame
            frame_index: Frame number (0-indexed)
            prompt: Segmentation prompt (use same prompt for consistency)
            previous_frame: Previous frame for motion analysis

        Returns:
            ProcessingResult with temporal smoothing applied
        """
        result = self.process(
            frame,
            prompt=prompt,
            previous_frame=previous_frame,
            generate_depth=True,
            generate_normals=True,
            generate_matte=True,
        )

        result.metadata["frame_index"] = frame_index

        return result

    def reset_temporal_state(self) -> None:
        """Reset temporal buffers (call between shots/scenes)."""
        if self.sam3:
            self.sam3.reset_temporal_buffer()
        if self.depth:
            self.depth.reset_temporal_buffer()
        if self.matte:
            self.matte.reset_temporal_buffer()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage of all models."""
        usage = {"total_gpu_mb": 0}

        if self.sam3:
            sam3_usage = self.sam3.get_memory_usage()
            usage["sam3"] = sam3_usage
            usage["total_gpu_mb"] += sam3_usage.get("gpu_allocated_mb", 0)

        if self.depth:
            depth_usage = self.depth.get_memory_usage()
            usage["depth"] = depth_usage
            usage["total_gpu_mb"] += depth_usage.get("gpu_allocated_mb", 0)

        if self.matte:
            matte_usage = self.matte.get_memory_usage()
            usage["matte"] = matte_usage
            usage["total_gpu_mb"] += matte_usage.get("gpu_allocated_mb", 0)

        return usage

    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_models()
        return False
