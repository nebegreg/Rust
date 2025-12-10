"""
Ultimate Rotoscopy Pipeline
============================

Cinema-quality rotoscopy pipeline integrating:
- SAM3 for intelligent segmentation
- Depth Anything 3 for depth-aware processing
- Advanced edge refinement with trimap
- Temporal consistency and tracking
- Professional matte extraction
- Keyframe-based workflow

Designed for VFX professionals working on:
- Feature films
- High-end commercials
- Streaming content
- Complex compositing shots
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


class RotoMode(Enum):
    """Rotoscopy processing modes."""
    AUTOMATIC = "automatic"          # Fully automatic segmentation
    SEMI_AUTO = "semi_auto"          # User-guided with AI assistance
    KEYFRAME = "keyframe"            # Keyframe-based propagation
    TRACKING = "tracking"            # Object tracking mode
    REFINEMENT = "refinement"        # Refine existing mattes


class EdgeMode(Enum):
    """Edge processing modes."""
    HARD = "hard"                    # Sharp binary edges
    SOFT = "soft"                    # Feathered edges
    MOTION_AWARE = "motion_aware"    # Motion blur compensation
    HAIR_DETAIL = "hair_detail"      # Fine detail preservation
    ADAPTIVE = "adaptive"            # Auto-select based on content


class OutputFormat(Enum):
    """Output format options."""
    EXR_16 = "exr_16"               # 16-bit half float EXR
    EXR_32 = "exr_32"               # 32-bit float EXR
    PNG_16 = "png_16"               # 16-bit PNG
    PNG_8 = "png_8"                 # 8-bit PNG
    TIFF_16 = "tiff_16"             # 16-bit TIFF
    DPX = "dpx"                     # DPX for film


class MatteChannel(Enum):
    """Matte output channels."""
    ALPHA = "alpha"                 # Main alpha channel
    CORE = "core"                   # Core matte (inner)
    EDGE = "edge"                   # Edge matte only
    DEPTH = "depth"                 # Depth channel
    MOTION = "motion"               # Motion vectors
    OBJECT_ID = "object_id"         # Object ID pass
    CRYPTOMATTE = "cryptomatte"     # Cryptomatte compatible


@dataclass
class Keyframe:
    """Keyframe data for rotoscopy."""
    frame_number: int
    mask: np.ndarray
    prompts: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    is_user_defined: bool = False
    interpolation: str = "linear"  # linear, smooth, hold
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RotoObject:
    """Tracked rotoscopy object."""
    object_id: int
    name: str
    color: Tuple[int, int, int] = (255, 0, 0)
    keyframes: List[Keyframe] = field(default_factory=list)
    tracking_data: Optional[Dict[int, np.ndarray]] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    # Advanced properties
    edge_mode: EdgeMode = EdgeMode.ADAPTIVE
    feather_amount: float = 0.0
    choke_amount: float = 0.0
    motion_blur_samples: int = 1
    depth_priority: float = 0.5  # 0 = background, 1 = foreground


@dataclass
class RotoConfig:
    """Configuration for rotoscopy pipeline."""
    # Processing modes
    mode: RotoMode = RotoMode.AUTOMATIC
    edge_mode: EdgeMode = EdgeMode.ADAPTIVE

    # AI Model settings
    sam_model_size: str = "large"
    depth_model_size: str = "large"
    use_text_prompts: bool = True
    use_depth_guidance: bool = True

    # Quality settings
    output_resolution: Optional[Tuple[int, int]] = None  # None = input resolution
    supersampling: float = 1.0  # 1.0 = native, 2.0 = 2x
    edge_refinement_iterations: int = 3
    temporal_smoothing: float = 0.85

    # Edge settings
    feather_radius: float = 0.0
    choke_radius: float = 0.0
    edge_blur: float = 0.0

    # Motion blur
    motion_blur_compensation: bool = True
    motion_blur_samples: int = 5

    # Hair/fine detail
    hair_detail_enhancement: bool = True
    hair_detail_threshold: float = 0.3

    # Depth integration
    depth_aware_edges: bool = True
    depth_feather_falloff: float = 1.0

    # Output settings
    output_format: OutputFormat = OutputFormat.EXR_16
    output_channels: List[MatteChannel] = field(
        default_factory=lambda: [MatteChannel.ALPHA]
    )

    # Performance
    batch_size: int = 4
    num_workers: int = 4
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    cache_embeddings: bool = True

    # Workflow
    auto_keyframe_interval: int = 24  # frames
    propagation_confidence_threshold: float = 0.7
    scene_detection: bool = True
    scene_detection_threshold: float = 0.3


@dataclass
class RotoResult:
    """Result from rotoscopy processing."""
    frame_number: int
    masks: Dict[int, np.ndarray]          # object_id -> mask
    alpha: np.ndarray                      # Combined alpha
    core_matte: Optional[np.ndarray] = None
    edge_matte: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    motion_vectors: Optional[np.ndarray] = None
    confidence: Dict[int, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UltimateRotoPipeline:
    """
    Ultimate Rotoscopy Pipeline for Cinema-Quality Results.

    Integrates:
    - SAM3 for intelligent, text-guided segmentation
    - Depth Anything 3 for depth-aware processing
    - Advanced trimap-based edge refinement
    - Temporal consistency with optical flow
    - Professional matte extraction pipeline
    - Keyframe-based workflow with smart propagation

    Example:
        >>> pipeline = UltimateRotoPipeline()
        >>> pipeline.load_models()
        >>>
        >>> # Automatic mode
        >>> results = pipeline.process_sequence(
        ...     frames,
        ...     text_prompts=["person in foreground"],
        ... )
        >>>
        >>> # Keyframe mode
        >>> pipeline.add_keyframe(0, user_mask, object_id=1)
        >>> pipeline.add_keyframe(100, user_mask, object_id=1)
        >>> results = pipeline.propagate_keyframes(frames)
    """

    def __init__(self, config: Optional[RotoConfig] = None):
        self.config = config or RotoConfig()
        self.sam_model = None
        self.depth_model = None
        self.matting_model = None
        self.flow_model = None
        self.objects: Dict[int, RotoObject] = {}
        self._next_object_id = 1
        self._frame_cache: Dict[int, np.ndarray] = {}
        self._embedding_cache: Dict[int, torch.Tensor] = {}
        self._depth_cache: Dict[int, np.ndarray] = {}
        self._flow_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._is_loaded = False
        self._device = None
        self._lock = threading.Lock()

    def load_models(self, progress_callback: Optional[Callable] = None) -> None:
        """Load all required AI models."""
        if self._is_loaded:
            return

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        )

        total_steps = 4
        current_step = 0

        def update_progress(msg: str):
            nonlocal current_step
            current_step += 1
            if progress_callback:
                progress_callback(current_step / total_steps, msg)
            print(f"[{current_step}/{total_steps}] {msg}")

        # Load SAM3
        update_progress("Loading SAM3 segmentation model...")
        self._load_sam_model()

        # Load Depth Anything 3
        update_progress("Loading Depth Anything 3 model...")
        self._load_depth_model()

        # Load matting refinement model
        update_progress("Loading matting refinement model...")
        self._load_matting_model()

        # Load optical flow model
        update_progress("Loading optical flow model...")
        self._load_flow_model()

        self._is_loaded = True
        print("All models loaded successfully!")

    def _load_sam_model(self) -> None:
        """Load SAM3 model."""
        from ultimate_rotoscopy.models.sam3 import (
            SAM3Segmentor, SAM3Config, SAM3ModelSize
        )

        size_map = {
            "tiny": SAM3ModelSize.TINY,
            "small": SAM3ModelSize.SMALL,
            "base": SAM3ModelSize.BASE,
            "large": SAM3ModelSize.LARGE,
        }

        config = SAM3Config(
            model_size=size_map.get(self.config.sam_model_size, SAM3ModelSize.LARGE),
            use_temporal_consistency=True,
            edge_refinement=True,
        )

        self.sam_model = SAM3Segmentor(config)
        self.sam_model.load()

    def _load_depth_model(self) -> None:
        """Load Depth Anything 3 model."""
        from ultimate_rotoscopy.models.depth_anything import (
            DepthAnythingV3, DepthConfig, DepthModelSize
        )

        size_map = {
            "small": DepthModelSize.SMALL,
            "base": DepthModelSize.BASE,
            "large": DepthModelSize.LARGE,
            "giant": DepthModelSize.GIANT,
        }

        config = DepthConfig(
            model_size=size_map.get(self.config.depth_model_size, DepthModelSize.LARGE),
            use_depth_ray=True,
            estimate_intrinsics=True,
            sky_segmentation=True,
            temporal_smoothing=True,
        )

        self.depth_model = DepthAnythingV3(config)
        self.depth_model.load()

    def _load_matting_model(self) -> None:
        """Load matting refinement model (ViTMatte or similar)."""
        try:
            from ultimate_rotoscopy.models.vitmatte import ViTMatte, ViTMatteConfig
            config = ViTMatteConfig()
            self.matting_model = ViTMatte(config)
            self.matting_model.load()
        except ImportError:
            # Fallback to guided filter refinement
            self.matting_model = None
            print("ViTMatte not available, using guided filter refinement")

    def _load_flow_model(self) -> None:
        """Load optical flow model for temporal consistency."""
        try:
            from ultimate_rotoscopy.models.raft import RAFTModel
            self.flow_model = RAFTModel()
            self.flow_model.load()
        except ImportError:
            # Fallback to OpenCV optical flow
            self.flow_model = None
            print("RAFT not available, using OpenCV Farneback flow")

    # =========================================================================
    # Object Management
    # =========================================================================

    def create_object(
        self,
        name: str,
        color: Optional[Tuple[int, int, int]] = None,
        edge_mode: EdgeMode = EdgeMode.ADAPTIVE,
    ) -> int:
        """Create a new rotoscopy object to track."""
        object_id = self._next_object_id
        self._next_object_id += 1

        if color is None:
            # Generate distinct color
            hue = (object_id * 137) % 360
            color = self._hsv_to_rgb(hue, 1.0, 1.0)

        self.objects[object_id] = RotoObject(
            object_id=object_id,
            name=name,
            color=color,
            edge_mode=edge_mode,
        )

        return object_id

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV to RGB."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    def add_keyframe(
        self,
        frame_number: int,
        mask: np.ndarray,
        object_id: int,
        prompts: Optional[Dict[str, Any]] = None,
        interpolation: str = "smooth",
    ) -> Keyframe:
        """Add a keyframe for an object."""
        if object_id not in self.objects:
            raise ValueError(f"Object {object_id} not found")

        keyframe = Keyframe(
            frame_number=frame_number,
            mask=mask.copy(),
            prompts=prompts,
            confidence=1.0,
            is_user_defined=True,
            interpolation=interpolation,
        )

        # Insert in sorted order
        obj = self.objects[object_id]
        obj.keyframes = sorted(
            obj.keyframes + [keyframe],
            key=lambda k: k.frame_number
        )

        return keyframe

    def remove_keyframe(self, frame_number: int, object_id: int) -> bool:
        """Remove a keyframe."""
        if object_id not in self.objects:
            return False

        obj = self.objects[object_id]
        obj.keyframes = [k for k in obj.keyframes if k.frame_number != frame_number]
        return True

    # =========================================================================
    # Main Processing Methods
    # =========================================================================

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        text_prompts: Optional[List[str]] = None,
        point_prompts: Optional[Dict[int, np.ndarray]] = None,
        box_prompts: Optional[Dict[int, np.ndarray]] = None,
        existing_masks: Optional[Dict[int, np.ndarray]] = None,
    ) -> RotoResult:
        """
        Process a single frame.

        Args:
            frame: Input frame (RGB, HxWx3)
            frame_number: Frame index
            text_prompts: List of text descriptions for objects
            point_prompts: Dict of object_id -> point coordinates
            box_prompts: Dict of object_id -> bounding boxes
            existing_masks: Dict of object_id -> existing masks to refine

        Returns:
            RotoResult with processed mattes
        """
        start_time = time.time()

        # Get depth map if needed
        depth = None
        if self.config.use_depth_guidance:
            depth = self._get_depth(frame, frame_number)

        # Segment objects
        masks = {}
        confidences = {}

        if text_prompts and self.config.use_text_prompts:
            # Text-based segmentation (SAM3)
            for i, text in enumerate(text_prompts):
                obj_id = self.create_object(text) if i >= len(self.objects) else list(self.objects.keys())[i]
                mask, conf = self._segment_with_text(frame, text, depth)
                masks[obj_id] = mask
                confidences[obj_id] = conf

        if point_prompts:
            for obj_id, points in point_prompts.items():
                if obj_id not in self.objects:
                    obj_id = self.create_object(f"Object {obj_id}")
                mask, conf = self._segment_with_points(frame, points, depth)
                masks[obj_id] = mask
                confidences[obj_id] = conf

        if box_prompts:
            for obj_id, boxes in box_prompts.items():
                if obj_id not in self.objects:
                    obj_id = self.create_object(f"Object {obj_id}")
                mask, conf = self._segment_with_boxes(frame, boxes, depth)
                masks[obj_id] = mask
                confidences[obj_id] = conf

        if existing_masks:
            for obj_id, mask in existing_masks.items():
                refined, conf = self._refine_mask(frame, mask, depth)
                masks[obj_id] = refined
                confidences[obj_id] = conf

        # Refine edges for all masks
        for obj_id, mask in masks.items():
            obj = self.objects.get(obj_id)
            edge_mode = obj.edge_mode if obj else self.config.edge_mode
            masks[obj_id] = self._refine_edges(frame, mask, depth, edge_mode)

        # Combine masks into final alpha
        alpha = self._combine_masks(masks, depth)

        # Generate additional outputs
        core_matte = None
        edge_matte = None

        if MatteChannel.CORE in self.config.output_channels:
            core_matte = self._generate_core_matte(alpha)

        if MatteChannel.EDGE in self.config.output_channels:
            edge_matte = self._generate_edge_matte(alpha)

        processing_time = time.time() - start_time

        return RotoResult(
            frame_number=frame_number,
            masks=masks,
            alpha=alpha,
            core_matte=core_matte,
            edge_matte=edge_matte,
            depth=depth,
            confidence=confidences,
            metadata={
                "processing_time_ms": processing_time * 1000,
                "num_objects": len(masks),
            }
        )

    def process_sequence(
        self,
        frames: List[np.ndarray],
        text_prompts: Optional[List[str]] = None,
        start_frame: int = 0,
        progress_callback: Optional[Callable] = None,
    ) -> List[RotoResult]:
        """
        Process a sequence of frames.

        Args:
            frames: List of input frames
            text_prompts: Text descriptions for automatic segmentation
            start_frame: Starting frame number
            progress_callback: Callback for progress updates

        Returns:
            List of RotoResults
        """
        results = []
        total_frames = len(frames)

        # Detect scene changes
        scene_changes = []
        if self.config.scene_detection:
            scene_changes = self._detect_scene_changes(frames)

        # Process frames
        prev_masks = None

        for i, frame in enumerate(frames):
            frame_num = start_frame + i

            # Check for scene change - reset tracking
            if frame_num in scene_changes:
                prev_masks = None
                self._clear_temporal_state()

            # Use previous masks for guidance if available
            if prev_masks is not None:
                # Propagate masks using optical flow
                propagated = self._propagate_masks(
                    frames[i-1] if i > 0 else frame,
                    frame,
                    prev_masks,
                    frame_num - 1,
                    frame_num,
                )
                result = self.process_frame(
                    frame,
                    frame_num,
                    text_prompts=text_prompts if i == 0 or frame_num in scene_changes else None,
                    existing_masks=propagated,
                )
            else:
                result = self.process_frame(
                    frame,
                    frame_num,
                    text_prompts=text_prompts,
                )

            results.append(result)
            prev_masks = result.masks

            if progress_callback:
                progress_callback(
                    (i + 1) / total_frames,
                    f"Processing frame {frame_num} ({i+1}/{total_frames})"
                )

        # Apply temporal smoothing
        if self.config.temporal_smoothing > 0:
            results = self._apply_temporal_smoothing(results)

        return results

    def propagate_keyframes(
        self,
        frames: List[np.ndarray],
        object_id: Optional[int] = None,
        start_frame: int = 0,
        progress_callback: Optional[Callable] = None,
    ) -> List[RotoResult]:
        """
        Propagate keyframes through a sequence.

        This is the professional workflow:
        1. User sets keyframes at important frames
        2. System interpolates/propagates between keyframes
        3. AI refines the propagated masks

        Args:
            frames: Input frames
            object_id: Specific object to propagate (None = all)
            start_frame: Starting frame number
            progress_callback: Progress callback

        Returns:
            List of RotoResults
        """
        results = []
        total_frames = len(frames)

        # Get objects to process
        objects_to_process = (
            {object_id: self.objects[object_id]}
            if object_id and object_id in self.objects
            else self.objects
        )

        for i, frame in enumerate(frames):
            frame_num = start_frame + i
            masks = {}
            confidences = {}

            for obj_id, obj in objects_to_process.items():
                # Find surrounding keyframes
                prev_kf, next_kf = self._find_surrounding_keyframes(
                    obj.keyframes, frame_num
                )

                if prev_kf is None and next_kf is None:
                    # No keyframes - skip or use AI
                    continue

                if prev_kf and prev_kf.frame_number == frame_num:
                    # Exact keyframe match
                    mask = prev_kf.mask
                    conf = prev_kf.confidence
                elif prev_kf and next_kf:
                    # Interpolate between keyframes
                    mask = self._interpolate_keyframes(
                        frames, prev_kf, next_kf, frame_num, start_frame
                    )
                    # Refine with AI
                    mask, conf = self._refine_mask(frame, mask)
                elif prev_kf:
                    # Forward propagate from previous keyframe
                    mask = self._propagate_single_mask(
                        frames, prev_kf, frame_num, start_frame, forward=True
                    )
                    mask, conf = self._refine_mask(frame, mask)
                else:
                    # Backward propagate from next keyframe
                    mask = self._propagate_single_mask(
                        frames, next_kf, frame_num, start_frame, forward=False
                    )
                    mask, conf = self._refine_mask(frame, mask)

                masks[obj_id] = mask
                confidences[obj_id] = conf

            # Combine and create result
            depth = self._get_depth(frame, frame_num) if self.config.use_depth_guidance else None
            alpha = self._combine_masks(masks, depth)

            results.append(RotoResult(
                frame_number=frame_num,
                masks=masks,
                alpha=alpha,
                depth=depth,
                confidence=confidences,
            ))

            if progress_callback:
                progress_callback(
                    (i + 1) / total_frames,
                    f"Propagating frame {frame_num}"
                )

        return results

    # =========================================================================
    # Segmentation Methods
    # =========================================================================

    def _segment_with_text(
        self,
        frame: np.ndarray,
        text: str,
        depth: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Segment using text prompt (SAM3 feature)."""
        from ultimate_rotoscopy.models.sam3 import (
            SegmentationPrompt, PromptType
        )

        prompt = SegmentationPrompt(
            prompt_type=PromptType.TEXT,
            text_prompt=text,
            use_presence_token=True,
        )

        result = self.sam_model.segment(frame, prompt)

        # Get best mask
        best_idx = result.scores.argmax()
        mask = result.masks[best_idx].astype(np.float32)
        confidence = float(result.scores[best_idx])

        return mask, confidence

    def _segment_with_points(
        self,
        frame: np.ndarray,
        points: np.ndarray,
        depth: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Segment using point prompts."""
        from ultimate_rotoscopy.models.sam3 import (
            SegmentationPrompt, PromptType
        )

        # Assume all points are foreground
        labels = np.ones(len(points), dtype=np.int32)

        prompt = SegmentationPrompt(
            prompt_type=PromptType.POINT,
            points=points,
            point_labels=labels,
        )

        result = self.sam_model.segment(frame, prompt)

        best_idx = result.scores.argmax()
        mask = result.masks[best_idx].astype(np.float32)
        confidence = float(result.scores[best_idx])

        return mask, confidence

    def _segment_with_boxes(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        depth: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Segment using bounding box prompts."""
        from ultimate_rotoscopy.models.sam3 import (
            SegmentationPrompt, PromptType
        )

        prompt = SegmentationPrompt(
            prompt_type=PromptType.BOX,
            boxes=boxes,
        )

        result = self.sam_model.segment(frame, prompt)

        best_idx = result.scores.argmax()
        mask = result.masks[best_idx].astype(np.float32)
        confidence = float(result.scores[best_idx])

        return mask, confidence

    def _refine_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        depth: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Refine an existing mask using AI."""
        from ultimate_rotoscopy.models.sam3 import (
            SegmentationPrompt, PromptType
        )

        # Use mask as input prompt
        prompt = SegmentationPrompt(
            prompt_type=PromptType.MASK,
            mask_input=mask,
        )

        result = self.sam_model.segment(frame, prompt)

        best_idx = result.scores.argmax()
        refined = result.masks[best_idx].astype(np.float32)
        confidence = float(result.scores[best_idx])

        return refined, confidence

    # =========================================================================
    # Edge Refinement
    # =========================================================================

    def _refine_edges(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        depth: Optional[np.ndarray],
        edge_mode: EdgeMode,
    ) -> np.ndarray:
        """Apply edge refinement based on mode."""
        if edge_mode == EdgeMode.HARD:
            return (mask > 0.5).astype(np.float32)

        elif edge_mode == EdgeMode.SOFT:
            return self._apply_feathering(mask, self.config.feather_radius)

        elif edge_mode == EdgeMode.MOTION_AWARE:
            return self._motion_aware_edges(frame, mask)

        elif edge_mode == EdgeMode.HAIR_DETAIL:
            return self._hair_detail_refinement(frame, mask)

        else:  # ADAPTIVE
            return self._adaptive_edge_refinement(frame, mask, depth)

    def _apply_feathering(self, mask: np.ndarray, radius: float) -> np.ndarray:
        """Apply Gaussian feathering to mask edges."""
        if radius <= 0:
            return mask

        # Convert radius to kernel size (must be odd)
        ksize = int(radius * 2) | 1
        return cv2.GaussianBlur(mask, (ksize, ksize), radius / 3)

    def _motion_aware_edges(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Handle motion blur in edges."""
        # Detect motion blur direction using gradient analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Estimate blur direction
        angles = np.arctan2(gy, gx)
        magnitudes = np.sqrt(gx**2 + gy**2)

        # Weighted average angle at mask edges
        edge_mask = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        edge_mask = edge_mask > 0

        if edge_mask.sum() > 0:
            avg_angle = np.average(angles[edge_mask], weights=magnitudes[edge_mask] + 1e-6)

            # Apply directional blur to edges
            blur_length = self.config.motion_blur_samples
            kernel = self._create_motion_blur_kernel(blur_length, avg_angle)

            # Blur only the edge region
            edge_region = cv2.dilate(edge_mask.astype(np.uint8), np.ones((5, 5)))
            blurred = cv2.filter2D(mask, -1, kernel)

            # Blend
            edge_region_f = edge_region.astype(np.float32)
            mask = mask * (1 - edge_region_f) + blurred * edge_region_f

        return mask

    def _create_motion_blur_kernel(self, length: int, angle: float) -> np.ndarray:
        """Create directional motion blur kernel."""
        kernel = np.zeros((length, length), dtype=np.float32)
        center = length // 2

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        for i in range(length):
            offset = i - center
            x = int(center + offset * cos_a)
            y = int(center + offset * sin_a)
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1.0

        return kernel / kernel.sum()

    def _hair_detail_refinement(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Preserve fine details like hair using guided filtering."""
        # Convert to appropriate format
        guide = frame.astype(np.float32) / 255.0
        mask_f = mask.astype(np.float32)

        # Multi-scale guided filtering
        scales = [1, 2, 4]
        refined = np.zeros_like(mask_f)

        for scale in scales:
            if scale > 1:
                guide_scaled = cv2.resize(
                    guide,
                    (guide.shape[1] // scale, guide.shape[0] // scale)
                )
                mask_scaled = cv2.resize(
                    mask_f,
                    (mask_f.shape[1] // scale, mask_f.shape[0] // scale)
                )
            else:
                guide_scaled = guide
                mask_scaled = mask_f

            # Apply guided filter
            filtered = self._guided_filter(
                guide_scaled,
                mask_scaled,
                radius=max(1, 8 // scale),
                eps=0.01 * scale
            )

            if scale > 1:
                filtered = cv2.resize(filtered, (mask_f.shape[1], mask_f.shape[0]))

            refined += filtered

        refined /= len(scales)

        # Enhance fine details
        detail = mask_f - cv2.GaussianBlur(mask_f, (15, 15), 0)
        threshold = self.config.hair_detail_threshold
        detail_mask = np.abs(detail) > threshold

        refined[detail_mask] = mask_f[detail_mask]

        return np.clip(refined, 0, 1)

    def _guided_filter(
        self,
        guide: np.ndarray,
        src: np.ndarray,
        radius: int,
        eps: float,
    ) -> np.ndarray:
        """Apply guided filter for edge-preserving smoothing."""
        if guide.ndim == 3:
            guide = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)

        guide = guide.astype(np.float32)
        src = src.astype(np.float32)

        mean_g = cv2.boxFilter(guide, -1, (radius, radius))
        mean_s = cv2.boxFilter(src, -1, (radius, radius))
        mean_gs = cv2.boxFilter(guide * src, -1, (radius, radius))
        mean_gg = cv2.boxFilter(guide * guide, -1, (radius, radius))

        cov_gs = mean_gs - mean_g * mean_s
        var_g = mean_gg - mean_g * mean_g

        a = cov_gs / (var_g + eps)
        b = mean_s - a * mean_g

        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))

        return mean_a * guide + mean_b

    def _adaptive_edge_refinement(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        depth: Optional[np.ndarray],
    ) -> np.ndarray:
        """Automatically choose and apply best edge refinement."""
        # Analyze edge characteristics
        edge_mask = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)

        # Compute edge complexity (high frequency content)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        edge_region = cv2.dilate(edge_mask, np.ones((11, 11)))

        # Laplacian variance in edge region
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        edge_complexity = np.var(laplacian[edge_region > 0]) if edge_region.sum() > 0 else 0

        # Decide refinement strategy
        if edge_complexity > 1000:  # High detail (hair, fur)
            refined = self._hair_detail_refinement(frame, mask)
        elif edge_complexity > 100:  # Medium detail
            # Use matting model if available
            if self.matting_model is not None:
                trimap = self._generate_trimap(mask)
                refined = self._apply_matting_model(frame, trimap)
            else:
                refined = self._hair_detail_refinement(frame, mask)
        else:  # Clean edges
            refined = self._apply_feathering(mask, self.config.feather_radius)

        # Apply depth-aware edge adjustment if available
        if depth is not None and self.config.depth_aware_edges:
            refined = self._depth_aware_edge_adjustment(refined, depth)

        return refined

    def _generate_trimap(self, mask: np.ndarray, dilation: int = 15) -> np.ndarray:
        """Generate trimap from binary/soft mask."""
        # Threshold to binary
        binary = (mask > 0.5).astype(np.uint8)

        # Erode for definite foreground
        kernel = np.ones((dilation, dilation), np.uint8)
        fg = cv2.erode(binary, kernel)

        # Dilate for definite background
        bg = cv2.dilate(binary, kernel)

        # Trimap: 0 = background, 128 = unknown, 255 = foreground
        trimap = np.zeros_like(mask, dtype=np.uint8)
        trimap[bg == 1] = 128  # Unknown (dilated region)
        trimap[fg == 1] = 255  # Definite foreground

        return trimap

    def _apply_matting_model(
        self,
        frame: np.ndarray,
        trimap: np.ndarray,
    ) -> np.ndarray:
        """Apply matting model for alpha refinement."""
        if self.matting_model is not None:
            return self.matting_model.predict(frame, trimap)

        # Fallback: use guided filter based matting
        return self._guided_filter_matting(frame, trimap)

    def _guided_filter_matting(
        self,
        frame: np.ndarray,
        trimap: np.ndarray,
    ) -> np.ndarray:
        """Guided filter based alpha matting."""
        # Initial alpha from trimap
        alpha = trimap.astype(np.float32) / 255.0

        # Unknown region
        unknown = (trimap == 128)

        if not unknown.any():
            return alpha

        # Apply guided filter in unknown region
        guide = frame.astype(np.float32) / 255.0
        refined = self._guided_filter(guide, alpha, radius=15, eps=0.001)

        # Keep known regions unchanged
        alpha[unknown] = refined[unknown]

        return np.clip(alpha, 0, 1)

    def _depth_aware_edge_adjustment(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        """Adjust edges based on depth information."""
        # Normalize depth
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

        # Compute depth gradient
        grad_x = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
        depth_edges = np.sqrt(grad_x**2 + grad_y**2)

        # Where depth edges are strong, preserve mask edges
        depth_edge_weight = np.clip(depth_edges * 5, 0, 1)

        # Soften mask where depth is smooth
        soft_mask = cv2.GaussianBlur(mask, (5, 5), 1)

        # Blend based on depth edges
        adjusted = mask * depth_edge_weight + soft_mask * (1 - depth_edge_weight)

        return adjusted

    # =========================================================================
    # Temporal Processing
    # =========================================================================

    def _get_depth(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Get depth map, using cache if available."""
        if frame_number in self._depth_cache:
            return self._depth_cache[frame_number]

        result = self.depth_model.estimate_depth(frame)
        depth = result.depth_normalized

        self._depth_cache[frame_number] = depth
        return depth

    def _propagate_masks(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        masks: Dict[int, np.ndarray],
        prev_frame_num: int,
        curr_frame_num: int,
    ) -> Dict[int, np.ndarray]:
        """Propagate masks from previous to current frame using optical flow."""
        flow = self._compute_flow(prev_frame, curr_frame, prev_frame_num, curr_frame_num)

        propagated = {}
        for obj_id, mask in masks.items():
            propagated[obj_id] = self._warp_mask(mask, flow)

        return propagated

    def _compute_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        frame1_num: int,
        frame2_num: int,
    ) -> np.ndarray:
        """Compute optical flow between frames."""
        cache_key = (frame1_num, frame2_num)
        if cache_key in self._flow_cache:
            return self._flow_cache[cache_key]

        if self.flow_model is not None:
            flow = self.flow_model.compute(frame1, frame2)
        else:
            # Fallback to Farneback
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

        self._flow_cache[cache_key] = flow
        return flow

    def _warp_mask(self, mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp mask using optical flow."""
        h, w = mask.shape[:2]

        # Create sampling grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # Apply flow
        x_new = x + flow[..., 0]
        y_new = y + flow[..., 1]

        # Warp using remap
        warped = cv2.remap(
            mask.astype(np.float32),
            x_new, y_new,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return warped

    def _detect_scene_changes(
        self,
        frames: List[np.ndarray],
    ) -> List[int]:
        """Detect scene changes in frame sequence."""
        scene_changes = []
        threshold = self.config.scene_detection_threshold

        for i in range(1, len(frames)):
            # Compute histogram difference
            hist1 = cv2.calcHist([frames[i-1]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frames[i]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()

            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            if diff < (1 - threshold):
                scene_changes.append(i)

        return scene_changes

    def _apply_temporal_smoothing(
        self,
        results: List[RotoResult],
    ) -> List[RotoResult]:
        """Apply temporal smoothing across results."""
        if len(results) < 3:
            return results

        alpha = self.config.temporal_smoothing

        for i in range(1, len(results) - 1):
            prev_alpha = results[i-1].alpha
            curr_alpha = results[i].alpha
            next_alpha = results[i+1].alpha

            # Weighted blend
            smoothed = (
                0.25 * prev_alpha +
                0.5 * curr_alpha +
                0.25 * next_alpha
            )

            # Blend with original based on alpha setting
            results[i].alpha = alpha * smoothed + (1 - alpha) * curr_alpha

            # Smooth individual masks too
            for obj_id in results[i].masks:
                if obj_id in results[i-1].masks and obj_id in results[i+1].masks:
                    prev_mask = results[i-1].masks[obj_id]
                    curr_mask = results[i].masks[obj_id]
                    next_mask = results[i+1].masks[obj_id]

                    smoothed_mask = (
                        0.25 * prev_mask +
                        0.5 * curr_mask +
                        0.25 * next_mask
                    )
                    results[i].masks[obj_id] = alpha * smoothed_mask + (1 - alpha) * curr_mask

        return results

    def _clear_temporal_state(self) -> None:
        """Clear temporal processing state (call on scene changes)."""
        if self.sam_model:
            self.sam_model.reset_temporal_buffer()
        if self.depth_model:
            self.depth_model.reset_temporal_buffer()

    # =========================================================================
    # Keyframe Processing
    # =========================================================================

    def _find_surrounding_keyframes(
        self,
        keyframes: List[Keyframe],
        frame_number: int,
    ) -> Tuple[Optional[Keyframe], Optional[Keyframe]]:
        """Find keyframes before and after given frame."""
        prev_kf = None
        next_kf = None

        for kf in keyframes:
            if kf.frame_number <= frame_number:
                prev_kf = kf
            elif kf.frame_number > frame_number and next_kf is None:
                next_kf = kf
                break

        return prev_kf, next_kf

    def _interpolate_keyframes(
        self,
        frames: List[np.ndarray],
        prev_kf: Keyframe,
        next_kf: Keyframe,
        target_frame: int,
        start_frame: int,
    ) -> np.ndarray:
        """Interpolate between two keyframes."""
        # Calculate interpolation factor
        total_distance = next_kf.frame_number - prev_kf.frame_number
        current_distance = target_frame - prev_kf.frame_number
        t = current_distance / total_distance

        # Apply interpolation curve
        if prev_kf.interpolation == "smooth":
            # Smooth step
            t = t * t * (3 - 2 * t)
        elif prev_kf.interpolation == "hold":
            t = 0.0 if t < 0.5 else 1.0
        # else: linear (t unchanged)

        # Get frames
        prev_idx = prev_kf.frame_number - start_frame
        next_idx = next_kf.frame_number - start_frame

        if 0 <= prev_idx < len(frames) and 0 <= next_idx < len(frames):
            # Compute flow for warping
            flow_forward = self._compute_flow(
                frames[prev_idx],
                frames[min(prev_idx + 1, len(frames) - 1)],
                prev_kf.frame_number,
                prev_kf.frame_number + 1,
            )

            flow_backward = self._compute_flow(
                frames[next_idx],
                frames[max(next_idx - 1, 0)],
                next_kf.frame_number,
                next_kf.frame_number - 1,
            )

            # Warp masks towards target
            warped_prev = self._warp_mask(prev_kf.mask, flow_forward * current_distance)
            warped_next = self._warp_mask(next_kf.mask, flow_backward * (total_distance - current_distance))

            # Blend
            interpolated = (1 - t) * warped_prev + t * warped_next
        else:
            # Simple blend without warping
            interpolated = (1 - t) * prev_kf.mask + t * next_kf.mask

        return interpolated

    def _propagate_single_mask(
        self,
        frames: List[np.ndarray],
        keyframe: Keyframe,
        target_frame: int,
        start_frame: int,
        forward: bool,
    ) -> np.ndarray:
        """Propagate a single mask forward or backward."""
        kf_idx = keyframe.frame_number - start_frame
        target_idx = target_frame - start_frame

        if not (0 <= kf_idx < len(frames) and 0 <= target_idx < len(frames)):
            return keyframe.mask

        mask = keyframe.mask.copy()

        if forward:
            indices = range(kf_idx, target_idx)
        else:
            indices = range(kf_idx, target_idx, -1)

        for i in indices:
            next_i = i + (1 if forward else -1)
            if 0 <= next_i < len(frames):
                flow = self._compute_flow(
                    frames[i], frames[next_i],
                    start_frame + i, start_frame + next_i
                )
                mask = self._warp_mask(mask, flow)

        return mask

    # =========================================================================
    # Output Generation
    # =========================================================================

    def _combine_masks(
        self,
        masks: Dict[int, np.ndarray],
        depth: Optional[np.ndarray],
    ) -> np.ndarray:
        """Combine multiple object masks into final alpha."""
        if not masks:
            return np.zeros((100, 100), dtype=np.float32)

        # Get shape from first mask
        first_mask = next(iter(masks.values()))
        h, w = first_mask.shape[:2]
        combined = np.zeros((h, w), dtype=np.float32)

        if depth is not None:
            # Depth-aware compositing
            # Sort objects by depth priority
            sorted_objs = sorted(
                masks.keys(),
                key=lambda oid: self.objects[oid].depth_priority if oid in self.objects else 0.5
            )

            for obj_id in sorted_objs:
                mask = masks[obj_id]
                # Over operation
                combined = combined + mask * (1 - combined)
        else:
            # Simple maximum blend
            for mask in masks.values():
                combined = np.maximum(combined, mask)

        return np.clip(combined, 0, 1)

    def _generate_core_matte(self, alpha: np.ndarray) -> np.ndarray:
        """Generate core (inner) matte."""
        # Erode alpha to get solid core
        kernel = np.ones((5, 5), np.uint8)
        core = cv2.erode((alpha * 255).astype(np.uint8), kernel, iterations=2)
        return core.astype(np.float32) / 255.0

    def _generate_edge_matte(self, alpha: np.ndarray) -> np.ndarray:
        """Generate edge-only matte."""
        core = self._generate_core_matte(alpha)
        edge = alpha - core
        return np.clip(edge, 0, 1)

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export_sequence(
        self,
        results: List[RotoResult],
        output_dir: Path,
        base_name: str = "roto",
        format: Optional[OutputFormat] = None,
        channels: Optional[List[MatteChannel]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[Path]:
        """
        Export rotoscopy results to files.

        Args:
            results: List of RotoResults
            output_dir: Output directory
            base_name: Base filename
            format: Output format (default from config)
            channels: Channels to export (default from config)
            progress_callback: Progress callback

        Returns:
            List of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        format = format or self.config.output_format
        channels = channels or self.config.output_channels

        output_paths = []
        total = len(results)

        for i, result in enumerate(results):
            frame_num = result.frame_number

            for channel in channels:
                # Get channel data
                if channel == MatteChannel.ALPHA:
                    data = result.alpha
                elif channel == MatteChannel.CORE:
                    data = result.core_matte or self._generate_core_matte(result.alpha)
                elif channel == MatteChannel.EDGE:
                    data = result.edge_matte or self._generate_edge_matte(result.alpha)
                elif channel == MatteChannel.DEPTH:
                    data = result.depth
                elif channel == MatteChannel.OBJECT_ID:
                    data = self._generate_object_id_pass(result.masks)
                else:
                    continue

                if data is None:
                    continue

                # Generate filename
                suffix = f"_{channel.value}" if len(channels) > 1 else ""
                filename = f"{base_name}{suffix}.{frame_num:04d}"

                # Export based on format
                output_path = self._export_frame(
                    data, output_dir / filename, format
                )
                output_paths.append(output_path)

            if progress_callback:
                progress_callback((i + 1) / total, f"Exporting frame {frame_num}")

        return output_paths

    def _export_frame(
        self,
        data: np.ndarray,
        base_path: Path,
        format: OutputFormat,
    ) -> Path:
        """Export single frame."""
        if format == OutputFormat.EXR_16 or format == OutputFormat.EXR_32:
            return self._export_exr(data, base_path, format == OutputFormat.EXR_32)
        elif format == OutputFormat.PNG_16:
            return self._export_png(data, base_path, 16)
        elif format == OutputFormat.PNG_8:
            return self._export_png(data, base_path, 8)
        elif format == OutputFormat.TIFF_16:
            return self._export_tiff(data, base_path)
        elif format == OutputFormat.DPX:
            return self._export_dpx(data, base_path)
        else:
            return self._export_png(data, base_path, 16)

    def _export_exr(
        self,
        data: np.ndarray,
        base_path: Path,
        use_float32: bool,
    ) -> Path:
        """Export as EXR."""
        try:
            import OpenEXR
            import Imath

            path = base_path.with_suffix(".exr")
            h, w = data.shape[:2]

            # Prepare data
            if use_float32:
                pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
                data_bytes = data.astype(np.float32).tobytes()
            else:
                pixel_type = Imath.PixelType(Imath.PixelType.HALF)
                data_bytes = data.astype(np.float16).tobytes()

            header = OpenEXR.Header(w, h)
            header["channels"] = {"A": Imath.Channel(pixel_type)}

            exr = OpenEXR.OutputFile(str(path), header)
            exr.writePixels({"A": data_bytes})
            exr.close()

            return path

        except ImportError:
            # Fallback to PNG
            print("OpenEXR not available, falling back to PNG")
            return self._export_png(data, base_path, 16)

    def _export_png(
        self,
        data: np.ndarray,
        base_path: Path,
        bit_depth: int,
    ) -> Path:
        """Export as PNG."""
        path = base_path.with_suffix(".png")

        if bit_depth == 16:
            data_int = (data * 65535).astype(np.uint16)
        else:
            data_int = (data * 255).astype(np.uint8)

        cv2.imwrite(str(path), data_int)
        return path

    def _export_tiff(self, data: np.ndarray, base_path: Path) -> Path:
        """Export as TIFF."""
        path = base_path.with_suffix(".tiff")
        data_int = (data * 65535).astype(np.uint16)
        cv2.imwrite(str(path), data_int)
        return path

    def _export_dpx(self, data: np.ndarray, base_path: Path) -> Path:
        """Export as DPX."""
        try:
            import imageio
            path = base_path.with_suffix(".dpx")
            data_int = (data * 1023).astype(np.uint16)  # 10-bit
            imageio.imwrite(str(path), data_int)
            return path
        except ImportError:
            return self._export_png(data, base_path, 16)

    def _generate_object_id_pass(
        self,
        masks: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Generate object ID pass."""
        if not masks:
            return None

        first_mask = next(iter(masks.values()))
        h, w = first_mask.shape[:2]

        # Create ID pass (each object gets unique integer ID)
        id_pass = np.zeros((h, w), dtype=np.float32)

        for obj_id, mask in masks.items():
            id_pass[mask > 0.5] = obj_id / 255.0  # Normalize for export

        return id_pass

    # =========================================================================
    # Project Save/Load
    # =========================================================================

    def save_project(self, path: Path) -> None:
        """Save project state to file."""
        project_data = {
            "config": {
                "mode": self.config.mode.value,
                "edge_mode": self.config.edge_mode.value,
                "sam_model_size": self.config.sam_model_size,
                "depth_model_size": self.config.depth_model_size,
                # Add other config fields...
            },
            "objects": {},
        }

        for obj_id, obj in self.objects.items():
            obj_data = {
                "name": obj.name,
                "color": obj.color,
                "edge_mode": obj.edge_mode.value,
                "keyframes": [],
            }

            for kf in obj.keyframes:
                kf_data = {
                    "frame_number": kf.frame_number,
                    "mask_file": f"masks/obj{obj_id}_frame{kf.frame_number}.npy",
                    "confidence": kf.confidence,
                    "is_user_defined": kf.is_user_defined,
                    "interpolation": kf.interpolation,
                }
                obj_data["keyframes"].append(kf_data)

                # Save mask
                mask_dir = path.parent / "masks"
                mask_dir.mkdir(exist_ok=True)
                np.save(mask_dir / f"obj{obj_id}_frame{kf.frame_number}.npy", kf.mask)

            project_data["objects"][str(obj_id)] = obj_data

        with open(path, "w") as f:
            json.dump(project_data, f, indent=2)

    def load_project(self, path: Path) -> None:
        """Load project state from file."""
        with open(path, "r") as f:
            project_data = json.load(f)

        # Load config
        config_data = project_data.get("config", {})
        if config_data:
            self.config.mode = RotoMode(config_data.get("mode", "automatic"))
            self.config.edge_mode = EdgeMode(config_data.get("edge_mode", "adaptive"))
            # Load other config fields...

        # Load objects
        self.objects.clear()
        for obj_id_str, obj_data in project_data.get("objects", {}).items():
            obj_id = int(obj_id_str)

            obj = RotoObject(
                object_id=obj_id,
                name=obj_data["name"],
                color=tuple(obj_data["color"]),
                edge_mode=EdgeMode(obj_data.get("edge_mode", "adaptive")),
            )

            for kf_data in obj_data.get("keyframes", []):
                mask_path = path.parent / kf_data["mask_file"]
                if mask_path.exists():
                    mask = np.load(mask_path)
                else:
                    continue

                kf = Keyframe(
                    frame_number=kf_data["frame_number"],
                    mask=mask,
                    confidence=kf_data.get("confidence", 1.0),
                    is_user_defined=kf_data.get("is_user_defined", True),
                    interpolation=kf_data.get("interpolation", "smooth"),
                )
                obj.keyframes.append(kf)

            self.objects[obj_id] = obj
            self._next_object_id = max(self._next_object_id, obj_id + 1)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def clear_caches(self) -> None:
        """Clear all caches."""
        self._frame_cache.clear()
        self._embedding_cache.clear()
        self._depth_cache.clear()
        self._flow_cache.clear()

    def unload_models(self) -> None:
        """Unload all models from memory."""
        if self.sam_model:
            self.sam_model.unload()
            self.sam_model = None
        if self.depth_model:
            self.depth_model.unload()
            self.depth_model = None
        if self.matting_model:
            self.matting_model = None
        if self.flow_model:
            self.flow_model = None

        self.clear_caches()
        self._is_loaded = False

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
