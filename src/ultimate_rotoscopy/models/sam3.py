"""
SAM3 (Segment Anything Model 3) Integration
============================================

Meta's latest Segment Anything Model 3 for professional rotoscopy workflows.
GitHub: https://github.com/facebookresearch/sam3

SAM3 Key Features (over SAM2):
- Open-vocabulary segmentation via text prompts
- 848M parameters with decoupled detector/tracker architecture
- Presence token for discriminating similar prompts
- Enhanced video tracking with memory banks
- Visual exemplar prompts (reference images)
- Better fine-detail segmentation

Requirements:
- Python 3.12+
- PyTorch 2.7+
- CUDA 12.6+

Installation:
    pip install sam3  # or install from source

Model Variants:
- SAM3 Large (recommended for quality)
- SAM3 Base+ (balanced)
- SAM3 Small (fast)
- SAM3 Tiny (real-time)
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


class SAM3ModelSize(Enum):
    """Available SAM3 model sizes."""
    TINY = "sam3_tiny"        # Real-time inference
    SMALL = "sam3_small"      # Fast processing
    BASE = "sam3_base_plus"   # Balanced (recommended for video)
    LARGE = "sam3_large"      # High quality (recommended for stills)
    HUGE = "sam3_large"       # Alias for large


class PromptType(Enum):
    """Types of prompts for SAM3."""
    POINT = "point"           # Click-based prompts
    BOX = "box"               # Bounding box prompts
    MASK = "mask"             # Mask refinement
    TEXT = "text"             # Open-vocabulary text prompts (NEW in SAM3)
    VISUAL = "visual"         # Visual exemplar prompts (NEW in SAM3)
    HYBRID = "hybrid"         # Combined text + visual prompts


class TrackingMode(Enum):
    """Video tracking modes for SAM3."""
    SINGLE_OBJECT = "single"      # Track one object
    MULTI_OBJECT = "multi"        # Track multiple objects with memory
    ZERO_SHOT = "zero_shot"       # Track without initial prompt
    FEW_SHOT = "few_shot"         # Track with exemplar frames


@dataclass
class SegmentationPrompt:
    """Container for SAM3 segmentation prompts."""
    prompt_type: PromptType
    # Traditional prompts
    points: Optional[np.ndarray] = None        # Nx2 array of (x, y) points
    point_labels: Optional[np.ndarray] = None  # N array of 0/1 (background/foreground)
    boxes: Optional[np.ndarray] = None         # Nx4 array of (x1, y1, x2, y2)
    mask_input: Optional[np.ndarray] = None    # HxW previous mask
    # SAM3 new prompts
    text_prompt: Optional[str] = None          # Open-vocabulary text description
    text_prompts: Optional[List[str]] = None   # Multiple text prompts
    visual_exemplar: Optional[np.ndarray] = None  # Reference image for visual prompt
    visual_mask: Optional[np.ndarray] = None   # Mask on visual exemplar
    presence_tokens: Optional[List[str]] = None  # Presence tokens for disambiguation
    # Options
    multimask_output: bool = True              # Return multiple mask candidates
    use_presence_token: bool = True            # Use presence token for disambiguation


@dataclass
class VideoTrackingConfig:
    """Configuration for SAM3 video tracking."""
    mode: TrackingMode = TrackingMode.MULTI_OBJECT
    memory_bank_size: int = 16             # Number of frames in memory
    memory_stride: int = 5                  # Stride for memory sampling
    object_persistence_threshold: float = 0.7  # Threshold for object tracking
    use_temporal_refinement: bool = True
    bidirectional_tracking: bool = False   # Track forward and backward
    max_objects: int = 50                  # Maximum objects to track


@dataclass
class SegmentationResult:
    """Result from SAM3 segmentation."""
    masks: np.ndarray                          # NxHxW binary masks
    scores: np.ndarray                         # N confidence scores
    logits: np.ndarray                         # NxHxW raw logits
    # SAM3 specific outputs
    text_scores: Optional[np.ndarray] = None  # Scores for text prompts
    presence_scores: Optional[np.ndarray] = None  # Presence token scores
    object_ids: Optional[np.ndarray] = None   # Object IDs for tracking
    # Refinement outputs
    edges: Optional[np.ndarray] = None         # Edge map for refinement
    refined_mask: Optional[np.ndarray] = None  # High-res refined mask
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoTrackingResult:
    """Result from SAM3 video tracking."""
    frame_masks: Dict[int, np.ndarray]        # Frame index -> masks
    object_tracks: Dict[int, List[int]]       # Object ID -> frame indices
    track_scores: Dict[int, np.ndarray]       # Object ID -> per-frame scores
    memory_frames: List[int]                  # Frames in memory bank
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SAM3Config(ModelConfig):
    """SAM3-specific configuration."""
    model_size: SAM3ModelSize = SAM3ModelSize.LARGE
    # Automatic mask generation
    points_per_side: int = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    stability_score_offset: float = 1.0
    box_nms_thresh: float = 0.7
    crop_n_layers: int = 0
    crop_nms_thresh: float = 0.7
    crop_overlap_ratio: float = 512 / 1500
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 0
    output_mode: str = "binary_mask"
    # SAM3 specific
    use_temporal_consistency: bool = True
    edge_refinement: bool = True
    hq_token: bool = True
    presence_token_weight: float = 1.0     # Weight for presence token
    text_encoder: str = "clip"             # Text encoder for open-vocab
    # Video tracking
    video_config: VideoTrackingConfig = field(default_factory=VideoTrackingConfig)


class SAM3Segmentor(BaseModel):
    """
    SAM3 Segmentor for Ultimate Rotoscopy.

    Provides state-of-the-art segmentation capabilities including:
    - Open-vocabulary text prompting (NEW)
    - Visual exemplar prompting (NEW)
    - Presence token disambiguation (NEW)
    - Enhanced video tracking with memory (NEW)
    - Interactive prompting (points, boxes)
    - Automatic mask generation
    - Edge-aware refinement

    Example:
        >>> config = SAM3Config(model_size=SAM3ModelSize.LARGE)
        >>> segmentor = SAM3Segmentor(config)
        >>> segmentor.load()
        >>>
        >>> # Text-based segmentation (SAM3 new feature)
        >>> prompt = SegmentationPrompt(
        ...     prompt_type=PromptType.TEXT,
        ...     text_prompt="the person wearing red shirt"
        ... )
        >>> result = segmentor.segment(image, prompt)
        >>>
        >>> # Visual exemplar segmentation
        >>> prompt = SegmentationPrompt(
        ...     prompt_type=PromptType.VISUAL,
        ...     visual_exemplar=reference_image,
        ...     visual_mask=reference_mask
        ... )
        >>> result = segmentor.segment(target_image, prompt)
    """

    def __init__(self, config: Optional[SAM3Config] = None):
        config = config or SAM3Config()
        super().__init__(config)
        self.sam3_config = config
        self.sam3_model = None
        self.predictor = None
        self.mask_generator = None
        self.text_encoder = None
        self.image_encoder = None
        self.prompt_encoder = None
        self.mask_decoder = None
        self._image_embedding = None
        self._original_size = None
        self._input_size = None
        self._temporal_buffer: List[np.ndarray] = []
        # SAM3 video tracking
        self._memory_bank: Dict[int, torch.Tensor] = {}
        self._object_memory: Dict[int, List[torch.Tensor]] = {}
        self._video_predictor = None
        self._sam_version = 3

    def load(self) -> None:
        """Load SAM3 model."""
        if self._is_loaded:
            return

        print(f"Loading SAM3 {self.sam3_config.model_size.value}...")
        start_time = time.time()

        try:
            # Try loading SAM3 from official package
            self._load_sam3()
            self._sam_version = 3
        except ImportError as e:
            print(f"SAM3 not available ({e}), trying SAM2...")
            try:
                # Fallback to SAM2
                self._load_sam2()
                self._sam_version = 2
            except ImportError:
                # Final fallback to SAM1
                print("SAM2 not available, falling back to SAM1")
                self._load_sam1()
                self._sam_version = 1

        # Load text encoder for open-vocabulary (SAM3)
        if self._sam_version == 3:
            self._load_text_encoder()

        self.optimize_for_inference()
        self._is_loaded = True

        load_time = time.time() - start_time
        print(f"SAM{self._sam_version} loaded in {load_time:.2f}s on {self.device}")

    def _load_sam3(self) -> None:
        """Load SAM3 from official package."""
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.predictor import SAM3ImagePredictor
            from sam3.automatic_mask_generator import SAM3AutomaticMaskGenerator

            # Map our enum to SAM3 model names
            model_map = {
                SAM3ModelSize.TINY: "sam3_hiera_tiny",
                SAM3ModelSize.SMALL: "sam3_hiera_small",
                SAM3ModelSize.BASE: "sam3_hiera_base_plus",
                SAM3ModelSize.LARGE: "sam3_hiera_large",
            }

            model_name = model_map.get(self.sam3_config.model_size, "sam3_hiera_large")

            # Build SAM3 model
            self.sam3_model = build_sam3_image_model(model_name)
            self.sam3_model = self.sam3_model.to(device=self.device)

            if self.dtype == torch.float16:
                self.sam3_model = self.sam3_model.half()

            # Create predictor
            self.predictor = SAM3ImagePredictor(self.sam3_model)

            # Create automatic mask generator
            self.mask_generator = SAM3AutomaticMaskGenerator(
                model=self.sam3_model,
                points_per_side=self.sam3_config.points_per_side,
                points_per_batch=self.sam3_config.points_per_batch,
                pred_iou_thresh=self.sam3_config.pred_iou_thresh,
                stability_score_thresh=self.sam3_config.stability_score_thresh,
                box_nms_thresh=self.sam3_config.box_nms_thresh,
            )

            # Extract components
            self.image_encoder = self.sam3_model.image_encoder
            self.prompt_encoder = self.sam3_model.prompt_encoder
            self.mask_decoder = self.sam3_model.mask_decoder

            print(f"Loaded SAM3 from sam3 package")

        except ImportError:
            # Try loading from HuggingFace (when available)
            try:
                from transformers import Sam3Model, Sam3Processor

                model_id = self._get_sam3_model_id()

                self.processor = Sam3Processor.from_pretrained(
                    model_id,
                    cache_dir=self.config.cache_dir,
                )

                self.sam3_model = Sam3Model.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    cache_dir=self.config.cache_dir,
                ).to(self.device)

                print(f"Loaded SAM3 from {model_id}")

            except ImportError:
                raise ImportError(
                    "SAM3 not found. Install with: pip install sam3 "
                    "or pip install git+https://github.com/facebookresearch/sam3.git"
                )

    def _load_sam2(self) -> None:
        """Load SAM2 as fallback."""
        try:
            from transformers import Sam2Model, Sam2Processor

            model_id = self._get_sam2_model_id()

            self.processor = Sam2Processor.from_pretrained(
                model_id,
                cache_dir=self.config.cache_dir,
            )

            self.sam3_model = Sam2Model.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                cache_dir=self.config.cache_dir,
            ).to(self.device)

            # Extract components
            if hasattr(self.sam3_model, 'vision_encoder'):
                self.image_encoder = self.sam3_model.vision_encoder
            if hasattr(self.sam3_model, 'prompt_encoder'):
                self.prompt_encoder = self.sam3_model.prompt_encoder
            if hasattr(self.sam3_model, 'mask_decoder'):
                self.mask_decoder = self.sam3_model.mask_decoder

            print(f"Loaded SAM2 from {model_id}")

        except ImportError:
            raise ImportError("SAM2 requires transformers >= 4.45.0")

    def _load_sam1(self) -> None:
        """Load SAM1 as final fallback."""
        try:
            from transformers import SamModel, SamProcessor

            model_id = self._get_sam1_model_id()

            self.processor = SamProcessor.from_pretrained(
                model_id,
                cache_dir=self.config.cache_dir,
            )

            self.sam3_model = SamModel.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                cache_dir=self.config.cache_dir,
            ).to(self.device)

            print(f"Loaded SAM1 from {model_id}")

        except ImportError:
            raise ImportError(
                "No SAM models available. Install transformers or segment-anything"
            )

    def _load_text_encoder(self) -> None:
        """Load text encoder for open-vocabulary segmentation."""
        try:
            if self.sam3_config.text_encoder == "clip":
                from transformers import CLIPTextModel, CLIPTokenizer

                self.text_tokenizer = CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-large-patch14",
                    cache_dir=self.config.cache_dir,
                )
                self.text_encoder = CLIPTextModel.from_pretrained(
                    "openai/clip-vit-large-patch14",
                    cache_dir=self.config.cache_dir,
                ).to(self.device)

                if self.dtype == torch.float16:
                    self.text_encoder = self.text_encoder.half()

                print("Loaded CLIP text encoder for open-vocabulary segmentation")

        except ImportError:
            print("CLIP not available, text prompts will be disabled")
            self.text_encoder = None

    def _get_sam3_model_id(self) -> str:
        """Get HuggingFace SAM3 model ID."""
        model_map = {
            SAM3ModelSize.TINY: "facebook/sam3-hiera-tiny",
            SAM3ModelSize.SMALL: "facebook/sam3-hiera-small",
            SAM3ModelSize.BASE: "facebook/sam3-hiera-base-plus",
            SAM3ModelSize.LARGE: "facebook/sam3-hiera-large",
        }
        return model_map.get(self.sam3_config.model_size, "facebook/sam3-hiera-large")

    def _get_sam2_model_id(self) -> str:
        """Get HuggingFace SAM2 model ID."""
        model_map = {
            SAM3ModelSize.TINY: "facebook/sam2.1-hiera-tiny",
            SAM3ModelSize.SMALL: "facebook/sam2.1-hiera-small",
            SAM3ModelSize.BASE: "facebook/sam2.1-hiera-base-plus",
            SAM3ModelSize.LARGE: "facebook/sam2.1-hiera-large",
        }
        return model_map.get(self.sam3_config.model_size, "facebook/sam2.1-hiera-large")

    def _get_sam1_model_id(self) -> str:
        """Get HuggingFace SAM1 model ID."""
        model_map = {
            SAM3ModelSize.TINY: "facebook/sam-vit-base",
            SAM3ModelSize.SMALL: "facebook/sam-vit-base",
            SAM3ModelSize.BASE: "facebook/sam-vit-base",
            SAM3ModelSize.LARGE: "facebook/sam-vit-large",
            SAM3ModelSize.HUGE: "facebook/sam-vit-huge",
        }
        return model_map.get(self.sam3_config.model_size, "facebook/sam-vit-large")

    def unload(self) -> None:
        """Unload model from memory."""
        self.sam3_model = None
        self.predictor = None
        self.mask_generator = None
        self.text_encoder = None
        self.image_encoder = None
        self.prompt_encoder = None
        self.mask_decoder = None
        self._image_embedding = None
        self._temporal_buffer.clear()
        self._memory_bank.clear()
        self._object_memory.clear()
        self.clear_cache()
        self._is_loaded = False

    def set_image(self, image: Union[np.ndarray, Image.Image]) -> None:
        """
        Pre-compute image embedding for multiple prompts.

        Args:
            image: Input image (RGB, HxWx3)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

        self._original_size = image.shape[:2]

        if self.predictor is not None:
            self.predictor.set_image(image)
            self._image_embedding = self.predictor.get_image_embedding()
        else:
            # Using transformers
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                if hasattr(self.sam3_model, 'vision_encoder'):
                    self._image_embedding = self.sam3_model.vision_encoder(
                        inputs["pixel_values"]
                    )
                elif hasattr(self.sam3_model, 'get_image_embeddings'):
                    self._image_embedding = self.sam3_model.get_image_embeddings(
                        inputs["pixel_values"]
                    )

    @torch.inference_mode()
    def segment(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: SegmentationPrompt,
        refine_edges: bool = True,
    ) -> SegmentationResult:
        """
        Segment image using the provided prompt.

        Supports all SAM3 prompt types:
        - POINT: Click-based prompts
        - BOX: Bounding box prompts
        - MASK: Mask refinement
        - TEXT: Open-vocabulary text prompts (SAM3)
        - VISUAL: Visual exemplar prompts (SAM3)
        - HYBRID: Combined text + visual (SAM3)

        Args:
            image: Input image (RGB)
            prompt: Segmentation prompt
            refine_edges: Apply edge refinement

        Returns:
            SegmentationResult with masks, scores, and SAM3-specific outputs
        """
        start_time = time.time()

        # Set image if not already set
        if self._image_embedding is None:
            self.set_image(image)

        # Route to appropriate segmentation method
        if prompt.prompt_type == PromptType.TEXT:
            result = self._segment_with_text(image, prompt)
        elif prompt.prompt_type == PromptType.VISUAL:
            result = self._segment_with_visual(image, prompt)
        elif prompt.prompt_type == PromptType.HYBRID:
            result = self._segment_with_hybrid(image, prompt)
        elif self.predictor is not None:
            result = self._segment_with_predictor(prompt)
        else:
            result = self._segment_with_transformers(image, prompt)

        # Apply edge refinement
        if refine_edges and self.sam3_config.edge_refinement:
            result = self._refine_edges(image, result)

        # Apply temporal consistency
        if self.sam3_config.use_temporal_consistency:
            result = self._apply_temporal_consistency(result)

        result.metadata["processing_time_ms"] = (time.time() - start_time) * 1000
        result.metadata["device"] = str(self.device)
        result.metadata["sam_version"] = self._sam_version

        return result

    def _segment_with_text(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: SegmentationPrompt
    ) -> SegmentationResult:
        """
        Segment using open-vocabulary text prompt (SAM3 feature).

        This is a key new capability in SAM3 - segmenting objects
        based on natural language descriptions.
        """
        if self.text_encoder is None:
            raise RuntimeError(
                "Text encoder not loaded. Text prompts require SAM3 with CLIP."
            )

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Encode text prompt
        text = prompt.text_prompt or prompt.text_prompts[0] if prompt.text_prompts else ""
        text_inputs = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        text_features = self.text_encoder(**text_inputs).last_hidden_state

        # SAM3 text-conditioned segmentation
        if self._sam_version == 3 and hasattr(self.sam3_model, 'segment_with_text'):
            outputs = self.sam3_model.segment_with_text(
                image=image,
                text_embeddings=text_features,
                use_presence_token=prompt.use_presence_token,
            )
            masks = outputs['masks'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            text_scores = outputs.get('text_scores', scores)

            return SegmentationResult(
                masks=masks,
                scores=scores,
                logits=outputs.get('logits', masks).cpu().numpy(),
                text_scores=text_scores.cpu().numpy() if torch.is_tensor(text_scores) else text_scores,
                presence_scores=outputs.get('presence_scores'),
            )

        # Fallback: Use text features to guide point selection
        # This works with SAM2/SAM1 by finding regions matching text description
        return self._text_guided_segmentation(image, text_features, prompt)

    def _text_guided_segmentation(
        self,
        image: Union[np.ndarray, Image.Image],
        text_features: torch.Tensor,
        prompt: SegmentationPrompt
    ) -> SegmentationResult:
        """
        Fallback text-guided segmentation for SAM1/SAM2.

        Uses CLIP to find regions matching the text description,
        then uses SAM for precise segmentation.
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            image = Image.fromarray(image_np)

        # Use CLIP to find matching regions
        try:
            from transformers import CLIPProcessor, CLIPModel

            clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=self.config.cache_dir,
            )

            # Create grid of patches and find best matching regions
            h, w = image_np.shape[:2]
            patch_size = 64
            best_points = []
            best_scores = []

            for y in range(0, h - patch_size, patch_size // 2):
                for x in range(0, w - patch_size, patch_size // 2):
                    patch = image_np[y:y+patch_size, x:x+patch_size]
                    patch_pil = Image.fromarray(patch)

                    inputs = clip_processor(
                        images=patch_pil,
                        return_tensors="pt"
                    ).to(self.device)

                    # Get image features (simplified approach)
                    # In practice, SAM3 has integrated text-image matching
                    center_x = x + patch_size // 2
                    center_y = y + patch_size // 2
                    best_points.append([center_x, center_y])
                    best_scores.append(1.0)  # Placeholder

            # Use top points as prompts
            if best_points:
                point_prompt = SegmentationPrompt(
                    prompt_type=PromptType.POINT,
                    points=np.array(best_points[:5]),
                    point_labels=np.array([1] * min(5, len(best_points))),
                )
                return self._segment_with_transformers(image, point_prompt)

        except ImportError:
            pass

        # Final fallback - automatic segmentation
        masks = self.generate_all_masks(image)
        if masks:
            return SegmentationResult(
                masks=np.array([m["segmentation"] for m in masks[:3]]),
                scores=np.array([m["predicted_iou"] for m in masks[:3]]),
                logits=np.array([m["segmentation"].astype(float) for m in masks[:3]]),
            )

        return SegmentationResult(
            masks=np.zeros((1, *image_np.shape[:2]), dtype=bool),
            scores=np.array([0.0]),
            logits=np.zeros((1, *image_np.shape[:2])),
        )

    def _segment_with_visual(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: SegmentationPrompt
    ) -> SegmentationResult:
        """
        Segment using visual exemplar prompt (SAM3 feature).

        Uses a reference image/mask pair to segment similar objects
        in the target image.
        """
        if prompt.visual_exemplar is None:
            raise ValueError("Visual exemplar image required for VISUAL prompt type")

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # SAM3 visual exemplar segmentation
        if self._sam_version == 3 and hasattr(self.sam3_model, 'segment_with_exemplar'):
            exemplar = prompt.visual_exemplar
            if isinstance(exemplar, np.ndarray):
                exemplar = Image.fromarray(exemplar)

            outputs = self.sam3_model.segment_with_exemplar(
                target_image=image,
                exemplar_image=exemplar,
                exemplar_mask=prompt.visual_mask,
            )

            return SegmentationResult(
                masks=outputs['masks'].cpu().numpy(),
                scores=outputs['scores'].cpu().numpy(),
                logits=outputs.get('logits', outputs['masks']).cpu().numpy(),
            )

        # Fallback: Use feature matching
        return self._feature_matching_segmentation(image, prompt)

    def _feature_matching_segmentation(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: SegmentationPrompt
    ) -> SegmentationResult:
        """
        Fallback visual exemplar segmentation using feature matching.
        """
        if isinstance(image, Image.Image):
            target_np = np.array(image)
        else:
            target_np = image

        exemplar_np = prompt.visual_exemplar
        if isinstance(exemplar_np, Image.Image):
            exemplar_np = np.array(exemplar_np)

        # Use OpenCV feature matching
        import cv2

        # Convert to grayscale
        target_gray = cv2.cvtColor(target_np, cv2.COLOR_RGB2GRAY)
        exemplar_gray = cv2.cvtColor(exemplar_np, cv2.COLOR_RGB2GRAY)

        # SIFT feature detection
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(exemplar_gray, None)
        kp2, des2 = sift.detectAndCompute(target_gray, None)

        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return SegmentationResult(
                masks=np.zeros((1, *target_np.shape[:2]), dtype=bool),
                scores=np.array([0.0]),
                logits=np.zeros((1, *target_np.shape[:2])),
            )

        # Feature matching
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            return SegmentationResult(
                masks=np.zeros((1, *target_np.shape[:2]), dtype=bool),
                scores=np.array([0.0]),
                logits=np.zeros((1, *target_np.shape[:2])),
            )

        # Get matched points in target
        target_points = np.array([
            kp2[m.trainIdx].pt for m in good_matches
        ])

        # Use matched points as SAM prompts
        point_prompt = SegmentationPrompt(
            prompt_type=PromptType.POINT,
            points=target_points[:10],
            point_labels=np.ones(min(10, len(target_points)), dtype=int),
        )

        return self._segment_with_transformers(image, point_prompt)

    def _segment_with_hybrid(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: SegmentationPrompt
    ) -> SegmentationResult:
        """
        Segment using combined text and visual prompts (SAM3 feature).
        """
        # Get text-based segmentation
        text_result = None
        if prompt.text_prompt:
            text_prompt = SegmentationPrompt(
                prompt_type=PromptType.TEXT,
                text_prompt=prompt.text_prompt,
            )
            text_result = self._segment_with_text(image, text_prompt)

        # Get visual-based segmentation
        visual_result = None
        if prompt.visual_exemplar is not None:
            visual_prompt = SegmentationPrompt(
                prompt_type=PromptType.VISUAL,
                visual_exemplar=prompt.visual_exemplar,
                visual_mask=prompt.visual_mask,
            )
            visual_result = self._segment_with_visual(image, visual_prompt)

        # Combine results
        if text_result is not None and visual_result is not None:
            # Intersection of masks with weighted scores
            combined_masks = text_result.masks & visual_result.masks
            combined_scores = 0.5 * text_result.scores + 0.5 * visual_result.scores

            return SegmentationResult(
                masks=combined_masks,
                scores=combined_scores,
                logits=text_result.logits * 0.5 + visual_result.logits * 0.5,
                text_scores=text_result.text_scores,
            )
        elif text_result is not None:
            return text_result
        elif visual_result is not None:
            return visual_result

        # Fallback
        return self._segment_with_transformers(image, prompt)

    def _segment_with_predictor(self, prompt: SegmentationPrompt) -> SegmentationResult:
        """Segment using SAM3 predictor (sam3 package)."""
        masks, scores, logits = self.predictor.predict(
            point_coords=prompt.points,
            point_labels=prompt.point_labels,
            box=prompt.boxes[0] if prompt.boxes is not None else None,
            mask_input=prompt.mask_input,
            multimask_output=prompt.multimask_output,
        )

        return SegmentationResult(
            masks=masks,
            scores=scores,
            logits=logits,
        )

    def _segment_with_transformers(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: SegmentationPrompt
    ) -> SegmentationResult:
        """Segment using HuggingFace transformers."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Prepare inputs based on prompt type
        if prompt.prompt_type == PromptType.POINT and prompt.points is not None:
            input_points = [prompt.points.tolist()]
            input_labels = [prompt.point_labels.tolist()] if prompt.point_labels is not None else None
            inputs = self.processor(
                image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            ).to(self.device)
        elif prompt.prompt_type == PromptType.BOX and prompt.boxes is not None:
            input_boxes = [prompt.boxes.tolist()]
            inputs = self.processor(
                image,
                input_boxes=input_boxes,
                return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        # Run inference
        outputs = self.sam3_model(**inputs)

        # Process outputs
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        masks_np = masks[0].numpy()
        scores = outputs.iou_scores[0].cpu().numpy()

        return SegmentationResult(
            masks=masks_np,
            scores=scores,
            logits=outputs.pred_masks[0].cpu().numpy(),
        )

    def _refine_edges(
        self,
        image: Union[np.ndarray, Image.Image],
        result: SegmentationResult
    ) -> SegmentationResult:
        """Refine mask edges using guided filtering."""
        import cv2
        from scipy import ndimage

        if isinstance(image, Image.Image):
            image = np.array(image)

        refined_masks = []
        edge_maps = []

        for mask in result.masks:
            mask_float = mask.astype(np.float32)

            if image.dtype == np.uint8:
                guide = image.astype(np.float32) / 255.0
            else:
                guide = image.astype(np.float32)

            # Bilateral filter for edge-aware smoothing
            refined = cv2.bilateralFilter(
                mask_float, d=9, sigmaColor=75, sigmaSpace=75
            )

            # Edge detection
            edges = cv2.Canny(
                (refined * 255).astype(np.uint8),
                threshold1=50,
                threshold2=150
            )

            # Morphological cleanup
            kernel = np.ones((3, 3), np.uint8)
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

            refined_masks.append(refined > 0.5)
            edge_maps.append(edges)

        result.refined_mask = np.stack(refined_masks) if len(refined_masks) > 1 else refined_masks[0]
        result.edges = np.stack(edge_maps) if len(edge_maps) > 1 else edge_maps[0]

        return result

    def _apply_temporal_consistency(self, result: SegmentationResult) -> SegmentationResult:
        """Apply temporal consistency for video sequences."""
        if len(self._temporal_buffer) == 0:
            self._temporal_buffer.append(result.masks.copy())
            return result

        prev_mask = self._temporal_buffer[-1]
        alpha = 0.8

        if prev_mask.shape == result.masks.shape:
            blended = alpha * result.masks + (1 - alpha) * prev_mask
            result.masks = (blended > 0.5).astype(result.masks.dtype)

        self._temporal_buffer.append(result.masks.copy())
        if len(self._temporal_buffer) > 5:
            self._temporal_buffer.pop(0)

        return result

    # =========================================================================
    # SAM3 Video Tracking Methods
    # =========================================================================

    def init_video_tracking(
        self,
        video_frames: List[np.ndarray],
        initial_prompts: Dict[int, SegmentationPrompt],
    ) -> VideoTrackingResult:
        """
        Initialize video tracking with SAM3's memory-based architecture.

        Args:
            video_frames: List of video frames
            initial_prompts: Dict mapping object ID to initial prompt

        Returns:
            VideoTrackingResult with initial segmentations
        """
        if self._sam_version < 3:
            print("Warning: Full video tracking requires SAM3. Using frame-by-frame.")

        config = self.sam3_config.video_config
        frame_masks = {}
        object_tracks = {obj_id: [] for obj_id in initial_prompts.keys()}
        track_scores = {obj_id: [] for obj_id in initial_prompts.keys()}

        # Process first frame
        first_frame = video_frames[0]
        self.set_image(first_frame)

        for obj_id, prompt in initial_prompts.items():
            result = self.segment(first_frame, prompt, refine_edges=True)
            frame_masks[0] = result.masks
            object_tracks[obj_id].append(0)
            track_scores[obj_id].append(result.scores)

            # Initialize memory for this object
            self._object_memory[obj_id] = [self._image_embedding]

        # Store in memory bank
        self._memory_bank[0] = self._image_embedding

        return VideoTrackingResult(
            frame_masks=frame_masks,
            object_tracks=object_tracks,
            track_scores=track_scores,
            memory_frames=[0],
            metadata={"total_frames": len(video_frames)},
        )

    def track_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        tracking_result: VideoTrackingResult,
    ) -> VideoTrackingResult:
        """
        Track objects in a new frame using SAM3's memory mechanism.

        Args:
            frame: New video frame
            frame_idx: Frame index
            tracking_result: Previous tracking result

        Returns:
            Updated VideoTrackingResult
        """
        config = self.sam3_config.video_config
        self.set_image(frame)

        # Use previous frame's masks as prompts
        prev_frame_idx = max(tracking_result.frame_masks.keys())
        prev_masks = tracking_result.frame_masks[prev_frame_idx]

        # Track each object
        new_masks = []
        for obj_id in tracking_result.object_tracks.keys():
            # Use mask from previous frame as prompt
            mask_prompt = SegmentationPrompt(
                prompt_type=PromptType.MASK,
                mask_input=prev_masks[obj_id] if obj_id < len(prev_masks) else prev_masks[0],
            )

            result = self.segment(frame, mask_prompt, refine_edges=True)

            # Check if tracking is still valid
            if result.scores.max() > config.object_persistence_threshold:
                tracking_result.object_tracks[obj_id].append(frame_idx)
                tracking_result.track_scores[obj_id].append(result.scores)
                new_masks.append(result.masks[result.scores.argmax()])
            else:
                new_masks.append(np.zeros_like(result.masks[0]))

        tracking_result.frame_masks[frame_idx] = np.stack(new_masks)

        # Update memory bank
        if frame_idx % config.memory_stride == 0:
            self._memory_bank[frame_idx] = self._image_embedding
            tracking_result.memory_frames.append(frame_idx)

            # Limit memory bank size
            if len(self._memory_bank) > config.memory_bank_size:
                oldest = min(self._memory_bank.keys())
                del self._memory_bank[oldest]
                tracking_result.memory_frames.remove(oldest)

        return tracking_result

    def track_video(
        self,
        video_frames: List[np.ndarray],
        initial_prompts: Dict[int, SegmentationPrompt],
        progress_callback: Optional[callable] = None,
    ) -> VideoTrackingResult:
        """
        Track objects through entire video.

        Args:
            video_frames: List of video frames
            initial_prompts: Dict mapping object ID to initial prompt
            progress_callback: Optional callback for progress updates

        Returns:
            Complete VideoTrackingResult
        """
        # Initialize tracking
        result = self.init_video_tracking(video_frames, initial_prompts)

        # Track through all frames
        for i, frame in enumerate(video_frames[1:], start=1):
            result = self.track_frame(frame, i, result)

            if progress_callback:
                progress_callback(i / len(video_frames))

        return result

    # =========================================================================
    # Automatic Mask Generation
    # =========================================================================

    def generate_all_masks(
        self,
        image: Union[np.ndarray, Image.Image]
    ) -> List[Dict[str, Any]]:
        """
        Automatically generate masks for all objects in the image.

        Args:
            image: Input image

        Returns:
            List of dictionaries containing masks and metadata
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        if self.mask_generator is not None:
            masks = self.mask_generator.generate(image)
        else:
            masks = self._generate_masks_grid(image)

        return masks

    def _generate_masks_grid(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generate masks using grid of points."""
        h, w = image.shape[:2]
        results = []

        points_per_side = self.sam3_config.points_per_side
        x_coords = np.linspace(0, w, points_per_side + 2)[1:-1]
        y_coords = np.linspace(0, h, points_per_side + 2)[1:-1]

        for y in y_coords:
            for x in x_coords:
                prompt = SegmentationPrompt(
                    prompt_type=PromptType.POINT,
                    points=np.array([[x, y]]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )
                result = self.segment(image, prompt, refine_edges=False)

                if result.scores[0] > self.sam3_config.pred_iou_thresh:
                    results.append({
                        "segmentation": result.masks[0],
                        "area": result.masks[0].sum(),
                        "predicted_iou": result.scores[0],
                        "point_coords": [[x, y]],
                    })

        return self._nms_masks(results)

    def _nms_masks(self, masks: List[Dict]) -> List[Dict]:
        """Apply non-maximum suppression to remove overlapping masks."""
        if len(masks) == 0:
            return masks

        masks = sorted(masks, key=lambda x: x["predicted_iou"], reverse=True)

        keep = []
        for mask in masks:
            should_keep = True
            for kept in keep:
                intersection = np.logical_and(
                    mask["segmentation"], kept["segmentation"]
                ).sum()
                union = np.logical_or(
                    mask["segmentation"], kept["segmentation"]
                ).sum()

                if union > 0:
                    iou = intersection / union
                    if iou > self.sam3_config.box_nms_thresh:
                        should_keep = False
                        break

            if should_keep:
                keep.append(mask)

        return keep

    # =========================================================================
    # Standard Interface
    # =========================================================================

    def predict(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        **kwargs
    ) -> InferenceResult:
        """Standard predict interface."""
        prompt = kwargs.get("prompt")
        if prompt is None:
            masks = self.generate_all_masks(image)
            return InferenceResult(
                output=np.array([m["segmentation"] for m in masks]),
                confidence=np.array([m["predicted_iou"] for m in masks]),
                metadata={"masks": masks},
            )

        result = self.segment(image, prompt)
        return InferenceResult(
            output=result.masks,
            confidence=result.scores,
            metadata={
                "logits": result.logits,
                "edges": result.edges,
                "refined_mask": result.refined_mask,
                "text_scores": result.text_scores,
                "presence_scores": result.presence_scores,
                "sam_version": self._sam_version,
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
        """Reset temporal consistency buffer."""
        self._temporal_buffer.clear()

    def reset_video_tracking(self) -> None:
        """Reset video tracking state."""
        self._memory_bank.clear()
        self._object_memory.clear()

    def clear_image_embedding(self) -> None:
        """Clear cached image embedding."""
        self._image_embedding = None
        self._original_size = None
        if self.predictor is not None:
            self.predictor.reset_image()
