"""
SAM2.1 (Segment Anything Model 2.1) Integration
================================================

Provides comprehensive integration with Meta's Segment Anything Model 2.1
for precise object segmentation in rotoscopy workflows.

SAM2.1 was released on 09/30/2024 with significant improvements:
- 6x faster than SAM1 for image segmentation
- 3x fewer interactions needed for video segmentation
- Better accuracy on fine details and edges
- Native video segmentation support

Model IDs (HuggingFace):
- facebook/sam2.1-hiera-large (recommended)
- facebook/sam2.1-hiera-base-plus
- facebook/sam2.1-hiera-small
- facebook/sam2.1-hiera-tiny

Features:
- Interactive point/box prompting
- Automatic mask generation
- Multi-object segmentation
- High-resolution mask refinement
- Edge-aware segmentation
- Temporal consistency for video sequences
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
    """Available SAM2.1 model sizes."""
    TINY = "sam2.1_tiny"        # Fastest, lower quality
    SMALL = "sam2.1_small"      # Good balance
    BASE = "sam2.1_base_plus"   # Base+ model
    LARGE = "sam2.1_large"      # High quality (recommended)
    HUGE = "sam2.1_large"       # Alias for large (no huge variant in SAM2)


class PromptType(Enum):
    """Types of prompts for SAM3."""
    POINT = "point"
    BOX = "box"
    MASK = "mask"
    TEXT = "text"  # SAM3 supports text prompts


@dataclass
class SegmentationPrompt:
    """Container for segmentation prompts."""
    prompt_type: PromptType
    points: Optional[np.ndarray] = None        # Nx2 array of (x, y) points
    point_labels: Optional[np.ndarray] = None  # N array of 0/1 (background/foreground)
    boxes: Optional[np.ndarray] = None         # Nx4 array of (x1, y1, x2, y2)
    mask_input: Optional[np.ndarray] = None    # HxW previous mask
    text_prompt: Optional[str] = None          # Text description
    multimask_output: bool = True              # Return multiple mask candidates


@dataclass
class SegmentationResult:
    """Result from SAM3 segmentation."""
    masks: np.ndarray                          # NxHxW binary masks
    scores: np.ndarray                         # N confidence scores
    logits: np.ndarray                         # NxHxW raw logits
    edges: Optional[np.ndarray] = None         # Edge map for refinement
    refined_mask: Optional[np.ndarray] = None  # High-res refined mask
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SAM3Config(ModelConfig):
    """SAM3-specific configuration."""
    model_size: SAM3ModelSize = SAM3ModelSize.LARGE
    points_per_side: int = 32                  # For automatic mask generation
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
    use_temporal_consistency: bool = True
    edge_refinement: bool = True
    hq_token: bool = True  # High-quality token for better edges


class SAM3Segmentor(BaseModel):
    """
    SAM3 Segmentor for Ultimate Rotoscopy.

    Provides state-of-the-art segmentation capabilities including:
    - Interactive prompting (points, boxes, text)
    - Automatic mask generation
    - Multi-resolution processing
    - Edge-aware refinement
    - Temporal consistency for video

    Example:
        >>> config = SAM3Config(model_size=SAM3ModelSize.LARGE)
        >>> segmentor = SAM3Segmentor(config)
        >>> segmentor.load()
        >>>
        >>> # Point-based segmentation
        >>> prompt = SegmentationPrompt(
        ...     prompt_type=PromptType.POINT,
        ...     points=np.array([[100, 200], [150, 250]]),
        ...     point_labels=np.array([1, 1])
        ... )
        >>> result = segmentor.segment(image, prompt)
    """

    def __init__(self, config: Optional[SAM3Config] = None):
        config = config or SAM3Config()
        super().__init__(config)
        self.sam3_config = config
        self.predictor = None
        self.mask_generator = None
        self.image_encoder = None
        self.prompt_encoder = None
        self.mask_decoder = None
        self._image_embedding = None
        self._original_size = None
        self._input_size = None
        self._temporal_buffer: List[np.ndarray] = []

    def load(self) -> None:
        """Load SAM2.1 model from HuggingFace or local path."""
        if self._is_loaded:
            return

        print(f"Loading SAM2.1 {self.sam3_config.model_size.value}...")
        start_time = time.time()

        model_id = self._get_model_id()

        try:
            # Try SAM2 first (requires transformers >= 4.45)
            from transformers import Sam2Model, Sam2Processor

            self.processor = Sam2Processor.from_pretrained(
                model_id,
                cache_dir=self.config.cache_dir,
            )

            self.model = Sam2Model.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                cache_dir=self.config.cache_dir,
            ).to(self.device)

            self._sam_version = 2
            print(f"Loaded SAM2.1 from {model_id}")

        except ImportError:
            print("SAM2 not available, falling back to SAM1")
            # Fallback to SAM1 if SAM2 not available
            try:
                from transformers import SamModel, SamProcessor

                # Use SAM1 model IDs
                sam1_model_id = self._get_sam1_fallback_id()

                self.processor = SamProcessor.from_pretrained(
                    sam1_model_id,
                    cache_dir=self.config.cache_dir,
                )

                self.model = SamModel.from_pretrained(
                    sam1_model_id,
                    torch_dtype=self.dtype,
                    cache_dir=self.config.cache_dir,
                ).to(self.device)

                self._sam_version = 1
                print(f"Loaded SAM1 from {sam1_model_id}")

            except ImportError:
                # Final fallback to segment-anything package
                self._load_from_segment_anything()

        # Extract components if available
        if hasattr(self.model, 'vision_encoder'):
            self.image_encoder = self.model.vision_encoder
        if hasattr(self.model, 'prompt_encoder'):
            self.prompt_encoder = self.model.prompt_encoder
        if hasattr(self.model, 'mask_decoder'):
            self.mask_decoder = self.model.mask_decoder

        self.optimize_for_inference()
        self._is_loaded = True

        load_time = time.time() - start_time
        print(f"SAM loaded in {load_time:.2f}s on {self.device}")

    def _get_model_id(self) -> str:
        """Get HuggingFace SAM2.1 model ID based on size."""
        model_map = {
            SAM3ModelSize.TINY: "facebook/sam2.1-hiera-tiny",
            SAM3ModelSize.SMALL: "facebook/sam2.1-hiera-small",
            SAM3ModelSize.BASE: "facebook/sam2.1-hiera-base-plus",
            SAM3ModelSize.LARGE: "facebook/sam2.1-hiera-large",
            SAM3ModelSize.HUGE: "facebook/sam2.1-hiera-large",
        }
        return model_map.get(self.sam3_config.model_size, "facebook/sam2.1-hiera-large")

    def _get_sam1_fallback_id(self) -> str:
        """Get SAM1 fallback model ID."""
        model_map = {
            SAM3ModelSize.TINY: "facebook/sam-vit-base",  # No tiny in SAM1
            SAM3ModelSize.SMALL: "facebook/sam-vit-base",
            SAM3ModelSize.BASE: "facebook/sam-vit-base",
            SAM3ModelSize.LARGE: "facebook/sam-vit-large",
            SAM3ModelSize.HUGE: "facebook/sam-vit-huge",
        }
        return model_map.get(self.sam3_config.model_size, "facebook/sam-vit-large")

    def _load_from_segment_anything(self) -> None:
        """Load using segment-anything package."""
        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

            # Download checkpoint if needed
            checkpoint_path = self._download_checkpoint()

            model_type = self.sam3_config.model_size.value.replace("sam3_", "vit_")
            if model_type == "vit_tiny":
                model_type = "vit_t"
            elif model_type == "vit_small":
                model_type = "vit_s"
            elif model_type == "vit_base":
                model_type = "vit_b"
            elif model_type == "vit_large":
                model_type = "vit_l"
            elif model_type == "vit_huge":
                model_type = "vit_h"

            self.model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
            self.model = self.model.to(device=self.device)

            if self.dtype == torch.float16:
                self.model = self.model.half()

            self.predictor = SamPredictor(self.model)
            self.mask_generator = SamAutomaticMaskGenerator(
                self.model,
                points_per_side=self.sam3_config.points_per_side,
                points_per_batch=self.sam3_config.points_per_batch,
                pred_iou_thresh=self.sam3_config.pred_iou_thresh,
                stability_score_thresh=self.sam3_config.stability_score_thresh,
                box_nms_thresh=self.sam3_config.box_nms_thresh,
                crop_n_layers=self.sam3_config.crop_n_layers,
                min_mask_region_area=self.sam3_config.min_mask_region_area,
            )

        except ImportError as e:
            raise ImportError(
                "Neither transformers nor segment-anything package found. "
                "Please install: pip install transformers or pip install segment-anything"
            ) from e

    def _download_checkpoint(self) -> Path:
        """Download SAM checkpoint if not present."""
        from huggingface_hub import hf_hub_download

        checkpoint_map = {
            SAM3ModelSize.BASE: ("facebook/sam-vit-base", "pytorch_model.bin"),
            SAM3ModelSize.LARGE: ("facebook/sam-vit-large", "pytorch_model.bin"),
            SAM3ModelSize.HUGE: ("facebook/sam-vit-huge", "pytorch_model.bin"),
        }

        repo_id, filename = checkpoint_map.get(
            self.sam3_config.model_size,
            ("facebook/sam-vit-large", "pytorch_model.bin")
        )

        return Path(hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=self.config.cache_dir,
        ))

    def unload(self) -> None:
        """Unload model from memory."""
        self.model = None
        self.predictor = None
        self.mask_generator = None
        self.image_encoder = None
        self.prompt_encoder = None
        self.mask_decoder = None
        self._image_embedding = None
        self._temporal_buffer.clear()
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
                self._image_embedding = self.image_encoder(inputs["pixel_values"])

    @torch.inference_mode()
    def segment(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: SegmentationPrompt,
        refine_edges: bool = True,
    ) -> SegmentationResult:
        """
        Segment image using the provided prompt.

        Args:
            image: Input image (RGB)
            prompt: Segmentation prompt (points, boxes, text, or mask)
            refine_edges: Apply edge refinement

        Returns:
            SegmentationResult with masks, scores, and optional refinements
        """
        start_time = time.time()

        # Set image if not already set
        if self._image_embedding is None:
            self.set_image(image)

        if self.predictor is not None:
            result = self._segment_with_predictor(prompt)
        else:
            result = self._segment_with_transformers(image, prompt)

        # Apply edge refinement
        if refine_edges and self.sam3_config.edge_refinement:
            result = self._refine_edges(image, result)

        # Apply temporal consistency for video
        if self.sam3_config.use_temporal_consistency:
            result = self._apply_temporal_consistency(result)

        result.metadata["processing_time_ms"] = (time.time() - start_time) * 1000
        result.metadata["device"] = str(self.device)

        return result

    def _segment_with_predictor(self, prompt: SegmentationPrompt) -> SegmentationResult:
        """Segment using segment-anything predictor."""
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
        if prompt.prompt_type == PromptType.POINT:
            input_points = [prompt.points.tolist()]
            input_labels = [prompt.point_labels.tolist()] if prompt.point_labels is not None else None
            inputs = self.processor(
                image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            ).to(self.device)
        elif prompt.prompt_type == PromptType.BOX:
            input_boxes = [prompt.boxes.tolist()]
            inputs = self.processor(
                image,
                input_boxes=input_boxes,
                return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        # Run inference
        outputs = self.model(**inputs)

        # Process outputs
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        masks = masks[0].numpy()
        scores = outputs.iou_scores[0].cpu().numpy()

        return SegmentationResult(
            masks=masks,
            scores=scores,
            logits=outputs.pred_masks[0].cpu().numpy(),
        )

    def _refine_edges(
        self,
        image: Union[np.ndarray, Image.Image],
        result: SegmentationResult
    ) -> SegmentationResult:
        """
        Refine mask edges using guided filtering and edge detection.

        This is crucial for rotoscopy work where clean edges are essential.
        """
        import cv2
        from scipy import ndimage

        if isinstance(image, Image.Image):
            image = np.array(image)

        refined_masks = []
        edge_maps = []

        for mask in result.masks:
            # Convert to float
            mask_float = mask.astype(np.float32)

            # Guided filter for edge-aware smoothing
            if image.dtype == np.uint8:
                guide = image.astype(np.float32) / 255.0
            else:
                guide = image.astype(np.float32)

            # Use bilateral filter as approximation of guided filter
            refined = cv2.bilateralFilter(
                mask_float,
                d=9,
                sigmaColor=75,
                sigmaSpace=75
            )

            # Edge detection on refined mask
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
        """
        Apply temporal consistency using mask history.

        Essential for video rotoscopy to prevent flickering.
        """
        if len(self._temporal_buffer) == 0:
            self._temporal_buffer.append(result.masks.copy())
            return result

        # Weight current and previous masks
        prev_mask = self._temporal_buffer[-1]
        alpha = 0.8  # Weight for current frame

        # Ensure shapes match
        if prev_mask.shape == result.masks.shape:
            blended = alpha * result.masks + (1 - alpha) * prev_mask
            result.masks = (blended > 0.5).astype(result.masks.dtype)

        # Update buffer (keep last 5 frames)
        self._temporal_buffer.append(result.masks.copy())
        if len(self._temporal_buffer) > 5:
            self._temporal_buffer.pop(0)

        return result

    def generate_all_masks(
        self,
        image: Union[np.ndarray, Image.Image]
    ) -> List[Dict[str, Any]]:
        """
        Automatically generate masks for all objects in the image.

        Useful for initial rotoscopy pass to identify all elements.

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
            # Implement grid-based prompting for transformers
            masks = self._generate_masks_grid(image)

        return masks

    def _generate_masks_grid(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generate masks using grid of points."""
        h, w = image.shape[:2]
        results = []

        # Create grid of points
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

        # Non-maximum suppression
        results = self._nms_masks(results)

        return results

    def _nms_masks(self, masks: List[Dict]) -> List[Dict]:
        """Apply non-maximum suppression to remove overlapping masks."""
        if len(masks) == 0:
            return masks

        # Sort by score
        masks = sorted(masks, key=lambda x: x["predicted_iou"], reverse=True)

        keep = []
        for mask in masks:
            should_keep = True
            for kept in keep:
                # Calculate IoU
                intersection = np.logical_and(
                    mask["segmentation"],
                    kept["segmentation"]
                ).sum()
                union = np.logical_or(
                    mask["segmentation"],
                    kept["segmentation"]
                ).sum()

                if union > 0:
                    iou = intersection / union
                    if iou > self.sam3_config.box_nms_thresh:
                        should_keep = False
                        break

            if should_keep:
                keep.append(mask)

        return keep

    def predict(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        **kwargs
    ) -> InferenceResult:
        """Standard predict interface."""
        prompt = kwargs.get("prompt")
        if prompt is None:
            # Default to automatic mask generation
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
        """Reset temporal consistency buffer (call between shots/scenes)."""
        self._temporal_buffer.clear()

    def clear_image_embedding(self) -> None:
        """Clear cached image embedding."""
        self._image_embedding = None
        self._original_size = None
        if self.predictor is not None:
            self.predictor.reset_image()
