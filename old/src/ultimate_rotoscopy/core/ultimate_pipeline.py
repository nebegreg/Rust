"""
Ultimate Pipeline for Ultimate Rotoscopy
=========================================

The complete professional VFX pipeline combining:

AI Segmentation & Matting:
- SAM3 for interactive segmentation
- RobustSAM for degraded/motion-blurred footage
- MatAnyone for temporal video matting
- GVM (Generative Video Matting) for fine detail

Professional Compositing:
- Despill for green/blue screen removal
- Pixel spread / edge extend
- Color harmonization
- Light wrap for photorealistic integration

This follows the ILM StageCraft-style workflow where:
"despilling is arguably the most important step" and
the background must be in place for proper evaluation.

Reference: Ben McEwan's despill guide, Digital Anarchy Light Wrap
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# AI Models
from ultimate_rotoscopy.models.sam3 import SAM3, SAM3Config, PromptType
from ultimate_rotoscopy.models.robust_sam import RobustSAM, RobustSAMConfig, DegradationType
from ultimate_rotoscopy.models.matanyone import MatAnyone, MatAnyoneConfig
from ultimate_rotoscopy.models.gvm import GVM, GVMConfig, GVMMode

# Compositing
from ultimate_rotoscopy.compositing.compositor import (
    UltimateCompositor,
    CompositorConfig,
    CompositeResult,
    CompositeMode,
)
from ultimate_rotoscopy.compositing.despill import DespillAlgorithm, SpillChannel
from ultimate_rotoscopy.compositing.light_wrap import WrapMode
from ultimate_rotoscopy.compositing.harmonization import HarmonizationMethod


class PipelineStage(Enum):
    """Pipeline processing stages."""
    SEGMENTATION = "segmentation"       # Initial SAM3/RobustSAM segmentation
    MATTING = "matting"                 # MatAnyone/GVM matte refinement
    DESPILL = "despill"                 # Spill removal
    EDGE_OPERATIONS = "edge_ops"        # Pixel spread, erode/dilate
    HARMONIZATION = "harmonization"     # Color matching
    LIGHT_WRAP = "light_wrap"           # Background light wrap
    COMPOSITE = "composite"             # Final composite


class MattingBackend(Enum):
    """Available matting backends."""
    MATANYONE = "matanyone"     # Memory-based video matting
    GVM = "gvm"                 # Diffusion-based matting
    HYBRID = "hybrid"          # Combine both for best results


@dataclass
class UltimatePipelineConfig:
    """Configuration for the ultimate pipeline."""
    # Segmentation
    use_robust_sam: bool = True             # Use RobustSAM for degraded images
    auto_detect_degradation: bool = True    # Auto-detect motion blur, noise

    # Matting
    matting_backend: MattingBackend = MattingBackend.MATANYONE
    enable_gvm_refinement: bool = True      # Use GVM for fine detail
    temporal_consistency: bool = True

    # Compositing
    enable_despill: bool = True
    despill_algorithm: DespillAlgorithm = DespillAlgorithm.ADAPTIVE
    despill_channel: SpillChannel = SpillChannel.GREEN
    despill_strength: float = 0.8

    enable_edge_operations: bool = True
    pixel_spread_amount: int = 5
    edge_erode: int = 0
    edge_dilate: int = 0

    enable_harmonization: bool = True
    harmonization_method: HarmonizationMethod = HarmonizationMethod.ADAPTIVE
    harmonization_strength: float = 0.4

    enable_light_wrap: bool = True
    light_wrap_intensity: float = 0.5
    light_wrap_width: int = 20
    light_wrap_mode: WrapMode = WrapMode.SCREEN

    # Output
    composite_mode: CompositeMode = CompositeMode.OVER
    output_premultiplied: bool = False

    # Device
    device: str = "cuda"


@dataclass
class UltimatePipelineResult:
    """Complete result from ultimate pipeline."""
    # Final outputs
    composite: np.ndarray           # Final composite
    alpha: np.ndarray               # Final alpha matte
    foreground: np.ndarray          # Processed foreground

    # Intermediate results
    segmentation_mask: np.ndarray   # Initial SAM3 segmentation
    refined_matte: np.ndarray       # After MatAnyone/GVM
    despilled: Optional[np.ndarray] = None
    edge_extended: Optional[np.ndarray] = None
    harmonized: Optional[np.ndarray] = None
    light_wrapped: Optional[np.ndarray] = None

    # Confidence and quality
    matte_confidence: Optional[np.ndarray] = None
    edge_detail: Optional[np.ndarray] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class UltimatePipeline:
    """
    The Ultimate Rotoscopy Pipeline.

    Combines state-of-the-art AI models with professional
    compositing techniques for cinema-quality results.

    Pipeline stages:
    1. Segmentation (SAM3/RobustSAM) - Interactive mask creation
    2. Matting (MatAnyone/GVM) - Refine to alpha matte
    3. Despill - Remove green/blue screen spill
    4. Edge Operations - Pixel spread, erode/dilate
    5. Harmonization - Match FG colors to BG
    6. Light Wrap - Blend BG light onto FG edges
    7. Composite - Final alpha over

    Example:
        >>> pipeline = UltimatePipeline(UltimatePipelineConfig(
        ...     matting_backend=MattingBackend.MATANYONE,
        ...     enable_light_wrap=True,
        ...     despill_algorithm=DespillAlgorithm.ADAPTIVE,
        ... ))
        >>>
        >>> # Process single image
        >>> result = pipeline.process_image(
        ...     foreground=green_screen_image,
        ...     background=background_plate,
        ...     prompt_points=[[500, 300]],
        ...     prompt_labels=[1],
        ... )
        >>>
        >>> # Or process video
        >>> pipeline.initialize_video(first_frame, prompt_points, prompt_labels)
        >>> for frame in video_frames:
        ...     result = pipeline.process_video_frame(frame, background)
    """

    def __init__(self, config: Optional[UltimatePipelineConfig] = None):
        self.config = config or UltimatePipelineConfig()

        # Initialize components lazily
        self._sam3: Optional[SAM3] = None
        self._robust_sam: Optional[RobustSAM] = None
        self._matanyone: Optional[MatAnyone] = None
        self._gvm: Optional[GVM] = None
        self._compositor: Optional[UltimateCompositor] = None

        # Video state
        self._video_initialized = False
        self._frame_idx = 0
        self._prev_mask: Optional[np.ndarray] = None

    def _get_sam3(self) -> SAM3:
        """Get or create SAM3 instance."""
        if self._sam3 is None:
            self._sam3 = SAM3(SAM3Config(device=self.config.device))
        return self._sam3

    def _get_robust_sam(self) -> RobustSAM:
        """Get or create RobustSAM instance."""
        if self._robust_sam is None:
            self._robust_sam = RobustSAM(RobustSAMConfig(
                degradation_type=DegradationType.AUTO_DETECT if self.config.auto_detect_degradation else DegradationType.MIXED,
                device=self.config.device,
            ))
        return self._robust_sam

    def _get_matanyone(self) -> MatAnyone:
        """Get or create MatAnyone instance."""
        if self._matanyone is None:
            self._matanyone = MatAnyone(MatAnyoneConfig(
                device=self.config.device,
            ))
        return self._matanyone

    def _get_gvm(self) -> GVM:
        """Get or create GVM instance."""
        if self._gvm is None:
            self._gvm = GVM(GVMConfig(
                mode=GVMMode.QUALITY if self.config.enable_gvm_refinement else GVMMode.FAST,
                temporal_attention=self.config.temporal_consistency,
                device=self.config.device,
            ))
        return self._gvm

    def _get_compositor(self) -> UltimateCompositor:
        """Get or create compositor instance."""
        if self._compositor is None:
            self._compositor = UltimateCompositor(CompositorConfig(
                enable_despill=self.config.enable_despill,
                despill_algorithm=self.config.despill_algorithm,
                despill_channel=self.config.despill_channel,
                despill_strength=self.config.despill_strength,
                enable_edge_operations=self.config.enable_edge_operations,
                pixel_spread_amount=self.config.pixel_spread_amount,
                edge_erode_amount=self.config.edge_erode,
                edge_dilate_amount=self.config.edge_dilate,
                enable_harmonization=self.config.enable_harmonization,
                harmonization_method=self.config.harmonization_method,
                harmonization_strength=self.config.harmonization_strength,
                enable_light_wrap=self.config.enable_light_wrap,
                light_wrap_intensity=self.config.light_wrap_intensity,
                light_wrap_width=self.config.light_wrap_width,
                light_wrap_mode=self.config.light_wrap_mode,
                composite_mode=self.config.composite_mode,
                output_premultiplied=self.config.output_premultiplied,
            ))
        return self._compositor

    def process_image(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        prompt_points: Optional[List[List[int]]] = None,
        prompt_labels: Optional[List[int]] = None,
        prompt_box: Optional[List[int]] = None,
        existing_mask: Optional[np.ndarray] = None,
        clean_plate: Optional[np.ndarray] = None,
    ) -> UltimatePipelineResult:
        """
        Process a single image through the complete pipeline.

        Args:
            foreground: Foreground/green screen image
            background: Background plate
            prompt_points: SAM3 prompt points [[x, y], ...]
            prompt_labels: Point labels (1=fg, 0=bg)
            prompt_box: Bounding box [x1, y1, x2, y2]
            existing_mask: Skip segmentation, use this mask
            clean_plate: Clean plate for better despill

        Returns:
            UltimatePipelineResult with all outputs
        """
        # Normalize inputs
        foreground = self._normalize(foreground)
        background = self._normalize(background)

        if clean_plate is not None:
            clean_plate = self._normalize(clean_plate)

        # Stage 1: Segmentation
        if existing_mask is not None:
            segmentation_mask = self._normalize_alpha(existing_mask)
        else:
            segmentation_mask = self._segment(
                foreground, prompt_points, prompt_labels, prompt_box
            )

        # Stage 2: Matting refinement
        refined_matte, matte_confidence, edge_detail = self._refine_matte(
            foreground, segmentation_mask
        )

        # Stage 3-7: Compositing
        compositor = self._get_compositor()
        comp_result = compositor.composite(
            foreground, background, refined_matte, clean_plate
        )

        return UltimatePipelineResult(
            composite=comp_result.composite,
            alpha=comp_result.alpha,
            foreground=comp_result.foreground,
            segmentation_mask=segmentation_mask,
            refined_matte=refined_matte,
            despilled=comp_result.despilled,
            edge_extended=comp_result.edge_extended,
            harmonized=comp_result.harmonized,
            light_wrapped=comp_result.light_wrapped,
            matte_confidence=matte_confidence,
            edge_detail=edge_detail,
            metadata={
                "stages_completed": [
                    PipelineStage.SEGMENTATION.value,
                    PipelineStage.MATTING.value,
                    PipelineStage.DESPILL.value if self.config.enable_despill else None,
                    PipelineStage.EDGE_OPERATIONS.value if self.config.enable_edge_operations else None,
                    PipelineStage.HARMONIZATION.value if self.config.enable_harmonization else None,
                    PipelineStage.LIGHT_WRAP.value if self.config.enable_light_wrap else None,
                    PipelineStage.COMPOSITE.value,
                ],
                "matting_backend": self.config.matting_backend.value,
            }
        )

    def initialize_video(
        self,
        first_frame: np.ndarray,
        prompt_points: Optional[List[List[int]]] = None,
        prompt_labels: Optional[List[int]] = None,
        prompt_box: Optional[List[int]] = None,
        existing_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Initialize video processing with first frame.

        Args:
            first_frame: First video frame
            prompt_points: SAM3 prompt points
            prompt_labels: Point labels
            prompt_box: Bounding box
            existing_mask: Use existing mask instead of segmenting

        Returns:
            Initial segmentation mask
        """
        first_frame = self._normalize(first_frame)

        # Get initial segmentation
        if existing_mask is not None:
            initial_mask = self._normalize_alpha(existing_mask)
        else:
            initial_mask = self._segment(
                first_frame, prompt_points, prompt_labels, prompt_box
            )

        # Initialize matting backend
        if self.config.matting_backend in (MattingBackend.MATANYONE, MattingBackend.HYBRID):
            matanyone = self._get_matanyone()
            matanyone.initialize(first_frame, initial_mask)

        if self.config.matting_backend in (MattingBackend.GVM, MattingBackend.HYBRID):
            gvm = self._get_gvm()
            gvm.initialize(first_frame, initial_mask)

        self._video_initialized = True
        self._frame_idx = 0
        self._prev_mask = initial_mask

        return initial_mask

    def process_video_frame(
        self,
        frame: np.ndarray,
        background: np.ndarray,
        clean_plate: Optional[np.ndarray] = None,
        correction_mask: Optional[np.ndarray] = None,
    ) -> UltimatePipelineResult:
        """
        Process a video frame through the pipeline.

        Args:
            frame: Current video frame
            background: Background plate for this frame
            clean_plate: Optional clean plate
            correction_mask: Optional user correction mask

        Returns:
            UltimatePipelineResult for this frame
        """
        if not self._video_initialized:
            raise RuntimeError(
                "Video not initialized. Call initialize_video() first."
            )

        frame = self._normalize(frame)
        background = self._normalize(background)

        if clean_plate is not None:
            clean_plate = self._normalize(clean_plate)

        self._frame_idx += 1

        # Get matte from video matting backend
        if self.config.matting_backend == MattingBackend.MATANYONE:
            matanyone = self._get_matanyone()
            matte_result = matanyone.process_frame(frame, self._frame_idx)
            refined_matte = matte_result.alpha
            matte_confidence = matte_result.confidence
            edge_detail = None

        elif self.config.matting_backend == MattingBackend.GVM:
            gvm = self._get_gvm()
            gvm_result = gvm.process_frame(frame, correction_mask)
            refined_matte = gvm_result.alpha
            matte_confidence = gvm_result.confidence
            edge_detail = gvm_result.edge_detail

        else:  # HYBRID
            # Use both and combine
            matanyone = self._get_matanyone()
            gvm = self._get_gvm()

            ma_result = matanyone.process_frame(frame, self._frame_idx)
            gvm_result = gvm.process_frame(frame, correction_mask)

            # Combine: MatAnyone for solid regions, GVM for edges
            edge_mask = self._get_edge_region(ma_result.alpha)
            refined_matte = (
                ma_result.alpha * (1 - edge_mask) +
                gvm_result.alpha * edge_mask
            )
            matte_confidence = (ma_result.confidence + gvm_result.confidence) / 2
            edge_detail = gvm_result.edge_detail

        # Apply user correction if provided
        if correction_mask is not None:
            correction_mask = self._normalize_alpha(correction_mask)
            refined_matte = self._apply_correction(refined_matte, correction_mask)

        # Store for next frame
        self._prev_mask = refined_matte

        # Composite
        compositor = self._get_compositor()
        comp_result = compositor.composite(
            frame, background, refined_matte, clean_plate
        )

        return UltimatePipelineResult(
            composite=comp_result.composite,
            alpha=comp_result.alpha,
            foreground=comp_result.foreground,
            segmentation_mask=self._prev_mask,  # Use propagated mask
            refined_matte=refined_matte,
            despilled=comp_result.despilled,
            edge_extended=comp_result.edge_extended,
            harmonized=comp_result.harmonized,
            light_wrapped=comp_result.light_wrapped,
            matte_confidence=matte_confidence,
            edge_detail=edge_detail,
            metadata={
                "frame_idx": self._frame_idx,
                "matting_backend": self.config.matting_backend.value,
            }
        )

    def _segment(
        self,
        image: np.ndarray,
        points: Optional[List[List[int]]],
        labels: Optional[List[int]],
        box: Optional[List[int]],
    ) -> np.ndarray:
        """Perform segmentation."""
        # Check if we should use RobustSAM for degraded images
        use_robust = self.config.use_robust_sam

        if use_robust and self.config.auto_detect_degradation:
            # Check image quality
            robust_sam = self._get_robust_sam()
            result = robust_sam.segment(
                image, points, labels, box, return_enhanced=False
            )
            return result.mask

        else:
            # Use standard SAM3
            sam3 = self._get_sam3()

            # Determine prompt type
            if points and labels:
                prompt_type = PromptType.POINT
            elif box:
                prompt_type = PromptType.BOX
            else:
                prompt_type = PromptType.AUTO

            result = sam3.segment(
                image,
                prompt_type=prompt_type,
                points=points,
                labels=labels,
                box=box,
            )
            return result.mask

    def _refine_matte(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Refine binary mask to alpha matte."""
        if self.config.matting_backend == MattingBackend.MATANYONE:
            matanyone = self._get_matanyone()
            result = matanyone.initialize(image, mask)
            return result.alpha, result.confidence, None

        elif self.config.matting_backend == MattingBackend.GVM:
            gvm = self._get_gvm()
            result = gvm.initialize(image, mask)
            return result.alpha, result.confidence, result.edge_detail

        else:  # HYBRID
            matanyone = self._get_matanyone()
            gvm = self._get_gvm()

            ma_result = matanyone.initialize(image, mask)
            gvm_result = gvm.initialize(image, mask)

            # Combine
            edge_mask = self._get_edge_region(ma_result.alpha)
            combined = (
                ma_result.alpha * (1 - edge_mask) +
                gvm_result.alpha * edge_mask
            )

            confidence = (ma_result.confidence + gvm_result.confidence) / 2

            return combined, confidence, gvm_result.edge_detail

    def _get_edge_region(self, alpha: np.ndarray) -> np.ndarray:
        """Get edge region mask for hybrid matting."""
        import cv2

        alpha_8bit = (alpha * 255).astype(np.uint8)

        # Dilate and erode to get edge band
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(alpha_8bit, kernel)
        eroded = cv2.erode(alpha_8bit, kernel)

        edge = (dilated.astype(np.float32) - eroded.astype(np.float32)) / 255.0

        # Smooth
        edge = cv2.GaussianBlur(edge, (7, 7), 0)

        return edge

    def _apply_correction(
        self,
        matte: np.ndarray,
        correction: np.ndarray,
    ) -> np.ndarray:
        """Apply user correction to matte."""
        # Where correction is > 0.5, use correction value
        # Where correction is < 0.5, use original
        # In between, blend

        weight = np.abs(correction - 0.5) * 2
        result = matte * (1 - weight) + correction * weight

        return np.clip(result, 0, 1)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 float."""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)

    def _normalize_alpha(self, alpha: np.ndarray) -> np.ndarray:
        """Normalize alpha to 0-1 float, single channel."""
        alpha = self._normalize(alpha)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        return alpha

    def reset(self):
        """Reset pipeline state for new video."""
        self._video_initialized = False
        self._frame_idx = 0
        self._prev_mask = None

        if self._matanyone:
            self._matanyone.reset()
        if self._gvm:
            self._gvm.reset()
        if self._robust_sam:
            self._robust_sam.reset_temporal()


def ultimate_composite(
    foreground: np.ndarray,
    background: np.ndarray,
    prompt_points: List[List[int]],
    prompt_labels: List[int],
    enable_all: bool = True,
) -> np.ndarray:
    """
    Quick function for ultimate compositing.

    Args:
        foreground: Foreground/green screen image
        background: Background plate
        prompt_points: Click points [[x, y], ...]
        prompt_labels: Labels (1=fg, 0=bg)
        enable_all: Enable all processing stages

    Returns:
        Final composite image
    """
    config = UltimatePipelineConfig(
        matting_backend=MattingBackend.MATANYONE,
        enable_despill=enable_all,
        enable_edge_operations=enable_all,
        enable_harmonization=enable_all,
        enable_light_wrap=enable_all,
    )

    pipeline = UltimatePipeline(config)
    result = pipeline.process_image(
        foreground, background,
        prompt_points=prompt_points,
        prompt_labels=prompt_labels,
    )

    return result.composite
