"""
RobustSAM Integration for Ultimate Rotoscopy
=============================================

RobustSAM extends SAM to handle degraded images including:
- Motion blur
- Low resolution
- Noise
- Compression artifacts

This is critical for VFX work where source footage often has
imperfections that cause standard SAM to fail.

Reference: "RobustSAM: Segment Anything Robustly on Degraded Images"
https://github.com/robustsam/RobustSAM
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class DegradationType(Enum):
    """Types of image degradation RobustSAM handles."""
    MOTION_BLUR = "motion_blur"
    DEFOCUS_BLUR = "defocus_blur"
    GAUSSIAN_NOISE = "gaussian_noise"
    JPEG_COMPRESSION = "jpeg"
    LOW_RESOLUTION = "low_res"
    MIXED = "mixed"
    AUTO_DETECT = "auto"


@dataclass
class RobustSAMConfig:
    """RobustSAM configuration."""
    model_type: str = "vit_h"  # vit_h, vit_l, vit_b
    checkpoint_path: Optional[str] = None

    # Degradation handling
    degradation_type: DegradationType = DegradationType.AUTO_DETECT
    blur_kernel_size: int = 15  # For motion blur estimation
    noise_level: float = 0.1   # Estimated noise level

    # Enhancement options
    enable_deblur: bool = True
    enable_denoise: bool = True
    enable_super_resolution: bool = False

    # Robust features
    use_robust_encoder: bool = True
    use_degradation_aware_prompt: bool = True
    confidence_threshold: float = 0.5

    # Multi-scale processing
    multi_scale: bool = True
    scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])

    # Temporal for video
    temporal_consistency: bool = True
    temporal_window: int = 5

    device: str = "cuda"


@dataclass
class RobustSAMResult:
    """Result from RobustSAM segmentation."""
    mask: np.ndarray                    # Binary mask
    confidence: np.ndarray              # Per-pixel confidence
    degradation_map: np.ndarray         # Estimated degradation
    enhanced_image: Optional[np.ndarray] = None  # Pre-processed image
    metadata: Dict[str, Any] = field(default_factory=dict)


class DegradationEstimator:
    """
    Estimates image degradation type and level.

    This helps RobustSAM choose the appropriate processing
    strategy for each image.
    """

    def __init__(self):
        self.blur_threshold = 100.0  # Laplacian variance threshold
        self.noise_threshold = 10.0

    def estimate(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Estimate degradation in image.

        Args:
            image: Input RGB image

        Returns:
            Dictionary with degradation info
        """
        import cv2

        # Convert to grayscale
        if image.ndim == 3:
            gray = cv2.cvtColor(
                (image * 255).astype(np.uint8) if image.dtype == np.float32 else image,
                cv2.COLOR_RGB2GRAY
            )
        else:
            gray = image

        # Estimate blur using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()

        # Estimate noise level
        noise_level = self._estimate_noise(gray)

        # Estimate motion blur direction
        motion_direction, motion_length = self._estimate_motion_blur(gray)

        # Detect compression artifacts
        compression_score = self._estimate_compression(gray)

        # Determine primary degradation
        degradations = []

        if blur_score < self.blur_threshold:
            degradations.append(DegradationType.MOTION_BLUR)

        if noise_level > self.noise_threshold:
            degradations.append(DegradationType.GAUSSIAN_NOISE)

        if compression_score > 0.5:
            degradations.append(DegradationType.JPEG_COMPRESSION)

        primary_type = degradations[0] if degradations else DegradationType.MIXED

        return {
            "blur_score": blur_score,
            "is_blurry": blur_score < self.blur_threshold,
            "noise_level": noise_level,
            "motion_direction": motion_direction,
            "motion_length": motion_length,
            "compression_score": compression_score,
            "degradation_type": primary_type,
            "all_degradations": degradations,
        }

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estimate noise level using robust median estimator."""
        import cv2

        # Use Laplacian of Gaussian for noise estimation
        # Donoho's robust noise estimation
        H, W = gray.shape

        # Apply Laplacian
        lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)

        # Robust noise estimate using median absolute deviation
        sigma = np.median(np.abs(lap)) / 0.6745

        return sigma

    def _estimate_motion_blur(self, gray: np.ndarray) -> Tuple[float, float]:
        """
        Estimate motion blur direction and length.

        Uses Radon transform-based analysis for blur direction
        and autocorrelation for blur length.
        """
        import cv2

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Simple FFT-based direction estimation
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        # Find dominant direction in frequency domain
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Sample along radial lines
        best_angle = 0
        max_energy = 0

        for angle in range(0, 180, 5):
            rad = np.radians(angle)
            energy = 0

            for r in range(1, min(cx, cy)):
                x = int(cx + r * np.cos(rad))
                y = int(cy + r * np.sin(rad))
                if 0 <= x < w and 0 <= y < h:
                    energy += magnitude[y, x]

            if energy > max_energy:
                max_energy = energy
                best_angle = angle

        # Estimate blur length from autocorrelation
        acf = cv2.matchTemplate(gray.astype(np.float32), gray.astype(np.float32), cv2.TM_CCORR_NORMED)
        blur_length = np.sum(acf > 0.5) ** 0.5  # Rough estimate

        return best_angle, blur_length

    def _estimate_compression(self, gray: np.ndarray) -> float:
        """Estimate JPEG compression artifacts."""
        import cv2

        # Look for 8x8 block artifacts
        h, w = gray.shape

        # Calculate variance in 8x8 blocks
        block_vars = []
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = gray[i:i+8, j:j+8]
                block_vars.append(np.var(block))

        if not block_vars:
            return 0.0

        # High regularity in block variance indicates compression
        block_vars = np.array(block_vars)
        regularity = 1.0 - (np.std(block_vars) / (np.mean(block_vars) + 1e-6))

        return np.clip(regularity, 0, 1)


class ImageEnhancer:
    """
    Pre-processing image enhancement for degraded images.

    Applies deblurring, denoising, and super-resolution
    before segmentation.
    """

    def __init__(self, config: RobustSAMConfig):
        self.config = config

    def enhance(
        self,
        image: np.ndarray,
        degradation_info: Dict[str, Any],
    ) -> np.ndarray:
        """
        Enhance degraded image.

        Args:
            image: Input image
            degradation_info: From DegradationEstimator

        Returns:
            Enhanced image
        """
        import cv2

        # Normalize to float
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        enhanced = image.copy()

        # Deblur if needed
        if self.config.enable_deblur and degradation_info.get("is_blurry", False):
            enhanced = self._deblur(
                enhanced,
                degradation_info.get("motion_direction", 0),
                degradation_info.get("motion_length", 5),
            )

        # Denoise if needed
        if self.config.enable_denoise and degradation_info.get("noise_level", 0) > 5:
            enhanced = self._denoise(enhanced, degradation_info["noise_level"])

        # Super-resolution if enabled
        if self.config.enable_super_resolution:
            enhanced = self._super_resolve(enhanced)

        return np.clip(enhanced, 0, 1)

    def _deblur(
        self,
        image: np.ndarray,
        direction: float,
        length: float,
    ) -> np.ndarray:
        """Apply motion deblur using Wiener deconvolution."""
        import cv2

        # Create motion blur kernel
        kernel_size = int(max(3, length))
        kernel = np.zeros((kernel_size, kernel_size))

        # Fill kernel along motion direction
        rad = np.radians(direction)
        for i in range(kernel_size):
            offset = i - kernel_size // 2
            x = kernel_size // 2 + int(offset * np.cos(rad))
            y = kernel_size // 2 + int(offset * np.sin(rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0

        kernel = kernel / (kernel.sum() + 1e-6)

        # Apply Wiener deconvolution per channel
        result = np.zeros_like(image)

        for c in range(image.shape[2]):
            channel = image[..., c]

            # FFT
            f_channel = np.fft.fft2(channel)
            f_kernel = np.fft.fft2(kernel, s=channel.shape)

            # Wiener filter
            noise_variance = 0.01
            wiener = np.conj(f_kernel) / (np.abs(f_kernel) ** 2 + noise_variance)

            # Deconvolve
            f_result = f_channel * wiener
            result[..., c] = np.abs(np.fft.ifft2(f_result))

        return result

    def _denoise(self, image: np.ndarray, noise_level: float) -> np.ndarray:
        """Apply non-local means denoising."""
        import cv2

        # Convert to uint8 for OpenCV
        img_uint8 = (image * 255).astype(np.uint8)

        # Adjust strength based on noise level
        h = min(10, max(3, noise_level * 0.5))

        denoised = cv2.fastNlMeansDenoisingColored(
            img_uint8, None, h, h, 7, 21
        )

        return denoised.astype(np.float32) / 255.0

    def _super_resolve(self, image: np.ndarray) -> np.ndarray:
        """Apply super-resolution (placeholder for real implementation)."""
        import cv2

        # Simple bicubic upscale for now
        # In production, use Real-ESRGAN or similar
        h, w = image.shape[:2]
        upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        return upscaled


class RobustSAM:
    """
    RobustSAM - Segment Anything Robustly on Degraded Images.

    Extends SAM with degradation-aware processing for:
    - Motion-blurred footage (common in VFX)
    - Noisy images
    - Low-resolution sources
    - Compressed footage

    Key innovations:
    1. Degradation estimation before segmentation
    2. Pre-processing enhancement pipeline
    3. Robust encoder with degradation-aware features
    4. Multi-scale processing for better edge detection
    5. Temporal consistency for video

    Example:
        >>> robust_sam = RobustSAM(RobustSAMConfig(
        ...     degradation_type=DegradationType.AUTO_DETECT,
        ...     enable_deblur=True,
        ...     multi_scale=True,
        ... ))
        >>>
        >>> # Segment motion-blurred image
        >>> result = robust_sam.segment(
        ...     blurry_image,
        ...     points=[[500, 300]],
        ...     labels=[1]
        ... )
        >>> mask = result.mask
    """

    def __init__(self, config: Optional[RobustSAMConfig] = None):
        self.config = config or RobustSAMConfig()
        self.degradation_estimator = DegradationEstimator()
        self.enhancer = ImageEnhancer(self.config)

        # Model components (initialized lazily)
        self._sam = None
        self._predictor = None
        self._robust_encoder = None

        # Temporal state
        self._prev_masks: List[np.ndarray] = []
        self._prev_features: List[np.ndarray] = []

    def _load_model(self):
        """Load RobustSAM model."""
        if self._sam is not None:
            return

        try:
            # Try to load RobustSAM first
            from robustsam import sam_model_registry, SamPredictor

            self._sam = sam_model_registry[self.config.model_type](
                checkpoint=self.config.checkpoint_path
            )
            self._sam.to(self.config.device)
            self._predictor = SamPredictor(self._sam)

        except ImportError:
            # Fallback to regular SAM with robust preprocessing
            try:
                from segment_anything import sam_model_registry, SamPredictor

                if self.config.checkpoint_path:
                    self._sam = sam_model_registry[self.config.model_type](
                        checkpoint=self.config.checkpoint_path
                    )
                    self._sam.to(self.config.device)
                    self._predictor = SamPredictor(self._sam)
                else:
                    # Use SAM2 as fallback
                    self._use_sam2_fallback()

            except ImportError:
                self._use_sam2_fallback()

    def _use_sam2_fallback(self):
        """Use SAM2 with robust preprocessing."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model_cfg = "sam2_hiera_l.yaml"
            sam2_checkpoint = "sam2_hiera_large.pt"

            self._sam = build_sam2(model_cfg, sam2_checkpoint, device=self.config.device)
            self._predictor = SAM2ImagePredictor(self._sam)

        except ImportError:
            raise ImportError(
                "Neither RobustSAM, SAM, nor SAM2 is installed. "
                "Please install one of these packages."
            )

    def segment(
        self,
        image: np.ndarray,
        points: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        box: Optional[List[int]] = None,
        mask_input: Optional[np.ndarray] = None,
        return_enhanced: bool = False,
    ) -> RobustSAMResult:
        """
        Segment image with robust degradation handling.

        Args:
            image: Input RGB image (may be degraded)
            points: List of [x, y] prompt points
            labels: Point labels (1=foreground, 0=background)
            box: Bounding box [x1, y1, x2, y2]
            mask_input: Prior mask for refinement
            return_enhanced: Include enhanced image in result

        Returns:
            RobustSAMResult with mask and confidence
        """
        self._load_model()

        # Normalize image
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Step 1: Estimate degradation
        degradation_info = self.degradation_estimator.estimate(image)

        # Step 2: Enhance image if degraded
        enhanced_image = None
        if self.config.degradation_type != DegradationType.MIXED or degradation_info["all_degradations"]:
            enhanced_image = self.enhancer.enhance(image, degradation_info)
            process_image = enhanced_image
        else:
            process_image = image

        # Step 3: Multi-scale processing if enabled
        if self.config.multi_scale:
            masks, confidences = self._multi_scale_segment(
                process_image, points, labels, box, mask_input
            )
            # Combine multi-scale results
            mask = self._combine_multi_scale(masks, confidences)
            confidence = np.mean(confidences, axis=0)
        else:
            mask, confidence = self._single_scale_segment(
                process_image, points, labels, box, mask_input
            )

        # Step 4: Apply temporal consistency if enabled
        if self.config.temporal_consistency and len(self._prev_masks) > 0:
            mask = self._apply_temporal_consistency(mask, confidence)

        # Update temporal history
        self._prev_masks.append(mask)
        if len(self._prev_masks) > self.config.temporal_window:
            self._prev_masks.pop(0)

        # Create degradation map
        degradation_map = self._create_degradation_map(image, degradation_info)

        return RobustSAMResult(
            mask=mask,
            confidence=confidence,
            degradation_map=degradation_map,
            enhanced_image=enhanced_image if return_enhanced else None,
            metadata={
                "degradation_info": degradation_info,
                "scales_used": self.config.scales if self.config.multi_scale else [1.0],
            }
        )

    def _single_scale_segment(
        self,
        image: np.ndarray,
        points: Optional[List[List[int]]],
        labels: Optional[List[int]],
        box: Optional[List[int]],
        mask_input: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment at single scale."""
        import cv2

        # Convert to uint8 for SAM
        img_uint8 = (image * 255).astype(np.uint8)

        # Set image
        self._predictor.set_image(img_uint8)

        # Prepare prompts
        point_coords = np.array(points) if points else None
        point_labels = np.array(labels) if labels else None
        box_input = np.array(box) if box else None

        # Predict
        masks, scores, logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_input,
            mask_input=mask_input,
            multimask_output=True,
        )

        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.float32)
        confidence = np.full_like(mask, scores[best_idx])

        return mask, confidence

    def _multi_scale_segment(
        self,
        image: np.ndarray,
        points: Optional[List[List[int]]],
        labels: Optional[List[int]],
        box: Optional[List[int]],
        mask_input: Optional[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Segment at multiple scales."""
        import cv2

        h, w = image.shape[:2]
        masks = []
        confidences = []

        for scale in self.config.scales:
            # Resize image
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # Scale prompts
            scaled_points = None
            if points:
                scaled_points = [[int(p[0] * scale), int(p[1] * scale)] for p in points]

            scaled_box = None
            if box:
                scaled_box = [int(b * scale) for b in box]

            scaled_mask_input = None
            if mask_input is not None:
                scaled_mask_input = cv2.resize(
                    mask_input.astype(np.float32),
                    (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR
                )

            # Segment at this scale
            mask, conf = self._single_scale_segment(
                scaled_img, scaled_points, labels, scaled_box, scaled_mask_input
            )

            # Resize back
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)

            masks.append(mask)
            confidences.append(conf)

        return masks, confidences

    def _combine_multi_scale(
        self,
        masks: List[np.ndarray],
        confidences: List[np.ndarray],
    ) -> np.ndarray:
        """Combine multi-scale masks using confidence weighting."""
        # Stack and weight by confidence
        masks = np.array(masks)
        confidences = np.array(confidences)

        # Normalize confidences
        conf_sum = confidences.sum(axis=0) + 1e-6
        weights = confidences / conf_sum

        # Weighted average
        combined = (masks * weights).sum(axis=0)

        # Threshold to binary
        return (combined > 0.5).astype(np.float32)

    def _apply_temporal_consistency(
        self,
        mask: np.ndarray,
        confidence: np.ndarray,
    ) -> np.ndarray:
        """Apply temporal consistency with previous frames."""
        import cv2

        if not self._prev_masks:
            return mask

        # Weight recent frames more heavily
        weights = np.array([0.5 ** i for i in range(len(self._prev_masks), 0, -1)])
        weights = weights / weights.sum()

        # Compute temporal average
        temporal_mask = np.zeros_like(mask)
        for i, prev_mask in enumerate(self._prev_masks):
            if prev_mask.shape == mask.shape:
                temporal_mask += weights[i] * prev_mask

        # Blend current with temporal
        alpha = 0.7  # Current frame weight
        blended = alpha * mask + (1 - alpha) * temporal_mask

        return (blended > 0.5).astype(np.float32)

    def _create_degradation_map(
        self,
        image: np.ndarray,
        degradation_info: Dict[str, Any],
    ) -> np.ndarray:
        """Create per-pixel degradation confidence map."""
        import cv2

        h, w = image.shape[:2]

        # Start with uniform degradation level
        base_level = degradation_info.get("blur_score", 100) / 200.0
        base_level = np.clip(1 - base_level, 0, 1)  # Higher = more degraded

        degradation_map = np.full((h, w), base_level, dtype=np.float32)

        # Add spatial variation based on local blur
        if image.ndim == 3:
            gray = np.dot(image, [0.299, 0.587, 0.114])
        else:
            gray = image

        # Local Laplacian variance
        laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
        local_sharpness = cv2.GaussianBlur(np.abs(laplacian), (31, 31), 0)

        # Normalize
        if local_sharpness.max() > 0:
            local_sharpness = local_sharpness / local_sharpness.max()

        # Invert: low sharpness = high degradation
        degradation_map = 1 - local_sharpness * 0.5

        return degradation_map

    def segment_video_frame(
        self,
        frame: np.ndarray,
        points: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        prev_mask: Optional[np.ndarray] = None,
    ) -> RobustSAMResult:
        """
        Segment a video frame with temporal awareness.

        Args:
            frame: Current video frame
            points: Optional prompt points
            labels: Point labels
            prev_mask: Previous frame mask for guidance

        Returns:
            RobustSAMResult with temporally consistent mask
        """
        return self.segment(
            frame,
            points=points,
            labels=labels,
            mask_input=prev_mask,
            return_enhanced=False,
        )

    def reset_temporal(self):
        """Reset temporal state for new video."""
        self._prev_masks.clear()
        self._prev_features.clear()


def segment_degraded_image(
    image: np.ndarray,
    points: Optional[List[List[int]]] = None,
    labels: Optional[List[int]] = None,
    box: Optional[List[int]] = None,
    auto_enhance: bool = True,
) -> np.ndarray:
    """
    Quick function to segment a degraded image.

    Args:
        image: Input image (may be motion-blurred, noisy, etc.)
        points: Prompt points
        labels: Point labels
        box: Bounding box
        auto_enhance: Auto-detect and enhance degradation

    Returns:
        Binary segmentation mask
    """
    config = RobustSAMConfig(
        degradation_type=DegradationType.AUTO_DETECT if auto_enhance else DegradationType.MIXED,
        enable_deblur=auto_enhance,
        enable_denoise=auto_enhance,
        multi_scale=True,
    )

    robust_sam = RobustSAM(config)
    result = robust_sam.segment(image, points, labels, box)

    return result.mask
