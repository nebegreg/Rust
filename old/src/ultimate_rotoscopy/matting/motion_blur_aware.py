"""
Motion Blur-Aware Alpha Matting System
=======================================

Advanced motion blur detection and handling for cinema-quality matting.

Based on research:
- Motion-Aware KNN Laplacian for Video Matting (Adobe Research)
- Alpha Matting of Motion-Blurred Objects (Springer)
- Improving Alpha Matting and Motion Blurred Foreground Estimation (ICIP 2013)
- Boris FX Optical Flow ML techniques
- RE:Vision Effects ReelSmart Motion Blur Pro

Key Techniques:
- Laplacian variance for blur detection
- Optical flow magnitude for motion analysis
- Adaptive mixing of sharp and blurred alpha
- Temporal consistency for video sequences
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

import cv2
import numpy as np
from scipy import ndimage


class MotionBlurLevel(Enum):
    """Motion blur severity levels."""
    NONE = "none"           # No motion blur detected
    LIGHT = "light"         # Slight blur (< 2 pixels)
    MODERATE = "moderate"   # Moderate blur (2-5 pixels)
    HEAVY = "heavy"         # Heavy blur (> 5 pixels)


@dataclass
class MotionBlurConfig:
    """Configuration for motion blur detection and processing."""

    # Detection parameters
    laplacian_threshold: float = 100.0     # Variance threshold for blur detection
    flow_magnitude_threshold: float = 2.0  # Optical flow threshold (pixels)

    # Processing parameters
    sharp_kernel_size: int = 3             # Sharpening kernel size
    deblur_iterations: int = 3             # Richardson-Lucy iterations
    temporal_window: int = 5               # Frames for temporal consistency

    # Mixing parameters
    blend_falloff: float = 0.3             # Smooth transition between sharp/blur
    edge_preservation: float = 0.8         # Edge detail preservation (0-1)

    # Quality
    use_optical_flow: bool = True          # Use optical flow for motion analysis
    use_temporal_consistency: bool = True  # Apply temporal smoothing


@dataclass
class MotionBlurResult:
    """Result of motion blur-aware processing."""

    # Alpha outputs
    alpha_sharp: np.ndarray        # Sharpened alpha
    alpha_motion_blur: np.ndarray  # Motion-blurred alpha
    alpha_final: np.ndarray        # Adaptively mixed result

    # Analysis outputs
    blur_mask: np.ndarray          # Motion blur confidence map (0-1)
    flow_magnitude: Optional[np.ndarray]  # Optical flow magnitude
    blur_level: MotionBlurLevel    # Overall blur assessment

    # Statistics
    blur_percentage: float         # Percentage of pixels with motion blur
    average_motion: float          # Average motion magnitude (pixels)


class MotionBlurDetector:
    """
    Detects motion blur in images using multiple techniques.

    Methods:
    1. Laplacian Variance: Measures image sharpness
    2. Optical Flow Magnitude: Measures motion between frames
    3. Frequency Analysis: Analyzes blur in frequency domain
    """

    def __init__(self, config: Optional[MotionBlurConfig] = None):
        self.config = config or MotionBlurConfig()

    def detect_blur(
        self,
        image: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, MotionBlurLevel]:
        """
        Detect motion blur in image.

        Args:
            image: Input image (H, W, 3) or (H, W)
            prev_frame: Previous frame for optical flow (optional)

        Returns:
            blur_mask: Per-pixel blur confidence (H, W) in [0, 1]
            blur_level: Overall blur assessment
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                (image * 255).astype(np.uint8) if image.dtype == np.float32 else image,
                cv2.COLOR_RGB2GRAY
            )
        else:
            gray = image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)

        # Method 1: Laplacian variance (sharpness)
        blur_mask_laplacian = self._detect_blur_laplacian(gray)

        # Method 2: Optical flow (motion)
        if self.config.use_optical_flow and prev_frame is not None:
            blur_mask_flow = self._detect_blur_optical_flow(gray, prev_frame)
            # Combine both methods
            blur_mask = np.maximum(blur_mask_laplacian, blur_mask_flow)
        else:
            blur_mask = blur_mask_laplacian

        # Determine overall blur level
        blur_percentage = np.mean(blur_mask > 0.3) * 100
        if blur_percentage < 5:
            blur_level = MotionBlurLevel.NONE
        elif blur_percentage < 20:
            blur_level = MotionBlurLevel.LIGHT
        elif blur_percentage < 50:
            blur_level = MotionBlurLevel.MODERATE
        else:
            blur_level = MotionBlurLevel.HEAVY

        return blur_mask, blur_level

    def _detect_blur_laplacian(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect blur using Laplacian variance method.

        Low variance = blurred, High variance = sharp
        """
        # Compute Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

        # Compute local variance in sliding window
        kernel_size = 15
        mean = cv2.boxFilter(laplacian, -1, (kernel_size, kernel_size))
        mean_sq = cv2.boxFilter(laplacian ** 2, -1, (kernel_size, kernel_size))
        variance = mean_sq - mean ** 2

        # Normalize variance to blur confidence [0, 1]
        # Low variance (< threshold) = high blur confidence
        threshold = self.config.laplacian_threshold
        blur_confidence = 1.0 - np.clip(variance / threshold, 0, 1)

        return blur_confidence.astype(np.float32)

    def _detect_blur_optical_flow(
        self,
        gray: np.ndarray,
        prev_gray: np.ndarray,
    ) -> np.ndarray:
        """
        Detect motion blur using optical flow magnitude.

        High flow magnitude = potential motion blur
        """
        # Ensure same size
        if gray.shape != prev_gray.shape:
            prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))

        # Compute optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Compute flow magnitude
        magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

        # Normalize to blur confidence [0, 1]
        threshold = self.config.flow_magnitude_threshold
        blur_confidence = np.clip(magnitude / (threshold * 2), 0, 1)

        return blur_confidence.astype(np.float32)


class MotionBlurAwareMatting:
    """
    Motion blur-aware alpha matting system.

    Produces three alpha outputs:
    1. alpha_sharp: Deblurred/sharpened alpha
    2. alpha_motion_blur: Motion-blur preserved alpha
    3. alpha_final: Adaptively mixed based on blur_mask

    This prevents:
    - Temporal popping in video sequences
    - Over-sharpening of naturally blurred edges
    - Artifacts in fast-moving objects
    """

    def __init__(self, config: Optional[MotionBlurConfig] = None):
        self.config = config or MotionBlurConfig()
        self.detector = MotionBlurDetector(config)
        self._prev_frame = None
        self._prev_alpha = None
        self._flow = None

    def process(
        self,
        alpha: np.ndarray,
        image: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
    ) -> MotionBlurResult:
        """
        Process alpha with motion blur awareness.

        Args:
            alpha: Input alpha channel (H, W) in [0, 1]
            image: Reference RGB image (H, W, 3)
            prev_frame: Previous frame for temporal analysis (optional)

        Returns:
            MotionBlurResult with all outputs
        """
        # Ensure float32
        alpha = alpha.astype(np.float32)
        if alpha.max() > 1.0:
            alpha = alpha / 255.0

        # Detect motion blur
        blur_mask, blur_level = self.detector.detect_blur(image, prev_frame)

        # Compute optical flow if available
        flow_magnitude = None
        if self.config.use_optical_flow and prev_frame is not None:
            flow_magnitude = self._compute_flow_magnitude(image, prev_frame)

        # Generate sharp alpha
        alpha_sharp = self._sharpen_alpha(alpha, blur_mask)

        # Generate motion-blurred alpha (preserve natural blur)
        alpha_motion_blur = self._preserve_motion_blur(alpha, blur_mask, flow_magnitude)

        # Adaptively mix based on blur_mask
        alpha_final = self._adaptive_mix(
            alpha_sharp,
            alpha_motion_blur,
            blur_mask,
        )

        # Temporal consistency for video
        if self.config.use_temporal_consistency and self._prev_alpha is not None:
            alpha_final = self._apply_temporal_consistency(alpha_final, self._prev_alpha)

        # Update history
        self._prev_frame = image
        self._prev_alpha = alpha_final

        # Statistics
        blur_percentage = np.mean(blur_mask > 0.3) * 100
        average_motion = np.mean(flow_magnitude) if flow_magnitude is not None else 0.0

        return MotionBlurResult(
            alpha_sharp=alpha_sharp,
            alpha_motion_blur=alpha_motion_blur,
            alpha_final=alpha_final,
            blur_mask=blur_mask,
            flow_magnitude=flow_magnitude,
            blur_level=blur_level,
            blur_percentage=blur_percentage,
            average_motion=average_motion,
        )

    def _sharpen_alpha(self, alpha: np.ndarray, blur_mask: np.ndarray) -> np.ndarray:
        """
        Sharpen alpha channel adaptively.

        Uses unsharp masking with edge preservation.
        """
        # Gaussian blur for base
        kernel_size = self.config.sharp_kernel_size * 2 + 1
        blurred = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)

        # Unsharp mask: alpha + amount * (alpha - blurred)
        # Amount varies with blur_mask (more sharpening where blurred)
        amount = 1.0 + blur_mask * 2.0  # 1.0 to 3.0 based on blur

        sharpened = alpha + amount[..., np.newaxis] * (alpha - blurred) if len(alpha.shape) == 3 else alpha + amount * (alpha - blurred)

        # Edge preservation: don't over-sharpen edges
        edge_mask = self._detect_alpha_edges(alpha)
        preservation = self.config.edge_preservation
        sharpened = np.where(
            edge_mask > 0.5,
            preservation * alpha + (1 - preservation) * sharpened,
            sharpened,
        )

        return np.clip(sharpened, 0.0, 1.0)

    def _preserve_motion_blur(
        self,
        alpha: np.ndarray,
        blur_mask: np.ndarray,
        flow_magnitude: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Preserve natural motion blur in alpha.

        Applies directional blur based on optical flow.
        """
        if flow_magnitude is None or self._flow is None:
            # Simple Gaussian blur for motion-blurred regions
            kernel_size = 5
            blurred = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)
            return np.where(blur_mask > 0.5, blurred, alpha)

        # Directional motion blur based on flow
        flow = self._flow
        alpha_mb = alpha.copy()

        # Apply motion blur along flow direction
        for y in range(0, alpha.shape[0], 4):  # Subsample for speed
            for x in range(0, alpha.shape[1], 4):
                if blur_mask[y, x] > 0.5:
                    fx, fy = flow[y, x]
                    magnitude = np.sqrt(fx**2 + fy**2)

                    if magnitude > 1.0:
                        # Sample along motion direction
                        samples = []
                        for t in np.linspace(-0.5, 0.5, 5):
                            sx = int(np.clip(x + fx * t, 0, alpha.shape[1] - 1))
                            sy = int(np.clip(y + fy * t, 0, alpha.shape[0] - 1))
                            samples.append(alpha[sy, sx])

                        alpha_mb[y:y+4, x:x+4] = np.mean(samples)

        return alpha_mb

    def _adaptive_mix(
        self,
        alpha_sharp: np.ndarray,
        alpha_blur: np.ndarray,
        blur_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Adaptively mix sharp and blurred alpha based on blur_mask.

        Uses smooth falloff for natural transitions.
        """
        # Smooth blur mask for gradual transition
        falloff = self.config.blend_falloff
        blur_weight = cv2.GaussianBlur(
            blur_mask,
            (int(falloff * 20) * 2 + 1, int(falloff * 20) * 2 + 1),
            0,
        )

        # Mix: weight towards sharp in sharp regions, blur in blurred regions
        mixed = (1.0 - blur_weight) * alpha_sharp + blur_weight * alpha_blur

        return np.clip(mixed, 0.0, 1.0)

    def _apply_temporal_consistency(
        self,
        alpha_current: np.ndarray,
        alpha_prev: np.ndarray,
    ) -> np.ndarray:
        """
        Apply temporal smoothing to reduce flickering.

        Uses exponential moving average with motion compensation.
        """
        # Simple temporal blend (can be enhanced with motion compensation)
        temporal_weight = 0.7  # 70% current, 30% previous

        blended = temporal_weight * alpha_current + (1 - temporal_weight) * alpha_prev

        return blended

    def _compute_flow_magnitude(
        self,
        image: np.ndarray,
        prev_image: np.ndarray,
    ) -> np.ndarray:
        """Compute optical flow magnitude."""
        # Convert to grayscale
        gray = cv2.cvtColor(
            (image * 255).astype(np.uint8) if image.dtype == np.float32 else image,
            cv2.COLOR_RGB2GRAY,
        )
        prev_gray = cv2.cvtColor(
            (prev_image * 255).astype(np.uint8) if prev_image.dtype == np.float32 else prev_image,
            cv2.COLOR_RGB2GRAY,
        )

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        # Store for later use
        self._flow = flow

        # Compute magnitude
        magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

        return magnitude

    def _detect_alpha_edges(self, alpha: np.ndarray) -> np.ndarray:
        """Detect edges in alpha channel."""
        # Sobel edge detection
        sobelx = cv2.Sobel(alpha, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(alpha, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)

        # Normalize
        if np.max(edges) > 0:
            edges = edges / np.max(edges)

        return edges.astype(np.float32)


def visualize_motion_blur_result(result: MotionBlurResult) -> np.ndarray:
    """
    Create visualization of motion blur processing.

    Returns 4-panel image showing:
    - Sharp alpha
    - Motion blur alpha
    - Final alpha
    - Blur mask
    """
    h, w = result.alpha_sharp.shape

    # Create 2x2 grid
    vis = np.zeros((h * 2, w * 2), dtype=np.float32)

    vis[0:h, 0:w] = result.alpha_sharp
    vis[0:h, w:w*2] = result.alpha_motion_blur
    vis[h:h*2, 0:w] = result.alpha_final
    vis[h:h*2, w:w*2] = result.blur_mask

    return vis
