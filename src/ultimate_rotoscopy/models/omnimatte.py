"""
OmniMatte - Associating Objects and Their Effects in Video
==========================================================

OmniMatte decomposes video into multiple RGBA layers that
include objects AND their associated effects (shadows, reflections).

This is critical for professional VFX where you need to:
- Separate an actor with their shadow
- Remove reflections with the object
- Re-composite with proper effect preservation

Reference: "Omnimatte: Associating Objects and Their Effects in Video"
https://github.com/erikalu/omnimatte
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class LayerType(Enum):
    """Types of OmniMatte layers."""
    FOREGROUND = "foreground"       # Main subject
    SHADOW = "shadow"               # Cast shadows
    REFLECTION = "reflection"       # Reflections
    BACKGROUND = "background"       # Background reconstruction
    EFFECT = "effect"               # Other associated effects


@dataclass
class OmniMatteConfig:
    """OmniMatte configuration."""
    # Model settings
    model_path: Optional[str] = None

    # Layer settings
    num_layers: int = 4             # Number of RGBA layers
    include_background: bool = True  # Reconstruct background

    # Optimization settings
    num_iterations: int = 2000
    learning_rate: float = 1e-4
    flow_weight: float = 1.0
    mask_weight: float = 1.0
    regularization_weight: float = 0.01

    # Video settings
    temporal_window: int = 5        # Frames for temporal consistency
    use_flow: bool = True           # Use optical flow

    # Output
    output_resolution: Optional[Tuple[int, int]] = None

    device: str = "cuda"


@dataclass
class OmniMatteLayer:
    """A single OmniMatte layer."""
    rgba: np.ndarray                # RGBA layer (H, W, 4)
    alpha: np.ndarray               # Alpha channel
    layer_type: LayerType
    confidence: np.ndarray          # Per-pixel confidence
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OmniMatteResult:
    """Result from OmniMatte decomposition."""
    layers: List[OmniMatteLayer]    # All decomposed layers
    background: np.ndarray          # Reconstructed background
    reconstruction: np.ndarray      # Full reconstruction
    flow: Optional[np.ndarray]      # Optical flow used
    metadata: Dict[str, Any] = field(default_factory=dict)


class FlowEstimator:
    """Optical flow estimation for temporal consistency."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def compute(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> np.ndarray:
        """
        Compute optical flow between frames.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Flow field (H, W, 2)
        """
        try:
            return self._neural_flow(frame1, frame2)
        except Exception:
            return self._opencv_flow(frame1, frame2)

    def _opencv_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> np.ndarray:
        """OpenCV Farneback flow."""
        import cv2

        # Convert to uint8 if needed
        if frame1.dtype == np.float32 or frame1.dtype == np.float64:
            frame1 = (frame1 * 255).astype(np.uint8)
            frame2 = (frame2 * 255).astype(np.uint8)

        # Convert to grayscale
        if frame1.ndim == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            gray1, gray2 = frame1, frame2

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        return flow

    def _neural_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> np.ndarray:
        """Neural network flow (RAFT-style)."""
        import torch

        try:
            from torchvision.models.optical_flow import raft_large

            if self._model is None:
                self._model = raft_large(pretrained=True).to(self.device).eval()

            # Prepare inputs
            if frame1.dtype == np.uint8:
                frame1 = frame1.astype(np.float32) / 255.0
                frame2 = frame2.astype(np.float32) / 255.0

            t1 = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).to(self.device)
            t2 = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                flow = self._model(t1 * 255, t2 * 255)[-1]

            return flow.squeeze().permute(1, 2, 0).cpu().numpy()

        except ImportError:
            return self._opencv_flow(frame1, frame2)


class LayerDecomposer:
    """
    Decompose video into RGBA layers.

    Uses optimization to find layers that:
    - Sum to original image
    - Have smooth alpha mattes
    - Are temporally consistent
    """

    def __init__(self, config: OmniMatteConfig):
        self.config = config
        self.flow_estimator = FlowEstimator(config.device)

    def decompose_frame(
        self,
        frame: np.ndarray,
        masks: List[np.ndarray],
        prev_layers: Optional[List[np.ndarray]] = None,
    ) -> List[OmniMatteLayer]:
        """
        Decompose a single frame into layers.

        Args:
            frame: Input RGB frame
            masks: Initial masks for each layer
            prev_layers: Previous frame layers for consistency

        Returns:
            List of OmniMatteLayer
        """
        import cv2

        h, w = frame.shape[:2]
        num_layers = len(masks)

        # Initialize layers
        layers = []

        for i, mask in enumerate(masks):
            # Expand mask to include associated effects
            expanded_mask = self._expand_mask_for_effects(frame, mask)

            # Extract layer RGB
            alpha = expanded_mask.astype(np.float32)
            if alpha.max() > 1:
                alpha = alpha / 255.0

            # Simple matting for RGB
            rgba = np.zeros((h, w, 4), dtype=np.float32)
            rgba[..., :3] = frame * alpha[..., np.newaxis]
            rgba[..., 3] = alpha

            # Determine layer type
            if i == 0:
                layer_type = LayerType.FOREGROUND
            elif i == num_layers - 1 and self.config.include_background:
                layer_type = LayerType.BACKGROUND
            else:
                layer_type = LayerType.EFFECT

            # Confidence based on mask certainty
            confidence = np.abs(alpha - 0.5) * 2

            layers.append(OmniMatteLayer(
                rgba=rgba,
                alpha=alpha,
                layer_type=layer_type,
                confidence=confidence,
            ))

        # Refine layers to sum to original
        layers = self._refine_layers(frame, layers)

        return layers

    def _expand_mask_for_effects(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Expand mask to include associated effects like shadows."""
        import cv2

        if mask.max() > 1:
            mask = mask / 255.0

        # Find shadow regions
        # Shadows are darker areas near the object

        # Convert to grayscale
        if frame.ndim == 3:
            if frame.dtype == np.float32:
                gray = (np.dot(frame, [0.299, 0.587, 0.114]) * 255).astype(np.uint8)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Find dark regions
        dark_threshold = np.percentile(gray, 30)
        dark_mask = (gray < dark_threshold).astype(np.float32)

        # Dilate object mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        dilated = cv2.dilate(mask.astype(np.float32), kernel)

        # Shadow = dark areas near object but not in object
        potential_shadow = dark_mask * dilated * (1 - mask)

        # Smooth
        potential_shadow = cv2.GaussianBlur(potential_shadow.astype(np.float32), (11, 11), 0)

        # Combine
        expanded = np.maximum(mask, potential_shadow * 0.5)

        return expanded

    def _refine_layers(
        self,
        frame: np.ndarray,
        layers: List[OmniMatteLayer],
    ) -> List[OmniMatteLayer]:
        """Refine layers so they sum to original."""
        # Compute current sum
        reconstruction = np.zeros_like(frame, dtype=np.float32)

        for layer in layers:
            reconstruction += layer.rgba[..., :3]

        # Compute residual
        residual = frame.astype(np.float32) - reconstruction

        # Distribute residual based on alpha
        total_alpha = sum(layer.alpha for layer in layers) + 1e-6

        for layer in layers:
            weight = layer.alpha / total_alpha
            layer.rgba[..., :3] += residual * weight[..., np.newaxis]
            layer.rgba = np.clip(layer.rgba, 0, 1)

        return layers


class OmniMatte:
    """
    OmniMatte - Video Layer Decomposition.

    Decomposes video into RGBA layers that include both
    objects and their associated effects (shadows, reflections).

    This enables:
    - Clean removal of objects WITH their effects
    - Realistic re-compositing with preserved shadows
    - Layer-based video editing

    Example:
        >>> omnimatte = OmniMatte(OmniMatteConfig(num_layers=3))
        >>>
        >>> # Process video with initial masks
        >>> result = omnimatte.process_video(
        ...     frames,
        ...     initial_masks=[person_mask, shadow_mask]
        ... )
        >>>
        >>> # Get person layer with their shadow
        >>> person_layer = result.layers[0]
    """

    def __init__(self, config: Optional[OmniMatteConfig] = None):
        self.config = config or OmniMatteConfig()
        self.decomposer = LayerDecomposer(self.config)
        self.flow_estimator = FlowEstimator(self.config.device)

        self._prev_layers: Optional[List[OmniMatteLayer]] = None
        self._prev_frame: Optional[np.ndarray] = None

    def process_frame(
        self,
        frame: np.ndarray,
        masks: List[np.ndarray],
    ) -> OmniMatteResult:
        """
        Process a single frame.

        Args:
            frame: Input RGB frame
            masks: List of initial masks for each layer

        Returns:
            OmniMatteResult with decomposed layers
        """
        # Normalize
        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / 255.0

        # Compute flow from previous frame
        flow = None
        if self._prev_frame is not None and self.config.use_flow:
            flow = self.flow_estimator.compute(self._prev_frame, frame)

        # Decompose
        layers = self.decomposer.decompose_frame(
            frame, masks, self._prev_layers
        )

        # Apply temporal consistency if we have previous layers
        if self._prev_layers is not None and flow is not None:
            layers = self._apply_temporal_consistency(layers, flow)

        # Update state
        self._prev_layers = layers
        self._prev_frame = frame.copy()

        # Reconstruct background
        background = self._reconstruct_background(frame, layers)

        # Full reconstruction
        reconstruction = self._composite_layers(layers, background)

        return OmniMatteResult(
            layers=layers,
            background=background,
            reconstruction=reconstruction,
            flow=flow,
            metadata={
                "num_layers": len(layers),
            }
        )

    def process_video(
        self,
        frames: List[np.ndarray],
        initial_masks: List[np.ndarray],
    ) -> List[OmniMatteResult]:
        """
        Process entire video.

        Args:
            frames: List of video frames
            initial_masks: Initial masks for first frame

        Returns:
            List of OmniMatteResult for each frame
        """
        results = []

        # Process first frame
        masks = initial_masks

        for i, frame in enumerate(frames):
            result = self.process_frame(frame, masks)
            results.append(result)

            # Propagate masks using flow
            if result.flow is not None and i < len(frames) - 1:
                masks = self._propagate_masks(
                    [layer.alpha for layer in result.layers],
                    result.flow
                )
            else:
                masks = [layer.alpha for layer in result.layers]

        return results

    def _apply_temporal_consistency(
        self,
        layers: List[OmniMatteLayer],
        flow: np.ndarray,
    ) -> List[OmniMatteLayer]:
        """Apply temporal consistency using optical flow."""
        import cv2

        for i, layer in enumerate(layers):
            if i < len(self._prev_layers):
                prev_layer = self._prev_layers[i]

                # Warp previous alpha
                h, w = layer.alpha.shape
                map_x = np.arange(w) + flow[..., 0]
                map_y = np.arange(h).reshape(-1, 1) + flow[..., 1]

                map_x = map_x.astype(np.float32)
                map_y = map_y.astype(np.float32)

                warped_alpha = cv2.remap(
                    prev_layer.alpha.astype(np.float32),
                    map_x, map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )

                # Blend with current
                layer.alpha = 0.7 * layer.alpha + 0.3 * warped_alpha

        return layers

    def _propagate_masks(
        self,
        masks: List[np.ndarray],
        flow: np.ndarray,
    ) -> List[np.ndarray]:
        """Propagate masks using optical flow."""
        import cv2

        propagated = []

        h, w = flow.shape[:2]
        map_x = np.arange(w) + flow[..., 0]
        map_y = np.arange(h).reshape(-1, 1) + flow[..., 1]

        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        for mask in masks:
            warped = cv2.remap(
                mask.astype(np.float32),
                map_x, map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            propagated.append(warped)

        return propagated

    def _reconstruct_background(
        self,
        frame: np.ndarray,
        layers: List[OmniMatteLayer],
    ) -> np.ndarray:
        """Reconstruct background from layers."""
        # Find background layer
        for layer in layers:
            if layer.layer_type == LayerType.BACKGROUND:
                return layer.rgba[..., :3]

        # If no explicit background, inpaint
        import cv2

        # Create mask of foreground regions
        fg_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for layer in layers:
            if layer.layer_type != LayerType.BACKGROUND:
                fg_mask[layer.alpha > 0.5] = 255

        # Inpaint background
        frame_uint8 = (frame * 255).astype(np.uint8) if frame.dtype == np.float32 else frame
        background = cv2.inpaint(frame_uint8, fg_mask, 10, cv2.INPAINT_TELEA)

        return background.astype(np.float32) / 255.0

    def _composite_layers(
        self,
        layers: List[OmniMatteLayer],
        background: np.ndarray,
    ) -> np.ndarray:
        """Composite all layers into final image."""
        result = background.copy()

        # Sort layers by type (background first)
        sorted_layers = sorted(
            layers,
            key=lambda l: (
                0 if l.layer_type == LayerType.BACKGROUND else
                1 if l.layer_type == LayerType.SHADOW else
                2
            )
        )

        for layer in sorted_layers:
            if layer.layer_type == LayerType.BACKGROUND:
                continue

            alpha = layer.alpha[..., np.newaxis]
            rgb = layer.rgba[..., :3]

            result = rgb * alpha + result * (1 - alpha)

        return result

    def reset(self):
        """Reset state for new video."""
        self._prev_layers = None
        self._prev_frame = None


def decompose_video_layers(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    num_layers: int = 3,
) -> List[OmniMatteResult]:
    """
    Quick video layer decomposition.

    Args:
        frames: Video frames
        masks: Initial masks for first frame
        num_layers: Number of layers to extract

    Returns:
        List of results with decomposed layers
    """
    config = OmniMatteConfig(num_layers=num_layers)
    omnimatte = OmniMatte(config)
    return omnimatte.process_video(frames, masks)
