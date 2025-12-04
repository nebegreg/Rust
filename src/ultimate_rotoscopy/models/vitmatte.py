"""
ViTMatte - Vision Transformer for Image Matting
================================================

State-of-the-art transformer-based matting that achieves superior
results on fine details like hair and fur.

Key innovations:
- Vision Transformer backbone for global context
- Detail capture module for fine structures
- Hybrid attention for both global and local features

Reference: "ViTMatte: Boosting Image Matting with Pre-trained Plain Vision Transformers"
https://github.com/hustvl/ViTMatte
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ViTMatteBackbone(Enum):
    """ViTMatte backbone variants."""
    VIT_S = "vit_s"      # Small - faster
    VIT_B = "vit_b"      # Base - balanced
    VIT_L = "vit_l"      # Large - best quality
    VIT_H = "vit_h"      # Huge - maximum quality


class TrimapSource(Enum):
    """Source for trimap generation."""
    MANUAL = "manual"           # User-provided trimap
    SAM3_AUTO = "sam3_auto"     # Auto-generate from SAM3 mask
    EDGE_BASED = "edge_based"   # Generate from edge detection
    DILATION = "dilation"       # Simple dilation-based


@dataclass
class ViTMatteConfig:
    """ViTMatte configuration."""
    backbone: ViTMatteBackbone = ViTMatteBackbone.VIT_B
    checkpoint_path: Optional[str] = None

    # Trimap settings
    trimap_source: TrimapSource = TrimapSource.SAM3_AUTO
    unknown_width: int = 25          # Width of unknown region
    erode_foreground: int = 5        # Erode FG for safety margin
    dilate_background: int = 5       # Dilate BG boundary

    # Detail capture
    enable_detail_capture: bool = True
    detail_kernel_size: int = 3

    # Processing
    patch_size: int = 16
    image_size: int = 1024           # Resize for processing
    use_fp16: bool = True

    # Post-processing
    refine_edges: bool = True
    guided_filter_radius: int = 16
    guided_filter_eps: float = 0.01

    device: str = "cuda"


@dataclass
class ViTMatteResult:
    """Result from ViTMatte."""
    alpha: np.ndarray               # Final alpha matte
    trimap: np.ndarray              # Used trimap
    detail_map: np.ndarray          # Fine detail map
    confidence: np.ndarray          # Per-pixel confidence
    foreground: Optional[np.ndarray] = None  # Estimated foreground
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrimapGenerator:
    """
    Automatic trimap generation from segmentation masks.

    Trimaps have three regions:
    - Foreground (255): Definitely foreground
    - Background (0): Definitely background
    - Unknown (128): Uncertain region for matting

    The unknown region is critical for good matting results.
    """

    def __init__(self, config: ViTMatteConfig):
        self.config = config

    def generate(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate trimap from binary mask.

        Args:
            mask: Binary segmentation mask
            image: Optional image for edge-guided generation

        Returns:
            Trimap with values 0 (BG), 128 (Unknown), 255 (FG)
        """
        import cv2

        # Normalize mask to 0-1
        if mask.max() > 1:
            mask = mask / 255.0
        mask = mask.astype(np.float32)

        if mask.ndim == 3:
            mask = mask[..., 0]

        # Convert to binary
        binary = (mask > 0.5).astype(np.uint8) * 255

        if self.config.trimap_source == TrimapSource.EDGE_BASED and image is not None:
            return self._edge_based_trimap(binary, image)
        else:
            return self._dilation_based_trimap(binary)

    def _dilation_based_trimap(self, mask: np.ndarray) -> np.ndarray:
        """Generate trimap using morphological operations."""
        import cv2

        # Create structuring elements
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.erode_foreground * 2 + 1, self.config.erode_foreground * 2 + 1)
        )
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.unknown_width * 2 + 1, self.config.unknown_width * 2 + 1)
        )

        # Erode for definite foreground
        fg = cv2.erode(mask, erode_kernel)

        # Dilate for potential foreground extent
        dilated = cv2.dilate(mask, dilate_kernel)

        # Create trimap
        trimap = np.zeros_like(mask)
        trimap[dilated > 127] = 128      # Unknown
        trimap[fg > 127] = 255           # Foreground

        return trimap

    def _edge_based_trimap(
        self,
        mask: np.ndarray,
        image: np.ndarray,
    ) -> np.ndarray:
        """Generate trimap with edge-aware unknown region."""
        import cv2

        # Get base trimap
        trimap = self._dilation_based_trimap(mask)

        # Convert image to grayscale
        if image.ndim == 3:
            if image.dtype == np.float32 or image.dtype == np.float64:
                gray = (np.dot(image, [0.299, 0.587, 0.114]) * 255).astype(np.uint8)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, edge_kernel)

        # Expand unknown region around edges
        unknown_mask = trimap == 128
        edge_unknown = edges_dilated > 0

        # Add edge regions to unknown if near boundary
        boundary = cv2.dilate(unknown_mask.astype(np.uint8) * 255, edge_kernel)
        additional_unknown = (edge_unknown & (boundary > 0))

        trimap[additional_unknown & (trimap == 0)] = 128
        trimap[additional_unknown & (trimap == 255)] = 128

        return trimap

    def refine_trimap(
        self,
        trimap: np.ndarray,
        image: np.ndarray,
    ) -> np.ndarray:
        """Refine trimap using image features."""
        import cv2

        # Use GrabCut-style refinement
        if image.dtype == np.float32 or image.dtype == np.float64:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image

        # Convert trimap to GrabCut mask
        gc_mask = np.zeros(trimap.shape, dtype=np.uint8)
        gc_mask[trimap == 0] = cv2.GC_BGD
        gc_mask[trimap == 255] = cv2.GC_FGD
        gc_mask[trimap == 128] = cv2.GC_PR_FGD

        # Run GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(img_uint8, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return trimap

        # Convert back to trimap
        refined = np.zeros_like(trimap)
        refined[(gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD)] = 0
        refined[(gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)] = 255

        # Keep original unknown regions uncertain
        refined[trimap == 128] = 128

        return refined


class DetailCaptureModule:
    """
    Detail Capture Module for fine structure preservation.

    Captures high-frequency details that the main ViT might miss,
    especially important for hair and fur.
    """

    def __init__(self, config: ViTMatteConfig):
        self.config = config

    def extract_details(
        self,
        image: np.ndarray,
        coarse_alpha: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract fine details from image guided by coarse alpha.

        Args:
            image: Input RGB image
            coarse_alpha: Coarse alpha from main network

        Returns:
            Tuple of (refined_alpha, detail_map)
        """
        import cv2

        # Normalize
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Find edge regions where details matter
        edge_region = self._get_edge_region(coarse_alpha)

        # High-pass filter for detail extraction
        detail_map = self._extract_high_frequency(image, coarse_alpha)

        # Refine alpha in edge regions using details
        refined = coarse_alpha.copy()

        # Apply detail enhancement only in edge regions
        detail_weight = edge_region * detail_map
        refined = refined + detail_weight * 0.1  # Subtle enhancement

        refined = np.clip(refined, 0, 1)

        return refined, detail_map

    def _get_edge_region(self, alpha: np.ndarray) -> np.ndarray:
        """Get mask of edge regions."""
        import cv2

        alpha_8bit = (alpha * 255).astype(np.uint8)

        # Gradient magnitude
        grad_x = cv2.Sobel(alpha_8bit, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(alpha_8bit, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize
        if gradient.max() > 0:
            gradient = gradient / gradient.max()

        # Threshold and dilate
        edge_mask = (gradient > 0.1).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edge_mask = cv2.dilate(edge_mask, kernel)

        return edge_mask.astype(np.float32)

    def _extract_high_frequency(
        self,
        image: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """Extract high-frequency detail map."""
        import cv2

        # Convert to grayscale
        if image.ndim == 3:
            gray = np.dot(image, [0.299, 0.587, 0.114])
        else:
            gray = image

        # Laplacian for high-frequency
        laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)

        # Normalize
        detail_map = np.abs(laplacian)
        if detail_map.max() > 0:
            detail_map = detail_map / detail_map.max()

        return detail_map


class ViTMatte:
    """
    ViTMatte - Vision Transformer for Image Matting.

    State-of-the-art matting using pre-trained Vision Transformers
    with a detail capture module for fine structures.

    The transformer backbone provides:
    - Global context understanding
    - Robust feature extraction
    - Better generalization

    Example:
        >>> vitmatte = ViTMatte(ViTMatteConfig(
        ...     backbone=ViTMatteBackbone.VIT_B,
        ...     trimap_source=TrimapSource.SAM3_AUTO,
        ... ))
        >>>
        >>> # With SAM3 mask
        >>> result = vitmatte.matte(image, sam3_mask=sam_mask)
        >>>
        >>> # With manual trimap
        >>> result = vitmatte.matte(image, trimap=manual_trimap)
    """

    def __init__(self, config: Optional[ViTMatteConfig] = None):
        self.config = config or ViTMatteConfig()
        self.trimap_generator = TrimapGenerator(self.config)
        self.detail_module = DetailCaptureModule(self.config)

        self._model = None
        self._loaded = False

    def _load_model(self):
        """Load ViTMatte model."""
        if self._loaded:
            return

        try:
            # Try to load official ViTMatte
            from vitmatte import ViTMatte as ViTMatteModel

            self._model = ViTMatteModel.from_pretrained(
                self.config.backbone.value,
                checkpoint_path=self.config.checkpoint_path,
            )
            self._model.to(self.config.device)
            self._model.eval()
            self._loaded = True

        except ImportError:
            # Use fallback implementation
            self._loaded = True
            self._model = None

    def matte(
        self,
        image: np.ndarray,
        trimap: Optional[np.ndarray] = None,
        sam3_mask: Optional[np.ndarray] = None,
        return_foreground: bool = False,
    ) -> ViTMatteResult:
        """
        Generate alpha matte from image.

        Args:
            image: Input RGB image
            trimap: Pre-computed trimap (optional)
            sam3_mask: SAM3 segmentation mask for auto trimap
            return_foreground: Also estimate foreground colors

        Returns:
            ViTMatteResult with alpha and metadata
        """
        self._load_model()

        # Normalize image
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Generate or use trimap
        if trimap is not None:
            used_trimap = trimap
        elif sam3_mask is not None:
            used_trimap = self.trimap_generator.generate(sam3_mask, image)
        else:
            raise ValueError("Either trimap or sam3_mask must be provided")

        # Run matting
        if self._model is not None:
            alpha = self._run_vitmatte(image, used_trimap)
        else:
            alpha = self._fallback_matte(image, used_trimap)

        # Detail capture
        if self.config.enable_detail_capture:
            alpha, detail_map = self.detail_module.extract_details(image, alpha)
        else:
            detail_map = np.zeros_like(alpha)

        # Post-processing
        if self.config.refine_edges:
            alpha = self._guided_filter_refinement(image, alpha)

        # Compute confidence
        confidence = self._compute_confidence(alpha, used_trimap)

        # Estimate foreground if requested
        foreground = None
        if return_foreground:
            foreground = self._estimate_foreground(image, alpha)

        return ViTMatteResult(
            alpha=alpha,
            trimap=used_trimap,
            detail_map=detail_map,
            confidence=confidence,
            foreground=foreground,
            metadata={
                "backbone": self.config.backbone.value,
                "trimap_source": self.config.trimap_source.value,
            }
        )

    def _run_vitmatte(
        self,
        image: np.ndarray,
        trimap: np.ndarray,
    ) -> np.ndarray:
        """Run ViTMatte model."""
        import torch

        # Prepare inputs
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        trimap_tensor = torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0)

        img_tensor = img_tensor.to(self.config.device)
        trimap_tensor = trimap_tensor.to(self.config.device)

        if self.config.use_fp16:
            img_tensor = img_tensor.half()
            trimap_tensor = trimap_tensor.half()

        # Normalize trimap to 0-1
        trimap_tensor = trimap_tensor / 255.0

        # Run model
        with torch.no_grad():
            alpha = self._model(img_tensor, trimap_tensor)

        alpha = alpha.squeeze().cpu().numpy()
        return np.clip(alpha, 0, 1)

    def _fallback_matte(
        self,
        image: np.ndarray,
        trimap: np.ndarray,
    ) -> np.ndarray:
        """Fallback matting without ViTMatte model."""
        import cv2

        # Use closed-form matting approach
        # This is a simplified version

        # Get known regions from trimap
        fg_mask = trimap == 255
        bg_mask = trimap == 0
        unknown_mask = trimap == 128

        # Initialize alpha
        alpha = np.zeros(trimap.shape, dtype=np.float32)
        alpha[fg_mask] = 1.0
        alpha[bg_mask] = 0.0

        # For unknown region, use color-based estimation
        if image.ndim == 3:
            gray = np.dot(image, [0.299, 0.587, 0.114])
        else:
            gray = image.copy()

        # Get FG and BG colors
        fg_color = np.mean(gray[fg_mask]) if np.any(fg_mask) else 1.0
        bg_color = np.mean(gray[bg_mask]) if np.any(bg_mask) else 0.0

        # Estimate alpha in unknown region
        if fg_color != bg_color:
            alpha_unknown = (gray - bg_color) / (fg_color - bg_color + 1e-6)
            alpha[unknown_mask] = np.clip(alpha_unknown[unknown_mask], 0, 1)
        else:
            alpha[unknown_mask] = 0.5

        # Refine with guided filter
        alpha = self._guided_filter(gray, alpha)

        return alpha

    def _guided_filter(
        self,
        guide: np.ndarray,
        source: np.ndarray,
        radius: int = 16,
        eps: float = 0.01,
    ) -> np.ndarray:
        """Apply guided filter."""
        import cv2

        def box_filter(img, r):
            return cv2.blur(img, (2*r+1, 2*r+1))

        guide = guide.astype(np.float32)
        source = source.astype(np.float32)

        mean_I = box_filter(guide, radius)
        mean_p = box_filter(source, radius)
        mean_Ip = box_filter(guide * source, radius)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = box_filter(guide * guide, radius)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = box_filter(a, radius)
        mean_b = box_filter(b, radius)

        return mean_a * guide + mean_b

    def _guided_filter_refinement(
        self,
        image: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """Refine alpha using guided filter."""
        if image.ndim == 3:
            gray = np.dot(image, [0.299, 0.587, 0.114])
        else:
            gray = image

        return self._guided_filter(
            gray.astype(np.float32),
            alpha.astype(np.float32),
            self.config.guided_filter_radius,
            self.config.guided_filter_eps,
        )

    def _compute_confidence(
        self,
        alpha: np.ndarray,
        trimap: np.ndarray,
    ) -> np.ndarray:
        """Compute per-pixel confidence."""
        # High confidence for definite regions
        confidence = np.ones_like(alpha)

        # Lower confidence in unknown regions
        unknown = trimap == 128
        confidence[unknown] = 0.5 + 0.5 * (2 * np.abs(alpha[unknown] - 0.5))

        return confidence

    def _estimate_foreground(
        self,
        image: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """Estimate foreground colors using alpha."""
        # Simple foreground estimation
        # F = (I - (1-alpha)*B) / alpha

        # Estimate background from low-alpha regions
        bg_mask = alpha < 0.1
        if np.any(bg_mask):
            bg_color = np.mean(image[bg_mask], axis=0)
        else:
            bg_color = np.zeros(3)

        # Estimate foreground
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        alpha_safe = np.maximum(alpha_3ch, 0.01)

        foreground = (image - (1 - alpha_3ch) * bg_color) / alpha_safe
        foreground = np.clip(foreground, 0, 1)

        return foreground


def auto_matte(
    image: np.ndarray,
    mask: np.ndarray,
    refine: bool = True,
) -> np.ndarray:
    """
    Quick function for automatic matting.

    Args:
        image: Input RGB image
        mask: Binary segmentation mask
        refine: Apply edge refinement

    Returns:
        Alpha matte
    """
    config = ViTMatteConfig(
        trimap_source=TrimapSource.SAM3_AUTO,
        refine_edges=refine,
    )

    vitmatte = ViTMatte(config)
    result = vitmatte.matte(image, sam3_mask=mask)

    return result.alpha
