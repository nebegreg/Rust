#!/usr/bin/env python3
"""
Ultimate Rotoscopy Application
==============================

Professional rotoscopy application integrating:
- SAM3: Initial segmentation with text/point/box prompts
- ViTMatte: High-quality alpha matting with trimap
- MatAnyone: Video matting with temporal consistency
- DepthAnything3: Depth estimation for VFX compositing

Features:
- Automatic trimap generation from SAM3 masks
- Hair and fine detail preservation
- Edge refinement and blur handling
- Multi-layer export (alpha, depth, normals)
- Professional VFX workflow support (Flame, Nuke)

Requirements:
    Python 3.10+
    PyTorch 2.0+
    CUDA 11.8+

Usage:
    # Image rotoscopy with text prompt
    python ultimate_roto.py image input.jpg --text "person" --output result/

    # Video rotoscopy
    python ultimate_roto.py video input.mp4 --text "person" --output result/

    # With depth estimation
    python ultimate_roto.py image input.jpg --text "person" --depth --output result/

Author: Ultimate Roto Pipeline
"""

import os
import sys
import argparse
import tempfile
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
from enum import Enum
import json

import numpy as np
import cv2
from PIL import Image

warnings.filterwarnings("ignore")

# Check PyTorch availability
try:
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import functional as TF
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch torchvision")


# =============================================================================
# CONFIGURATION
# =============================================================================

class MattingBackend(Enum):
    """Available matting backends."""
    VITMATTE = "vitmatte"
    MATANYONE = "matanyone"
    AUTO = "auto"  # Auto-select based on input type


class DepthBackend(Enum):
    """Available depth estimation backends."""
    DEPTH_ANYTHING_3 = "da3"
    NONE = "none"


@dataclass
class RotoConfig:
    """Configuration for the rotoscopy pipeline."""
    # Device settings
    device: str = "cuda"

    # SAM3 settings
    sam3_model: str = "sam3"

    # Matting settings
    matting_backend: MattingBackend = MattingBackend.AUTO
    vitmatte_model: str = "vitmatte-b"  # vitmatte-s or vitmatte-b

    # Trimap settings
    trimap_erosion: int = 15  # Erosion kernel for foreground
    trimap_dilation: int = 30  # Dilation kernel for unknown region
    trimap_blur: int = 5  # Blur for smooth trimap

    # Hair/Detail refinement
    hair_refinement: bool = True
    hair_threshold: float = 0.3  # Threshold for hair detection
    edge_refinement: bool = True
    edge_blur_radius: int = 3

    # MatAnyone video settings
    video_warmup_frames: int = 10
    video_erode_kernel: int = 10
    video_dilate_kernel: int = 10

    # Depth settings
    depth_enabled: bool = False
    depth_model: str = "da3-large"  # da3-small, da3-large, da3-giant

    # Output settings
    output_format: str = "png"  # png, exr
    output_colorspace: str = "srgb"  # srgb, acescg
    save_trimap: bool = True
    save_layers: bool = True  # Save decomposed layers (core, edge, hair)


@dataclass
class RotoResult:
    """Result from rotoscopy processing."""
    # Core outputs
    alpha: np.ndarray  # Final alpha matte (H, W) float32 [0, 1]
    foreground: np.ndarray  # Extracted foreground (H, W, 3) uint8

    # Optional outputs
    trimap: Optional[np.ndarray] = None  # Trimap used (H, W) uint8
    depth: Optional[np.ndarray] = None  # Depth map (H, W) float32

    # Layer decomposition
    core_mask: Optional[np.ndarray] = None  # Solid foreground
    edge_mask: Optional[np.ndarray] = None  # Edge/transition region
    hair_mask: Optional[np.ndarray] = None  # Hair/fine detail region

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_composite(self, background: np.ndarray) -> np.ndarray:
        """Composite foreground over a background."""
        alpha_3ch = np.stack([self.alpha] * 3, axis=-1)
        fg = self.foreground.astype(np.float32) / 255.0
        bg = background.astype(np.float32) / 255.0
        composite = fg * alpha_3ch + bg * (1 - alpha_3ch)
        return (composite * 255).astype(np.uint8)


# =============================================================================
# TRIMAP GENERATION
# =============================================================================

class TrimapGenerator:
    """
    Generate trimap from binary mask.

    Trimap values:
    - 0: Background (definite)
    - 128: Unknown (transition region)
    - 255: Foreground (definite)
    """

    def __init__(self, config: RotoConfig):
        self.config = config

    def generate(
        self,
        mask: np.ndarray,
        erosion: Optional[int] = None,
        dilation: Optional[int] = None,
        blur: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate trimap from binary mask.

        Args:
            mask: Binary mask (H, W) with values 0 or 255
            erosion: Erosion kernel size (default from config)
            dilation: Dilation kernel size (default from config)
            blur: Blur kernel size (default from config)

        Returns:
            Trimap (H, W) with values 0, 128, 255
        """
        erosion = erosion or self.config.trimap_erosion
        dilation = dilation or self.config.trimap_dilation
        blur = blur or self.config.trimap_blur

        # Ensure binary mask
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # Create kernels
        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erosion, erosion)
        )
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation, dilation)
        )

        # Erode to get definite foreground
        foreground = cv2.erode(mask, erosion_kernel, iterations=1)

        # Dilate to get potential foreground region
        dilated = cv2.dilate(mask, dilation_kernel, iterations=1)

        # Unknown region = dilated - eroded
        unknown = dilated - foreground

        # Create trimap
        trimap = np.zeros_like(mask, dtype=np.uint8)
        trimap[foreground > 127] = 255  # Definite foreground
        trimap[unknown > 127] = 128  # Unknown region
        # Background remains 0

        # Optional blur for smoother transitions
        if blur > 0:
            # Only blur the unknown region boundaries
            trimap_float = trimap.astype(np.float32)
            trimap_blurred = cv2.GaussianBlur(trimap_float, (blur * 2 + 1, blur * 2 + 1), 0)

            # Requantize while preserving structure
            trimap_out = np.zeros_like(trimap)
            trimap_out[trimap_blurred > 191] = 255
            trimap_out[(trimap_blurred > 64) & (trimap_blurred <= 191)] = 128
            trimap = trimap_out

        return trimap

    def generate_adaptive(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate adaptive trimap based on image content.

        Uses edge detection to create wider unknown regions
        around high-frequency areas (hair, fur, etc.)
        """
        # Base trimap
        trimap = self.generate(mask)

        if image is not None and self.config.hair_refinement:
            # Detect edges in the image
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150)

            # Dilate edges
            edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges_dilated = cv2.dilate(edges, edge_kernel, iterations=2)

            # Expand unknown region around edges that intersect with mask boundary
            mask_boundary = cv2.dilate(mask, edge_kernel) - cv2.erode(mask, edge_kernel)
            hair_region = edges_dilated & (mask_boundary > 0)

            # Add to unknown region
            trimap[hair_region > 0] = 128

        return trimap


# =============================================================================
# SAM3 INTEGRATION
# =============================================================================

class SAM3Segmenter:
    """SAM3 segmentation wrapper."""

    def __init__(self, config: RotoConfig):
        self.config = config
        self.device = config.device
        self.model = None
        self.processor = None
        self._initialized = False

    def _init_model(self):
        """Lazy initialization of SAM3 model."""
        if self._initialized:
            return

        print("Loading SAM3 model...")
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)

            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to(device=self.device)
                self.model.eval()
                print(f"  SAM3 loaded on {self.device}")
            else:
                self.device = "cpu"
                self.model.to(device=self.device)
                self.model.eval()
                print(f"  SAM3 loaded on CPU")

            self._initialized = True

        except ImportError as e:
            print(f"SAM3 not available: {e}")
            print("Install: pip install git+https://github.com/facebookresearch/sam3.git")
            raise

    def segment_text(self, image: Union[str, Path, np.ndarray], text: str) -> np.ndarray:
        """
        Segment image using text prompt.

        Args:
            image: Image path or numpy array
            text: Text description

        Returns:
            Binary mask (H, W) with values 0 or 255
        """
        self._init_model()

        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(str(image)).convert("RGB")
        else:
            pil_image = Image.fromarray(image)

        print(f"  Segmenting with text: '{text}'")

        # Set image and get segmentation
        inference_state = self.processor.set_image(pil_image)
        output = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text
        )

        # Get best mask
        masks = output["masks"]
        scores = output["scores"]

        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        best_idx = scores.argmax()
        mask = masks[best_idx]

        # Convert to uint8
        mask = (mask * 255).astype(np.uint8)

        print(f"  Found object with score: {scores[best_idx]:.3f}")

        return mask

    def segment_points(
        self,
        image: Union[str, Path, np.ndarray],
        points: List[Tuple[int, int]],
        labels: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Segment image using point prompts.

        Args:
            image: Image path or numpy array
            points: List of (x, y) points
            labels: Point labels (1 = foreground, 0 = background)

        Returns:
            Binary mask
        """
        self._init_model()

        if isinstance(image, (str, Path)):
            pil_image = Image.open(str(image)).convert("RGB")
        else:
            pil_image = Image.fromarray(image)

        if labels is None:
            labels = [1] * len(points)

        # Set image
        inference_state = self.processor.set_image(pil_image)

        # Convert points to format expected by SAM3
        points_np = np.array(points)
        labels_np = np.array(labels)

        output = self.processor.set_point_prompt(
            state=inference_state,
            points=points_np,
            labels=labels_np
        )

        masks = output["masks"]
        scores = output["scores"]

        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        best_idx = scores.argmax()
        mask = (masks[best_idx] * 255).astype(np.uint8)

        return mask

    def segment_box(
        self,
        image: Union[str, Path, np.ndarray],
        box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Segment image using bounding box.

        Args:
            image: Image path or numpy array
            box: Bounding box (x1, y1, x2, y2)

        Returns:
            Binary mask
        """
        self._init_model()

        if isinstance(image, (str, Path)):
            pil_image = Image.open(str(image)).convert("RGB")
        else:
            pil_image = Image.fromarray(image)

        inference_state = self.processor.set_image(pil_image)

        output = self.processor.set_box_prompt(
            state=inference_state,
            box=np.array(box)
        )

        masks = output["masks"]
        scores = output["scores"]

        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        best_idx = scores.argmax()
        mask = (masks[best_idx] * 255).astype(np.uint8)

        return mask


# =============================================================================
# VITMATTE INTEGRATION
# =============================================================================

class ViTMatteProcessor:
    """ViTMatte alpha matting processor."""

    def __init__(self, config: RotoConfig):
        self.config = config
        self.device = config.device
        self.model = None
        self._initialized = False

    def _init_model(self):
        """Lazy initialization of ViTMatte model."""
        if self._initialized:
            return

        print("Loading ViTMatte model...")
        try:
            from detectron2.config import LazyConfig, instantiate
            from detectron2.checkpoint import DetectionCheckpointer

            # Determine config based on model type
            vitmatte_dir = Path(__file__).parent / "ViTMatte-main"
            config_path = vitmatte_dir / "configs" / "common" / "model.py"

            if not config_path.exists():
                raise FileNotFoundError(f"ViTMatte config not found: {config_path}")

            cfg = LazyConfig.load(str(config_path))

            # Adjust for model variant
            if self.config.vitmatte_model == "vitmatte-b":
                cfg.model.backbone.embed_dim = 768
                cfg.model.backbone.num_heads = 12
                cfg.model.decoder.in_chans = 768

            self.model = instantiate(cfg.model)

            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to(self.device)
            else:
                self.device = "cpu"
                self.model.to(self.device)

            self.model.eval()

            # Try to load checkpoint if available
            ckpt_path = vitmatte_dir / "pretrained" / f"{self.config.vitmatte_model}.pth"
            if ckpt_path.exists():
                DetectionCheckpointer(self.model).load(str(ckpt_path))
                print(f"  Loaded checkpoint: {ckpt_path}")

            print(f"  ViTMatte loaded on {self.device}")
            self._initialized = True

        except ImportError as e:
            print(f"ViTMatte dependencies not available: {e}")
            print("Install: pip install detectron2")
            raise

    def process(
        self,
        image: np.ndarray,
        trimap: np.ndarray
    ) -> np.ndarray:
        """
        Process image with trimap to get alpha matte.

        Args:
            image: RGB image (H, W, 3) uint8
            trimap: Trimap (H, W) with values 0, 128, 255

        Returns:
            Alpha matte (H, W) float32 [0, 1]
        """
        self._init_model()

        # Convert to tensors
        image_tensor = TF.to_tensor(Image.fromarray(image)).unsqueeze(0)
        trimap_tensor = TF.to_tensor(Image.fromarray(trimap)).unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self.device)
        trimap_tensor = trimap_tensor.to(self.device)

        # Prepare input
        input_dict = {
            'image': image_tensor,
            'trimap': trimap_tensor
        }

        # Run inference
        with torch.no_grad():
            output = self.model(input_dict)

        # Get alpha
        alpha = output['phas'].flatten(0, 2)
        alpha = alpha.cpu().numpy()

        return alpha.astype(np.float32)


# =============================================================================
# MATANYONE INTEGRATION
# =============================================================================

class MatAnyoneProcessor:
    """MatAnyone video matting processor."""

    def __init__(self, config: RotoConfig):
        self.config = config
        self.device = config.device
        self.processor = None
        self._initialized = False

    def _init_model(self):
        """Lazy initialization of MatAnyone model."""
        if self._initialized:
            return

        print("Loading MatAnyone model...")
        try:
            # Add matanyone to path
            matanyone_dir = Path(__file__).parent / "matanyone"
            sys.path.insert(0, str(matanyone_dir.parent))

            from matanyone.inference.inference_core import InferenceCore
            from matanyone.utils.get_default_model import get_matanyone_model
            from hugging_face.tools.download_util import load_file_from_url

            # Download checkpoint if needed
            pretrain_url = "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
            ckpt_path = load_file_from_url(pretrain_url, 'pretrained_models')

            # Load model
            model = get_matanyone_model(ckpt_path, self.device)
            self.processor = InferenceCore(model, cfg=model.cfg)

            print(f"  MatAnyone loaded on {self.device}")
            self._initialized = True

        except ImportError as e:
            print(f"MatAnyone not available: {e}")
            raise

    def process_video(
        self,
        video_path: Union[str, Path],
        mask_path: Union[str, Path],
        output_dir: Union[str, Path],
        warmup: Optional[int] = None,
        erode: Optional[int] = None,
        dilate: Optional[int] = None,
        save_frames: bool = True
    ) -> Tuple[str, str]:
        """
        Process video for matting.

        Args:
            video_path: Input video path
            mask_path: First frame mask path
            output_dir: Output directory
            warmup: Number of warmup frames
            erode: Erosion kernel size
            dilate: Dilation kernel size
            save_frames: Save individual frames

        Returns:
            Tuple of (foreground_video_path, alpha_video_path)
        """
        self._init_model()

        warmup = warmup or self.config.video_warmup_frames
        erode = erode or self.config.video_erode_kernel
        dilate = dilate or self.config.video_dilate_kernel

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        return self.processor.process_video(
            input_path=str(video_path),
            mask_path=str(mask_path),
            output_path=str(output_dir),
            n_warmup=warmup,
            r_erode=erode,
            r_dilate=dilate,
            save_image=save_frames
        )

    def process_frames(
        self,
        frames: List[np.ndarray],
        mask: np.ndarray,
        warmup: Optional[int] = None,
        erode: Optional[int] = None,
        dilate: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process frames for matting.

        Args:
            frames: List of RGB frames (H, W, 3) uint8
            mask: First frame mask (H, W) uint8

        Returns:
            Tuple of (foreground_frames, alpha_frames)
        """
        self._init_model()

        warmup = warmup or self.config.video_warmup_frames
        erode = erode or self.config.video_erode_kernel
        dilate = dilate or self.config.video_dilate_kernel

        # Use the matanyone wrapper function
        from hugging_face.matanyone_wrapper import matanyone

        return matanyone(
            self.processor,
            frames,
            mask,
            r_erode=erode,
            r_dilate=dilate,
            n_warmup=warmup
        )


# =============================================================================
# DEPTH ANYTHING 3 INTEGRATION
# =============================================================================

class DepthAnything3Processor:
    """Depth Anything 3 depth estimation processor."""

    def __init__(self, config: RotoConfig):
        self.config = config
        self.device = config.device
        self.model = None
        self._initialized = False

    def _init_model(self):
        """Lazy initialization of Depth Anything 3 model."""
        if self._initialized:
            return

        print("Loading Depth Anything 3 model...")
        try:
            # Add DA3 to path
            da3_dir = Path(__file__).parent / "Depth-Anything-3-main" / "src"
            sys.path.insert(0, str(da3_dir))

            from depth_anything_3.api import DepthAnything3

            self.model = DepthAnything3.from_pretrained(
                f"depth-anything/{self.config.depth_model}"
            )

            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to(self.device)
            else:
                self.device = "cpu"
                self.model.to(self.device)

            self.model.eval()

            print(f"  Depth Anything 3 loaded on {self.device}")
            self._initialized = True

        except Exception as e:
            print(f"Depth Anything 3 not available: {e}")
            # Try alternative loading method
            try:
                from depth_anything_3.api import DepthAnything3
                self.model = DepthAnything3(model_name=self.config.depth_model)
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model.to(self.device)
                self._initialized = True
                print(f"  Depth Anything 3 loaded (local) on {self.device}")
            except Exception as e2:
                print(f"Could not load Depth Anything 3: {e2}")
                raise

    def estimate_depth(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """
        Estimate depth from image.

        Args:
            image: Input image

        Returns:
            Depth map (H, W) float32, normalized to [0, 1]
        """
        self._init_model()

        # Prepare input
        if isinstance(image, (str, Path)):
            image_input = [str(image)]
        elif isinstance(image, np.ndarray):
            image_input = [Image.fromarray(image)]
        else:
            image_input = [image]

        # Run inference
        prediction = self.model.inference(image_input)

        # Get depth
        depth = prediction.depth[0]  # (H, W)

        # Normalize to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)

        return depth.astype(np.float32)


# =============================================================================
# LAYER DECOMPOSITION
# =============================================================================

class LayerDecomposer:
    """
    Decompose alpha matte into semantic layers.

    Layers:
    - Core: Solid foreground (alpha > 0.95)
    - Edge: Transition region (0.05 < alpha < 0.95)
    - Hair: Fine detail region (detected by high-frequency analysis)
    """

    def __init__(self, config: RotoConfig):
        self.config = config

    def decompose(
        self,
        alpha: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Decompose alpha into layers.

        Args:
            alpha: Alpha matte (H, W) float32 [0, 1]
            image: Original image for hair detection

        Returns:
            Dictionary with 'core', 'edge', 'hair' masks
        """
        # Core: solid foreground
        core = (alpha > 0.95).astype(np.float32)

        # Edge: transition region
        edge = ((alpha > 0.05) & (alpha <= 0.95)).astype(np.float32)

        # Hair: detected from image if available
        hair = np.zeros_like(alpha)

        if image is not None and self.config.hair_refinement:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Detect high-frequency regions (hair, fur)
            # Using Laplacian for edge detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.abs(laplacian)

            # Normalize
            lap_norm = laplacian / (laplacian.max() + 1e-8)

            # Hair region = high frequency within edge region
            hair_candidate = (lap_norm > self.config.hair_threshold).astype(np.float32)

            # Intersect with edge region
            hair = hair_candidate * edge

            # Remove hair from edge
            edge = edge * (1 - hair)

        return {
            'core': core,
            'edge': edge,
            'hair': hair
        }


# =============================================================================
# EDGE REFINEMENT
# =============================================================================

class EdgeRefiner:
    """Refine alpha matte edges."""

    def __init__(self, config: RotoConfig):
        self.config = config

    def refine(
        self,
        alpha: np.ndarray,
        image: Optional[np.ndarray] = None,
        guided: bool = True
    ) -> np.ndarray:
        """
        Refine alpha matte edges.

        Args:
            alpha: Alpha matte (H, W) float32 [0, 1]
            image: Original image for guided filtering
            guided: Use guided filter if True

        Returns:
            Refined alpha matte
        """
        if not self.config.edge_refinement:
            return alpha

        # Apply Gaussian blur for smoothing
        blur_radius = self.config.edge_blur_radius
        if blur_radius > 0:
            alpha = cv2.GaussianBlur(
                alpha,
                (blur_radius * 2 + 1, blur_radius * 2 + 1),
                0
            )

        # Guided filter if image provided
        if guided and image is not None:
            try:
                # Use OpenCV's guided filter
                alpha_uint8 = (alpha * 255).astype(np.uint8)
                if len(image.shape) == 3:
                    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    guide = image

                # Guided filter parameters
                radius = 8
                eps = 0.01 * 255 * 255

                alpha_refined = cv2.ximgproc.guidedFilter(
                    guide, alpha_uint8, radius, eps
                )
                alpha = alpha_refined.astype(np.float32) / 255.0
            except AttributeError:
                # cv2.ximgproc not available
                pass

        # Ensure valid range
        alpha = np.clip(alpha, 0, 1)

        return alpha


# =============================================================================
# ULTIMATE ROTOSCOPY PIPELINE
# =============================================================================

class UltimateRoto:
    """
    Ultimate Rotoscopy Pipeline.

    Integrates SAM3, ViTMatte, MatAnyone, and Depth Anything 3
    for professional-quality rotoscopy with:
    - Automatic trimap generation
    - Hair and fine detail preservation
    - Depth estimation for compositing
    - Layer decomposition
    """

    def __init__(self, config: Optional[RotoConfig] = None):
        self.config = config or RotoConfig()

        # Initialize components (lazy loading)
        self.sam3 = SAM3Segmenter(self.config)
        self.trimap_gen = TrimapGenerator(self.config)
        self.vitmatte = ViTMatteProcessor(self.config)
        self.matanyone = MatAnyoneProcessor(self.config)
        self.depth = DepthAnything3Processor(self.config)
        self.decomposer = LayerDecomposer(self.config)
        self.refiner = EdgeRefiner(self.config)

    def process_image(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        points: Optional[List[Tuple[int, int]]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
        mask: Optional[np.ndarray] = None,
        estimate_depth: bool = False
    ) -> RotoResult:
        """
        Process single image for rotoscopy.

        Args:
            image_path: Path to input image
            prompt: Text prompt for SAM3
            points: Point prompts for SAM3
            box: Box prompt for SAM3
            mask: Pre-computed mask (skip SAM3)
            estimate_depth: Whether to estimate depth

        Returns:
            RotoResult with alpha, foreground, and optional layers
        """
        image_path = Path(image_path)
        print(f"\n{'='*60}")
        print(f"  Ultimate Roto - Processing: {image_path.name}")
        print(f"{'='*60}")

        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

        # Step 1: Get initial mask from SAM3
        if mask is None:
            print("\n[1/5] SAM3 Segmentation...")
            if prompt:
                mask = self.sam3.segment_text(image, prompt)
            elif points:
                mask = self.sam3.segment_points(image, points)
            elif box:
                mask = self.sam3.segment_box(image, box)
            else:
                raise ValueError("Must provide prompt, points, box, or mask")
        else:
            print("\n[1/5] Using provided mask...")

        # Step 2: Generate trimap
        print("\n[2/5] Generating Trimap...")
        trimap = self.trimap_gen.generate_adaptive(mask, image)

        # Step 3: Alpha matting with ViTMatte
        print("\n[3/5] ViTMatte Alpha Matting...")
        try:
            alpha = self.vitmatte.process(image, trimap)
        except Exception as e:
            print(f"  ViTMatte failed: {e}")
            print("  Falling back to trimap-based alpha...")
            # Fallback: use trimap as alpha
            alpha = trimap.astype(np.float32) / 255.0

        # Step 4: Edge refinement
        print("\n[4/5] Edge Refinement...")
        alpha = self.refiner.refine(alpha, image)

        # Step 5: Optional depth estimation
        depth_map = None
        if estimate_depth or self.config.depth_enabled:
            print("\n[5/5] Depth Estimation...")
            try:
                depth_map = self.depth.estimate_depth(image)
            except Exception as e:
                print(f"  Depth estimation failed: {e}")
        else:
            print("\n[5/5] Skipping depth estimation...")

        # Extract foreground
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        foreground = (image * alpha_3ch).astype(np.uint8)

        # Decompose layers
        layers = self.decomposer.decompose(alpha, image)

        # Create result
        result = RotoResult(
            alpha=alpha,
            foreground=foreground,
            trimap=trimap,
            depth=depth_map,
            core_mask=layers['core'],
            edge_mask=layers['edge'],
            hair_mask=layers['hair'],
            metadata={
                'source': str(image_path),
                'prompt': prompt,
                'image_size': (image.shape[1], image.shape[0]),
                'config': {
                    'trimap_erosion': self.config.trimap_erosion,
                    'trimap_dilation': self.config.trimap_dilation,
                    'hair_refinement': self.config.hair_refinement,
                }
            }
        )

        print(f"\n{'='*60}")
        print(f"  Processing Complete!")
        print(f"{'='*60}")

        return result

    def process_video(
        self,
        video_path: Union[str, Path],
        prompt: str,
        output_dir: Union[str, Path],
        first_frame_mask: Optional[np.ndarray] = None
    ) -> Tuple[str, str]:
        """
        Process video for rotoscopy using MatAnyone.

        Args:
            video_path: Path to input video
            prompt: Text prompt for first frame segmentation
            output_dir: Output directory
            first_frame_mask: Optional pre-computed first frame mask

        Returns:
            Tuple of (foreground_video_path, alpha_video_path)
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Ultimate Roto - Video Processing")
        print(f"  Input: {video_path.name}")
        print(f"{'='*60}")

        # Extract first frame if no mask provided
        if first_frame_mask is None:
            print("\n[1/3] Extracting first frame...")
            cap = cv2.VideoCapture(str(video_path))
            ret, first_frame = cap.read()
            cap.release()

            if not ret:
                raise ValueError(f"Could not read video: {video_path}")

            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

            print("\n[2/3] SAM3 Segmentation on first frame...")
            first_frame_mask = self.sam3.segment_text(first_frame, prompt)
        else:
            print("\n[1-2/3] Using provided first frame mask...")

        # Save first frame mask
        mask_path = output_dir / "first_frame_mask.png"
        cv2.imwrite(str(mask_path), first_frame_mask)

        # Process with MatAnyone
        print("\n[3/3] MatAnyone Video Processing...")
        return self.matanyone.process_video(
            video_path,
            mask_path,
            output_dir,
            save_frames=True
        )

    def save_result(
        self,
        result: RotoResult,
        output_dir: Union[str, Path],
        prefix: str = "roto"
    ) -> Dict[str, str]:
        """
        Save rotoscopy result to disk.

        Args:
            result: RotoResult to save
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Save alpha
        alpha_path = output_dir / f"{prefix}_alpha.png"
        alpha_uint8 = (result.alpha * 255).astype(np.uint8)
        cv2.imwrite(str(alpha_path), alpha_uint8)
        saved['alpha'] = str(alpha_path)

        # Save foreground with alpha
        fg_path = output_dir / f"{prefix}_foreground.png"
        fg_rgba = np.concatenate([
            result.foreground,
            alpha_uint8[:, :, np.newaxis]
        ], axis=-1)
        Image.fromarray(fg_rgba).save(fg_path)
        saved['foreground'] = str(fg_path)

        # Save trimap
        if result.trimap is not None and self.config.save_trimap:
            trimap_path = output_dir / f"{prefix}_trimap.png"
            cv2.imwrite(str(trimap_path), result.trimap)
            saved['trimap'] = str(trimap_path)

        # Save depth
        if result.depth is not None:
            depth_path = output_dir / f"{prefix}_depth.png"
            depth_uint8 = (result.depth * 255).astype(np.uint8)
            # Apply colormap for visualization
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
            cv2.imwrite(str(depth_path), depth_colored)
            saved['depth'] = str(depth_path)

            # Also save raw depth as 16-bit
            depth_raw_path = output_dir / f"{prefix}_depth_raw.png"
            depth_uint16 = (result.depth * 65535).astype(np.uint16)
            cv2.imwrite(str(depth_raw_path), depth_uint16)
            saved['depth_raw'] = str(depth_raw_path)

        # Save layers
        if self.config.save_layers:
            if result.core_mask is not None:
                core_path = output_dir / f"{prefix}_layer_core.png"
                cv2.imwrite(str(core_path), (result.core_mask * 255).astype(np.uint8))
                saved['layer_core'] = str(core_path)

            if result.edge_mask is not None:
                edge_path = output_dir / f"{prefix}_layer_edge.png"
                cv2.imwrite(str(edge_path), (result.edge_mask * 255).astype(np.uint8))
                saved['layer_edge'] = str(edge_path)

            if result.hair_mask is not None:
                hair_path = output_dir / f"{prefix}_layer_hair.png"
                cv2.imwrite(str(hair_path), (result.hair_mask * 255).astype(np.uint8))
                saved['layer_hair'] = str(hair_path)

        # Save metadata
        meta_path = output_dir / f"{prefix}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(result.metadata, f, indent=2)
        saved['metadata'] = str(meta_path)

        print(f"\nSaved {len(saved)} files to {output_dir}")
        for key, path in saved.items():
            print(f"  {key}: {Path(path).name}")

        return saved


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ultimate Rotoscopy - Professional rotoscopy with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image with text prompt
  python ultimate_roto.py image photo.jpg --text "person" -o results/

  # Image with point prompt
  python ultimate_roto.py image photo.jpg --point 100,200 --point 150,250 -o results/

  # Image with depth estimation
  python ultimate_roto.py image photo.jpg --text "person" --depth -o results/

  # Video processing
  python ultimate_roto.py video clip.mp4 --text "person" -o results/
        """
    )

    subparsers = parser.add_subparsers(dest="mode", help="Processing mode")

    # Image mode
    img_parser = subparsers.add_parser("image", help="Process single image")
    img_parser.add_argument("input", type=Path, help="Input image path")
    img_parser.add_argument("--text", "-t", type=str, help="Text prompt")
    img_parser.add_argument("--point", action="append", help="Point prompt (x,y)")
    img_parser.add_argument("--box", type=str, help="Box prompt (x1,y1,x2,y2)")
    img_parser.add_argument("--output", "-o", type=Path, default=Path("output"), help="Output directory")
    img_parser.add_argument("--depth", action="store_true", help="Estimate depth")
    img_parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    img_parser.add_argument("--erosion", type=int, default=15, help="Trimap erosion")
    img_parser.add_argument("--dilation", type=int, default=30, help="Trimap dilation")

    # Video mode
    vid_parser = subparsers.add_parser("video", help="Process video")
    vid_parser.add_argument("input", type=Path, help="Input video path")
    vid_parser.add_argument("--text", "-t", type=str, required=True, help="Text prompt")
    vid_parser.add_argument("--output", "-o", type=Path, default=Path("output"), help="Output directory")
    vid_parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    vid_parser.add_argument("--warmup", type=int, default=10, help="Warmup frames")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    # Create config
    config = RotoConfig(device=args.device)

    if args.mode == "image":
        config.trimap_erosion = args.erosion
        config.trimap_dilation = args.dilation
        config.depth_enabled = args.depth

        # Create pipeline
        roto = UltimateRoto(config)

        # Parse prompts
        points = None
        if args.point:
            points = [tuple(map(int, p.split(","))) for p in args.point]

        box = None
        if args.box:
            box = tuple(map(int, args.box.split(",")))

        # Process
        result = roto.process_image(
            args.input,
            prompt=args.text,
            points=points,
            box=box,
            estimate_depth=args.depth
        )

        # Save
        roto.save_result(result, args.output, prefix=args.input.stem)

    elif args.mode == "video":
        config.video_warmup_frames = args.warmup

        # Create pipeline
        roto = UltimateRoto(config)

        # Process
        fg_path, alpha_path = roto.process_video(
            args.input,
            args.text,
            args.output
        )

        print(f"\nVideo processing complete:")
        print(f"  Foreground: {fg_path}")
        print(f"  Alpha: {alpha_path}")


if __name__ == "__main__":
    main()
