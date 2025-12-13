#!/usr/bin/env python3
"""
Ultimate Rotoscopy Application
==============================

Professional rotoscopy pipeline integrating:
- SAM3: Initial rough segmentation (text/point/box prompts)
- ViTMatte: High-quality alpha matting from trimap
- MatAnyone: Video matting with temporal consistency
- DepthAnything3: Depth estimation for VFX compositing

Pipeline Flow:
    Input Image/Video
           ↓
    SAM3 (rough mask)
           ↓
    Trimap Generation (erosion/dilation)
           ↓
    ViTMatte (refined alpha) or MatAnyone (video)
           ↓
    Edge Refinement (guided filter)
           ↓
    Layer Decomposition (core/edge/hair)
           ↓
    Optional: Depth Estimation
           ↓
    Export (PNG/EXR with alpha)

Usage:
    python ultimate_roto.py image input.jpg --text "person" -o results/
    python ultimate_roto.py video input.mp4 --text "person" -o results/
"""

import os
import sys
import argparse
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

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RotoConfig:
    """Configuration for the rotoscopy pipeline."""
    # Device
    device: str = "cuda"

    # Trimap generation
    trimap_erosion: int = 15
    trimap_dilation: int = 30
    trimap_blur: int = 5

    # ViTMatte
    vitmatte_model: str = "vitmatte-b"  # vitmatte-s or vitmatte-b
    vitmatte_checkpoint: str = ""  # Path to checkpoint

    # MatAnyone video settings
    matanyone_warmup: int = 10
    matanyone_erode: int = 10
    matanyone_dilate: int = 10

    # Depth settings
    depth_model: str = "da3-large"  # da3-small, da3-large, da3-giant

    # Refinement
    hair_refinement: bool = True
    hair_threshold: float = 0.3
    edge_refinement: bool = True
    guided_filter_radius: int = 8
    guided_filter_eps: float = 0.01

    # Output
    save_trimap: bool = True
    save_layers: bool = True
    output_format: str = "png"


@dataclass
class RotoResult:
    """Result from rotoscopy processing."""
    alpha: np.ndarray  # (H, W) float32 [0, 1]
    foreground: np.ndarray  # (H, W, 4) RGBA uint8
    trimap: Optional[np.ndarray] = None  # (H, W) uint8
    depth: Optional[np.ndarray] = None  # (H, W) float32
    core_mask: Optional[np.ndarray] = None
    edge_mask: Optional[np.ndarray] = None
    hair_mask: Optional[np.ndarray] = None
    sam_mask: Optional[np.ndarray] = None  # Original SAM3 mask
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# TRIMAP GENERATOR
# =============================================================================

class TrimapGenerator:
    """
    Generate trimap from binary mask.

    Trimap values:
    - 0: Definite background
    - 128: Unknown/transition region
    - 255: Definite foreground
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
        """Generate trimap from binary mask."""
        erosion = erosion if erosion is not None else self.config.trimap_erosion
        dilation = dilation if dilation is not None else self.config.trimap_dilation
        blur = blur if blur is not None else self.config.trimap_blur

        # Ensure binary mask in uint8
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask = (mask * 255).astype(np.uint8)
        mask = mask.astype(np.uint8)

        # Threshold to ensure binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Create kernels
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))

        # Erode for definite foreground
        foreground = cv2.erode(mask, erode_kernel, iterations=1)

        # Dilate for potential region
        dilated = cv2.dilate(mask, dilate_kernel, iterations=1)

        # Unknown = dilated - foreground
        unknown = cv2.subtract(dilated, foreground)

        # Build trimap
        trimap = np.zeros_like(mask, dtype=np.uint8)
        trimap[foreground > 127] = 255  # Definite foreground
        trimap[unknown > 127] = 128  # Unknown region
        # Background stays 0

        # Optional blur for smoother boundaries
        if blur > 0:
            trimap_float = trimap.astype(np.float32)
            trimap_blurred = cv2.GaussianBlur(trimap_float, (blur * 2 + 1, blur * 2 + 1), 0)
            # Re-quantize
            trimap = np.zeros_like(mask, dtype=np.uint8)
            trimap[trimap_blurred > 191] = 255
            trimap[(trimap_blurred > 64) & (trimap_blurred <= 191)] = 128

        return trimap

    def generate_adaptive(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None,
        erosion: Optional[int] = None,
        dilation: Optional[int] = None
    ) -> np.ndarray:
        """Generate adaptive trimap with edge-aware unknown region."""
        # Base trimap
        trimap = self.generate(mask, erosion, dilation)

        if image is not None and self.config.hair_refinement:
            # Detect edges in image
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150)

            # Dilate edges
            edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges_dilated = cv2.dilate(edges, edge_kernel, iterations=2)

            # Find mask boundary
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_binary = (mask > 127).astype(np.uint8) * 255
            boundary = cv2.dilate(mask_binary, erode_kernel) - cv2.erode(mask_binary, erode_kernel)

            # Expand unknown where edges intersect boundary
            hair_region = cv2.bitwise_and(edges_dilated, boundary)
            trimap[hair_region > 0] = 128

        return trimap


# =============================================================================
# SAM3 SEGMENTER
# =============================================================================

class SAM3Segmenter:
    """
    SAM3 segmentation for initial rough mask.

    Supports three prompt types:
    - Text prompts (via Sam3Processor)
    - Point prompts (via SAM3InteractiveImagePredictor)
    - Box prompts (via Sam3Processor or SAM3InteractiveImagePredictor)
    """

    def __init__(self, config: RotoConfig):
        self.config = config
        self.device = config.device
        self._model = None
        self._processor = None  # Sam3Processor for text/box prompts
        self._interactive_predictor = None  # For point/box prompts

    def _load_model(self):
        """Lazy load SAM3 model."""
        if self._model is not None:
            return

        print("Loading SAM3 model...")
        try:
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            # Build model with instance interactivity for point/box prompts
            self._model = build_sam3_image_model(
                device=self.device if torch.cuda.is_available() else "cpu",
                enable_inst_interactivity=True  # Enable point/box prompts
            )

            # Sam3Processor for text prompts
            self._processor = Sam3Processor(
                self._model,
                device=self.device if torch.cuda.is_available() else "cpu"
            )

            # Interactive predictor for point/box prompts
            if self._model.inst_interactive_predictor is not None:
                self._interactive_predictor = self._model.inst_interactive_predictor

            if not torch.cuda.is_available():
                self.device = "cpu"

            self._model.eval()
            print(f"  SAM3 loaded on {self.device}")
            print(f"    - Text prompts: {'✓' if self._processor else '✗'}")
            print(f"    - Point/Box prompts: {'✓' if self._interactive_predictor else '✗'}")

        except ImportError as e:
            raise ImportError(f"SAM3 not installed: {e}\nInstall: pip install sam3")

    def segment_text(self, image: Union[str, Path, np.ndarray], text: str) -> np.ndarray:
        """
        Segment with text prompt using Sam3Processor.

        Args:
            image: Input image (path, Path, or numpy array RGB)
            text: Text description of object to segment (e.g., "person", "cat")

        Returns:
            Binary mask (H, W) uint8 with values 0 or 255
        """
        self._load_model()

        if isinstance(image, (str, Path)):
            pil_image = Image.open(str(image)).convert("RGB")
        else:
            pil_image = Image.fromarray(image)

        print(f"  Segmenting with text: '{text}'")

        # Use Sam3Processor for text prompts
        state = self._processor.set_image(pil_image)
        output = self._processor.set_text_prompt(prompt=text, state=state)

        # Extract masks and scores
        masks = output.get("masks", output.get("out_binary_masks"))
        scores = output.get("scores", output.get("out_probs"))

        if masks is None:
            raise ValueError("No masks returned from SAM3 text prompt")

        # Convert to numpy
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        # Ensure arrays
        masks = np.asarray(masks)
        scores = np.asarray(scores)

        if len(masks) == 0:
            raise ValueError(f"No objects found for text prompt: '{text}'")

        # Get best mask
        best_idx = scores.argmax()
        mask = masks[best_idx]

        # Ensure 2D
        while mask.ndim > 2:
            mask = mask.squeeze(0)

        # Convert to binary mask
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        print(f"  Found object (score: {scores[best_idx]:.3f})")

        return mask

    def segment_points(
        self,
        image: Union[str, Path, np.ndarray],
        points: List[Tuple[int, int]],
        labels: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Segment with point prompts using SAM3InteractiveImagePredictor.

        Args:
            image: Input image (path, Path, or numpy array RGB)
            points: List of (x, y) point coordinates
            labels: List of labels (1=foreground, 0=background). Default: all foreground

        Returns:
            Binary mask (H, W) uint8 with values 0 or 255
        """
        self._load_model()

        if self._interactive_predictor is None:
            raise RuntimeError("SAM3 interactive predictor not available. Point prompts not supported.")

        if isinstance(image, (str, Path)):
            img_array = np.array(Image.open(str(image)).convert("RGB"))
        elif isinstance(image, np.ndarray):
            img_array = image
        else:
            img_array = np.array(image)

        if labels is None:
            labels = [1] * len(points)  # All foreground by default

        print(f"  Segmenting with {len(points)} points")

        # Set image
        self._interactive_predictor.set_image(img_array)

        # Prepare points
        points_np = np.array(points, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int32)

        # Predict
        masks, scores, _ = self._interactive_predictor.predict(
            point_coords=points_np,
            point_labels=labels_np,
            multimask_output=True
        )

        # Get best mask
        best_idx = scores.argmax()
        mask = masks[best_idx]

        # Ensure 2D
        while mask.ndim > 2:
            mask = mask.squeeze(0)

        # Convert to uint8
        mask = (mask * 255).astype(np.uint8)
        print(f"  Segmentation done (score: {scores[best_idx]:.3f})")

        return mask

    def segment_box(
        self,
        image: Union[str, Path, np.ndarray],
        box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Segment with bounding box prompt.

        Args:
            image: Input image (path, Path, or numpy array RGB)
            box: Bounding box (x1, y1, x2, y2) in pixel coordinates

        Returns:
            Binary mask (H, W) uint8 with values 0 or 255
        """
        self._load_model()

        if isinstance(image, (str, Path)):
            img_array = np.array(Image.open(str(image)).convert("RGB"))
            pil_image = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            img_array = image
            pil_image = Image.fromarray(image)
        else:
            img_array = np.array(image)
            pil_image = image

        print(f"  Segmenting with box: {box}")

        # Try interactive predictor first (more accurate for box prompts)
        if self._interactive_predictor is not None:
            self._interactive_predictor.set_image(img_array)

            box_np = np.array(box, dtype=np.float32)

            masks, scores, _ = self._interactive_predictor.predict(
                box=box_np,
                multimask_output=True
            )

            best_idx = scores.argmax()
            mask = masks[best_idx]

            while mask.ndim > 2:
                mask = mask.squeeze(0)

            mask = (mask * 255).astype(np.uint8)
            print(f"  Segmentation done (score: {scores[best_idx]:.3f})")
            return mask

        # Fallback to Sam3Processor with geometric prompt
        h, w = img_array.shape[:2]
        x1, y1, x2, y2 = box

        # Convert to normalized center format [cx, cy, w, h]
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        normalized_box = [cx, cy, bw, bh]

        state = self._processor.set_image(pil_image)
        output = self._processor.add_geometric_prompt(
            box=normalized_box,
            label=True,  # Positive box
            state=state
        )

        masks = output.get("masks", output.get("out_binary_masks"))
        scores = output.get("scores", output.get("out_probs"))

        if masks is None or len(masks) == 0:
            raise ValueError("No masks returned from SAM3 box prompt")

        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        best_idx = np.asarray(scores).argmax()
        mask = np.asarray(masks)[best_idx]

        while mask.ndim > 2:
            mask = mask.squeeze(0)

        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        print(f"  Segmentation done (score: {scores[best_idx]:.3f})")
        return mask


# =============================================================================
# VITMATTE PROCESSOR
# =============================================================================

class ViTMatteProcessor:
    """
    ViTMatte alpha matting from image + trimap.

    Based on: ViTMatte-main/run_one_image.py
    """

    def __init__(self, config: RotoConfig):
        self.config = config
        self.device = config.device
        self._model = None

    def _load_model(self):
        """Load ViTMatte model using the official API."""
        if self._model is not None:
            return

        print("Loading ViTMatte model...")

        try:
            import torch
            from torchvision.transforms import functional as F

            # Add ViTMatte to path
            vitmatte_dir = Path(__file__).parent / "ViTMatte-main"
            if str(vitmatte_dir) not in sys.path:
                sys.path.insert(0, str(vitmatte_dir))

            from detectron2.config import LazyConfig, instantiate
            from detectron2.checkpoint import DetectionCheckpointer

            # Load config
            config_path = vitmatte_dir / "configs" / "common" / "model.py"
            if not config_path.exists():
                raise FileNotFoundError(f"ViTMatte config not found: {config_path}")

            cfg = LazyConfig.load(str(config_path))

            # Adjust for model variant
            if self.config.vitmatte_model == "vitmatte-b":
                cfg.model.backbone.embed_dim = 768
                cfg.model.backbone.num_heads = 12
                cfg.model.decoder.in_chans = 768

            # Instantiate model
            self._model = instantiate(cfg.model)

            # Move to device
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self._model.to("cuda")
                else:
                    self.device = "cpu"
                    self._model.to("cpu")
            else:
                self._model.to(self.device)

            self._model.eval()

            # Load checkpoint if available
            ckpt_path = self.config.vitmatte_checkpoint
            if not ckpt_path:
                # Try default locations
                possible_paths = [
                    vitmatte_dir / "pretrained" / f"{self.config.vitmatte_model}.pth",
                    vitmatte_dir / f"{self.config.vitmatte_model}.pth",
                    Path(f"pretrained_models/{self.config.vitmatte_model}.pth"),
                ]
                for p in possible_paths:
                    if p.exists():
                        ckpt_path = str(p)
                        break

            if ckpt_path and Path(ckpt_path).exists():
                DetectionCheckpointer(self._model).load(ckpt_path)
                print(f"  Loaded checkpoint: {ckpt_path}")
            else:
                print("  Warning: No checkpoint loaded - using random weights")

            print(f"  ViTMatte loaded on {self.device}")

        except ImportError as e:
            raise ImportError(f"ViTMatte dependencies missing: {e}\nInstall: pip install detectron2")

    def process(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """
        Process image with trimap to get alpha matte.

        Args:
            image: RGB image (H, W, 3) uint8
            trimap: Trimap (H, W) uint8 with values 0, 128, 255

        Returns:
            Alpha matte (H, W) float32 [0, 1]
        """
        self._load_model()

        import torch
        from torchvision.transforms import functional as F

        # Convert to PIL for transforms
        image_pil = Image.fromarray(image)
        trimap_pil = Image.fromarray(trimap).convert('L')

        # Convert to tensors
        image_tensor = F.to_tensor(image_pil).unsqueeze(0)
        trimap_tensor = F.to_tensor(trimap_pil).unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self.device)
        trimap_tensor = trimap_tensor.to(self.device)

        # Prepare input dict (as per ViTMatte API)
        input_dict = {
            'image': image_tensor,
            'trimap': trimap_tensor
        }

        # Run inference
        with torch.no_grad():
            output = self._model(input_dict)

        # Extract alpha
        alpha = output['phas'].flatten(0, 2)
        alpha = alpha.cpu().numpy()

        return alpha.astype(np.float32)


# =============================================================================
# MATANYONE VIDEO PROCESSOR
# =============================================================================

class MatAnyoneProcessor:
    """
    MatAnyone video matting with temporal consistency.

    Based on: matanyone/inference/inference_core.py
    """

    def __init__(self, config: RotoConfig):
        self.config = config
        self.device = config.device
        self._processor = None

    def _load_model(self):
        """Load MatAnyone model."""
        if self._processor is not None:
            return

        print("Loading MatAnyone model...")

        try:
            import torch

            # Add matanyone to path
            matanyone_dir = Path(__file__).parent / "matanyone"
            if str(matanyone_dir.parent) not in sys.path:
                sys.path.insert(0, str(matanyone_dir.parent))

            from matanyone.inference.inference_core import InferenceCore
            from matanyone.utils.get_default_model import get_matanyone_model

            # Try to find or download checkpoint
            ckpt_path = None
            possible_paths = [
                Path("pretrained_models/matanyone.pth"),
                Path(__file__).parent / "pretrained_models" / "matanyone.pth",
            ]

            for p in possible_paths:
                if p.exists():
                    ckpt_path = str(p)
                    break

            if ckpt_path is None:
                # Try downloading
                try:
                    from hugging_face.tools.download_util import load_file_from_url
                    pretrain_url = "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
                    ckpt_path = load_file_from_url(pretrain_url, 'pretrained_models')
                except:
                    raise FileNotFoundError("MatAnyone checkpoint not found. Download from: https://github.com/pq-yang/MatAnyone/releases")

            # Load model
            model = get_matanyone_model(ckpt_path, self.device)
            self._processor = InferenceCore(model, cfg=model.cfg)

            print(f"  MatAnyone loaded on {self.device}")

        except ImportError as e:
            raise ImportError(f"MatAnyone dependencies missing: {e}")

    def process_video(
        self,
        video_path: Union[str, Path],
        mask_path: Union[str, Path],
        output_dir: Union[str, Path],
        warmup: Optional[int] = None,
        erode: Optional[int] = None,
        dilate: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Process video for matting.

        Returns:
            Tuple of (foreground_video_path, alpha_video_path)
        """
        self._load_model()

        warmup = warmup if warmup is not None else self.config.matanyone_warmup
        erode = erode if erode is not None else self.config.matanyone_erode
        dilate = dilate if dilate is not None else self.config.matanyone_dilate

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        return self._processor.process_video(
            input_path=str(video_path),
            mask_path=str(mask_path),
            output_path=str(output_dir),
            n_warmup=warmup,
            r_erode=erode,
            r_dilate=dilate,
            save_image=True
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
        self._load_model()

        warmup = warmup if warmup is not None else self.config.matanyone_warmup
        erode = erode if erode is not None else self.config.matanyone_erode
        dilate = dilate if dilate is not None else self.config.matanyone_dilate

        # Use the matanyone wrapper
        from hugging_face.matanyone_wrapper import matanyone

        return matanyone(
            self._processor,
            frames,
            mask,
            r_erode=erode,
            r_dilate=dilate,
            n_warmup=warmup
        )


# =============================================================================
# DEPTH ANYTHING 3 PROCESSOR
# =============================================================================

class DepthAnything3Processor:
    """
    Depth Anything 3 depth estimation with full feature support.

    Features:
    - Multiple model variants (da3-large, da3-giant, da3metric-large, da3nested-giant-large)
    - Export formats: mini_npz, npz, glb, ply, gs, gs_video, colmap, feat_vis
    - Multi-view depth estimation
    - Camera pose estimation
    - 3D Gaussian Splatting
    - Point cloud generation

    Based on: Depth-Anything-3-main/src/depth_anything_3/api.py
    """

    # Available model variants
    MODEL_VARIANTS = {
        "da3-large": "Standard large model",
        "da3-giant": "Giant model (more accurate)",
        "da3metric-large": "Metric depth (real-world scale)",
        "da3nested-giant-large": "Nested architecture"
    }

    # Available export formats
    EXPORT_FORMATS = {
        "mini_npz": "Minimal NPZ (depth + camera)",
        "npz": "Full NPZ (all data)",
        "glb": "GLTF Binary with point cloud",
        "ply": "Point cloud PLY format",
        "gs": "Gaussian Splatting",
        "gs_video": "Gaussian Splatting video render",
        "colmap": "COLMAP format",
        "feat_vis": "Feature visualization video"
    }

    def __init__(self, config: RotoConfig):
        self.config = config
        self.device = config.device
        self._model = None
        self._model_name = config.depth_model

    def _load_model(self, model_name: Optional[str] = None):
        """Load Depth Anything 3 model."""
        model_name = model_name or self._model_name

        # Reload if different model requested
        if self._model is not None and model_name == self._model_name:
            return

        print(f"Loading Depth Anything 3 model ({model_name})...")

        try:
            import torch

            # Add DA3 to path
            da3_dir = Path(__file__).parent / "Depth-Anything-3-main" / "src"
            if str(da3_dir) not in sys.path:
                sys.path.insert(0, str(da3_dir))

            from depth_anything_3.api import DepthAnything3

            # Create model
            self._model = DepthAnything3(model_name=model_name)
            self._model_name = model_name

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self._model.to("cuda")
                self._model.device = torch.device("cuda")
            else:
                self.device = "cpu"
                self._model.to("cpu")
                self._model.device = torch.device("cpu")

            print(f"  Depth Anything 3 ({model_name}) loaded on {self.device}")

        except ImportError as e:
            raise ImportError(f"Depth Anything 3 dependencies missing: {e}")

    def estimate(
        self,
        image: Union[str, Path, np.ndarray, List],
        model_name: Optional[str] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Estimate depth from image(s).

        Args:
            image: Input image(s) - path, numpy array, or list of images
            model_name: Model variant to use (default: config.depth_model)
            normalize: Normalize depth to [0, 1] range

        Returns:
            Depth map (H, W) float32, optionally normalized
        """
        self._load_model(model_name)

        # Prepare input as list
        if isinstance(image, (str, Path)):
            image_input = [str(image)]
        elif isinstance(image, np.ndarray):
            image_input = [Image.fromarray(image)]
        elif isinstance(image, list):
            image_input = []
            for img in image:
                if isinstance(img, (str, Path)):
                    image_input.append(str(img))
                elif isinstance(img, np.ndarray):
                    image_input.append(Image.fromarray(img))
                else:
                    image_input.append(img)
        else:
            image_input = [image]

        # Run inference
        prediction = self._model.inference(image_input)

        # Extract depth
        depth = prediction.depth[0]  # (H, W)

        if normalize:
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = np.zeros_like(depth)

        return depth.astype(np.float32)

    def estimate_with_export(
        self,
        images: Union[str, Path, np.ndarray, List],
        export_dir: Union[str, Path],
        export_format: str = "mini_npz",
        model_name: Optional[str] = None,
        # Processing options
        process_res: int = 504,
        # GLB export options
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        # Gaussian Splatting options
        infer_gs: bool = False,
        # Multi-view options
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Estimate depth and export to various formats.

        Args:
            images: Input image(s)
            export_dir: Output directory
            export_format: One of: mini_npz, npz, glb, ply, gs, gs_video, colmap, feat_vis
            model_name: Model variant to use
            process_res: Processing resolution
            conf_thresh_percentile: Confidence threshold for GLB export
            num_max_points: Maximum points for point cloud
            show_cameras: Show camera wireframes in GLB
            infer_gs: Enable Gaussian Splatting (required for gs/gs_video export)
            extrinsics: Camera extrinsics for multi-view (N, 4, 4)
            intrinsics: Camera intrinsics for multi-view (N, 3, 3)

        Returns:
            Dictionary with prediction results and export paths
        """
        self._load_model(model_name)

        # Prepare input as list
        if isinstance(images, (str, Path)):
            image_input = [str(images)]
        elif isinstance(images, np.ndarray):
            image_input = [Image.fromarray(images)]
        elif isinstance(images, list):
            image_input = []
            for img in images:
                if isinstance(img, (str, Path)):
                    image_input.append(str(img))
                elif isinstance(img, np.ndarray):
                    image_input.append(Image.fromarray(img))
                else:
                    image_input.append(img)
        else:
            image_input = [images]

        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Enable GS if needed
        if "gs" in export_format and not infer_gs:
            infer_gs = True
            print("  Enabling Gaussian Splatting for GS export")

        print(f"  Processing {len(image_input)} image(s), export format: {export_format}")

        # Run inference with export
        prediction = self._model.inference(
            image=image_input,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            infer_gs=infer_gs,
            process_res=process_res,
            export_dir=str(export_dir),
            export_format=export_format,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
        )

        result = {
            "depth": prediction.depth,
            "export_dir": str(export_dir),
            "export_format": export_format,
        }

        # Add camera parameters if available
        if hasattr(prediction, 'extrinsics') and prediction.extrinsics is not None:
            result["extrinsics"] = prediction.extrinsics
        if hasattr(prediction, 'intrinsics') and prediction.intrinsics is not None:
            result["intrinsics"] = prediction.intrinsics

        # Add processed images if available
        if hasattr(prediction, 'processed_images'):
            result["processed_images"] = prediction.processed_images

        print(f"  Exported to: {export_dir}")
        return result

    def estimate_multi_view(
        self,
        images: List[Union[str, Path, np.ndarray]],
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate depth from multiple views with camera pose estimation.

        Args:
            images: List of input images
            extrinsics: Known camera extrinsics (N, 4, 4), or None to estimate
            intrinsics: Known camera intrinsics (N, 3, 3), or None to estimate

        Returns:
            Tuple of (depths, estimated_extrinsics, estimated_intrinsics)
        """
        self._load_model(model_name)

        # Prepare images
        image_input = []
        for img in images:
            if isinstance(img, (str, Path)):
                image_input.append(str(img))
            elif isinstance(img, np.ndarray):
                image_input.append(Image.fromarray(img))
            else:
                image_input.append(img)

        print(f"  Multi-view depth estimation: {len(image_input)} images")

        # Run inference
        prediction = self._model.inference(
            image=image_input,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )

        depths = prediction.depth
        est_extrinsics = getattr(prediction, 'extrinsics', None)
        est_intrinsics = getattr(prediction, 'intrinsics', None)

        return depths, est_extrinsics, est_intrinsics

    def generate_point_cloud(
        self,
        image: Union[str, Path, np.ndarray],
        output_path: Union[str, Path],
        format: str = "ply",
        num_max_points: int = 1_000_000,
        conf_thresh_percentile: float = 40.0,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Generate a 3D point cloud from depth estimation.

        Args:
            image: Input image
            output_path: Output file path
            format: Export format ("ply" or "glb")
            num_max_points: Maximum number of points
            conf_thresh_percentile: Confidence threshold

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        export_dir = output_path.parent
        export_dir.mkdir(parents=True, exist_ok=True)

        result = self.estimate_with_export(
            images=image,
            export_dir=export_dir,
            export_format=format,
            model_name=model_name,
            num_max_points=num_max_points,
            conf_thresh_percentile=conf_thresh_percentile,
        )

        return str(export_dir)

    @classmethod
    def available_models(cls) -> dict:
        """Return available model variants with descriptions."""
        return cls.MODEL_VARIANTS.copy()

    @classmethod
    def available_export_formats(cls) -> dict:
        """Return available export formats with descriptions."""
        return cls.EXPORT_FORMATS.copy()


# =============================================================================
# EDGE REFINER
# =============================================================================

class EdgeRefiner:
    """Refine alpha matte edges using guided filtering."""

    def __init__(self, config: RotoConfig):
        self.config = config

    def refine(
        self,
        alpha: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Refine alpha matte edges.

        Args:
            alpha: Alpha matte (H, W) float32 [0, 1]
            image: Guide image for guided filtering

        Returns:
            Refined alpha matte
        """
        if not self.config.edge_refinement:
            return alpha

        # Apply guided filter if image provided
        if image is not None:
            try:
                alpha_uint8 = (alpha * 255).astype(np.uint8)

                if len(image.shape) == 3:
                    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    guide = image

                radius = self.config.guided_filter_radius
                eps = self.config.guided_filter_eps * 255 * 255

                alpha_refined = cv2.ximgproc.guidedFilter(guide, alpha_uint8, radius, eps)
                alpha = alpha_refined.astype(np.float32) / 255.0

            except AttributeError:
                # cv2.ximgproc not available, use bilateral filter as fallback
                alpha_uint8 = (alpha * 255).astype(np.uint8)
                alpha_refined = cv2.bilateralFilter(alpha_uint8, 9, 75, 75)
                alpha = alpha_refined.astype(np.float32) / 255.0

        return np.clip(alpha, 0, 1)


# =============================================================================
# LAYER DECOMPOSER
# =============================================================================

class LayerDecomposer:
    """Decompose alpha into semantic layers (core, edge, hair)."""

    def __init__(self, config: RotoConfig):
        self.config = config

    def decompose(
        self,
        alpha: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Decompose alpha into layers.

        Returns:
            Dict with 'core', 'edge', 'hair' masks
        """
        # Core: solid foreground (alpha > 0.95)
        core = (alpha > 0.95).astype(np.float32)

        # Edge: transition region
        edge = ((alpha > 0.05) & (alpha <= 0.95)).astype(np.float32)

        # Hair: high-frequency detail in edge region
        hair = np.zeros_like(alpha)

        if image is not None and self.config.hair_refinement:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

            # Detect high-frequency with Laplacian
            laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
            lap_norm = laplacian / (laplacian.max() + 1e-8)

            # Hair = high frequency within edge region
            hair_candidate = (lap_norm > self.config.hair_threshold).astype(np.float32)
            hair = hair_candidate * edge

            # Remove hair from edge
            edge = edge * (1 - hair)

        return {
            'core': core,
            'edge': edge,
            'hair': hair
        }


# =============================================================================
# ULTIMATE ROTO PIPELINE
# =============================================================================

class UltimateRoto:
    """
    Ultimate Rotoscopy Pipeline.

    Workflow:
    1. SAM3: Get rough segmentation mask
    2. Trimap: Generate trimap from mask
    3. ViTMatte/MatAnyone: Refine to alpha matte
    4. Edge Refinement: Clean up edges
    5. Layer Decomposition: Separate core/edge/hair
    6. Optional: Depth estimation
    """

    def __init__(self, config: Optional[RotoConfig] = None):
        self.config = config or RotoConfig()

        # Initialize components (lazy loading)
        self.sam3 = SAM3Segmenter(self.config)
        self.trimap_gen = TrimapGenerator(self.config)
        self.vitmatte = ViTMatteProcessor(self.config)
        self.matanyone = MatAnyoneProcessor(self.config)
        self.depth_processor = DepthAnything3Processor(self.config)
        self.refiner = EdgeRefiner(self.config)
        self.decomposer = LayerDecomposer(self.config)

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
            points: Point prompts [(x, y), ...]
            box: Box prompt (x1, y1, x2, y2)
            mask: Pre-computed mask (skip SAM3)
            estimate_depth: Whether to estimate depth

        Returns:
            RotoResult with alpha, foreground, layers
        """
        image_path = Path(image_path)
        print(f"\n{'='*60}")
        print(f"  Ultimate Roto - {image_path.name}")
        print(f"{'='*60}")

        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        print(f"  Image: {w}x{h}")

        # Step 1: SAM3 segmentation
        sam_mask = None
        if mask is None:
            print("\n[1/5] SAM3 Segmentation...")
            if prompt:
                mask = self.sam3.segment_text(image, prompt)
            elif points:
                mask = self.sam3.segment_points(image, points)
            elif box:
                mask = self.sam3.segment_box(image, box)
            else:
                raise ValueError("Provide prompt, points, box, or mask")
            sam_mask = mask.copy()
        else:
            print("\n[1/5] Using provided mask...")
            sam_mask = mask.copy()

        # Step 2: Generate trimap
        print("\n[2/5] Generating Trimap...")
        trimap = self.trimap_gen.generate_adaptive(mask, image)

        # Step 3: ViTMatte refinement
        print("\n[3/5] ViTMatte Alpha Matting...")
        try:
            alpha = self.vitmatte.process(image, trimap)
        except Exception as e:
            print(f"  ViTMatte failed: {e}")
            print("  Falling back to trimap-based alpha...")
            alpha = trimap.astype(np.float32) / 255.0

        # Step 4: Edge refinement
        print("\n[4/5] Edge Refinement...")
        alpha = self.refiner.refine(alpha, image)

        # Step 5: Depth estimation (optional)
        depth = None
        if estimate_depth:
            print("\n[5/5] Depth Estimation...")
            try:
                depth = self.depth_processor.estimate(image)
            except Exception as e:
                print(f"  Depth estimation failed: {e}")
        else:
            print("\n[5/5] Skipping depth estimation...")

        # Create foreground RGBA
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        foreground = np.dstack([image, alpha_uint8])

        # Decompose layers
        layers = self.decomposer.decompose(alpha, image)

        result = RotoResult(
            alpha=alpha,
            foreground=foreground,
            trimap=trimap,
            depth=depth,
            core_mask=layers['core'],
            edge_mask=layers['edge'],
            hair_mask=layers['hair'],
            sam_mask=sam_mask,
            metadata={
                'source': str(image_path),
                'prompt': prompt,
                'size': (w, h),
                'config': {
                    'trimap_erosion': self.config.trimap_erosion,
                    'trimap_dilation': self.config.trimap_dilation,
                }
            }
        )

        print(f"\n{'='*60}")
        print(f"  Complete!")
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

        Returns:
            Tuple of (foreground_video_path, alpha_video_path)
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Ultimate Roto - Video: {video_path.name}")
        print(f"{'='*60}")

        # Get first frame mask
        if first_frame_mask is None:
            print("\n[1/2] SAM3 on first frame...")
            cap = cv2.VideoCapture(str(video_path))
            ret, first_frame = cap.read()
            cap.release()

            if not ret:
                raise ValueError(f"Cannot read video: {video_path}")

            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            first_frame_mask = self.sam3.segment_text(first_frame, prompt)
        else:
            print("\n[1/2] Using provided mask...")

        # Save mask for MatAnyone
        mask_path = output_dir / "first_frame_mask.png"
        cv2.imwrite(str(mask_path), first_frame_mask)

        # Process with MatAnyone
        print("\n[2/2] MatAnyone video processing...")
        return self.matanyone.process_video(video_path, mask_path, output_dir)

    def save_result(
        self,
        result: RotoResult,
        output_dir: Union[str, Path],
        prefix: str = "roto"
    ) -> Dict[str, str]:
        """Save rotoscopy result to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Save alpha
        alpha_path = output_dir / f"{prefix}_alpha.png"
        alpha_uint8 = (result.alpha * 255).astype(np.uint8)
        cv2.imwrite(str(alpha_path), alpha_uint8)
        saved['alpha'] = str(alpha_path)

        # Save foreground with alpha (RGBA PNG)
        fg_path = output_dir / f"{prefix}_foreground.png"
        # Convert RGB to BGR for OpenCV, keep alpha
        fg_bgra = cv2.cvtColor(result.foreground[:, :, :3], cv2.COLOR_RGB2BGR)
        fg_bgra = np.dstack([fg_bgra, result.foreground[:, :, 3]])
        cv2.imwrite(str(fg_path), fg_bgra)
        saved['foreground'] = str(fg_path)

        # Save trimap
        if result.trimap is not None and self.config.save_trimap:
            trimap_path = output_dir / f"{prefix}_trimap.png"
            cv2.imwrite(str(trimap_path), result.trimap)
            saved['trimap'] = str(trimap_path)

        # Save SAM mask
        if result.sam_mask is not None:
            sam_path = output_dir / f"{prefix}_sam_mask.png"
            cv2.imwrite(str(sam_path), result.sam_mask)
            saved['sam_mask'] = str(sam_path)

        # Save depth
        if result.depth is not None:
            # Colorized depth
            depth_path = output_dir / f"{prefix}_depth.png"
            depth_uint8 = (result.depth * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
            cv2.imwrite(str(depth_path), depth_colored)
            saved['depth'] = str(depth_path)

            # Raw depth (16-bit)
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
        description="Ultimate Rotoscopy - Professional AI-powered rotoscopy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ultimate_roto.py image photo.jpg --text "person" -o results/
  python ultimate_roto.py image photo.jpg --point 100,200 -o results/
  python ultimate_roto.py image photo.jpg --text "dog" --depth -o results/
  python ultimate_roto.py video clip.mp4 --text "person" -o results/
        """
    )

    subparsers = parser.add_subparsers(dest="mode", help="Processing mode")

    # Image mode
    img_parser = subparsers.add_parser("image", help="Process single image")
    img_parser.add_argument("input", type=Path, help="Input image")
    img_parser.add_argument("--text", "-t", type=str, help="Text prompt")
    img_parser.add_argument("--point", action="append", help="Point x,y (can repeat)")
    img_parser.add_argument("--box", type=str, help="Box x1,y1,x2,y2")
    img_parser.add_argument("--output", "-o", type=Path, default=Path("output"))
    img_parser.add_argument("--depth", action="store_true", help="Estimate depth")
    img_parser.add_argument("--device", default="cuda")
    img_parser.add_argument("--erosion", type=int, default=15)
    img_parser.add_argument("--dilation", type=int, default=30)

    # Video mode
    vid_parser = subparsers.add_parser("video", help="Process video")
    vid_parser.add_argument("input", type=Path, help="Input video")
    vid_parser.add_argument("--text", "-t", type=str, required=True)
    vid_parser.add_argument("--output", "-o", type=Path, default=Path("output"))
    vid_parser.add_argument("--device", default="cuda")
    vid_parser.add_argument("--warmup", type=int, default=10)

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    # Create config
    config = RotoConfig(device=args.device)

    if args.mode == "image":
        config.trimap_erosion = args.erosion
        config.trimap_dilation = args.dilation

        roto = UltimateRoto(config)

        # Parse prompts
        points = None
        if args.point:
            points = [tuple(map(int, p.split(","))) for p in args.point]

        box = None
        if args.box:
            box = tuple(map(int, args.box.split(",")))

        result = roto.process_image(
            args.input,
            prompt=args.text,
            points=points,
            box=box,
            estimate_depth=args.depth
        )

        roto.save_result(result, args.output, prefix=args.input.stem)

    elif args.mode == "video":
        config.matanyone_warmup = args.warmup

        roto = UltimateRoto(config)

        fg_path, alpha_path = roto.process_video(args.input, args.text, args.output)

        print(f"\nVideo complete:")
        print(f"  Foreground: {fg_path}")
        print(f"  Alpha: {alpha_path}")


if __name__ == "__main__":
    main()
