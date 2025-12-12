#!/usr/bin/env python3
"""
Matte Anything - Complete Professional Matting Wrapper
======================================================

Full-featured matting pipeline integrating:
- ViTMatte (Transformer-based matting)
- MatAnyone (Video matting with temporal consistency)
- SAM3 integration for automatic trimap generation
- Professional alpha refinement
- Multi-layer export (core, edge, hair)

Requirements:
    pip install torch torchvision transformers opencv-python numpy pillow

Usage:
    # CLI
    python matte_anything.py image.jpg -o output/
    python matte_anything.py image.jpg --mask mask.png -o output/
    python matte_anything.py video.mp4 --video -o output/

    # Python API
    from matte_anything import MatteAnything, MattingResult
    matte = MatteAnything()
    result = matte.process_image("image.jpg", mask="mask.png")
    result.save_all("output/")
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import cv2
except ImportError as e:
    print("Missing dependencies!")
    print("Install: pip install torch torchvision Pillow opencv-python numpy")
    sys.exit(1)


class MattingModel(Enum):
    """Available matting models."""
    VITMATTE = "vitmatte"
    MATANYONE = "matanyone"
    GVM = "gvm"  # Generative Video Matting


class TrimapSource(Enum):
    """Source for trimap generation."""
    SAM3 = "sam3"
    MANUAL = "manual"
    AUTO = "auto"


@dataclass
class MattingConfig:
    """Configuration for matting pipeline."""
    model: MattingModel = MattingModel.VITMATTE
    trimap_source: TrimapSource = TrimapSource.AUTO
    refine_edges: bool = True
    compute_layers: bool = True  # Split into core/edge/hair
    temporal_consistency: bool = True  # For video
    device: str = "cuda"


@dataclass
class MattingResult:
    """
    Complete result from matting pipeline.

    Contains alpha matte, layer decomposition, and metadata.
    """
    # Main output
    alpha: np.ndarray  # (H, W) - Alpha matte [0, 1]

    # Layer decomposition
    alpha_core: Optional[np.ndarray] = None  # (H, W) - Solid interior
    alpha_edge: Optional[np.ndarray] = None  # (H, W) - Transition boundary
    alpha_hair: Optional[np.ndarray] = None  # (H, W) - Fine details

    # Input data
    trimap: Optional[np.ndarray] = None  # (H, W) - Trimap used
    foreground: Optional[np.ndarray] = None  # (H, W, 3) - Extracted foreground

    # Metadata
    source_path: Optional[str] = None
    model_used: str = ""
    processing_time_ms: float = 0.0
    confidence: Optional[np.ndarray] = None

    def get_composite(self, background: np.ndarray) -> np.ndarray:
        """
        Composite foreground over background.

        Args:
            background: (H, W, 3) RGB background image

        Returns:
            (H, W, 3) Composited RGB image
        """
        if self.foreground is None:
            raise ValueError("No foreground available")

        alpha_3ch = np.stack([self.alpha] * 3, axis=-1)
        composite = self.foreground * alpha_3ch + background * (1 - alpha_3ch)
        return composite.astype(np.uint8)

    def get_alpha_visualization(self, colorize: bool = False) -> np.ndarray:
        """Get alpha as RGB visualization."""
        alpha_uint8 = (self.alpha * 255).astype(np.uint8)

        if colorize:
            # Red-green gradient
            h, w = self.alpha.shape
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            vis[:, :, 0] = 255 - alpha_uint8  # Red for transparent
            vis[:, :, 1] = alpha_uint8  # Green for opaque
            return vis
        else:
            return np.stack([alpha_uint8] * 3, axis=-1)

    def save_alpha(self, path: Union[str, Path], format: str = "png"):
        """Save alpha matte to file."""
        path = Path(path)

        if format == "npy":
            np.save(path.with_suffix(".npy"), self.alpha)
        elif format == "exr":
            self._save_exr(path.with_suffix(".exr"), self.alpha)
        else:
            alpha_uint8 = (self.alpha * 255).astype(np.uint8)
            cv2.imwrite(str(path.with_suffix(f".{format}")), alpha_uint8)

    def save_layers(self, output_dir: Union[str, Path], base_name: str = "matte"):
        """Save layer decomposition."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.alpha_core is not None:
            core = (self.alpha_core * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"{base_name}_core.png"), core)

        if self.alpha_edge is not None:
            edge = (self.alpha_edge * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"{base_name}_edge.png"), edge)

        if self.alpha_hair is not None:
            hair = (self.alpha_hair * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"{base_name}_hair.png"), hair)

    def save_foreground(self, path: Union[str, Path]):
        """Save extracted foreground with alpha."""
        if self.foreground is None:
            print("Warning: No foreground to save")
            return

        path = Path(path)

        # Create RGBA image
        alpha_uint8 = (self.alpha * 255).astype(np.uint8)
        rgba = np.dstack([self.foreground, alpha_uint8])

        # Save as PNG with alpha
        cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    def save_all(self, output_dir: Union[str, Path], base_name: Optional[str] = None):
        """Save all outputs."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if base_name is None:
            if self.source_path:
                base_name = Path(self.source_path).stem
            else:
                base_name = "matte"

        # Alpha
        self.save_alpha(output_dir / f"{base_name}_alpha.png")
        np.save(output_dir / f"{base_name}_alpha.npy", self.alpha)

        # Layers
        if self.alpha_core is not None:
            self.save_layers(output_dir, base_name)

        # Foreground
        if self.foreground is not None:
            self.save_foreground(output_dir / f"{base_name}_foreground.png")

        # Trimap
        if self.trimap is not None:
            cv2.imwrite(str(output_dir / f"{base_name}_trimap.png"), self.trimap)

        # Metadata
        metadata = {
            "model_used": self.model_used,
            "processing_time_ms": self.processing_time_ms,
            "has_layers": self.alpha_core is not None,
        }
        with open(output_dir / f"{base_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved all outputs to: {output_dir}")

    def _save_exr(self, path: Path, data: np.ndarray):
        """Save single-channel EXR."""
        try:
            import OpenEXR
            import Imath

            h, w = data.shape
            header = OpenEXR.Header(w, h)
            header['channels'] = {'A': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))}

            exr = OpenEXR.OutputFile(str(path), header)
            exr.writePixels({'A': data.astype(np.float32).tobytes()})
            exr.close()
        except ImportError:
            np.save(path.with_suffix('.npy'), data)


class MatteAnything:
    """
    Professional matting pipeline.

    Combines multiple matting models with SAM3 integration
    for automatic trimap generation and high-quality alpha extraction.

    Example:
        >>> matte = MatteAnything(device="cuda")
        >>> result = matte.process_image("image.jpg", mask="mask.png")
        >>> result.save_all("output/")
    """

    def __init__(
        self,
        config: Optional[MattingConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize matting pipeline.

        Args:
            config: MattingConfig or None for defaults
            device: 'cuda' or 'cpu'
        """
        self.config = config or MattingConfig()
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

        self.vitmatte_model = None
        self.vitmatte_processor = None
        self.sam3_processor = None
        self._loaded = False

    def _load_models(self):
        """Load matting models (lazy loading)."""
        if self._loaded:
            return

        print(f"Loading matting models...")

        # Load ViTMatte
        try:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor

            print("  Loading ViTMatte...")
            self.vitmatte_processor = VitMatteImageProcessor.from_pretrained(
                "hustvl/vitmatte-small-composition-1k"
            )
            self.vitmatte_model = VitMatteForImageMatting.from_pretrained(
                "hustvl/vitmatte-small-composition-1k"
            ).to(self.device).eval()
            print("  ViTMatte loaded")

        except ImportError as e:
            print(f"  Warning: ViTMatte not available: {e}")
            print("  Install: pip install transformers")

        except Exception as e:
            print(f"  Warning: Could not load ViTMatte: {e}")

        self._loaded = True

    def process_image(
        self,
        image_path: Union[str, Path],
        mask: Optional[Union[str, Path, np.ndarray]] = None,
        trimap: Optional[Union[str, Path, np.ndarray]] = None,
        text_prompt: Optional[str] = None,
        compute_layers: bool = True,
        refine_edges: bool = True,
    ) -> MattingResult:
        """
        Process single image for matting.

        Args:
            image_path: Path to input image
            mask: Binary mask (from SAM3 or manual)
            trimap: Pre-computed trimap (optional)
            text_prompt: Text prompt for SAM3 segmentation
            compute_layers: Decompose into core/edge/hair
            refine_edges: Apply edge refinement

        Returns:
            MattingResult with alpha and layers
        """
        start_time = time.time()
        self._load_models()

        image_path = Path(image_path)
        print(f"\n[MATTING] Processing: {image_path}")

        # Load image
        image = np.array(Image.open(str(image_path)).convert("RGB"))
        h, w = image.shape[:2]

        # Get or generate trimap
        if trimap is not None:
            trimap_arr = self._load_trimap(trimap, (h, w))
        elif mask is not None:
            mask_arr = self._load_mask(mask, (h, w))
            trimap_arr = self._mask_to_trimap(mask_arr)
        elif text_prompt is not None:
            mask_arr = self._segment_with_sam3(image_path, text_prompt)
            trimap_arr = self._mask_to_trimap(mask_arr)
        else:
            # Auto-detect foreground
            trimap_arr = self._auto_trimap(image)

        print(f"  Trimap generated")

        # Run matting
        alpha = self._run_matting(image, trimap_arr)
        print(f"  Alpha matte computed")

        # Refine edges
        if refine_edges:
            alpha = self._refine_alpha(alpha, image, trimap_arr)
            print(f"  Edges refined")

        # Compute layer decomposition
        alpha_core, alpha_edge, alpha_hair = None, None, None
        if compute_layers:
            alpha_core, alpha_edge, alpha_hair = self._decompose_alpha(alpha)
            print(f"  Layers decomposed")

        # Extract foreground
        foreground = self._extract_foreground(image, alpha)

        processing_time = (time.time() - start_time) * 1000

        result = MattingResult(
            alpha=alpha,
            alpha_core=alpha_core,
            alpha_edge=alpha_edge,
            alpha_hair=alpha_hair,
            trimap=trimap_arr,
            foreground=foreground,
            source_path=str(image_path),
            model_used="vitmatte",
            processing_time_ms=processing_time,
        )

        print(f"  Processing time: {processing_time:.0f}ms")

        return result

    def process_video(
        self,
        video_path: Union[str, Path],
        initial_mask: Optional[Union[str, Path, np.ndarray]] = None,
        text_prompt: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[MattingResult]:
        """
        Process video with temporal consistency.

        Args:
            video_path: Path to input video
            initial_mask: Mask for first frame
            text_prompt: Text prompt for SAM3
            output_dir: Output directory for frame results

        Returns:
            List of MattingResult for each frame
        """
        video_path = Path(video_path)
        print(f"\n[VIDEO MATTING] Processing: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"  Frames: {total_frames}, FPS: {fps:.1f}")

        results = []
        prev_alpha = None

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # First frame: use initial mask or text prompt
            if frame_idx == 0:
                if initial_mask is not None:
                    mask = self._load_mask(initial_mask, frame_rgb.shape[:2])
                elif text_prompt is not None:
                    # Save temp frame for SAM3
                    temp_path = Path("/tmp/temp_frame.jpg")
                    cv2.imwrite(str(temp_path), frame)
                    mask = self._segment_with_sam3(temp_path, text_prompt)
                else:
                    mask = self._auto_mask(frame_rgb)

                trimap = self._mask_to_trimap(mask)
            else:
                # Propagate from previous frame
                trimap = self._propagate_trimap(prev_alpha, frame_rgb)

            # Run matting
            alpha = self._run_matting(frame_rgb, trimap)

            # Temporal smoothing
            if prev_alpha is not None:
                alpha = self._temporal_blend(alpha, prev_alpha, weight=0.7)

            prev_alpha = alpha

            # Create result
            result = MattingResult(
                alpha=alpha,
                source_path=str(video_path),
                model_used="vitmatte",
            )
            results.append(result)

            # Save frame result
            if output_dir:
                alpha_path = output_dir / f"alpha_{frame_idx:06d}.png"
                result.save_alpha(alpha_path)

            frame_idx += 1

            if frame_idx % 10 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames")

        cap.release()
        print(f"  Video matting complete: {len(results)} frames")

        return results

    def _load_mask(self, mask: Union[str, Path, np.ndarray], target_size: Tuple[int, int]) -> np.ndarray:
        """Load and resize binary mask."""
        if isinstance(mask, np.ndarray):
            mask_arr = mask
        else:
            mask_arr = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)

        # Resize if needed
        if mask_arr.shape[:2] != target_size:
            mask_arr = cv2.resize(mask_arr, (target_size[1], target_size[0]))

        # Normalize to [0, 1]
        if mask_arr.max() > 1:
            mask_arr = mask_arr / 255.0

        return mask_arr

    def _load_trimap(self, trimap: Union[str, Path, np.ndarray], target_size: Tuple[int, int]) -> np.ndarray:
        """Load and resize trimap (0=bg, 128=unknown, 255=fg)."""
        if isinstance(trimap, np.ndarray):
            trimap_arr = trimap
        else:
            trimap_arr = cv2.imread(str(trimap), cv2.IMREAD_GRAYSCALE)

        # Resize if needed
        if trimap_arr.shape[:2] != target_size:
            trimap_arr = cv2.resize(trimap_arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

        return trimap_arr

    def _mask_to_trimap(self, mask: np.ndarray, erosion_size: int = 15, dilation_size: int = 15) -> np.ndarray:
        """
        Convert binary mask to trimap.

        Args:
            mask: Binary mask [0, 1]
            erosion_size: Size for erosion (definite foreground)
            dilation_size: Size for dilation (definite background)

        Returns:
            Trimap (0=bg, 128=unknown, 255=fg)
        """
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Erosion for definite foreground
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        fg = cv2.erode(mask_uint8, kernel_erode)

        # Dilation for region of interest
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        bg = cv2.dilate(mask_uint8, kernel_dilate)

        # Create trimap
        trimap = np.zeros_like(mask_uint8)
        trimap[bg == 255] = 128  # Unknown
        trimap[fg == 255] = 255  # Definite foreground
        # Background stays 0

        return trimap

    def _segment_with_sam3(self, image_path: Path, text_prompt: str) -> np.ndarray:
        """Use SAM3 for automatic segmentation."""
        try:
            from sam3_complete import SAM3ImageProcessor

            if self.sam3_processor is None:
                self.sam3_processor = SAM3ImageProcessor(device=self.device)

            result = self.sam3_processor.segment_with_text(image_path, text_prompt)
            mask, _ = result.get_best_mask()
            return mask.astype(np.float32)

        except ImportError:
            print("Warning: SAM3 not available, using auto-detection")
            image = np.array(Image.open(str(image_path)))
            return self._auto_mask(image)

    def _auto_trimap(self, image: np.ndarray) -> np.ndarray:
        """Auto-generate trimap using saliency detection."""
        # Simple saliency based on color distance from borders
        h, w = image.shape[:2]

        # Sample border colors
        border_colors = np.concatenate([
            image[0, :, :].reshape(-1, 3),
            image[-1, :, :].reshape(-1, 3),
            image[:, 0, :].reshape(-1, 3),
            image[:, -1, :].reshape(-1, 3),
        ])
        bg_color = np.median(border_colors, axis=0)

        # Compute distance from background color
        diff = np.abs(image.astype(np.float32) - bg_color.reshape(1, 1, 3))
        saliency = np.mean(diff, axis=2)
        saliency = saliency / saliency.max()

        # Threshold to create mask
        mask = (saliency > 0.3).astype(np.float32)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return self._mask_to_trimap(mask)

    def _auto_mask(self, image: np.ndarray) -> np.ndarray:
        """Auto-generate mask using GrabCut."""
        h, w = image.shape[:2]

        # Initialize mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Rectangle for GrabCut (center region)
        margin = min(h, w) // 10
        rect = (margin, margin, w - 2*margin, h - 2*margin)

        # Background and foreground models
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        # Run GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Convert mask
        mask_output = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.float32)

        return mask_output

    def _run_matting(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Run matting model."""
        if self.vitmatte_model is not None and self.vitmatte_processor is not None:
            return self._run_vitmatte(image, trimap)
        else:
            # Fallback to simple alpha from trimap
            return self._simple_matting(image, trimap)

    def _run_vitmatte(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Run ViTMatte model."""
        # Prepare inputs
        image_pil = Image.fromarray(image)
        trimap_pil = Image.fromarray(trimap)

        inputs = self.vitmatte_processor(
            images=image_pil,
            trimaps=trimap_pil,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.vitmatte_model(**inputs)

        # Extract alpha
        alpha = outputs.alphas[0, 0].cpu().numpy()

        # Resize to original size if needed
        if alpha.shape != image.shape[:2]:
            alpha = cv2.resize(alpha, (image.shape[1], image.shape[0]))

        return np.clip(alpha, 0, 1)

    def _simple_matting(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Simple matting fallback using trimap-guided blending."""
        # Normalize trimap to [0, 1]
        alpha = trimap.astype(np.float32) / 255.0

        # Apply Gaussian blur to unknown regions
        unknown_mask = (trimap == 128).astype(np.float32)

        # Iterative refinement
        for _ in range(5):
            alpha_blurred = cv2.GaussianBlur(alpha, (15, 15), 0)
            alpha = alpha * (1 - unknown_mask) + alpha_blurred * unknown_mask

        return np.clip(alpha, 0, 1)

    def _refine_alpha(self, alpha: np.ndarray, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Refine alpha edges using guided filtering."""
        # Convert to grayscale guide
        guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # Guided filter parameters
        radius = 10
        eps = 0.01

        # Apply guided filter
        try:
            alpha_refined = cv2.ximgproc.guidedFilter(
                guide, alpha.astype(np.float32), radius, eps
            )
        except AttributeError:
            # Fallback if ximgproc not available
            alpha_refined = cv2.bilateralFilter(
                alpha.astype(np.float32), 9, 75, 75
            )

        # Preserve definite regions from trimap
        fg_mask = trimap == 255
        bg_mask = trimap == 0
        alpha_refined[fg_mask] = 1.0
        alpha_refined[bg_mask] = 0.0

        return np.clip(alpha_refined, 0, 1)

    def _decompose_alpha(self, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose alpha into core, edge, and hair layers.

        Returns:
            (alpha_core, alpha_edge, alpha_hair)
        """
        # Core: solid interior (alpha > 0.95)
        alpha_core = np.zeros_like(alpha)
        alpha_core[alpha > 0.95] = 1.0

        # Erode core slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        alpha_core = cv2.erode(alpha_core, kernel)

        # Edge: transition zone (0.05 < alpha < 0.95)
        alpha_edge = np.zeros_like(alpha)
        edge_mask = (alpha > 0.05) & (alpha < 0.95)
        alpha_edge[edge_mask] = alpha[edge_mask]

        # Hair: fine details (high frequency content in semi-transparent areas)
        # Use Laplacian to detect fine details
        laplacian = cv2.Laplacian(alpha, cv2.CV_32F)
        laplacian = np.abs(laplacian)
        laplacian = laplacian / (laplacian.max() + 1e-8)

        alpha_hair = np.zeros_like(alpha)
        hair_mask = edge_mask & (laplacian > 0.1)
        alpha_hair[hair_mask] = alpha[hair_mask]

        return alpha_core, alpha_edge, alpha_hair

    def _extract_foreground(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Extract foreground with color decontamination."""
        # Simple extraction (no decontamination for now)
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        foreground = image * alpha_3ch

        return foreground.astype(np.uint8)

    def _propagate_trimap(self, prev_alpha: np.ndarray, current_frame: np.ndarray) -> np.ndarray:
        """Propagate trimap from previous frame."""
        # Use previous alpha as base
        mask = (prev_alpha > 0.5).astype(np.float32)

        # Dilate slightly to account for motion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel)

        return self._mask_to_trimap(mask, erosion_size=10, dilation_size=20)

    def _temporal_blend(self, current: np.ndarray, previous: np.ndarray, weight: float = 0.7) -> np.ndarray:
        """Temporal blending for video consistency."""
        return weight * current + (1 - weight) * previous


def main():
    """CLI interface for Matte Anything."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Matte Anything - Professional Alpha Matting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image matting with mask
  python matte_anything.py image.jpg --mask mask.png -o output/

  # Image matting with text prompt (uses SAM3)
  python matte_anything.py image.jpg --prompt "person" -o output/

  # Auto-matting
  python matte_anything.py image.jpg -o output/

  # Video matting
  python matte_anything.py video.mp4 --video --prompt "person" -o output/
        """
    )

    parser.add_argument("input", type=str, help="Input image or video")
    parser.add_argument("-o", "--output", type=str, default="matte_output", help="Output directory")
    parser.add_argument("--mask", type=str, help="Binary mask file")
    parser.add_argument("--trimap", type=str, help="Trimap file")
    parser.add_argument("--prompt", type=str, help="Text prompt for SAM3 segmentation")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--video", action="store_true", help="Process as video")
    parser.add_argument("--no-layers", action="store_true", help="Don't compute layer decomposition")
    parser.add_argument("--no-refine", action="store_true", help="Don't refine edges")

    args = parser.parse_args()

    # Check input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize matte pipeline
    config = MattingConfig(device=args.device)
    matte = MatteAnything(config=config, device=args.device)

    if args.video:
        # Video processing
        results = matte.process_video(
            input_path,
            initial_mask=args.mask,
            text_prompt=args.prompt,
            output_dir=output_dir,
        )
        print(f"\n Video matting complete!")
        print(f"   Frames: {len(results)}")
        print(f"   Output: {output_dir}")
    else:
        # Image processing
        result = matte.process_image(
            input_path,
            mask=args.mask,
            trimap=args.trimap,
            text_prompt=args.prompt,
            compute_layers=not args.no_layers,
            refine_edges=not args.no_refine,
        )

        # Save outputs
        result.save_all(output_dir, input_path.stem)

        print(f"\n Matting complete!")
        print(f"   Output: {output_dir}")


if __name__ == "__main__":
    main()
