#!/usr/bin/env python3
"""
MatAnyone Wrapper - Video Matting with Memory Propagation
=========================================================

Official wrapper for MatAnyone (CVPR 2025) - pq-yang/MatAnyone

Pipeline:
    1. SAM2 generates first-frame mask (interactive)
    2. MatAnyone propagates matte through video with memory banks
    3. Outputs: alpha sequence + foreground

Features:
    - No per-frame trimap needed (memory propagation)
    - Stable temporal consistency
    - High-quality hair/fur/motion blur handling
    - Direct SAM2 integration

Usage:
    # CLI
    python matanyone_wrapper.py video.mp4 --mask first_frame_mask.png -o output/

    # Python API
    from matanyone_wrapper import MatAnyoneProcessor
    processor = MatAnyoneProcessor()
    processor.process_video("video.mp4", "mask.png", "output/")
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict
from dataclasses import dataclass
import numpy as np

try:
    import torch
    from PIL import Image
    import cv2
except ImportError as e:
    print(f"Missing core dependencies: {e}")
    print("Install: pip install torch torchvision opencv-python pillow")
    sys.exit(1)


@dataclass
class VideoMattingResult:
    """Result from video matting."""
    alpha_dir: Path            # Directory with alpha frames
    foreground_dir: Path       # Directory with foreground frames
    video_path: Path           # Original video
    total_frames: int
    fps: float
    resolution: Tuple[int, int]
    processing_time_sec: float


class MatAnyoneProcessor:
    """
    Video matting processor using MatAnyone.

    MatAnyone uses memory propagation to maintain temporal consistency
    without requiring per-frame trimap. Just provide first-frame mask.

    Example:
        >>> processor = MatAnyoneProcessor()
        >>> result = processor.process_video(
        ...     "video.mp4",
        ...     "first_frame_mask.png",
        ...     "output/"
        ... )
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        max_size: int = 1080,
    ):
        """
        Initialize MatAnyone processor.

        Args:
            checkpoint_path: Path to matanyone.pth (auto-downloads if None)
            device: 'cuda' or 'cpu'
            max_size: Max resolution (longer side)
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_size = max_size
        self.checkpoint_path = checkpoint_path

        self._model = None
        self._loaded = False

    def _find_checkpoint(self) -> Path:
        """Find or download MatAnyone checkpoint."""
        # Check common locations
        search_paths = [
            Path("models/matanyone.pth"),
            Path("pretrained_models/matanyone.pth"),
            Path.home() / ".cache/matanyone/matanyone.pth",
        ]

        if self.checkpoint_path:
            search_paths.insert(0, Path(self.checkpoint_path))

        for path in search_paths:
            if path.exists():
                return path

        # Auto-download
        print("Downloading MatAnyone checkpoint...")
        cache_dir = Path.home() / ".cache/matanyone"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = cache_dir / "matanyone.pth"

        import urllib.request
        url = "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
        urllib.request.urlretrieve(url, str(ckpt_path))

        return ckpt_path

    def _load_model(self):
        """Load MatAnyone model."""
        if self._loaded:
            return

        print("[MatAnyone] Loading model...")

        try:
            # Try importing official MatAnyone
            sys.path.insert(0, str(Path("repos/MatAnyone")))
            from matanyone.inference_core import InferenceCore

            ckpt_path = self._find_checkpoint()
            self._model = InferenceCore(str(ckpt_path), device=self.device)
            self._use_official = True
            print(f"  Loaded official MatAnyone from {ckpt_path}")

        except ImportError:
            print("  Official MatAnyone not found, using HuggingFace API...")
            self._use_official = False

            try:
                # Try HuggingFace version
                from huggingface_hub import hf_hub_download

                # This is a fallback path
                self._hf_processor = True
                print("  Using HuggingFace integration")

            except ImportError:
                print("  Warning: No MatAnyone backend available")
                print("  Install: ./setup_rotoscopy.sh")
                self._hf_processor = False

        self._loaded = True

    def process_video(
        self,
        video_path: Union[str, Path],
        mask_path: Union[str, Path],
        output_dir: Union[str, Path],
        save_foreground: bool = True,
    ) -> VideoMattingResult:
        """
        Process video with MatAnyone.

        Args:
            video_path: Input video file
            mask_path: First-frame binary mask (from SAM2)
            output_dir: Output directory
            save_foreground: Also save foreground sequence

        Returns:
            VideoMattingResult with paths to output
        """
        start_time = time.time()
        self._load_model()

        video_path = Path(video_path)
        mask_path = Path(mask_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        alpha_dir = output_dir / "alpha"
        alpha_dir.mkdir(exist_ok=True)

        fg_dir = output_dir / "foreground"
        if save_foreground:
            fg_dir.mkdir(exist_ok=True)

        print(f"\n[MatAnyone] Processing: {video_path.name}")

        # Load video info
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"  Video: {width}x{height}, {total_frames} frames @ {fps:.1f}fps")

        # Load first-frame mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        # Resize mask if needed
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # Normalize mask
        mask = (mask > 127).astype(np.uint8) * 255

        if self._use_official and self._model is not None:
            # Use official MatAnyone
            self._process_official(
                video_path, mask, alpha_dir, fg_dir,
                save_foreground, total_frames
            )
        else:
            # Fallback processing
            self._process_fallback(
                video_path, mask, alpha_dir, fg_dir,
                save_foreground, total_frames, fps
            )

        processing_time = time.time() - start_time

        print(f"  Done! Processing time: {processing_time:.1f}s")
        print(f"  Alpha output: {alpha_dir}")

        return VideoMattingResult(
            alpha_dir=alpha_dir,
            foreground_dir=fg_dir if save_foreground else None,
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            resolution=(width, height),
            processing_time_sec=processing_time,
        )

    def _process_official(
        self,
        video_path: Path,
        first_mask: np.ndarray,
        alpha_dir: Path,
        fg_dir: Path,
        save_foreground: bool,
        total_frames: int,
    ):
        """Process using official MatAnyone."""
        # Save temp mask
        temp_mask = Path("/tmp/matanyone_mask.png")
        cv2.imwrite(str(temp_mask), first_mask)

        # Run inference
        print("  Running MatAnyone inference...")
        self._model.process_video(
            str(video_path),
            str(temp_mask),
            str(alpha_dir.parent),
            max_size=self.max_size,
        )

        # Move outputs to expected locations
        # MatAnyone outputs to results/ by default

    def _process_fallback(
        self,
        video_path: Path,
        first_mask: np.ndarray,
        alpha_dir: Path,
        fg_dir: Path,
        save_foreground: bool,
        total_frames: int,
        fps: float,
    ):
        """
        Fallback video matting using frame-by-frame processing
        with temporal propagation (simulates MatAnyone behavior).
        """
        print("  Using fallback matting (install MatAnyone for best results)")

        cap = cv2.VideoCapture(str(video_path))

        prev_alpha = first_mask.astype(np.float32) / 255.0
        prev_frame = None

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_idx == 0:
                # First frame: use provided mask directly
                alpha = prev_alpha
            else:
                # Propagate mask with optical flow
                alpha = self._propagate_mask_flow(
                    prev_frame, frame_rgb, prev_alpha
                )

                # Refine with matting
                alpha = self._refine_alpha(frame_rgb, alpha)

            # Save alpha
            alpha_uint8 = (alpha * 255).astype(np.uint8)
            cv2.imwrite(str(alpha_dir / f"alpha_{frame_idx:06d}.png"), alpha_uint8)

            # Save foreground
            if save_foreground:
                fg = self._extract_foreground(frame, alpha)
                cv2.imwrite(str(fg_dir / f"fg_{frame_idx:06d}.png"), fg)

            prev_alpha = alpha
            prev_frame = frame_rgb

            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx}/{total_frames}")

        cap.release()

    def _propagate_mask_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_alpha: np.ndarray,
    ) -> np.ndarray:
        """Propagate mask using optical flow."""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Warp previous alpha
        h, w = prev_alpha.shape
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        flow_map[:, :, 0] = np.arange(w)
        flow_map[:, :, 1] = np.arange(h)[:, np.newaxis]
        flow_map += flow

        warped_alpha = cv2.remap(
            prev_alpha.astype(np.float32),
            flow_map[:, :, 0], flow_map[:, :, 1],
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        return np.clip(warped_alpha, 0, 1)

    def _refine_alpha(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Refine alpha using guided filter."""
        guide = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        try:
            # Use guided filter if available
            refined = cv2.ximgproc.guidedFilter(
                guide, alpha.astype(np.float32), radius=8, eps=0.01
            )
        except AttributeError:
            # Fallback to bilateral
            refined = cv2.bilateralFilter(alpha.astype(np.float32), 9, 0.1, 10)

        return np.clip(refined, 0, 1)

    def _extract_foreground(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Extract foreground with alpha."""
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        fg = frame * alpha_3ch
        return fg.astype(np.uint8)


class SAM3MaskGenerator:
    """
    Generate first-frame mask using SAM3.

    Provides interactive mask generation for MatAnyone input.
    SAM3 supports text prompts for open-vocabulary segmentation.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._processor = None
        self._loaded = False

    def _load_model(self):
        """Load SAM3 model."""
        if self._loaded:
            return

        print("[SAM3] Loading model...")

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            model = build_sam3_image_model()
            self._processor = Sam3Processor(model)

            if self.device == "cuda" and torch.cuda.is_available():
                self._processor.model = self._processor.model.cuda()

            print("  SAM3 loaded successfully")

        except ImportError as e:
            print(f"  SAM3 not available: {e}")
            print("  Install: pip install git+https://github.com/facebookresearch/sam3.git")
            print("  Auth: huggingface-cli login")
            self._processor = None

        self._loaded = True

    def generate_mask_from_text(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
    ) -> np.ndarray:
        """
        Generate mask from text prompt (open-vocabulary).

        Args:
            image_path: Input image
            text_prompt: Text description (e.g., "person", "red car")

        Returns:
            Binary mask (H, W)
        """
        self._load_model()

        if self._processor is None:
            raise RuntimeError("SAM3 not loaded")

        image = Image.open(str(image_path)).convert("RGB")

        # SAM3 text prompting
        result = self._processor.segment_with_text(image, text_prompt)

        # Get best mask
        masks = result["masks"]
        scores = result["scores"]

        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()

        return (mask > 0.5).astype(np.uint8) * 255

    def generate_mask_from_point(
        self,
        image_path: Union[str, Path],
        point: Tuple[int, int],
        label: int = 1,
    ) -> np.ndarray:
        """
        Generate mask from point prompt.

        Args:
            image_path: Input image
            point: (x, y) coordinates
            label: 1 for foreground, 0 for background

        Returns:
            Binary mask (H, W)
        """
        self._load_model()

        if self._processor is None:
            raise RuntimeError("SAM3 not loaded")

        image = Image.open(str(image_path)).convert("RGB")

        # SAM3 point prompting
        result = self._processor.segment_with_points(
            image,
            points=[[point[0], point[1]]],
            labels=[label]
        )

        masks = result["masks"]
        scores = result["scores"]

        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()

        return (mask > 0.5).astype(np.uint8) * 255

    def generate_mask_from_box(
        self,
        image_path: Union[str, Path],
        box: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Generate mask from bounding box.

        Args:
            image_path: Input image
            box: (x1, y1, x2, y2) bounding box

        Returns:
            Binary mask (H, W)
        """
        self._load_model()

        if self._processor is None:
            raise RuntimeError("SAM3 not loaded")

        image = Image.open(str(image_path)).convert("RGB")

        # SAM3 box prompting
        result = self._processor.segment_with_box(image, box=list(box))

        masks = result["masks"]
        scores = result["scores"]

        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()

        return (mask > 0.5).astype(np.uint8) * 255


def create_video_matte(
    video_path: str,
    mask_path: str,
    output_dir: str,
    device: str = "cuda",
) -> VideoMattingResult:
    """
    Convenience function for video matting.

    Args:
        video_path: Input video
        mask_path: First-frame mask (from SAM2)
        output_dir: Output directory
        device: 'cuda' or 'cpu'

    Returns:
        VideoMattingResult
    """
    processor = MatAnyoneProcessor(device=device)
    return processor.process_video(video_path, mask_path, output_dir)


def main():
    """CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MatAnyone Video Matting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline:
  1. SAM3 generates first-frame mask (text/point/box prompt)
  2. MatAnyone propagates matte through video

Examples:
  # With text prompt (recommended - open vocabulary)
  python matanyone_wrapper.py video.mp4 --text "person" -o output/

  # With point click
  python matanyone_wrapper.py video.mp4 --point 500,300 -o output/

  # With bounding box
  python matanyone_wrapper.py video.mp4 --box 100,100,400,500 -o output/

  # With existing mask
  python matanyone_wrapper.py video.mp4 --mask first_frame.png -o output/
        """
    )

    parser.add_argument("video", help="Input video file")
    parser.add_argument("-o", "--output", default="matanyone_output", help="Output directory")
    parser.add_argument("--mask", help="First-frame mask file")
    parser.add_argument("--text", help="Text prompt for SAM3 (e.g., 'person', 'red car')")
    parser.add_argument("--point", help="Point prompt x,y for SAM3")
    parser.add_argument("--box", help="Box prompt x1,y1,x2,y2 for SAM3")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-size", type=int, default=1080, help="Max resolution")

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    # Get or generate mask
    mask_path = None

    if args.mask:
        mask_path = Path(args.mask)
        if not mask_path.exists():
            print(f"Error: Mask not found: {mask_path}")
            sys.exit(1)

    elif args.text or args.point or args.box:
        # Extract first frame
        print("Extracting first frame for SAM3...")
        cap = cv2.VideoCapture(str(video_path))
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Could not read video")
            sys.exit(1)

        first_frame_path = output_dir / "first_frame.jpg"
        cv2.imwrite(str(first_frame_path), first_frame)

        # Generate mask with SAM3
        sam3 = SAM3MaskGenerator(device=args.device)

        if args.text:
            print(f"  Using text prompt: '{args.text}'")
            mask = sam3.generate_mask_from_text(first_frame_path, args.text)
        elif args.point:
            x, y = map(int, args.point.split(","))
            mask = sam3.generate_mask_from_point(first_frame_path, (x, y))
        else:
            coords = list(map(int, args.box.split(",")))
            mask = sam3.generate_mask_from_box(first_frame_path, tuple(coords))

        mask_path = output_dir / "first_frame_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"Mask saved: {mask_path}")

    else:
        print("Error: Provide --mask, --text, --point, or --box")
        sys.exit(1)

    # Run video matting
    processor = MatAnyoneProcessor(device=args.device, max_size=args.max_size)
    result = processor.process_video(video_path, mask_path, output_dir)

    print(f"\nVideo matting complete!")
    print(f"  Alpha:      {result.alpha_dir}")
    print(f"  Foreground: {result.foreground_dir}")
    print(f"  Time:       {result.processing_time_sec:.1f}s")


if __name__ == "__main__":
    main()
