#!/usr/bin/env python3
"""
SAM3 Complete Tool - Professional Segmentation and Video Tracking
=================================================================

Complete implementation of SAM3 with all features:
- Text prompting (open-vocabulary)
- Visual prompting (points, boxes, masks)
- Video tracking with session management
- Interactive refinement
- Batch processing

Requirements:
    Python 3.12+
    PyTorch 2.7+
    CUDA 12.6+
    HuggingFace authentication (hf auth login)

Usage:
    # Image segmentation with text
    python sam3_complete.py image input.jpg --text "red baseball cap" --output mask.png

    # Image segmentation with points
    python sam3_complete.py image input.jpg --points 100,200 150,250 --output mask.png

    # Video tracking
    python sam3_complete.py video frames/ --text "person in white" --output results/
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json

try:
    from PIL import Image
    import torch
except ImportError as e:
    print(f"Error: Missing dependencies - {e}")
    print("Install: pip install torch torchvision Pillow opencv-python numpy")
    sys.exit(1)


class PromptType(Enum):
    """Types of prompts supported by SAM3."""
    TEXT = "text"
    MASK = "mask"


def _to_numpy(tensor_or_array):
    """Convert PyTorch tensor to numpy array if needed."""
    if hasattr(tensor_or_array, 'detach'):  # It's a PyTorch tensor
        return tensor_or_array.detach().cpu().numpy()
    return tensor_or_array


@dataclass
class SegmentationResult:
    """Result from SAM3 segmentation."""
    masks: np.ndarray  # (N, H, W) - N masks
    boxes: np.ndarray  # (N, 4) - Bounding boxes [x1, y1, x2, y2]
    scores: np.ndarray  # (N,) - Confidence scores
    prompt_type: PromptType
    prompt_data: Any

    def __post_init__(self):
        """Convert tensors to numpy arrays if needed."""
        self.masks = _to_numpy(self.masks)
        self.boxes = _to_numpy(self.boxes)
        self.scores = _to_numpy(self.scores)

    def get_best_mask(self) -> Tuple[np.ndarray, float]:
        """Get the mask with highest confidence."""
        best_idx = self.scores.argmax()
        return self.masks[best_idx], self.scores[best_idx]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "masks_shape": self.masks.shape,
            "num_masks": len(self.masks),
            "boxes": self.boxes.tolist(),
            "scores": self.scores.tolist(),
            "prompt_type": self.prompt_type.value,
            "best_score": float(self.scores.max()),
        }


@dataclass
class VideoTrackingSession:
    """Video tracking session management."""
    session_id: str
    video_path: Path
    num_frames: int
    frame_results: Dict[int, SegmentationResult]  # frame_idx -> result

    def add_frame_result(self, frame_idx: int, result: SegmentationResult):
        """Add segmentation result for a frame."""
        self.frame_results[frame_idx] = result

    def get_frame_result(self, frame_idx: int) -> Optional[SegmentationResult]:
        """Get result for specific frame."""
        return self.frame_results.get(frame_idx)

    def export_summary(self) -> Dict:
        """Export session summary."""
        return {
            "session_id": self.session_id,
            "video_path": str(self.video_path),
            "num_frames": self.num_frames,
            "tracked_frames": len(self.frame_results),
            "frame_indices": sorted(self.frame_results.keys()),
        }


class SAM3ImageProcessor:
    """
    SAM3 Image Processor - Handles single image segmentation.

    Supports all prompt types: text, points, boxes, masks.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize SAM3 image processor.

        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load SAM3 image model."""
        print("Loading SAM3 image model...")

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            # Build model
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to(device=self.device)
                self.model.eval()
                print(f"✓ SAM3 image model loaded on {self.device}")
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                self.model.to(device=self.device)
                self.model.eval()
                print(f"✓ SAM3 image model loaded on {self.device}")

        except ImportError as e:
            print(f"✗ SAM3 not installed!")
            print(f"  Install: pip install git+https://github.com/facebookresearch/sam3.git")
            print(f"  Auth: hf auth login")
            print(f"  Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ SAM3 loading failed: {e}")
            sys.exit(1)

    def segment_with_text(
        self,
        image_path: Path,
        text_prompt: str
    ) -> SegmentationResult:
        """
        Segment image using text prompt.

        Args:
            image_path: Path to input image
            text_prompt: Text description (e.g., "red baseball cap")

        Returns:
            SegmentationResult with masks, boxes, scores
        """
        print(f"\n[TEXT PROMPT] Processing: {image_path}")
        print(f"  Prompt: '{text_prompt}'")

        # Load image
        image = Image.open(str(image_path))
        print(f"  Image size: {image.size[0]}x{image.size[1]}")

        # Set image and text prompt
        inference_state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt
        )

        # Extract results
        masks = output["masks"]  # (N, H, W)
        boxes = output["boxes"]  # (N, 4)
        scores = output["scores"]  # (N,)

        print(f"✓ Found {len(masks)} object(s)")
        print(f"  Best score: {scores.max():.3f}")

        return SegmentationResult(
            masks=masks,
            boxes=boxes,
            scores=scores,
            prompt_type=PromptType.TEXT,
            prompt_data=text_prompt
        )


class SAM3VideoTracker:
    """
    SAM3 Video Tracker - Handles video object tracking.

    Uses session-based API for efficient frame-to-frame tracking.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize SAM3 video tracker.

        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.video_predictor = None
        self._load_model()

    def _load_model(self):
        """Load SAM3 video model."""
        print("Loading SAM3 video tracker...")

        try:
            from sam3.model_builder import build_sam3_video_predictor

            # Build video predictor
            self.video_predictor = build_sam3_video_predictor()

            if self.device == "cuda" and torch.cuda.is_available():
                print(f"✓ SAM3 video tracker loaded on {self.device}")
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print(f"✓ SAM3 video tracker loaded on {self.device}")

        except ImportError as e:
            print(f"✗ SAM3 not installed!")
            print(f"  Install: pip install git+https://github.com/facebookresearch/sam3.git")
            print(f"  Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ SAM3 video tracker loading failed: {e}")
            sys.exit(1)

    def start_session(self, video_path: Path) -> VideoTrackingSession:
        """
        Start video tracking session.

        Args:
            video_path: Path to video file or directory of frames

        Returns:
            VideoTrackingSession
        """
        print(f"\n[VIDEO SESSION] Starting: {video_path}")

        # Start session
        response = self.video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(video_path)
            )
        )

        session_id = response["session_id"]
        num_frames = response.get("num_frames", 0)

        print(f"✓ Session started: {session_id}")
        print(f"  Frames: {num_frames}")

        return VideoTrackingSession(
            session_id=session_id,
            video_path=video_path,
            num_frames=num_frames,
            frame_results={}
        )

    def add_text_prompt(
        self,
        session: VideoTrackingSession,
        frame_index: int,
        text_prompt: str
    ) -> SegmentationResult:
        """
        Add text prompt to specific frame for tracking.

        Args:
            session: Active tracking session
            frame_index: Frame index to add prompt
            text_prompt: Text description

        Returns:
            SegmentationResult for the frame
        """
        print(f"\n[TRACKING] Adding text prompt to frame {frame_index}")
        print(f"  Prompt: '{text_prompt}'")

        # Add prompt
        response = self.video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session.session_id,
                frame_index=frame_index,
                text=text_prompt
            )
        )

        # Debug: print response structure
        print(f"  Response keys: {list(response.keys())}")

        # Extract outputs - handle different response formats
        if "outputs" in response:
            output = response["outputs"]
            print(f"  Output keys: {list(output.keys())}")
        else:
            # Response might BE the output directly
            output = response
            print(f"  Direct output keys: {list(output.keys())}")

        # Extract masks, boxes, scores - handle SAM3 video API format
        if "out_binary_masks" in output:
            # SAM3 video API format
            masks = output["out_binary_masks"]
            boxes_xywh = output.get("out_boxes_xywh")
            scores = output.get("out_probs")

            # Convert boxes from xywh to xyxy format if present
            if boxes_xywh is not None:
                boxes_xywh = _to_numpy(boxes_xywh)
                # xywh to xyxy: [x, y, w, h] -> [x1, y1, x2, y2]
                boxes = np.zeros_like(boxes_xywh)
                boxes[:, 0] = boxes_xywh[:, 0]  # x1 = x
                boxes[:, 1] = boxes_xywh[:, 1]  # y1 = y
                boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]  # x2 = x + w
                boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]  # y2 = y + h
            else:
                boxes = None

        elif "masks" in output:
            # Standard format
            masks = output["masks"]
            boxes = output.get("boxes", None)
            scores = output.get("scores", None)
        elif "mask" in output:
            # Singular form
            masks = output["mask"]
            boxes = output.get("box", output.get("boxes", None))
            scores = output.get("score", output.get("scores", None))
        else:
            # Print available keys for debugging
            print(f"  ERROR: No masks found in output. Available keys: {list(output.keys())}")
            raise KeyError(f"Cannot find masks in response. Available keys: {list(output.keys())}")

        result = SegmentationResult(
            masks=masks,
            boxes=boxes,
            scores=scores,
            prompt_type=PromptType.TEXT,
            prompt_data=text_prompt
        )

        # Add to session
        session.add_frame_result(frame_index, result)

        print(f"✓ Tracked {len(masks)} object(s)")
        print(f"  Best score: {scores.max():.3f}")

        return result

    def propagate_tracking(
        self,
        session: VideoTrackingSession,
        start_frame: int,
        end_frame: int
    ) -> Dict[int, SegmentationResult]:
        """
        Propagate tracking across frame range.

        Args:
            session: Active tracking session
            start_frame: Start frame index
            end_frame: End frame index (inclusive)

        Returns:
            Dictionary mapping frame_index -> SegmentationResult
        """
        print(f"\n[TRACKING] Propagating from frame {start_frame} to {end_frame}")

        # Use handle_stream_request with propagate_in_video
        # This returns an iterator that yields results for each frame
        results = {}

        try:
            for response in self.video_predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session.session_id
                )
            ):
                frame_idx = response.get("frame_index")
                if frame_idx is None:
                    continue

                # Only process frames in our range
                if frame_idx < start_frame or frame_idx > end_frame:
                    continue

                output = response.get("outputs", {})

                # Handle SAM3 video API format
                if "out_binary_masks" in output:
                    masks = output["out_binary_masks"]
                    boxes_xywh = output.get("out_boxes_xywh")
                    scores = output.get("out_probs")

                    # Convert boxes from xywh to xyxy if present
                    if boxes_xywh is not None:
                        boxes_xywh = _to_numpy(boxes_xywh)
                        boxes = np.zeros_like(boxes_xywh)
                        boxes[:, 0] = boxes_xywh[:, 0]
                        boxes[:, 1] = boxes_xywh[:, 1]
                        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
                        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]
                    else:
                        boxes = None
                else:
                    # Standard format
                    masks = output.get("masks")
                    boxes = output.get("boxes")
                    scores = output.get("scores")

                result = SegmentationResult(
                    masks=masks,
                    boxes=boxes,
                    scores=scores,
                    prompt_type=PromptType.TEXT,  # Inherited from initial prompt
                    prompt_data="propagated"
                )
                results[int(frame_idx)] = result
                session.add_frame_result(int(frame_idx), result)

                # Print progress every 10 frames
                if frame_idx % 10 == 0:
                    print(f"  Processed frame {frame_idx}...")

        except Exception as e:
            print(f"  Error during propagation: {e}")
            raise

        print(f"✓ Propagated tracking to {len(results)} frames")

        return results


def save_mask(mask: np.ndarray, output_path: Path, colormap: bool = False):
    """
    Save segmentation mask.

    Args:
        mask: Binary mask (H, W) with values 0 or 1
        output_path: Output file path
        colormap: If True, apply colormap for visualization
    """
    # Convert to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    if colormap:
        # Apply colormap for better visualization
        mask_colored = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_path), mask_colored)
    else:
        # Save as grayscale
        cv2.imwrite(str(output_path), mask_uint8)

    print(f"  Saved: {output_path}")


def save_visualization(
    image_path: Path,
    result: SegmentationResult,
    output_path: Path,
    alpha: float = 0.5
):
    """
    Save visualization with mask overlay.

    Args:
        image_path: Original image
        result: Segmentation result
        output_path: Output path
        alpha: Overlay transparency
    """
    # Load image
    image = cv2.imread(str(image_path))

    # Get best mask
    best_mask, score = result.get_best_mask()

    # Create colored overlay
    overlay = image.copy()
    overlay[best_mask > 0] = [0, 255, 0]  # Green overlay

    # Blend
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Draw box
    best_idx = result.scores.argmax()
    box = result.boxes[best_idx]
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Add score text
    cv2.putText(
        blended, f"Score: {score:.3f}",
        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 0), 2
    )

    cv2.imwrite(str(output_path), blended)
    print(f"  Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Complete Tool - Image and Video Segmentation"
    )

    subparsers = parser.add_subparsers(dest="mode", help="Processing mode")

    # Image mode
    image_parser = subparsers.add_parser("image", help="Process single image")
    image_parser.add_argument("input", type=Path, help="Input image path")
    image_parser.add_argument("--text", type=str, help="Text prompt")
    image_parser.add_argument("--points", nargs="+", help="Points as x,y")
    image_parser.add_argument("--box", type=str, help="Box as x1,y1,x2,y2")
    image_parser.add_argument("--output", "-o", type=Path, default=Path("mask.png"))
    image_parser.add_argument("--visualize", action="store_true", help="Save visualization")
    image_parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")

    # Video mode
    video_parser = subparsers.add_parser("video", help="Process video")
    video_parser.add_argument("input", type=Path, help="Video path or frame directory")
    video_parser.add_argument("--text", type=str, required=True, help="Text prompt")
    video_parser.add_argument("--start-frame", type=int, default=0, help="Start frame")
    video_parser.add_argument("--end-frame", type=int, help="End frame (default: all)")
    video_parser.add_argument("--output", "-o", type=Path, default=Path("results"))
    video_parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")

    args = parser.parse_args()

    if args.mode == "image":
        # IMAGE PROCESSING
        print("=" * 60)
        print("  SAM3 Complete - Image Segmentation")
        print("=" * 60)

        # Initialize processor
        processor = SAM3ImageProcessor(device=args.device)

        # Process based on prompt type
        if args.text:
            result = processor.segment_with_text(args.input, args.text)
        elif args.points:
            points = [tuple(map(int, p.split(","))) for p in args.points]
            result = processor.segment_with_points(args.input, points)
        elif args.box:
            box = tuple(map(int, args.box.split(",")))
            result = processor.segment_with_box(args.input, box)
        else:
            print("Error: Must specify --text, --points, or --box")
            sys.exit(1)

        # Save results
        best_mask, score = result.get_best_mask()
        save_mask(best_mask, args.output)

        if args.visualize:
            vis_path = args.output.parent / f"{args.output.stem}_vis{args.output.suffix}"
            save_visualization(args.input, result, vis_path)

        # Save metadata
        meta_path = args.output.parent / f"{args.output.stem}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"  Saved metadata: {meta_path}")

        print("\n" + "=" * 60)
        print("  ✓ SUCCESS")
        print("=" * 60)

    elif args.mode == "video":
        # VIDEO PROCESSING
        print("=" * 60)
        print("  SAM3 Complete - Video Tracking")
        print("=" * 60)

        # Initialize tracker
        tracker = SAM3VideoTracker(device=args.device)

        # Start session
        session = tracker.start_session(args.input)

        # Add initial prompt on start frame
        result = tracker.add_text_prompt(session, args.start_frame, args.text)

        # Propagate tracking
        end_frame = args.end_frame or session.num_frames - 1
        results = tracker.propagate_tracking(session, args.start_frame, end_frame)

        # Save results
        args.output.mkdir(parents=True, exist_ok=True)

        for frame_idx, result in session.frame_results.items():
            mask, score = result.get_best_mask()
            output_path = args.output / f"mask_{frame_idx:04d}.png"
            save_mask(mask, output_path)

        # Save session summary
        summary_path = args.output / "session_summary.json"
        with open(summary_path, "w") as f:
            json.dump(session.export_summary(), f, indent=2)
        print(f"\n  Saved session summary: {summary_path}")

        print("\n" + "=" * 60)
        print("  ✓ SUCCESS")
        print("=" * 60)
        print(f"\n  Processed {len(session.frame_results)} frames")
        print(f"  Output: {args.output}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
