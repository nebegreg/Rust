"""
Video Pipeline for Ultimate Rotoscopy
======================================

Specialized pipeline for video sequence processing with
temporal consistency and batch optimization.
"""

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ultimate_rotoscopy.core.engine import RotoscopyEngine, ProcessingResult
from ultimate_rotoscopy.models.sam3 import SegmentationPrompt


@dataclass
class VideoConfig:
    """Video processing configuration."""
    frame_rate: float = 24.0
    start_frame: int = 0
    end_frame: Optional[int] = None
    temporal_window: int = 5
    keyframe_interval: int = 24
    motion_threshold: float = 0.1
    parallel_decode: bool = True
    prefetch_frames: int = 4
    output_format: str = "exr"


class VideoPipeline:
    """
    Specialized pipeline for video sequence processing.

    Features:
    - Temporal consistency across frames
    - Motion-aware processing
    - Keyframe propagation
    - Parallel frame decoding
    - Memory-efficient streaming

    Example:
        >>> pipeline = VideoPipeline(engine)
        >>>
        >>> # Process video file
        >>> for result in pipeline.process_video("input.mp4"):
        ...     print(f"Frame {result.metadata['frame_index']}")
        >>>
        >>> # Process image sequence
        >>> for result in pipeline.process_sequence("frames/", pattern="*.png"):
        ...     save_result(result)
    """

    def __init__(
        self,
        engine: RotoscopyEngine,
        config: Optional[VideoConfig] = None,
    ):
        self.engine = engine
        self.config = config or VideoConfig()

        self._frame_queue: Optional[queue.Queue] = None
        self._decode_thread: Optional[threading.Thread] = None
        self._keyframe_cache: Dict[int, ProcessingResult] = {}
        self._motion_vectors: List[np.ndarray] = []

    def process_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[SegmentationPrompt] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Generator[ProcessingResult, None, None]:
        """
        Process a video file.

        Args:
            video_path: Path to video file
            prompt: Segmentation prompt (applied to all frames)
            progress_callback: Callback for progress updates

        Yields:
            ProcessingResult for each frame
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required for video processing: pip install opencv-python")

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate frame range
        start_frame = self.config.start_frame
        end_frame = self.config.end_frame or total_frames - 1
        end_frame = min(end_frame, total_frames - 1)

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        previous_frame = None
        frame_idx = start_frame

        try:
            while frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Check if keyframe
                is_keyframe = (frame_idx - start_frame) % self.config.keyframe_interval == 0

                # Process frame
                result = self._process_frame(
                    frame_rgb,
                    frame_idx,
                    prompt,
                    previous_frame,
                    is_keyframe,
                )

                # Cache keyframe
                if is_keyframe:
                    self._keyframe_cache[frame_idx] = result

                # Update progress
                if progress_callback:
                    progress_callback(frame_idx - start_frame + 1, end_frame - start_frame + 1)

                previous_frame = frame_rgb
                frame_idx += 1

                yield result

        finally:
            cap.release()
            self.engine.reset_temporal_state()
            self._keyframe_cache.clear()

    def process_sequence(
        self,
        input_dir: Union[str, Path],
        pattern: str = "*.png",
        prompt: Optional[SegmentationPrompt] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Generator[ProcessingResult, None, None]:
        """
        Process an image sequence.

        Args:
            input_dir: Directory containing frames
            pattern: Glob pattern for frame files
            prompt: Segmentation prompt
            progress_callback: Callback for progress updates

        Yields:
            ProcessingResult for each frame
        """
        input_dir = Path(input_dir)
        frames = sorted(input_dir.glob(pattern))

        if not frames:
            raise ValueError(f"No frames found matching pattern: {pattern}")

        start_idx = self.config.start_frame
        end_idx = self.config.end_frame or len(frames) - 1
        end_idx = min(end_idx, len(frames) - 1)

        frames = frames[start_idx:end_idx + 1]
        total_frames = len(frames)

        previous_frame = None

        for i, frame_path in enumerate(frames):
            frame_idx = start_idx + i

            # Load frame
            frame = np.array(Image.open(frame_path))

            # Check if keyframe
            is_keyframe = i % self.config.keyframe_interval == 0

            # Process
            result = self._process_frame(
                frame,
                frame_idx,
                prompt,
                previous_frame,
                is_keyframe,
            )

            # Cache keyframe
            if is_keyframe:
                self._keyframe_cache[frame_idx] = result

            # Update progress
            if progress_callback:
                progress_callback(i + 1, total_frames)

            previous_frame = frame

            yield result

        self.engine.reset_temporal_state()
        self._keyframe_cache.clear()

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        prompt: Optional[SegmentationPrompt],
        previous_frame: Optional[np.ndarray],
        is_keyframe: bool,
    ) -> ProcessingResult:
        """Process a single frame with temporal awareness."""
        # Estimate motion if we have a previous frame
        motion_mask = None
        if previous_frame is not None:
            motion_mask = self._estimate_motion(frame, previous_frame)

        # Process through engine
        result = self.engine.process_video_frame(
            frame,
            frame_index=frame_idx,
            prompt=prompt,
            previous_frame=previous_frame,
        )

        # Add motion information
        result.motion_mask = motion_mask
        result.metadata["is_keyframe"] = is_keyframe
        result.metadata["frame_index"] = frame_idx

        return result

    def _estimate_motion(
        self,
        current: np.ndarray,
        previous: np.ndarray
    ) -> np.ndarray:
        """Estimate motion between frames."""
        try:
            import cv2

            # Convert to grayscale
            if current.ndim == 3:
                curr_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
                prev_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
            else:
                curr_gray = current
                prev_gray = previous

            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            # Compute flow magnitude
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

            # Normalize
            motion_mask = mag / (mag.max() + 1e-6)

            return motion_mask

        except Exception:
            # Return zero motion mask on failure
            return np.zeros(current.shape[:2], dtype=np.float32)

    def get_keyframe(self, frame_idx: int) -> Optional[ProcessingResult]:
        """Get cached keyframe result."""
        # Find nearest keyframe
        keyframes = sorted(self._keyframe_cache.keys())
        if not keyframes:
            return None

        nearest = min(keyframes, key=lambda k: abs(k - frame_idx))
        return self._keyframe_cache.get(nearest)


class BatchProcessor:
    """
    Batch processor for parallel image processing.

    Features:
    - Multi-threaded processing
    - Memory-efficient batching
    - Progress tracking

    Example:
        >>> processor = BatchProcessor(engine, num_workers=4)
        >>> results = processor.process_batch(image_paths)
    """

    def __init__(
        self,
        engine: RotoscopyEngine,
        num_workers: int = 4,
        batch_size: int = 4,
    ):
        self.engine = engine
        self.num_workers = num_workers
        self.batch_size = batch_size

    def process_batch(
        self,
        images: List[Union[np.ndarray, Path]],
        prompts: Optional[List[SegmentationPrompt]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ProcessingResult]:
        """
        Process a batch of images.

        Args:
            images: List of images or paths
            prompts: Optional list of prompts (one per image)
            progress_callback: Progress callback

        Returns:
            List of ProcessingResult
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(images)
        total = len(images)

        def process_single(idx: int) -> Tuple[int, ProcessingResult]:
            image = images[idx]
            if isinstance(image, Path):
                image = np.array(Image.open(image))

            prompt = prompts[idx] if prompts and idx < len(prompts) else None

            result = self.engine.process(
                image,
                prompt=prompt,
            )

            return idx, result

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(process_single, i): i
                for i in range(total)
            }

            completed = 0
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed += 1

                if progress_callback:
                    progress_callback(completed, total)

        return results
