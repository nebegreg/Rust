"""
Unified Processing Pipeline
============================

High-level pipeline that combines all models for complete rotoscopy workflow.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from ultimate_rotoscopy.core.engine import (
    RotoscopyEngine,
    EngineConfig,
    ProcessingMode,
    ProcessingResult,
)
from ultimate_rotoscopy.core.session import Session, SessionConfig
from ultimate_rotoscopy.models.sam3 import SegmentationPrompt, PromptType
from ultimate_rotoscopy.export.exr_writer import EXRWriter
from ultimate_rotoscopy.export.aov_manager import AOVManager


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Engine settings
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    device: str = "cuda"

    # Output settings
    output_directory: Path = field(default_factory=lambda: Path.cwd() / "output")
    output_format: str = "exr"  # exr, png, tiff
    output_prefix: str = "frame"
    output_padding: int = 6

    # Processing settings
    batch_size: int = 1
    num_workers: int = 4

    # Features
    generate_depth: bool = True
    generate_normals: bool = True
    generate_matte: bool = True
    generate_point_cloud: bool = False
    generate_aovs: bool = True

    # Video settings
    temporal_consistency: bool = True
    keyframe_interval: int = 24  # Frames between keyframes

    # Quality settings
    edge_refinement: bool = True
    motion_blur_handling: bool = True
    hair_detail: bool = True

    # Callbacks
    progress_callback: Optional[Callable[[int, int, ProcessingResult], None]] = None


class UnifiedPipeline:
    """
    Unified processing pipeline for Ultimate Rotoscopy.

    Provides a complete workflow for:
    - Single image processing
    - Batch image processing
    - Video sequence processing
    - EXR/AOV export

    Example:
        >>> pipeline = UnifiedPipeline()
        >>>
        >>> # Process single image
        >>> result = pipeline.process_image(
        ...     image,
        ...     points=np.array([[100, 200]])
        ... )
        >>>
        >>> # Process video sequence
        >>> pipeline.process_sequence(
        ...     input_dir="frames/",
        ...     output_dir="output/",
        ...     points=np.array([[100, 200]])
        ... )
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize engine
        engine_config = EngineConfig(
            processing_mode=self.config.processing_mode,
            device=self.config.device,
            enable_sam3=True,
            enable_depth=self.config.generate_depth,
            enable_matte=self.config.generate_matte,
            parallel_processing=True,
        )
        self.engine = RotoscopyEngine(engine_config)

        # Initialize exporters
        self.exr_writer = EXRWriter()
        self.aov_manager = AOVManager()

        # Session for state management
        session_config = SessionConfig(
            output_directory=self.config.output_directory,
        )
        self.session = Session(session_config)

        # Ensure output directory exists
        self.config.output_directory.mkdir(parents=True, exist_ok=True)

    def process_image(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        points: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        save_output: bool = True,
        output_name: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Process a single image.

        Args:
            image: Input image
            points: Foreground point prompts (Nx2)
            boxes: Bounding box prompts (Nx4)
            mask: Existing mask for matte refinement
            save_output: Save results to disk
            output_name: Custom output filename

        Returns:
            ProcessingResult with all outputs
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image = np.array(Image.open(image_path))
            if output_name is None:
                output_name = image_path.stem
        elif isinstance(image, Image.Image):
            image = np.array(image)

        if output_name is None:
            output_name = "image"

        # Create prompt if points or boxes provided
        prompt = None
        if points is not None:
            prompt = SegmentationPrompt(
                prompt_type=PromptType.POINT,
                points=points,
                point_labels=np.ones(len(points), dtype=np.int32),
            )
        elif boxes is not None:
            prompt = SegmentationPrompt(
                prompt_type=PromptType.BOX,
                boxes=boxes,
            )

        # Process
        result = self.engine.process(
            image,
            prompt=prompt,
            generate_depth=self.config.generate_depth,
            generate_normals=self.config.generate_normals,
            generate_matte=self.config.generate_matte,
            generate_point_cloud=self.config.generate_point_cloud,
        )

        # Save output
        if save_output:
            self._save_result(result, output_name, image)

        return result

    def process_sequence(
        self,
        input_path: Union[str, Path, List[Path]],
        output_path: Optional[Union[str, Path]] = None,
        points: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        frame_pattern: str = "*.png",
    ) -> Generator[ProcessingResult, None, None]:
        """
        Process a sequence of frames.

        Args:
            input_path: Input directory or list of frame paths
            output_path: Output directory (defaults to config)
            points: Foreground point prompts (used for all frames)
            boxes: Bounding box prompts
            start_frame: First frame to process
            end_frame: Last frame to process (None = all)
            frame_pattern: Glob pattern for frame files

        Yields:
            ProcessingResult for each frame
        """
        # Get frame list
        if isinstance(input_path, list):
            frame_paths = input_path
        else:
            input_dir = Path(input_path)
            frame_paths = sorted(input_dir.glob(frame_pattern))

        if end_frame is None:
            end_frame = len(frame_paths) - 1

        frame_paths = frame_paths[start_frame:end_frame + 1]

        if output_path:
            output_dir = Path(output_path)
        else:
            output_dir = self.config.output_directory

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create prompt
        prompt = None
        if points is not None:
            prompt = SegmentationPrompt(
                prompt_type=PromptType.POINT,
                points=points,
                point_labels=np.ones(len(points), dtype=np.int32),
            )
        elif boxes is not None:
            prompt = SegmentationPrompt(
                prompt_type=PromptType.BOX,
                boxes=boxes,
            )

        # Process frames
        previous_frame = None
        total_frames = len(frame_paths)

        for i, frame_path in enumerate(tqdm(frame_paths, desc="Processing frames")):
            frame_idx = start_frame + i

            # Load frame
            frame = np.array(Image.open(frame_path))

            # Process
            result = self.engine.process_video_frame(
                frame,
                frame_index=frame_idx,
                prompt=prompt,
                previous_frame=previous_frame,
            )

            # Update session
            self.session.add_frame(frame_idx, image_path=frame_path)
            self.session.update_frame(frame_idx, result)

            # Save output
            output_name = f"{self.config.output_prefix}_{frame_idx:0{self.config.output_padding}d}"
            self._save_result(result, output_name, frame, output_dir)

            # Callback
            if self.config.progress_callback:
                self.config.progress_callback(i + 1, total_frames, result)

            # Update for next frame
            previous_frame = frame

            yield result

        # Reset temporal state at end of sequence
        self.engine.reset_temporal_state()

    def process_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, Path]],
        prompts: Optional[List[SegmentationPrompt]] = None,
        output_names: Optional[List[str]] = None,
    ) -> List[ProcessingResult]:
        """
        Process a batch of images.

        Args:
            images: List of input images
            prompts: List of prompts (one per image, or None for all)
            output_names: Custom output names

        Returns:
            List of ProcessingResult
        """
        results = []

        for i, image in enumerate(tqdm(images, desc="Processing batch")):
            prompt = prompts[i] if prompts and i < len(prompts) else None
            output_name = output_names[i] if output_names and i < len(output_names) else f"batch_{i:06d}"

            result = self.process_image(
                image,
                save_output=True,
                output_name=output_name,
            )

            # Apply prompt if provided separately
            if prompt:
                result = self.engine.process(
                    image if isinstance(image, np.ndarray) else np.array(Image.open(image)),
                    prompt=prompt,
                )

            results.append(result)

        return results

    def _save_result(
        self,
        result: ProcessingResult,
        output_name: str,
        source_image: np.ndarray,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Save processing result to disk."""
        output_dir = output_dir or self.config.output_directory

        if self.config.output_format == "exr":
            self._save_exr(result, output_name, source_image, output_dir)
        elif self.config.output_format == "png":
            self._save_png(result, output_name, source_image, output_dir)
        elif self.config.output_format == "tiff":
            self._save_tiff(result, output_name, source_image, output_dir)

    def _save_exr(
        self,
        result: ProcessingResult,
        output_name: str,
        source_image: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Save result as multi-channel EXR."""
        if self.config.generate_aovs and result.aov_package:
            # Save as multi-layer EXR with all AOVs
            output_path = output_dir / f"{output_name}.exr"
            self.exr_writer.write_multilayer(
                str(output_path),
                result.aov_package,
                source_image,
            )
        else:
            # Save individual EXR files
            if result.alpha is not None:
                self.exr_writer.write_single(
                    str(output_dir / f"{output_name}_alpha.exr"),
                    result.alpha,
                )
            if result.depth_normalized is not None:
                self.exr_writer.write_single(
                    str(output_dir / f"{output_name}_depth.exr"),
                    result.depth_normalized,
                )
            if result.normals is not None:
                self.exr_writer.write_rgb(
                    str(output_dir / f"{output_name}_normals.exr"),
                    (result.normals + 1) / 2,  # Convert to 0-1
                )

    def _save_png(
        self,
        result: ProcessingResult,
        output_name: str,
        source_image: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Save result as PNG files."""
        if result.alpha is not None:
            alpha_8bit = (result.alpha * 255).astype(np.uint8)
            Image.fromarray(alpha_8bit).save(output_dir / f"{output_name}_alpha.png")

        if result.depth_normalized is not None:
            depth_8bit = (result.depth_normalized * 255).astype(np.uint8)
            Image.fromarray(depth_8bit).save(output_dir / f"{output_name}_depth.png")

        if result.normals is not None:
            normals_8bit = ((result.normals + 1) / 2 * 255).astype(np.uint8)
            Image.fromarray(normals_8bit).save(output_dir / f"{output_name}_normals.png")

        if result.foreground is not None:
            if result.foreground.max() <= 1:
                fg_8bit = (result.foreground * 255).astype(np.uint8)
            else:
                fg_8bit = result.foreground.astype(np.uint8)
            Image.fromarray(fg_8bit).save(output_dir / f"{output_name}_foreground.png")

    def _save_tiff(
        self,
        result: ProcessingResult,
        output_name: str,
        source_image: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Save result as TIFF files (16-bit)."""
        import imageio

        if result.alpha is not None:
            alpha_16bit = (result.alpha * 65535).astype(np.uint16)
            imageio.imwrite(output_dir / f"{output_name}_alpha.tiff", alpha_16bit)

        if result.depth_normalized is not None:
            depth_16bit = (result.depth_normalized * 65535).astype(np.uint16)
            imageio.imwrite(output_dir / f"{output_name}_depth.tiff", depth_16bit)

    def interactive_segment(
        self,
        image: np.ndarray,
    ) -> "InteractiveSession":
        """
        Start an interactive segmentation session.

        Returns an object that allows adding/removing points
        and updating the segmentation in real-time.
        """
        return InteractiveSession(self.engine, image)

    def save_session(self, path: Union[str, Path]) -> None:
        """Save current session to file."""
        self.session.save(path)

    def load_session(self, path: Union[str, Path]) -> None:
        """Load session from file."""
        self.session = Session.load(path)

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "session": self.session.get_statistics(),
            "engine_memory": self.engine.get_memory_usage(),
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        self.engine.unload_models()


class InteractiveSession:
    """
    Interactive segmentation session for real-time editing.

    Example:
        >>> session = pipeline.interactive_segment(image)
        >>> session.add_point(100, 200, foreground=True)
        >>> session.add_point(50, 50, foreground=False)
        >>> result = session.get_result()
    """

    def __init__(self, engine: RotoscopyEngine, image: np.ndarray):
        self.engine = engine
        self.image = image
        self.points: List[Tuple[int, int]] = []
        self.labels: List[int] = []
        self.boxes: List[Tuple[int, int, int, int]] = []
        self._cached_result: Optional[ProcessingResult] = None

    def add_point(self, x: int, y: int, foreground: bool = True) -> None:
        """Add a point prompt."""
        self.points.append((x, y))
        self.labels.append(1 if foreground else 0)
        self._cached_result = None

    def remove_point(self, index: int) -> None:
        """Remove a point by index."""
        if 0 <= index < len(self.points):
            self.points.pop(index)
            self.labels.pop(index)
            self._cached_result = None

    def add_box(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Add a box prompt."""
        self.boxes.append((x1, y1, x2, y2))
        self._cached_result = None

    def remove_box(self, index: int) -> None:
        """Remove a box by index."""
        if 0 <= index < len(self.boxes):
            self.boxes.pop(index)
            self._cached_result = None

    def clear(self) -> None:
        """Clear all prompts."""
        self.points.clear()
        self.labels.clear()
        self.boxes.clear()
        self._cached_result = None

    def get_result(self, force_refresh: bool = False) -> ProcessingResult:
        """Get the current segmentation result."""
        if self._cached_result is not None and not force_refresh:
            return self._cached_result

        prompt = None

        if self.points:
            prompt = SegmentationPrompt(
                prompt_type=PromptType.POINT,
                points=np.array(self.points),
                point_labels=np.array(self.labels),
            )
        elif self.boxes:
            prompt = SegmentationPrompt(
                prompt_type=PromptType.BOX,
                boxes=np.array(self.boxes),
            )

        if prompt is None:
            # Return empty result
            h, w = self.image.shape[:2]
            return ProcessingResult(
                masks=np.zeros((1, h, w), dtype=bool),
            )

        self._cached_result = self.engine.process(
            self.image,
            prompt=prompt,
        )

        return self._cached_result
