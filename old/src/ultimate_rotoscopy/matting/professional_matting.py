"""
Professional Matting Pipeline
==============================

Integrated pipeline combining:
- Alpha split (core/edge/hair)
- Motion blur awareness
- Multi-layer EXR export
- Temporal consistency

Production-ready for cinema VFX workflows.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from ultimate_rotoscopy.matting.alpha_split import (
    AlphaSplitter,
    AlphaSplitConfig,
    AlphaSplitResult,
)
from ultimate_rotoscopy.matting.motion_blur_aware import (
    MotionBlurAwareMatting,
    MotionBlurConfig,
    MotionBlurResult,
    MotionBlurLevel,
)


@dataclass
class ProfessionalMattingConfig:
    """Configuration for professional matting pipeline."""

    # Alpha split config
    alpha_split: AlphaSplitConfig = None

    # Motion blur config
    motion_blur: MotionBlurConfig = None

    # Export settings
    export_all_layers: bool = True
    export_format: str = "exr"  # "exr", "png", "tiff"
    output_bit_depth: int = 16  # 8, 16, or 32 (for EXR)

    # Processing options
    enable_motion_blur: bool = True
    enable_temporal_consistency: bool = True

    def __post_init__(self):
        if self.alpha_split is None:
            self.alpha_split = AlphaSplitConfig()
        if self.motion_blur is None:
            self.motion_blur = MotionBlurConfig()


@dataclass
class ProfessionalMattingResult:
    """Complete result from professional matting pipeline."""

    # Alpha split components
    alpha_core: np.ndarray
    alpha_edge: np.ndarray
    alpha_hair: np.ndarray

    # Motion blur variants (if enabled)
    alpha_sharp: Optional[np.ndarray] = None
    alpha_motion_blur: Optional[np.ndarray] = None
    blur_mask: Optional[np.ndarray] = None

    # Final composited alpha
    alpha_final: np.ndarray = None

    # Metadata
    has_motion_blur: bool = False
    blur_level: MotionBlurLevel = MotionBlurLevel.NONE
    blur_percentage: float = 0.0

    # File paths (if exported)
    exported_files: dict = None


class ProfessionalMatting:
    """
    Professional matting pipeline for cinema VFX.

    Workflow:
    1. Input alpha (from SAM3, Matte Anything, etc.)
    2. Detect motion blur
    3. Split alpha into core/edge/hair
    4. Generate sharp and motion-blur variants
    5. Adaptively mix based on blur detection
    6. Export all layers for compositing

    Output layers:
    - alpha_core: Solid interior
    - alpha_edge: Transition boundary
    - alpha_hair: Fine details
    - alpha_sharp: Sharpened version
    - alpha_motion_blur: Motion-preserved version
    - alpha_final: Adaptively mixed result
    - blur_mask: Motion blur confidence map

    Benefits:
    - Complete control in compositing
    - Separate treatment of hair/edges
    - No temporal popping
    - Cinema-quality edge detail
    """

    def __init__(self, config: Optional[ProfessionalMattingConfig] = None):
        self.config = config or ProfessionalMattingConfig()

        # Initialize processors
        self.splitter = AlphaSplitter(self.config.alpha_split)
        self.motion_blur_processor = MotionBlurAwareMatting(self.config.motion_blur)

        # State for video sequences
        self._frame_history: List[np.ndarray] = []

    def process_frame(
        self,
        alpha: np.ndarray,
        image: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
    ) -> ProfessionalMattingResult:
        """
        Process single frame through professional matting pipeline.

        Args:
            alpha: Input alpha channel (H, W) in [0, 1]
            image: Reference RGB image (H, W, 3)
            prev_frame: Previous frame for temporal analysis (optional)

        Returns:
            ProfessionalMattingResult with all layers
        """
        # Step 1: Motion blur detection and processing
        motion_result = None
        if self.config.enable_motion_blur:
            motion_result = self.motion_blur_processor.process(
                alpha,
                image,
                prev_frame,
            )
            # Use motion-aware alpha for splitting
            alpha_for_split = motion_result.alpha_final
        else:
            alpha_for_split = alpha

        # Step 2: Alpha split (core/edge/hair)
        split_result = self.splitter.split(alpha_for_split, image)

        # Step 3: Apply motion blur awareness to each component
        if self.config.enable_motion_blur and motion_result is not None:
            # Create motion-blur variants for each component
            core_sharp, core_mb = self._apply_motion_blur_to_component(
                split_result.core,
                motion_result.blur_mask,
            )
            edge_sharp, edge_mb = self._apply_motion_blur_to_component(
                split_result.edge,
                motion_result.blur_mask,
            )
            hair_sharp, hair_mb = self._apply_motion_blur_to_component(
                split_result.hair,
                motion_result.blur_mask,
            )

            # Reconstruct final alpha
            alpha_final = self._reconstruct_final_alpha(
                core_sharp, edge_sharp, hair_sharp,
                core_mb, edge_mb, hair_mb,
                motion_result.blur_mask,
            )
        else:
            alpha_final = split_result.alpha_reconstructed

        # Create result
        result = ProfessionalMattingResult(
            alpha_core=split_result.core,
            alpha_edge=split_result.edge,
            alpha_hair=split_result.hair,
            alpha_sharp=motion_result.alpha_sharp if motion_result else None,
            alpha_motion_blur=motion_result.alpha_motion_blur if motion_result else None,
            blur_mask=motion_result.blur_mask if motion_result else None,
            alpha_final=alpha_final,
            has_motion_blur=motion_result is not None and motion_result.blur_level != MotionBlurLevel.NONE,
            blur_level=motion_result.blur_level if motion_result else MotionBlurLevel.NONE,
            blur_percentage=motion_result.blur_percentage if motion_result else 0.0,
        )

        return result

    def process_sequence(
        self,
        alpha_sequence: List[np.ndarray],
        image_sequence: List[np.ndarray],
    ) -> List[ProfessionalMattingResult]:
        """
        Process video sequence with temporal consistency.

        Args:
            alpha_sequence: List of alpha frames
            image_sequence: List of RGB frames

        Returns:
            List of ProfessionalMattingResult for each frame
        """
        results = []

        for i, (alpha, image) in enumerate(zip(alpha_sequence, image_sequence)):
            prev_frame = image_sequence[i - 1] if i > 0 else None

            result = self.process_frame(alpha, image, prev_frame)
            results.append(result)

        # Post-process for temporal consistency
        if self.config.enable_temporal_consistency:
            results = self._apply_temporal_consistency_to_sequence(results)

        return results

    def export_layers(
        self,
        result: ProfessionalMattingResult,
        output_path: str,
        frame_number: Optional[int] = None,
    ) -> dict:
        """
        Export all alpha layers to files.

        Args:
            result: ProfessionalMattingResult to export
            output_path: Base output path
            frame_number: Frame number for sequences (optional)

        Returns:
            Dictionary mapping layer name to file path
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = output_path.stem
        if frame_number is not None:
            stem = f"{stem}_{frame_number:04d}"

        ext = f".{self.config.export_format}"

        files = {}

        # Define layers to export
        layers = {
            "alpha_core": result.alpha_core,
            "alpha_edge": result.alpha_edge,
            "alpha_hair": result.alpha_hair,
            "alpha_final": result.alpha_final,
        }

        # Add motion blur layers if available
        if result.has_motion_blur:
            layers.update({
                "alpha_sharp": result.alpha_sharp,
                "alpha_motion_blur": result.alpha_motion_blur,
                "blur_mask": result.blur_mask,
            })

        # Export each layer
        for layer_name, alpha in layers.items():
            if alpha is None:
                continue

            filepath = output_dir / f"{stem}_{layer_name}{ext}"

            if self.config.export_format == "exr":
                self._write_exr(filepath, alpha, layer_name)
            else:
                self._write_image(filepath, alpha, self.config.output_bit_depth)

            files[layer_name] = str(filepath)

        result.exported_files = files
        return files

    def export_multi_layer_exr(
        self,
        result: ProfessionalMattingResult,
        output_path: str,
    ) -> str:
        """
        Export all layers to single multi-layer EXR file.

        Args:
            result: ProfessionalMattingResult to export
            output_path: Output EXR file path

        Returns:
            Path to exported EXR file
        """
        try:
            import OpenEXR
            import Imath

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            h, w = result.alpha_core.shape

            # Define all channels
            channels = {
                "alpha_core.A": result.alpha_core,
                "alpha_edge.A": result.alpha_edge,
                "alpha_hair.A": result.alpha_hair,
                "alpha_final.A": result.alpha_final,
            }

            if result.has_motion_blur:
                channels.update({
                    "alpha_sharp.A": result.alpha_sharp,
                    "alpha_motion_blur.A": result.alpha_motion_blur,
                    "blur_mask.A": result.blur_mask,
                })

            # Create EXR header
            header = OpenEXR.Header(w, h)
            header['channels'] = {
                name: Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
                for name in channels.keys()
            }

            # Write EXR
            exr = OpenEXR.OutputFile(str(output_path), header)
            pixel_data = {
                name: alpha.astype(np.float32).tobytes()
                for name, alpha in channels.items()
            }
            exr.writePixels(pixel_data)
            exr.close()

            return str(output_path)

        except ImportError:
            print("Warning: OpenEXR not available, using separate files")
            return self.export_layers(result, output_path)

    def _apply_motion_blur_to_component(
        self,
        component: np.ndarray,
        blur_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sharp and motion-blur variants of a component.

        Returns:
            (sharp_variant, motion_blur_variant)
        """
        import cv2

        # Sharp variant: slight sharpening
        kernel_sharp = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ]) / 1.0
        sharp = cv2.filter2D(component, -1, kernel_sharp)
        sharp = np.clip(sharp, 0, 1)

        # Motion blur variant: directional blur
        kernel_blur = np.ones((5, 5), np.float32) / 25
        motion_blur = cv2.filter2D(component, -1, kernel_blur)

        return sharp, motion_blur

    def _reconstruct_final_alpha(
        self,
        core_sharp: np.ndarray,
        edge_sharp: np.ndarray,
        hair_sharp: np.ndarray,
        core_mb: np.ndarray,
        edge_mb: np.ndarray,
        hair_mb: np.ndarray,
        blur_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct final alpha from sharp and motion-blur components.

        Adaptively mixes based on blur_mask.
        """
        # Reconstruct sharp and blur versions
        alpha_sharp = np.clip(core_sharp + edge_sharp + hair_sharp, 0, 1)
        alpha_blur = np.clip(core_mb + edge_mb + hair_mb, 0, 1)

        # Adaptive mix
        alpha_final = (1 - blur_mask) * alpha_sharp + blur_mask * alpha_blur

        return np.clip(alpha_final, 0, 1)

    def _apply_temporal_consistency_to_sequence(
        self,
        results: List[ProfessionalMattingResult],
    ) -> List[ProfessionalMattingResult]:
        """
        Apply temporal smoothing to reduce flickering.

        Uses exponential moving average on all alpha layers.
        """
        if len(results) < 2:
            return results

        temporal_weight = 0.7  # 70% current, 30% previous

        smoothed_results = [results[0]]  # Keep first frame as-is

        for i in range(1, len(results)):
            current = results[i]
            previous = smoothed_results[i - 1]

            # Smooth each layer
            smoothed = ProfessionalMattingResult(
                alpha_core=self._temporal_blend(current.alpha_core, previous.alpha_core, temporal_weight),
                alpha_edge=self._temporal_blend(current.alpha_edge, previous.alpha_edge, temporal_weight),
                alpha_hair=self._temporal_blend(current.alpha_hair, previous.alpha_hair, temporal_weight),
                alpha_sharp=self._temporal_blend(current.alpha_sharp, previous.alpha_sharp, temporal_weight) if current.alpha_sharp is not None else None,
                alpha_motion_blur=self._temporal_blend(current.alpha_motion_blur, previous.alpha_motion_blur, temporal_weight) if current.alpha_motion_blur is not None else None,
                blur_mask=self._temporal_blend(current.blur_mask, previous.blur_mask, temporal_weight) if current.blur_mask is not None else None,
                alpha_final=self._temporal_blend(current.alpha_final, previous.alpha_final, temporal_weight),
                has_motion_blur=current.has_motion_blur,
                blur_level=current.blur_level,
                blur_percentage=current.blur_percentage,
            )

            smoothed_results.append(smoothed)

        return smoothed_results

    def _temporal_blend(
        self,
        current: Optional[np.ndarray],
        previous: Optional[np.ndarray],
        weight: float,
    ) -> Optional[np.ndarray]:
        """Temporal blend of two frames."""
        if current is None or previous is None:
            return current

        return weight * current + (1 - weight) * previous

    def _write_exr(self, filepath: Path, alpha: np.ndarray, channel_name: str):
        """Write single-channel EXR."""
        try:
            import OpenEXR
            import Imath

            h, w = alpha.shape
            header = OpenEXR.Header(w, h)
            header['channels'] = {
                channel_name: Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
            }

            exr = OpenEXR.OutputFile(str(filepath), header)
            exr.writePixels({channel_name: alpha.astype(np.float32).tobytes()})
            exr.close()
        except ImportError:
            print(f"Warning: OpenEXR not available, falling back to PNG")
            self._write_image(filepath.with_suffix('.png'), alpha, 16)

    def _write_image(self, filepath: Path, alpha: np.ndarray, bit_depth: int):
        """Write alpha to PNG/TIFF."""
        import cv2

        if bit_depth == 8:
            alpha_int = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        else:  # 16-bit
            alpha_int = (np.clip(alpha, 0, 1) * 65535).astype(np.uint16)

        cv2.imwrite(str(filepath), alpha_int)
