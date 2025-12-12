#!/usr/bin/env python3
"""
Flame Export - Professional VFX Output for Autodesk Flame
==========================================================

Export utilities for Autodesk Flame compatible output:
- Multi-layer OpenEXR with proper channel naming
- AOV management (alpha, depth, normals, etc.)
- Flame clip XML generation
- Batch setup templates
- Sequence support

Requirements:
    pip install numpy opencv-python

For OpenEXR support:
    pip install openexr

Usage:
    # CLI
    python flame_export.py --input results/ --output flame_output/

    # Python API
    from flame_export import FlameExporter
    exporter = FlameExporter("output/", clip_name="my_shot")
    exporter.add_rgba(image, alpha)
    exporter.add_depth(depth_map)
    exporter.add_normals(normal_map)
    exporter.export_frame(1)
    exporter.finalize()
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import cv2
except ImportError:
    print("Warning: OpenCV not available")
    cv2 = None


@dataclass
class FlameClipInfo:
    """Information for Flame clip creation."""
    name: str
    width: int
    height: int
    frame_rate: float = 24.0
    start_frame: int = 1
    end_frame: int = 1
    bit_depth: int = 16
    color_space: str = "ACEScg"


class AOVChannel:
    """AOV channel definition."""

    # Standard AOV names for Flame
    RGBA = "rgba"
    ALPHA = "alpha"
    DEPTH = "depth"
    NORMALS = "normals"
    POSITION = "position"
    UV = "uv"
    MOTION = "motion"
    CRYPTO = "crypto"
    MATTE = "matte"

    # Layer suffixes
    CORE = "_core"
    EDGE = "_edge"
    HAIR = "_hair"


class FlameExporter:
    """
    Professional exporter for Autodesk Flame.

    Supports:
    - Multi-layer EXR export
    - AOV channel management
    - Flame clip XML generation
    - Batch setup templates
    - Sequence numbering

    Example:
        >>> exporter = FlameExporter("output/", "my_shot")
        >>> exporter.add_rgba(image, alpha)
        >>> exporter.add_depth(depth)
        >>> exporter.add_normals(normals)
        >>> exporter.export_frame(1)
        >>> exporter.finalize()
    """

    COLOR_SPACES = [
        "ACEScg",
        "ACES",
        "ACEScct",
        "Rec.709",
        "Rec.2020",
        "sRGB",
        "Linear",
        "Log3G10",
        "LogC",
    ]

    def __init__(
        self,
        output_dir: Union[str, Path],
        clip_name: str = "rotoscopy",
        frame_rate: float = 24.0,
        color_space: str = "ACEScg",
    ):
        """
        Initialize Flame exporter.

        Args:
            output_dir: Output directory
            clip_name: Name for the clip
            frame_rate: Frame rate
            color_space: Color space (ACEScg, ACES, etc.)
        """
        self.output_dir = Path(output_dir)
        self.clip_name = clip_name
        self.frame_rate = frame_rate
        self.color_space = color_space

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "exr").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

        # Frame data storage
        self._current_frame_data: Dict[str, np.ndarray] = {}
        self._exported_frames: List[int] = []
        self._resolution: Optional[Tuple[int, int]] = None

    def add_rgba(self, image: np.ndarray, alpha: Optional[np.ndarray] = None):
        """
        Add RGBA beauty pass.

        Args:
            image: RGB image (H, W, 3)
            alpha: Alpha channel (H, W) optional
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        self._current_frame_data["R"] = image[:, :, 0]
        self._current_frame_data["G"] = image[:, :, 1]
        self._current_frame_data["B"] = image[:, :, 2]

        if alpha is not None:
            if alpha.dtype == np.uint8:
                alpha = alpha.astype(np.float32) / 255.0
            self._current_frame_data["A"] = alpha

        # Set resolution
        if self._resolution is None:
            self._resolution = (image.shape[0], image.shape[1])

    def add_alpha(self, alpha: np.ndarray, layer_name: str = ""):
        """
        Add alpha channel.

        Args:
            alpha: Alpha channel (H, W)
            layer_name: Optional layer suffix (e.g., "_core", "_edge")
        """
        if alpha.dtype == np.uint8:
            alpha = alpha.astype(np.float32) / 255.0

        channel_name = f"alpha{layer_name}.A"
        self._current_frame_data[channel_name] = alpha

    def add_depth(self, depth: np.ndarray, normalize: bool = True):
        """
        Add depth channel.

        Args:
            depth: Depth map (H, W)
            normalize: Normalize to [0, 1] range
        """
        if normalize:
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)

        self._current_frame_data["depth.Z"] = depth.astype(np.float32)

    def add_normals(self, normals: np.ndarray):
        """
        Add normal map.

        Args:
            normals: Normal map (H, W, 3) in range [-1, 1]
        """
        # Ensure float32
        normals = normals.astype(np.float32)

        self._current_frame_data["normal.X"] = normals[:, :, 0]
        self._current_frame_data["normal.Y"] = normals[:, :, 1]
        self._current_frame_data["normal.Z"] = normals[:, :, 2]

    def add_matte_layers(
        self,
        core: Optional[np.ndarray] = None,
        edge: Optional[np.ndarray] = None,
        hair: Optional[np.ndarray] = None,
    ):
        """
        Add matte layer decomposition.

        Args:
            core: Core alpha (solid interior)
            edge: Edge alpha (transition)
            hair: Hair alpha (fine details)
        """
        if core is not None:
            self.add_alpha(core, "_core")
        if edge is not None:
            self.add_alpha(edge, "_edge")
        if hair is not None:
            self.add_alpha(hair, "_hair")

    def add_custom_channel(self, name: str, data: np.ndarray):
        """
        Add custom AOV channel.

        Args:
            name: Channel name (e.g., "custom.R")
            data: Channel data (H, W)
        """
        self._current_frame_data[name] = data.astype(np.float32)

    def export_frame(self, frame_number: int) -> Path:
        """
        Export current frame data to EXR.

        Args:
            frame_number: Frame number

        Returns:
            Path to exported file
        """
        if not self._current_frame_data:
            raise ValueError("No data added for export")

        output_path = self.output_dir / "exr" / f"{self.clip_name}.{frame_number:06d}.exr"

        # Try OpenEXR export
        try:
            self._export_exr(output_path, frame_number)
        except ImportError:
            # Fallback to individual files
            self._export_fallback(output_path.parent, frame_number)

        # Save metadata
        self._save_frame_metadata(frame_number)

        # Track frame
        self._exported_frames.append(frame_number)

        # Clear current frame data
        self._current_frame_data = {}

        return output_path

    def _export_exr(self, output_path: Path, frame_number: int):
        """Export as multi-layer EXR."""
        import OpenEXR
        import Imath

        h, w = self._resolution

        # Build header
        header = OpenEXR.Header(w, h)
        header['compression'] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)

        # Add channels
        channels = {}
        for name in self._current_frame_data.keys():
            channels[name] = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))

        header['channels'] = channels

        # Add Flame metadata
        header['software'] = 'Ultimate Rotoscopy'
        header['capDate'] = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        header['framesPerSecond'] = f"{int(self.frame_rate * 1000)}/1000"
        header['timeCode'] = self._frame_to_timecode(frame_number)
        header['colorSpace'] = self.color_space

        # Write file
        exr = OpenEXR.OutputFile(str(output_path), header)

        pixel_data = {}
        for name, data in self._current_frame_data.items():
            pixel_data[name] = data.astype(np.float32).tobytes()

        exr.writePixels(pixel_data)
        exr.close()

    def _export_fallback(self, output_dir: Path, frame_number: int):
        """Fallback export when OpenEXR not available."""
        output_dir.mkdir(exist_ok=True)

        for name, data in self._current_frame_data.items():
            safe_name = name.replace(".", "_").replace("/", "_")
            file_path = output_dir / f"{self.clip_name}_{safe_name}.{frame_number:06d}.png"

            # Normalize to 16-bit PNG
            if data.max() <= 1.0:
                data_16bit = (data * 65535).astype(np.uint16)
            else:
                data_16bit = data.astype(np.uint16)

            if cv2 is not None:
                cv2.imwrite(str(file_path), data_16bit)
            else:
                np.save(file_path.with_suffix('.npy'), data)

    def _save_frame_metadata(self, frame_number: int):
        """Save frame metadata."""
        metadata_path = self.output_dir / "metadata" / f"{self.clip_name}.{frame_number:06d}.json"

        metadata = {
            "frame_number": frame_number,
            "timecode": self._frame_to_timecode(frame_number),
            "resolution": self._resolution,
            "channels": list(self._current_frame_data.keys()),
            "color_space": self.color_space,
            "exported_at": datetime.now().isoformat(),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _frame_to_timecode(self, frame: int) -> str:
        """Convert frame to timecode string."""
        fps = int(self.frame_rate)
        hours = frame // (fps * 60 * 60)
        minutes = (frame // (fps * 60)) % 60
        seconds = (frame // fps) % 60
        frames = frame % fps

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    def generate_clip_xml(self) -> Path:
        """
        Generate Flame clip XML.

        Returns:
            Path to XML file
        """
        if not self._exported_frames:
            raise RuntimeError("No frames exported")

        start_frame = min(self._exported_frames)
        end_frame = max(self._exported_frames)

        clip_info = FlameClipInfo(
            name=self.clip_name,
            width=self._resolution[1],
            height=self._resolution[0],
            frame_rate=self.frame_rate,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        xml_content = self._create_clip_xml(clip_info)

        xml_path = self.output_dir / f"{self.clip_name}.clip"
        with open(xml_path, "w") as f:
            f.write(xml_content)

        return xml_path

    def _create_clip_xml(self, clip_info: FlameClipInfo) -> str:
        """Create Flame clip XML content."""
        exr_pattern = f"exr/{self.clip_name}.######.exr"

        return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE clip>
<clip version="2">
    <name>{clip_info.name}</name>
    <handler>OpenEXR</handler>
    <sourceName>{clip_info.name}</sourceName>
    <channelsType>OpenEXR Multi-Channel</channelsType>

    <tracks type="video">
        <track>
            <trackName>video</trackName>
            <trackUid>1</trackUid>
            <feeds type="video" currentVersion="v0">
                <feed>
                    <feedName>v0</feedName>
                    <feedUid>1</feedUid>
                    <spans type="video">
                        <span>
                            <path>{exr_pattern}</path>
                            <startFrame>{clip_info.start_frame}</startFrame>
                            <endFrame>{clip_info.end_frame}</endFrame>
                        </span>
                    </spans>
                </feed>
            </feeds>
        </track>
    </tracks>

    <essenceData>
        <format>
            <height>{clip_info.height}</height>
            <width>{clip_info.width}</width>
            <pixelRatio>1.0</pixelRatio>
            <fieldDominance>progressive</fieldDominance>
            <bitDepth>{clip_info.bit_depth}</bitDepth>
            <colourSpace>{self.color_space}</colourSpace>
        </format>
        <frameRate>{clip_info.frame_rate}</frameRate>
        <duration>
            <type>frames</type>
            <value>{clip_info.end_frame - clip_info.start_frame + 1}</value>
        </duration>
    </essenceData>

    <userData>
        <generator>Ultimate Rotoscopy</generator>
        <version>1.0</version>
        <createdAt>{datetime.now().isoformat()}</createdAt>
    </userData>
</clip>'''

    def generate_batch_setup(self) -> Path:
        """
        Generate Flame batch setup template.

        Returns:
            Path to batch file
        """
        batch_content = f'''# Flame Batch Setup
# Generated by Ultimate Rotoscopy
# Clip: {self.clip_name}
# Resolution: {self._resolution}
# Frames: {min(self._exported_frames)} - {max(self._exported_frames)}

BATCH_VERSION 2021.1

SCHEMATIC {{
    NODE Read1 {{
        TYPE Read
        POSITION 0 0
        CLIP "{self.output_dir / 'exr' / f'{self.clip_name}.######.exr'}"
    }}

    NODE Action1 {{
        TYPE Action
        POSITION 200 0
    }}

    NODE Write1 {{
        TYPE Write
        POSITION 400 0
    }}

    LINK Read1 Action1 "Front"
    LINK Action1 Write1
}}

# Available AOV Channels:
'''
        # Add channel list
        if self._exported_frames and hasattr(self, '_last_channels'):
            for channel in self._last_channels:
                batch_content += f"# - {channel}\n"

        batch_path = self.output_dir / f"{self.clip_name}.batch"
        with open(batch_path, "w") as f:
            f.write(batch_content)

        return batch_path

    def finalize(self) -> Dict[str, Path]:
        """
        Finalize export and generate all support files.

        Returns:
            Dictionary of generated files
        """
        files = {}

        # Generate clip XML
        try:
            files["clip"] = self.generate_clip_xml()
        except Exception as e:
            print(f"Warning: Could not generate clip XML: {e}")

        # Generate batch setup
        try:
            files["batch"] = self.generate_batch_setup()
        except Exception as e:
            print(f"Warning: Could not generate batch setup: {e}")

        # Generate summary
        summary_path = self.output_dir / f"{self.clip_name}_summary.json"
        summary = {
            "clip_name": self.clip_name,
            "frame_rate": self.frame_rate,
            "resolution": self._resolution,
            "color_space": self.color_space,
            "frame_range": {
                "start": min(self._exported_frames) if self._exported_frames else 0,
                "end": max(self._exported_frames) if self._exported_frames else 0,
                "count": len(self._exported_frames),
            },
            "output_directory": str(self.output_dir),
            "generated_at": datetime.now().isoformat(),
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        files["summary"] = summary_path

        print(f"\nFlame export complete!")
        print(f"  Frames: {len(self._exported_frames)}")
        print(f"  Output: {self.output_dir}")

        return files


def export_rotoscopy_result(
    output_dir: Union[str, Path],
    clip_name: str,
    image: np.ndarray,
    alpha: np.ndarray,
    depth: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    alpha_core: Optional[np.ndarray] = None,
    alpha_edge: Optional[np.ndarray] = None,
    alpha_hair: Optional[np.ndarray] = None,
    frame_number: int = 1,
) -> Path:
    """
    Convenience function to export a complete rotoscopy result.

    Args:
        output_dir: Output directory
        clip_name: Clip name
        image: RGB image
        alpha: Alpha matte
        depth: Depth map (optional)
        normals: Normal map (optional)
        alpha_core: Core alpha layer (optional)
        alpha_edge: Edge alpha layer (optional)
        alpha_hair: Hair alpha layer (optional)
        frame_number: Frame number

    Returns:
        Path to exported EXR
    """
    exporter = FlameExporter(output_dir, clip_name)

    exporter.add_rgba(image, alpha)

    if depth is not None:
        exporter.add_depth(depth)

    if normals is not None:
        exporter.add_normals(normals)

    if any([alpha_core, alpha_edge, alpha_hair]):
        exporter.add_matte_layers(alpha_core, alpha_edge, alpha_hair)

    output_path = exporter.export_frame(frame_number)
    exporter.finalize()

    return output_path


def main():
    """CLI for Flame export testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Flame Export Utility")
    parser.add_argument("--test", action="store_true", help="Run test export")
    parser.add_argument("-o", "--output", type=str, default="flame_test", help="Output directory")

    args = parser.parse_args()

    if args.test:
        print("Running Flame export test...")

        # Create test data
        h, w = 1080, 1920
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        alpha = np.random.random((h, w)).astype(np.float32)
        depth = np.random.random((h, w)).astype(np.float32)
        normals = np.random.random((h, w, 3)).astype(np.float32) * 2 - 1

        # Export
        output_path = export_rotoscopy_result(
            args.output,
            "test_clip",
            image,
            alpha,
            depth=depth,
            normals=normals,
            frame_number=1
        )

        print(f"Test export complete: {output_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
