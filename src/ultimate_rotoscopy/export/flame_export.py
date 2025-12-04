"""
Flame Exporter for Ultimate Rotoscopy
======================================

Export functionality specifically designed for Autodesk Flame workflows.
Supports Flame's clip structure, batch setups, and channel naming conventions.
"""

import json
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ultimate_rotoscopy.export.exr_writer import EXRWriter, EXRCompression
from ultimate_rotoscopy.export.aov_manager import AOVManager, AOVType


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
    color_space: str = "ACES"
    aspect_ratio: float = 1.0


class FlameExporter:
    """
    Exporter for Autodesk Flame compatible output.

    Features:
    - Multi-layer EXR with Flame-standard naming
    - Clip XML generation for Flame import
    - Batch setup templates
    - Action/Gmask format export
    - Timeline clip metadata

    Example:
        >>> exporter = FlameExporter(output_dir="output/")
        >>>
        >>> # Export a sequence
        >>> for i, result in enumerate(results):
        ...     exporter.export_frame(result, frame_number=i)
        >>>
        >>> # Generate Flame clip
        >>> exporter.generate_clip_xml("my_clip")
    """

    # Flame color spaces
    COLOR_SPACES = [
        "ACES",
        "ACEScg",
        "ACEScct",
        "Rec.709",
        "Rec.2020",
        "sRGB",
        "Linear",
        "Log3G10",
        "LogC",
        "REDLog3G10",
    ]

    def __init__(
        self,
        output_dir: Union[str, Path],
        clip_name: str = "rotoscopy_output",
        frame_rate: float = 24.0,
        color_space: str = "ACES",
    ):
        self.output_dir = Path(output_dir)
        self.clip_name = clip_name
        self.frame_rate = frame_rate
        self.color_space = color_space

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "exr").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

        # Initialize components
        self.exr_writer = EXRWriter(compression=EXRCompression.ZIP)
        self.aov_manager = AOVManager()

        # Track exported frames
        self._exported_frames: List[int] = []
        self._resolution: Optional[Tuple[int, int]] = None

    def export_frame(
        self,
        result: Any,
        frame_number: int,
        source_image: Optional[np.ndarray] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export a single frame in Flame-compatible format.

        Args:
            result: ProcessingResult from the engine
            frame_number: Frame number
            source_image: Original source image (for beauty pass)
            additional_metadata: Extra metadata to embed

        Returns:
            Path to exported EXR file
        """
        # Create AOVs from result
        self.aov_manager.clear()
        self.aov_manager.create_from_result(result)

        # Get Flame-formatted package
        package = self.aov_manager.get_flame_package()

        # Set resolution from first frame
        if self._resolution is None:
            first_aov = list(package.values())[0]
            self._resolution = first_aov.shape[:2]

        # Generate output path
        output_path = self.output_dir / "exr" / f"{self.clip_name}.{frame_number:06d}.exr"

        # Write EXR with Flame metadata
        metadata = self._create_flame_metadata(frame_number, additional_metadata)

        self.exr_writer.write_multilayer(
            str(output_path),
            package,
            beauty=source_image,
            metadata=metadata,
        )

        # Track frame
        self._exported_frames.append(frame_number)

        # Save frame metadata
        self._save_frame_metadata(frame_number, result, additional_metadata)

        return output_path

    def _create_flame_metadata(
        self,
        frame_number: int,
        additional: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Create Flame-compatible EXR metadata."""
        metadata = {
            "software": "Ultimate Rotoscopy 1.0",
            "hostComputer": "rotoscopy_engine",
            "writer": "Ultimate Rotoscopy EXR Writer",
            "capDate": datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            "owner": "Ultimate Rotoscopy",
            "comments": f"Frame {frame_number} - {self.clip_name}",
            "framesPerSecond": str(int(self.frame_rate * 1000)),
            "timeCode": self._frame_to_timecode(frame_number),
            "type": "rotoscopy_pass",
            "colorSpace": self.color_space,
        }

        if additional:
            for key, value in additional.items():
                metadata[key] = str(value)

        return metadata

    def _frame_to_timecode(self, frame: int) -> str:
        """Convert frame number to timecode string."""
        fps = int(self.frame_rate)
        hours = frame // (fps * 60 * 60)
        minutes = (frame // (fps * 60)) % 60
        seconds = (frame // fps) % 60
        frames = frame % fps

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    def _save_frame_metadata(
        self,
        frame_number: int,
        result: Any,
        additional: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save frame metadata as JSON."""
        metadata_path = self.output_dir / "metadata" / f"{self.clip_name}.{frame_number:06d}.json"

        metadata = {
            "frame_number": frame_number,
            "timecode": self._frame_to_timecode(frame_number),
            "resolution": self._resolution,
            "aovs": self.aov_manager.list_aovs(),
            "processing_time_ms": getattr(result, "processing_time_ms", 0),
            "exported_at": datetime.now().isoformat(),
        }

        if hasattr(result, "metadata"):
            metadata["result_metadata"] = result.metadata

        if additional:
            metadata["additional"] = additional

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def generate_clip_xml(self) -> Path:
        """
        Generate Flame clip XML file.

        Returns:
            Path to generated XML file
        """
        if not self._exported_frames:
            raise RuntimeError("No frames exported yet")

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
        # EXR pattern
        exr_pattern = f"exr/{self.clip_name}.######.exr"

        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
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
            <aspectRatio>{clip_info.aspect_ratio}</aspectRatio>
            <fieldDominance>progressive</fieldDominance>
            <bitDepth>{clip_info.bit_depth}</bitDepth>
            <colourSpace>{self.color_space}</colourSpace>
        </format>
        <frameRate>{clip_info.frame_rate}</frameRate>
        <sampleRate>48000</sampleRate>
        <duration>
            <type>frames</type>
            <value>{clip_info.end_frame - clip_info.start_frame + 1}</value>
        </duration>
        <dropMode>false</dropMode>
    </essenceData>

    <userData>
        <generator>Ultimate Rotoscopy</generator>
        <version>1.0</version>
        <createdAt>{datetime.now().isoformat()}</createdAt>
        <aovCount>{len(self.aov_manager.list_aovs())}</aovCount>
    </userData>
</clip>'''

        return xml

    def generate_batch_setup(self) -> Path:
        """
        Generate Flame batch setup template.

        Returns:
            Path to generated batch file
        """
        batch_content = self._create_batch_setup()

        batch_path = self.output_dir / f"{self.clip_name}.batch"
        with open(batch_path, "w") as f:
            f.write(batch_content)

        return batch_path

    def _create_batch_setup(self) -> str:
        """Create Flame batch setup content."""
        # This creates a basic batch setup template
        # Full implementation would include node connections

        setup = f'''# Flame Batch Setup
# Generated by Ultimate Rotoscopy
# Clip: {self.clip_name}
# Resolution: {self._resolution}
# Frames: {min(self._exported_frames)} - {max(self._exported_frames)}

# Nodes:
# - Read: {self.clip_name}
# - Action: rotoscopy_comp
# - Write: output

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
        SETUP rotoscopy_action
    }}

    NODE Write1 {{
        TYPE Write
        POSITION 400 0
    }}

    LINK Read1 Action1 "Front"
    LINK Action1 Write1
}}

# Available AOVs:
'''

        for aov_name in self.aov_manager.list_aovs():
            setup += f"# - {aov_name}\n"

        return setup

    def export_gmask(
        self,
        mask_data: np.ndarray,
        frame_number: int,
        mask_name: str = "rotoscopy_mask",
    ) -> Path:
        """
        Export mask as Flame GMask format.

        Args:
            mask_data: Binary or alpha mask
            frame_number: Frame number
            mask_name: Name for the mask

        Returns:
            Path to exported gmask file
        """
        # Flame GMask is essentially a clip with specific metadata
        # For simplicity, export as a separate EXR that can be used in GMask

        gmask_dir = self.output_dir / "gmask"
        gmask_dir.mkdir(exist_ok=True)

        output_path = gmask_dir / f"{mask_name}.{frame_number:06d}.exr"

        self.exr_writer.write_single(
            str(output_path),
            mask_data.astype(np.float32),
            channel_name="A",
        )

        return output_path

    def export_action_setup(
        self,
        result: Any,
        frame_number: int,
    ) -> Path:
        """
        Export an Action node setup for the current result.

        This creates a basic Action setup that:
        - Has the source connected
        - Uses the matte from the result
        - Has depth and normal passes available

        Args:
            result: ProcessingResult
            frame_number: Frame number

        Returns:
            Path to Action setup file
        """
        action_dir = self.output_dir / "action"
        action_dir.mkdir(exist_ok=True)

        action_path = action_dir / f"{self.clip_name}_{frame_number:06d}.action"

        action_content = self._create_action_setup(result, frame_number)

        with open(action_path, "w") as f:
            f.write(action_content)

        return action_path

    def _create_action_setup(self, result: Any, frame_number: int) -> str:
        """Create Action node setup content."""
        setup = f'''# Flame Action Setup
# Generated by Ultimate Rotoscopy
# Frame: {frame_number}

ACTION_VERSION 2021.1

SCHEMATIC {{
    # Source media
    MEDIA Source {{
        TYPE Media
        CLIP "{self.output_dir / 'exr' / f'{self.clip_name}.{frame_number:06d}.exr'}"
        CHANNELS {{
            R -> R
            G -> G
            B -> B
            A -> A
        }}
    }}

    # Matte input
    MATTE AlphaMatte {{
        TYPE Media
        CHANNEL A
        SOURCE Source
    }}

    # Depth channel
    CHANNEL Depth {{
        TYPE Media
        CHANNEL "depth.Z"
        SOURCE Source
    }}

    # Normal channels
    CHANNEL NormalX {{
        TYPE Media
        CHANNEL "normal.X"
        SOURCE Source
    }}

    CHANNEL NormalY {{
        TYPE Media
        CHANNEL "normal.Y"
        SOURCE Source
    }}

    CHANNEL NormalZ {{
        TYPE Media
        CHANNEL "normal.Z"
        SOURCE Source
    }}

    # Axis for 3D compositing
    AXIS MainAxis {{
        TYPE Axis
        POSITION 0 0 0
        ROTATION 0 0 0
        SCALE 1 1 1
    }}

    # Result node
    RESULT Output {{
        TYPE Result
        MEDIA Source
        MATTE AlphaMatte
    }}
}}

# Processing metadata
METADATA {{
    processing_time_ms: {getattr(result, 'processing_time_ms', 0)}
    has_depth: {result.depth_map is not None if hasattr(result, 'depth_map') else False}
    has_normals: {result.normals is not None if hasattr(result, 'normals') else False}
    has_matte: {result.alpha is not None if hasattr(result, 'alpha') else False}
}}
'''
        return setup

    def finalize(self) -> Dict[str, Path]:
        """
        Finalize export and generate all support files.

        Returns:
            Dictionary of generated files
        """
        files = {}

        # Generate clip XML
        files["clip"] = self.generate_clip_xml()

        # Generate batch setup
        files["batch"] = self.generate_batch_setup()

        # Generate summary JSON
        summary_path = self.output_dir / f"{self.clip_name}_summary.json"
        summary = {
            "clip_name": self.clip_name,
            "frame_rate": self.frame_rate,
            "resolution": self._resolution,
            "color_space": self.color_space,
            "frame_range": {
                "start": min(self._exported_frames),
                "end": max(self._exported_frames),
                "count": len(self._exported_frames),
            },
            "aovs": self.aov_manager.list_aovs(),
            "output_directory": str(self.output_dir),
            "files": {
                "exr_pattern": f"exr/{self.clip_name}.######.exr",
                "clip": str(files["clip"]),
                "batch": str(files["batch"]),
            },
            "generated_at": datetime.now().isoformat(),
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        files["summary"] = summary_path

        return files

    def get_export_statistics(self) -> Dict[str, Any]:
        """Get statistics about the export."""
        return {
            "clip_name": self.clip_name,
            "frames_exported": len(self._exported_frames),
            "frame_range": (
                min(self._exported_frames) if self._exported_frames else 0,
                max(self._exported_frames) if self._exported_frames else 0,
            ),
            "resolution": self._resolution,
            "output_directory": str(self.output_dir),
            "aov_statistics": self.aov_manager.get_statistics(),
        }
