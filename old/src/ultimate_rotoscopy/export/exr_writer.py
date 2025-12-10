"""
EXR Writer for Ultimate Rotoscopy
==================================

Professional OpenEXR writer with multi-layer support for VFX workflows.
Supports Flame, Nuke, Fusion, and other compositing software.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class EXRCompression(Enum):
    """EXR compression methods."""
    NONE = "none"
    RLE = "rle"
    ZIPS = "zips"
    ZIP = "zip"
    PIZ = "piz"
    PXR24 = "pxr24"
    B44 = "b44"
    B44A = "b44a"
    DWAA = "dwaa"
    DWAB = "dwab"


class EXRPixelType(Enum):
    """EXR pixel types."""
    HALF = "half"      # 16-bit float
    FLOAT = "float"    # 32-bit float
    UINT = "uint"      # 32-bit unsigned int


@dataclass
class EXRChannel:
    """Configuration for an EXR channel."""
    name: str
    data: np.ndarray
    pixel_type: EXRPixelType = EXRPixelType.HALF


@dataclass
class EXRLayer:
    """Configuration for an EXR layer."""
    name: str
    channels: Dict[str, np.ndarray]  # channel_name -> data
    pixel_type: EXRPixelType = EXRPixelType.HALF


class EXRWriter:
    """
    Professional EXR writer for VFX workflows.

    Features:
    - Multi-layer/multi-channel EXR support
    - Flame-compatible layer naming
    - Nuke-compatible AOV structure
    - Various compression options
    - Half/Float precision

    Example:
        >>> writer = EXRWriter()
        >>>
        >>> # Write single channel
        >>> writer.write_single("alpha.exr", alpha_data)
        >>>
        >>> # Write multi-layer
        >>> layers = {
        ...     "alpha": alpha_data,
        ...     "depth.Z": depth_data,
        ...     "normal.X": normal_x,
        ...     "normal.Y": normal_y,
        ...     "normal.Z": normal_z,
        ... }
        >>> writer.write_multilayer("output.exr", layers, rgb_image)
    """

    def __init__(
        self,
        compression: EXRCompression = EXRCompression.ZIP,
        pixel_type: EXRPixelType = EXRPixelType.HALF,
    ):
        self.compression = compression
        self.pixel_type = pixel_type
        self._check_openexr()

    def _check_openexr(self) -> None:
        """Check if OpenEXR is available."""
        try:
            import OpenEXR
            import Imath
            self._openexr_available = True
        except ImportError:
            self._openexr_available = False
            print("Warning: OpenEXR not available. Using fallback writer.")

    def write_single(
        self,
        path: Union[str, Path],
        data: np.ndarray,
        channel_name: str = "Y",
    ) -> None:
        """
        Write a single-channel EXR file.

        Args:
            path: Output file path
            data: 2D array of pixel data
            channel_name: Channel name (Y, R, G, B, A, Z, etc.)
        """
        path = Path(path)

        if self._openexr_available:
            self._write_single_openexr(path, data, channel_name)
        else:
            self._write_single_fallback(path, data)

    def write_rgb(
        self,
        path: Union[str, Path],
        data: np.ndarray,
        alpha: Optional[np.ndarray] = None,
    ) -> None:
        """
        Write an RGB(A) EXR file.

        Args:
            path: Output file path
            data: HxWx3 or HxWx4 array
            alpha: Optional separate alpha channel
        """
        path = Path(path)

        if data.ndim != 3 or data.shape[2] < 3:
            raise ValueError("Data must be HxWx3 or HxWx4")

        if self._openexr_available:
            self._write_rgb_openexr(path, data, alpha)
        else:
            self._write_rgb_fallback(path, data, alpha)

    def write_multilayer(
        self,
        path: Union[str, Path],
        aovs: Dict[str, np.ndarray],
        beauty: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Write a multi-layer EXR file with AOVs.

        AOV naming conventions (Flame/Nuke compatible):
        - "alpha" -> A channel
        - "depth" or "depth.Z" -> depth.Z
        - "normal.X/Y/Z" -> normal.X, normal.Y, normal.Z
        - "matte" -> matte.R (or matte.A)
        - "motion.X/Y" -> motion vectors

        Args:
            path: Output file path
            aovs: Dictionary of AOV name -> data
            beauty: Optional beauty/RGB pass
            metadata: Optional EXR metadata
        """
        path = Path(path)

        if self._openexr_available:
            self._write_multilayer_openexr(path, aovs, beauty, metadata)
        else:
            self._write_multilayer_fallback(path, aovs, beauty)

    def _write_single_openexr(
        self,
        path: Path,
        data: np.ndarray,
        channel_name: str,
    ) -> None:
        """Write single channel using OpenEXR."""
        import OpenEXR
        import Imath

        h, w = data.shape[:2]

        # Convert to float32
        data = data.astype(np.float32)

        # Create header
        header = OpenEXR.Header(w, h)
        header["compression"] = self._get_compression_type()

        # Create channel
        half = Imath.PixelType(Imath.PixelType.HALF)
        header["channels"] = {channel_name: Imath.Channel(half)}

        # Write file
        exr = OpenEXR.OutputFile(str(path), header)
        exr.writePixels({channel_name: data.tobytes()})
        exr.close()

    def _write_rgb_openexr(
        self,
        path: Path,
        data: np.ndarray,
        alpha: Optional[np.ndarray],
    ) -> None:
        """Write RGB(A) using OpenEXR."""
        import OpenEXR
        import Imath

        h, w = data.shape[:2]

        # Ensure float32
        data = data.astype(np.float32)

        # Create header
        header = OpenEXR.Header(w, h)
        header["compression"] = self._get_compression_type()

        half = Imath.PixelType(Imath.PixelType.HALF)

        channels = {
            "R": Imath.Channel(half),
            "G": Imath.Channel(half),
            "B": Imath.Channel(half),
        }

        channel_data = {
            "R": data[..., 0].tobytes(),
            "G": data[..., 1].tobytes(),
            "B": data[..., 2].tobytes(),
        }

        # Add alpha
        if alpha is not None:
            channels["A"] = Imath.Channel(half)
            channel_data["A"] = alpha.astype(np.float32).tobytes()
        elif data.shape[2] == 4:
            channels["A"] = Imath.Channel(half)
            channel_data["A"] = data[..., 3].tobytes()

        header["channels"] = channels

        # Write
        exr = OpenEXR.OutputFile(str(path), header)
        exr.writePixels(channel_data)
        exr.close()

    def _write_multilayer_openexr(
        self,
        path: Path,
        aovs: Dict[str, np.ndarray],
        beauty: Optional[np.ndarray],
        metadata: Optional[Dict[str, str]],
    ) -> None:
        """Write multi-layer EXR using OpenEXR."""
        import OpenEXR
        import Imath

        # Determine dimensions from first AOV
        first_aov = list(aovs.values())[0]
        h, w = first_aov.shape[:2]

        # Create header
        header = OpenEXR.Header(w, h)
        header["compression"] = self._get_compression_type()

        half = Imath.PixelType(Imath.PixelType.HALF)

        channels = {}
        channel_data = {}

        # Add beauty pass (RGBA)
        if beauty is not None:
            beauty = beauty.astype(np.float32)
            if beauty.max() > 1:
                beauty = beauty / 255.0

            channels["R"] = Imath.Channel(half)
            channels["G"] = Imath.Channel(half)
            channels["B"] = Imath.Channel(half)

            channel_data["R"] = beauty[..., 0].tobytes()
            channel_data["G"] = beauty[..., 1].tobytes()
            channel_data["B"] = beauty[..., 2].tobytes()

            if beauty.shape[2] == 4:
                channels["A"] = Imath.Channel(half)
                channel_data["A"] = beauty[..., 3].tobytes()

        # Add AOVs
        for aov_name, aov_data in aovs.items():
            aov_data = aov_data.astype(np.float32)

            # Parse AOV name for layer structure
            channel_names = self._get_channel_names(aov_name, aov_data)

            for ch_name, ch_data in channel_names:
                channels[ch_name] = Imath.Channel(half)
                channel_data[ch_name] = ch_data.tobytes()

        header["channels"] = channels

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                header[key] = value

        # Add standard metadata
        header["software"] = "Ultimate Rotoscopy 1.0"
        header["comments"] = "Generated by Ultimate Rotoscopy"

        # Write
        exr = OpenEXR.OutputFile(str(path), header)
        exr.writePixels(channel_data)
        exr.close()

    def _get_channel_names(
        self,
        aov_name: str,
        data: np.ndarray
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Convert AOV name to EXR channel names.

        Follows Nuke/Flame conventions.
        """
        channels = []

        if data.ndim == 2:
            # Single channel
            if aov_name in ("alpha", "A", "matte"):
                channels.append(("A", data))
            elif aov_name in ("depth", "Z", "z_depth"):
                channels.append(("depth.Z", data))
            elif "." in aov_name:
                channels.append((aov_name, data))
            else:
                # Default to layer.R for single channel
                channels.append((f"{aov_name}.R", data))

        elif data.ndim == 3:
            if data.shape[2] == 3:
                # RGB channels
                if aov_name in ("normals", "normal", "N"):
                    channels.append(("normal.X", data[..., 0]))
                    channels.append(("normal.Y", data[..., 1]))
                    channels.append(("normal.Z", data[..., 2]))
                elif aov_name in ("motion", "velocity", "mv"):
                    channels.append(("motion.X", data[..., 0]))
                    channels.append(("motion.Y", data[..., 1]))
                    if data.shape[2] > 2:
                        channels.append(("motion.Z", data[..., 2]))
                elif aov_name in ("position", "P", "world_position"):
                    channels.append(("position.X", data[..., 0]))
                    channels.append(("position.Y", data[..., 1]))
                    channels.append(("position.Z", data[..., 2]))
                else:
                    channels.append((f"{aov_name}.R", data[..., 0]))
                    channels.append((f"{aov_name}.G", data[..., 1]))
                    channels.append((f"{aov_name}.B", data[..., 2]))

            elif data.shape[2] == 4:
                # RGBA
                channels.append((f"{aov_name}.R", data[..., 0]))
                channels.append((f"{aov_name}.G", data[..., 1]))
                channels.append((f"{aov_name}.B", data[..., 2]))
                channels.append((f"{aov_name}.A", data[..., 3]))

        return channels

    def _get_compression_type(self):
        """Get OpenEXR compression type."""
        import Imath

        compression_map = {
            EXRCompression.NONE: Imath.Compression.NO_COMPRESSION,
            EXRCompression.RLE: Imath.Compression.RLE_COMPRESSION,
            EXRCompression.ZIPS: Imath.Compression.ZIPS_COMPRESSION,
            EXRCompression.ZIP: Imath.Compression.ZIP_COMPRESSION,
            EXRCompression.PIZ: Imath.Compression.PIZ_COMPRESSION,
            EXRCompression.PXR24: Imath.Compression.PXR24_COMPRESSION,
            EXRCompression.B44: Imath.Compression.B44_COMPRESSION,
            EXRCompression.B44A: Imath.Compression.B44A_COMPRESSION,
            EXRCompression.DWAA: Imath.Compression.DWAA_COMPRESSION,
            EXRCompression.DWAB: Imath.Compression.DWAB_COMPRESSION,
        }
        return compression_map.get(self.compression, Imath.Compression.ZIP_COMPRESSION)

    def _write_single_fallback(self, path: Path, data: np.ndarray) -> None:
        """Fallback writer using imageio."""
        try:
            import imageio
            imageio.imwrite(str(path), data.astype(np.float32))
        except Exception:
            # Last resort: save as 16-bit TIFF
            from PIL import Image
            data_16bit = (np.clip(data, 0, 1) * 65535).astype(np.uint16)
            Image.fromarray(data_16bit).save(path.with_suffix(".tiff"))

    def _write_rgb_fallback(
        self,
        path: Path,
        data: np.ndarray,
        alpha: Optional[np.ndarray],
    ) -> None:
        """Fallback RGB writer."""
        try:
            import imageio

            if alpha is not None:
                data = np.concatenate([data, alpha[..., None]], axis=-1)

            imageio.imwrite(str(path), data.astype(np.float32))
        except Exception:
            from PIL import Image
            data_8bit = (np.clip(data, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(data_8bit).save(path.with_suffix(".png"))

    def _write_multilayer_fallback(
        self,
        path: Path,
        aovs: Dict[str, np.ndarray],
        beauty: Optional[np.ndarray],
    ) -> None:
        """Fallback: write each AOV as separate file."""
        base_path = path.parent / path.stem

        if beauty is not None:
            self._write_rgb_fallback(
                Path(f"{base_path}_beauty.exr"),
                beauty,
                None
            )

        for aov_name, aov_data in aovs.items():
            aov_path = Path(f"{base_path}_{aov_name}.exr")

            if aov_data.ndim == 2:
                self._write_single_fallback(aov_path, aov_data)
            else:
                self._write_rgb_fallback(aov_path, aov_data, None)


class EXRReader:
    """
    EXR file reader for loading multi-channel EXR files.

    Example:
        >>> reader = EXRReader()
        >>> data = reader.read("input.exr")
        >>> alpha = data.get("A")
        >>> depth = data.get("depth.Z")
    """

    def __init__(self):
        self._check_openexr()

    def _check_openexr(self) -> None:
        """Check OpenEXR availability."""
        try:
            import OpenEXR
            self._openexr_available = True
        except ImportError:
            self._openexr_available = False

    def read(self, path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Read all channels from an EXR file.

        Returns:
            Dictionary mapping channel names to numpy arrays
        """
        path = Path(path)

        if not self._openexr_available:
            return self._read_fallback(path)

        return self._read_openexr(path)

    def _read_openexr(self, path: Path) -> Dict[str, np.ndarray]:
        """Read using OpenEXR."""
        import OpenEXR
        import Imath

        exr = OpenEXR.InputFile(str(path))
        header = exr.header()

        # Get dimensions
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1

        # Read all channels
        channels = {}
        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        for ch_name in header["channels"].keys():
            raw = exr.channel(ch_name, pt)
            data = np.frombuffer(raw, dtype=np.float32)
            data = data.reshape(h, w)
            channels[ch_name] = data

        exr.close()
        return channels

    def _read_fallback(self, path: Path) -> Dict[str, np.ndarray]:
        """Fallback reader."""
        try:
            import imageio
            data = imageio.imread(str(path))

            result = {}
            if data.ndim == 2:
                result["Y"] = data
            else:
                if data.shape[2] >= 1:
                    result["R"] = data[..., 0]
                if data.shape[2] >= 2:
                    result["G"] = data[..., 1]
                if data.shape[2] >= 3:
                    result["B"] = data[..., 2]
                if data.shape[2] >= 4:
                    result["A"] = data[..., 3]

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to read EXR: {e}")

    def get_channels(self, path: Union[str, Path]) -> List[str]:
        """Get list of channel names in an EXR file."""
        if not self._openexr_available:
            return []

        import OpenEXR

        exr = OpenEXR.InputFile(str(path))
        channels = list(exr.header()["channels"].keys())
        exr.close()
        return channels

    def get_metadata(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Get metadata from EXR file."""
        if not self._openexr_available:
            return {}

        import OpenEXR

        exr = OpenEXR.InputFile(str(path))
        header = exr.header()

        metadata = {}
        for key, value in header.items():
            if key not in ("channels", "compression", "dataWindow", "displayWindow"):
                metadata[key] = str(value)

        exr.close()
        return metadata
