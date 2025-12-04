"""
AOV Manager for Ultimate Rotoscopy
===================================

Manages Arbitrary Output Variables (AOVs) for professional compositing workflows.
Compatible with Flame, Nuke, Fusion, and other compositing software.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class AOVType(Enum):
    """Standard AOV types for VFX workflows."""
    # Matte/Alpha
    ALPHA = "alpha"
    MATTE = "matte"
    OBJECT_ID = "object_id"
    CRYPTO_OBJECT = "crypto_object"
    CRYPTO_MATERIAL = "crypto_material"
    CRYPTO_ASSET = "crypto_asset"

    # Depth
    DEPTH = "depth"
    Z_DEPTH = "z_depth"
    DEPTH_NORMALIZED = "depth_normalized"
    DISPARITY = "disparity"

    # Normals
    NORMAL = "normal"
    NORMAL_CAMERA = "normal_camera"
    NORMAL_WORLD = "normal_world"

    # Position
    POSITION = "position"
    POSITION_WORLD = "position_world"

    # Motion
    MOTION = "motion"
    MOTION_VECTOR = "motion_vector"
    MOTION_BLUR_MASK = "motion_blur_mask"

    # Detail masks
    EDGE_MASK = "edge_mask"
    HAIR_MASK = "hair_mask"
    TRANSPARENCY_MASK = "transparency_mask"

    # Beauty
    BEAUTY = "beauty"
    RGBA = "rgba"
    FOREGROUND = "foreground"

    # Lighting (for future)
    DIFFUSE = "diffuse"
    SPECULAR = "specular"
    AMBIENT_OCCLUSION = "ambient_occlusion"

    # Custom
    CUSTOM = "custom"


@dataclass
class AOVDefinition:
    """Definition of an AOV."""
    name: str
    aov_type: AOVType
    channels: int = 1  # Number of channels (1=single, 3=RGB, 4=RGBA)
    data_range: Tuple[float, float] = (0.0, 1.0)
    description: str = ""
    layer_name: Optional[str] = None  # For multi-layer EXR
    metadata: Dict[str, Any] = field(default_factory=dict)


class AOVManager:
    """
    Manages AOVs (Arbitrary Output Variables) for compositing.

    Features:
    - Standard AOV definitions for common passes
    - Flame-compatible naming conventions
    - Nuke-compatible layer structure
    - AOV validation and normalization
    - Multi-layer EXR organization

    Example:
        >>> manager = AOVManager()
        >>>
        >>> # Add processing results as AOVs
        >>> manager.add_aov("alpha", result.alpha, AOVType.ALPHA)
        >>> manager.add_aov("depth", result.depth_normalized, AOVType.DEPTH)
        >>> manager.add_aov("normal", result.normals, AOVType.NORMAL)
        >>>
        >>> # Get Flame-compatible package
        >>> package = manager.get_flame_package()
    """

    # Standard AOV definitions
    STANDARD_AOVS = {
        AOVType.ALPHA: AOVDefinition(
            name="alpha",
            aov_type=AOVType.ALPHA,
            channels=1,
            data_range=(0.0, 1.0),
            description="Alpha/transparency matte",
            layer_name="A",
        ),
        AOVType.MATTE: AOVDefinition(
            name="matte",
            aov_type=AOVType.MATTE,
            channels=1,
            data_range=(0.0, 1.0),
            description="Object matte",
            layer_name="matte",
        ),
        AOVType.DEPTH: AOVDefinition(
            name="depth",
            aov_type=AOVType.DEPTH,
            channels=1,
            data_range=(0.0, 1.0),
            description="Normalized depth",
            layer_name="depth",
        ),
        AOVType.Z_DEPTH: AOVDefinition(
            name="z_depth",
            aov_type=AOVType.Z_DEPTH,
            channels=1,
            data_range=(0.1, 100.0),
            description="Z-depth in scene units",
            layer_name="depth",
        ),
        AOVType.NORMAL: AOVDefinition(
            name="normal",
            aov_type=AOVType.NORMAL,
            channels=3,
            data_range=(-1.0, 1.0),
            description="Camera-space normals",
            layer_name="normal",
        ),
        AOVType.NORMAL_WORLD: AOVDefinition(
            name="normal_world",
            aov_type=AOVType.NORMAL_WORLD,
            channels=3,
            data_range=(-1.0, 1.0),
            description="World-space normals",
            layer_name="normal_world",
        ),
        AOVType.POSITION: AOVDefinition(
            name="position",
            aov_type=AOVType.POSITION,
            channels=3,
            description="3D position",
            layer_name="position",
        ),
        AOVType.MOTION: AOVDefinition(
            name="motion",
            aov_type=AOVType.MOTION,
            channels=2,
            description="Motion vectors",
            layer_name="motion",
        ),
        AOVType.DISPARITY: AOVDefinition(
            name="disparity",
            aov_type=AOVType.DISPARITY,
            channels=1,
            data_range=(0.0, 1.0),
            description="Disparity (inverse depth)",
            layer_name="disparity",
        ),
        AOVType.EDGE_MASK: AOVDefinition(
            name="edge_mask",
            aov_type=AOVType.EDGE_MASK,
            channels=1,
            data_range=(0.0, 1.0),
            description="Edge regions mask",
            layer_name="edge",
        ),
        AOVType.HAIR_MASK: AOVDefinition(
            name="hair_mask",
            aov_type=AOVType.HAIR_MASK,
            channels=1,
            data_range=(0.0, 1.0),
            description="Hair/fine detail mask",
            layer_name="hair",
        ),
        AOVType.FOREGROUND: AOVDefinition(
            name="foreground",
            aov_type=AOVType.FOREGROUND,
            channels=3,
            data_range=(0.0, 1.0),
            description="Extracted foreground",
            layer_name="foreground",
        ),
        AOVType.BEAUTY: AOVDefinition(
            name="beauty",
            aov_type=AOVType.BEAUTY,
            channels=4,
            data_range=(0.0, 1.0),
            description="Beauty/RGBA pass",
            layer_name=None,  # Uses R, G, B, A directly
        ),
    }

    def __init__(self):
        self.aovs: Dict[str, Tuple[AOVDefinition, np.ndarray]] = {}
        self._resolution: Optional[Tuple[int, int]] = None

    def add_aov(
        self,
        name: str,
        data: np.ndarray,
        aov_type: Optional[AOVType] = None,
        normalize: bool = True,
    ) -> None:
        """
        Add an AOV to the manager.

        Args:
            name: AOV name
            data: AOV data array
            aov_type: Type of AOV (auto-detected if not provided)
            normalize: Normalize data to expected range
        """
        if data is None:
            return

        # Detect type if not provided
        if aov_type is None:
            aov_type = self._detect_aov_type(name, data)

        # Get or create definition
        if aov_type in self.STANDARD_AOVS:
            definition = self.STANDARD_AOVS[aov_type]
            definition = AOVDefinition(
                name=name,
                aov_type=aov_type,
                channels=definition.channels,
                data_range=definition.data_range,
                description=definition.description,
                layer_name=definition.layer_name,
            )
        else:
            channels = data.shape[2] if data.ndim == 3 else 1
            definition = AOVDefinition(
                name=name,
                aov_type=AOVType.CUSTOM,
                channels=channels,
            )

        # Set resolution from first AOV
        if self._resolution is None:
            self._resolution = data.shape[:2]

        # Validate and normalize
        data = self._validate_data(data, definition)
        if normalize:
            data = self._normalize_data(data, definition)

        self.aovs[name] = (definition, data)

    def _detect_aov_type(self, name: str, data: np.ndarray) -> AOVType:
        """Detect AOV type from name and data shape."""
        name_lower = name.lower()

        if "alpha" in name_lower or name_lower == "a":
            return AOVType.ALPHA
        elif "matte" in name_lower:
            return AOVType.MATTE
        elif "depth" in name_lower or name_lower == "z":
            if "norm" in name_lower:
                return AOVType.DEPTH
            return AOVType.Z_DEPTH
        elif "normal" in name_lower or name_lower == "n":
            if "world" in name_lower:
                return AOVType.NORMAL_WORLD
            return AOVType.NORMAL
        elif "position" in name_lower or name_lower == "p":
            return AOVType.POSITION
        elif "motion" in name_lower or "velocity" in name_lower or name_lower == "mv":
            return AOVType.MOTION
        elif "disparity" in name_lower:
            return AOVType.DISPARITY
        elif "edge" in name_lower:
            return AOVType.EDGE_MASK
        elif "hair" in name_lower:
            return AOVType.HAIR_MASK
        elif "foreground" in name_lower or name_lower == "fg":
            return AOVType.FOREGROUND
        elif "beauty" in name_lower or name_lower == "rgba":
            return AOVType.BEAUTY

        return AOVType.CUSTOM

    def _validate_data(
        self,
        data: np.ndarray,
        definition: AOVDefinition
    ) -> np.ndarray:
        """Validate and fix AOV data."""
        # Ensure float32
        if data.dtype != np.float32:
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0
            elif data.dtype == np.uint16:
                data = data.astype(np.float32) / 65535.0
            else:
                data = data.astype(np.float32)

        # Check resolution
        if self._resolution and data.shape[:2] != self._resolution:
            import cv2
            data = cv2.resize(
                data,
                (self._resolution[1], self._resolution[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # Ensure correct channel count
        if data.ndim == 2 and definition.channels > 1:
            data = np.stack([data] * definition.channels, axis=-1)
        elif data.ndim == 3 and data.shape[2] != definition.channels:
            if data.shape[2] > definition.channels:
                data = data[..., :definition.channels]
            else:
                # Pad with zeros
                padding = np.zeros(
                    (*data.shape[:2], definition.channels - data.shape[2]),
                    dtype=np.float32
                )
                data = np.concatenate([data, padding], axis=-1)

        return data

    def _normalize_data(
        self,
        data: np.ndarray,
        definition: AOVDefinition
    ) -> np.ndarray:
        """Normalize data to expected range."""
        data_min, data_max = definition.data_range

        if definition.aov_type == AOVType.NORMAL:
            # Ensure normals are in [-1, 1] range
            if data.min() >= 0:
                # Convert from [0, 1] to [-1, 1]
                data = data * 2 - 1

            # Re-normalize vectors
            norm = np.linalg.norm(data, axis=-1, keepdims=True)
            data = data / (norm + 1e-8)

        elif definition.aov_type in (AOVType.ALPHA, AOVType.MATTE, AOVType.DEPTH):
            # Clip to [0, 1]
            data = np.clip(data, 0, 1)

        return data

    def get_aov(self, name: str) -> Optional[np.ndarray]:
        """Get AOV data by name."""
        if name in self.aovs:
            return self.aovs[name][1]
        return None

    def get_definition(self, name: str) -> Optional[AOVDefinition]:
        """Get AOV definition by name."""
        if name in self.aovs:
            return self.aovs[name][0]
        return None

    def list_aovs(self) -> List[str]:
        """List all AOV names."""
        return list(self.aovs.keys())

    def get_package(self) -> Dict[str, np.ndarray]:
        """Get all AOVs as a dictionary."""
        return {name: data for name, (_, data) in self.aovs.items()}

    def get_flame_package(self) -> Dict[str, np.ndarray]:
        """
        Get AOVs formatted for Autodesk Flame.

        Flame expects specific naming conventions:
        - Alpha channel as 'A'
        - Depth as 'depth.Z'
        - Normals as 'normal.X', 'normal.Y', 'normal.Z'
        """
        package = {}

        for name, (definition, data) in self.aovs.items():
            if definition.aov_type == AOVType.ALPHA:
                package["A"] = data
            elif definition.aov_type == AOVType.MATTE:
                package["matte.A"] = data
            elif definition.aov_type in (AOVType.DEPTH, AOVType.Z_DEPTH):
                package["depth.Z"] = data
            elif definition.aov_type in (AOVType.NORMAL, AOVType.NORMAL_WORLD):
                prefix = "normal" if definition.aov_type == AOVType.NORMAL else "normal_world"
                package[f"{prefix}.X"] = data[..., 0]
                package[f"{prefix}.Y"] = data[..., 1]
                package[f"{prefix}.Z"] = data[..., 2]
            elif definition.aov_type == AOVType.POSITION:
                package["position.X"] = data[..., 0]
                package["position.Y"] = data[..., 1]
                package["position.Z"] = data[..., 2]
            elif definition.aov_type == AOVType.MOTION:
                package["motion.X"] = data[..., 0]
                package["motion.Y"] = data[..., 1]
            elif definition.aov_type == AOVType.FOREGROUND:
                package["foreground.R"] = data[..., 0]
                package["foreground.G"] = data[..., 1]
                package["foreground.B"] = data[..., 2]
            elif definition.aov_type == AOVType.DISPARITY:
                package["disparity.Z"] = data
            elif definition.aov_type == AOVType.BEAUTY:
                package["R"] = data[..., 0]
                package["G"] = data[..., 1]
                package["B"] = data[..., 2]
                if data.shape[2] == 4:
                    package["A"] = data[..., 3]
            else:
                # Custom AOV
                if data.ndim == 2:
                    package[f"{name}.R"] = data
                else:
                    for i, suffix in enumerate(["R", "G", "B", "A"][:data.shape[2]]):
                        package[f"{name}.{suffix}"] = data[..., i]

        return package

    def get_nuke_package(self) -> Dict[str, np.ndarray]:
        """
        Get AOVs formatted for Nuke.

        Nuke uses similar conventions to Flame but with some differences.
        """
        return self.get_flame_package()  # Similar format

    def get_fusion_package(self) -> Dict[str, np.ndarray]:
        """
        Get AOVs formatted for DaVinci Resolve Fusion.
        """
        package = {}

        for name, (definition, data) in self.aovs.items():
            if definition.aov_type == AOVType.ALPHA:
                package["Alpha"] = data
            elif definition.aov_type == AOVType.DEPTH:
                package["Z"] = data
            elif definition.aov_type == AOVType.NORMAL:
                package["NX"] = data[..., 0]
                package["NY"] = data[..., 1]
                package["NZ"] = data[..., 2]
            else:
                # Use simple naming
                package[name] = data

        return package

    def create_from_result(self, result: Any) -> None:
        """
        Create AOVs from a ProcessingResult object.

        Args:
            result: ProcessingResult from the engine
        """
        if hasattr(result, "alpha") and result.alpha is not None:
            self.add_aov("alpha", result.alpha, AOVType.ALPHA)

        if hasattr(result, "depth_normalized") and result.depth_normalized is not None:
            self.add_aov("depth", result.depth_normalized, AOVType.DEPTH)

        if hasattr(result, "z_depth") and result.z_depth is not None:
            self.add_aov("z_depth", result.z_depth, AOVType.Z_DEPTH)

        if hasattr(result, "normals") and result.normals is not None:
            self.add_aov("normal", result.normals, AOVType.NORMAL)

        if hasattr(result, "disparity") and result.disparity is not None:
            self.add_aov("disparity", result.disparity, AOVType.DISPARITY)

        if hasattr(result, "foreground") and result.foreground is not None:
            self.add_aov("foreground", result.foreground, AOVType.FOREGROUND)

        if hasattr(result, "edge_mask") and result.edge_mask is not None:
            self.add_aov("edge_mask", result.edge_mask, AOVType.EDGE_MASK)

        if hasattr(result, "hair_mask") and result.hair_mask is not None:
            self.add_aov("hair_mask", result.hair_mask, AOVType.HAIR_MASK)

        if hasattr(result, "motion_mask") and result.motion_mask is not None:
            self.add_aov("motion_blur_mask", result.motion_mask, AOVType.MOTION_BLUR_MASK)

        if hasattr(result, "masks") and result.masks is not None:
            # Add first mask as segmentation
            if len(result.masks) > 0:
                self.add_aov("segmentation", result.masks[0].astype(np.float32), AOVType.MATTE)

    def clear(self) -> None:
        """Clear all AOVs."""
        self.aovs.clear()
        self._resolution = None

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored AOVs."""
        total_size = 0
        aov_info = {}

        for name, (definition, data) in self.aovs.items():
            size_mb = data.nbytes / (1024 * 1024)
            total_size += size_mb
            aov_info[name] = {
                "type": definition.aov_type.value,
                "shape": data.shape,
                "dtype": str(data.dtype),
                "size_mb": size_mb,
                "min": float(data.min()),
                "max": float(data.max()),
            }

        return {
            "num_aovs": len(self.aovs),
            "resolution": self._resolution,
            "total_size_mb": total_size,
            "aovs": aov_info,
        }
