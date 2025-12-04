"""
OpenFX Plugin Wrapper for Ultimate Rotoscopy
=============================================

Provides OpenFX (OFX) plugin interface for using Ultimate Rotoscopy
in professional compositing applications:
- Nuke
- Flame
- DaVinci Resolve
- Natron
- Vegas Pro

OpenFX is the standard plugin API for visual effects software.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np


class OFXParamType(Enum):
    """OpenFX parameter types."""
    DOUBLE = "double"
    INT = "int"
    BOOL = "bool"
    CHOICE = "choice"
    STRING = "string"
    RGBA = "rgba"
    RGB = "rgb"
    DOUBLE2D = "double2d"
    INT2D = "int2d"
    CUSTOM = "custom"
    PUSH_BUTTON = "pushbutton"
    GROUP = "group"
    PAGE = "page"


class OFXImageField(Enum):
    """Image field order."""
    NONE = "none"
    LOWER = "lower"
    UPPER = "upper"
    BOTH = "both"


class OFXContext(Enum):
    """OpenFX contexts."""
    FILTER = "filter"
    GENERATOR = "generator"
    TRANSITION = "transition"
    GENERAL = "general"
    RETIMER = "retimer"


@dataclass
class OFXParameter:
    """Definition of an OpenFX parameter."""
    name: str
    param_type: OFXParamType
    label: str
    hint: str = ""
    default: Any = None
    min_value: Any = None
    max_value: Any = None
    choices: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    enabled: bool = True
    secret: bool = False


@dataclass
class OFXClip:
    """Definition of an OpenFX clip (input/output)."""
    name: str
    optional: bool = False
    supports_tiles: bool = True
    temporal_access: bool = False


@dataclass
class OFXPluginInfo:
    """OpenFX plugin metadata."""
    identifier: str
    label: str
    version_major: int = 1
    version_minor: int = 0
    group: str = "Ultimate Rotoscopy"
    description: str = ""
    context: OFXContext = OFXContext.FILTER


class OFXImageEffect:
    """
    Base class for OpenFX image effects.

    Implement this class to create an OFX-compatible plugin
    that can be loaded in Nuke, Flame, etc.
    """

    def __init__(self, info: OFXPluginInfo):
        self.info = info
        self._parameters: Dict[str, OFXParameter] = {}
        self._clips: Dict[str, OFXClip] = {}
        self._param_values: Dict[str, Any] = {}

    def define_parameter(self, param: OFXParameter):
        """Define a plugin parameter."""
        self._parameters[param.name] = param
        self._param_values[param.name] = param.default

    def define_clip(self, clip: OFXClip):
        """Define an input/output clip."""
        self._clips[clip.name] = clip

    def get_param(self, name: str) -> Any:
        """Get parameter value."""
        return self._param_values.get(name)

    def set_param(self, name: str, value: Any):
        """Set parameter value."""
        self._param_values[name] = value

    def describe(self) -> Dict[str, Any]:
        """Describe the plugin for the host."""
        return {
            "identifier": self.info.identifier,
            "label": self.info.label,
            "version": (self.info.version_major, self.info.version_minor),
            "group": self.info.group,
            "description": self.info.description,
            "context": self.info.context.value,
            "parameters": {k: self._param_to_dict(v) for k, v in self._parameters.items()},
            "clips": {k: self._clip_to_dict(v) for k, v in self._clips.items()},
        }

    def _param_to_dict(self, param: OFXParameter) -> Dict[str, Any]:
        """Convert parameter to dictionary."""
        return {
            "type": param.param_type.value,
            "label": param.label,
            "hint": param.hint,
            "default": param.default,
            "min": param.min_value,
            "max": param.max_value,
            "choices": param.choices,
            "parent": param.parent,
            "enabled": param.enabled,
        }

    def _clip_to_dict(self, clip: OFXClip) -> Dict[str, Any]:
        """Convert clip to dictionary."""
        return {
            "optional": clip.optional,
            "supports_tiles": clip.supports_tiles,
            "temporal_access": clip.temporal_access,
        }

    def render(
        self,
        time: float,
        render_scale: Tuple[float, float],
        render_window: Tuple[int, int, int, int],
        inputs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Render the effect. Override in subclass.

        Args:
            time: Current time
            render_scale: X and Y scale factors
            render_window: (x1, y1, x2, y2) region to render
            inputs: Dictionary of input images by clip name

        Returns:
            Rendered output image
        """
        raise NotImplementedError("Subclass must implement render()")

    def get_region_of_interest(
        self,
        time: float,
        render_window: Tuple[int, int, int, int],
    ) -> Dict[str, Tuple[int, int, int, int]]:
        """Get region of interest for each input."""
        # Default: same as output
        return {name: render_window for name in self._clips if name != "Output"}

    def get_frames_needed(
        self,
        time: float,
    ) -> Dict[str, List[float]]:
        """Get frames needed from each input."""
        # Default: just current frame
        return {name: [time] for name in self._clips if name != "Output"}


class UltimateMatteOFX(OFXImageEffect):
    """
    Ultimate Matte - OpenFX Plugin

    Professional AI-powered matting for Nuke/Flame/Resolve.
    """

    def __init__(self):
        super().__init__(OFXPluginInfo(
            identifier="com.ultimaterotoscopy.ultimatematte",
            label="Ultimate Matte",
            version_major=2,
            version_minor=0,
            group="Ultimate Rotoscopy",
            description="AI-powered professional matting with SAM3, MatAnyone, and GVM",
        ))

        self._setup_clips()
        self._setup_parameters()
        self._engine = None

    def _setup_clips(self):
        """Define input/output clips."""
        self.define_clip(OFXClip("Source", optional=False))
        self.define_clip(OFXClip("Background", optional=True))
        self.define_clip(OFXClip("CleanPlate", optional=True))
        self.define_clip(OFXClip("Mask", optional=True))
        self.define_clip(OFXClip("Output", optional=False))

    def _setup_parameters(self):
        """Define plugin parameters."""
        # Model selection
        self.define_parameter(OFXParameter(
            name="mattingModel",
            param_type=OFXParamType.CHOICE,
            label="Matting Model",
            hint="Select the AI model for matting",
            default=0,
            choices=["MatAnyone", "GVM", "ViTMatte", "Hybrid"],
        ))

        # Prompt points page
        self.define_parameter(OFXParameter(
            name="promptPage",
            param_type=OFXParamType.PAGE,
            label="Prompt Points",
        ))

        self.define_parameter(OFXParameter(
            name="promptPoint1",
            param_type=OFXParamType.DOUBLE2D,
            label="Foreground Point 1",
            hint="Click on foreground object",
            default=(0.5, 0.5),
            parent="promptPage",
        ))

        self.define_parameter(OFXParameter(
            name="promptPoint2",
            param_type=OFXParamType.DOUBLE2D,
            label="Foreground Point 2",
            hint="Additional foreground point (optional)",
            default=(0.0, 0.0),
            parent="promptPage",
        ))

        self.define_parameter(OFXParameter(
            name="bgPoint1",
            param_type=OFXParamType.DOUBLE2D,
            label="Background Point 1",
            hint="Click on background",
            default=(0.0, 0.0),
            parent="promptPage",
        ))

        # Matte refinement page
        self.define_parameter(OFXParameter(
            name="mattePage",
            param_type=OFXParamType.PAGE,
            label="Matte Settings",
        ))

        self.define_parameter(OFXParameter(
            name="edgeRefinement",
            param_type=OFXParamType.DOUBLE,
            label="Edge Refinement",
            hint="Amount of edge refinement",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            parent="mattePage",
        ))

        self.define_parameter(OFXParameter(
            name="erode",
            param_type=OFXParamType.INT,
            label="Erode",
            hint="Erode matte edges",
            default=0,
            min_value=-50,
            max_value=50,
            parent="mattePage",
        ))

        self.define_parameter(OFXParameter(
            name="blur",
            param_type=OFXParamType.DOUBLE,
            label="Edge Blur",
            hint="Blur matte edges",
            default=0.0,
            min_value=0.0,
            max_value=50.0,
            parent="mattePage",
        ))

        # Compositing page
        self.define_parameter(OFXParameter(
            name="compPage",
            param_type=OFXParamType.PAGE,
            label="Compositing",
        ))

        self.define_parameter(OFXParameter(
            name="enableDespill",
            param_type=OFXParamType.BOOL,
            label="Enable Despill",
            default=True,
            parent="compPage",
        ))

        self.define_parameter(OFXParameter(
            name="despillAlgorithm",
            param_type=OFXParamType.CHOICE,
            label="Despill Algorithm",
            default=0,
            choices=["Average", "Maximum", "Double Average", "Adaptive"],
            parent="compPage",
        ))

        self.define_parameter(OFXParameter(
            name="despillStrength",
            param_type=OFXParamType.DOUBLE,
            label="Despill Strength",
            default=0.8,
            min_value=0.0,
            max_value=1.0,
            parent="compPage",
        ))

        self.define_parameter(OFXParameter(
            name="enableLightWrap",
            param_type=OFXParamType.BOOL,
            label="Enable Light Wrap",
            default=False,
            parent="compPage",
        ))

        self.define_parameter(OFXParameter(
            name="lightWrapWidth",
            param_type=OFXParamType.INT,
            label="Light Wrap Width",
            default=20,
            min_value=1,
            max_value=100,
            parent="compPage",
        ))

        self.define_parameter(OFXParameter(
            name="lightWrapIntensity",
            param_type=OFXParamType.DOUBLE,
            label="Light Wrap Intensity",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            parent="compPage",
        ))

        # Output options
        self.define_parameter(OFXParameter(
            name="outputPage",
            param_type=OFXParamType.PAGE,
            label="Output",
        ))

        self.define_parameter(OFXParameter(
            name="outputMode",
            param_type=OFXParamType.CHOICE,
            label="Output Mode",
            default=0,
            choices=["Composite", "Matte", "Premultiplied FG", "Status"],
            parent="outputPage",
        ))

    def _get_engine(self):
        """Get or create the processing engine."""
        if self._engine is None:
            from ultimate_rotoscopy.core.ultimate_pipeline import UltimatePipeline
            self._engine = UltimatePipeline()
        return self._engine

    def render(
        self,
        time: float,
        render_scale: Tuple[float, float],
        render_window: Tuple[int, int, int, int],
        inputs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Render the matte effect."""
        source = inputs.get("Source")
        if source is None:
            return np.zeros((render_window[3] - render_window[1],
                           render_window[2] - render_window[0], 4))

        background = inputs.get("Background")
        clean_plate = inputs.get("CleanPlate")
        mask_input = inputs.get("Mask")

        # Get parameters
        model_idx = self.get_param("mattingModel")
        fg_point1 = self.get_param("promptPoint1")
        fg_point2 = self.get_param("promptPoint2")
        bg_point1 = self.get_param("bgPoint1")

        # Build prompt points
        h, w = source.shape[:2]
        points = []
        labels = []

        if fg_point1 != (0.0, 0.0):
            points.append([int(fg_point1[0] * w), int(fg_point1[1] * h)])
            labels.append(1)

        if fg_point2 != (0.0, 0.0):
            points.append([int(fg_point2[0] * w), int(fg_point2[1] * h)])
            labels.append(1)

        if bg_point1 != (0.0, 0.0):
            points.append([int(bg_point1[0] * w), int(bg_point1[1] * h)])
            labels.append(0)

        # Process
        try:
            engine = self._get_engine()

            if background is not None:
                result = engine.process_image(
                    source, background,
                    prompt_points=points if points else None,
                    prompt_labels=labels if labels else None,
                    existing_mask=mask_input,
                    clean_plate=clean_plate,
                )
            else:
                # Just matting without composite
                result = engine.process_image(
                    source, source,  # Use source as BG placeholder
                    prompt_points=points if points else None,
                    prompt_labels=labels if labels else None,
                    existing_mask=mask_input,
                )

            # Output based on mode
            output_mode = self.get_param("outputMode")

            if output_mode == 0:  # Composite
                return result.composite
            elif output_mode == 1:  # Matte
                alpha = result.alpha
                return np.stack([alpha, alpha, alpha, np.ones_like(alpha)], axis=-1)
            elif output_mode == 2:  # Premultiplied FG
                return result.foreground
            else:  # Status
                return source

        except Exception as e:
            # Return source on error
            print(f"Ultimate Matte error: {e}")
            return source


class DepthExtractOFX(OFXImageEffect):
    """
    Depth Extract - OpenFX Plugin

    AI depth estimation with normal maps and 3D export.
    """

    def __init__(self):
        super().__init__(OFXPluginInfo(
            identifier="com.ultimaterotoscopy.depthextract",
            label="Depth Extract",
            version_major=2,
            version_minor=0,
            group="Ultimate Rotoscopy",
            description="AI depth estimation with Depth Anything V3",
        ))

        self._setup_clips()
        self._setup_parameters()

    def _setup_clips(self):
        self.define_clip(OFXClip("Source", optional=False))
        self.define_clip(OFXClip("Output", optional=False))

    def _setup_parameters(self):
        self.define_parameter(OFXParameter(
            name="outputType",
            param_type=OFXParamType.CHOICE,
            label="Output Type",
            default=0,
            choices=["Depth", "Normals", "Point Cloud Preview", "AO"],
        ))

        self.define_parameter(OFXParameter(
            name="depthScale",
            param_type=OFXParamType.DOUBLE,
            label="Depth Scale",
            default=1.0,
            min_value=0.1,
            max_value=10.0,
        ))

        self.define_parameter(OFXParameter(
            name="normalIntensity",
            param_type=OFXParamType.DOUBLE,
            label="Normal Intensity",
            default=1.0,
            min_value=0.0,
            max_value=5.0,
        ))

    def render(
        self,
        time: float,
        render_scale: Tuple[float, float],
        render_window: Tuple[int, int, int, int],
        inputs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Render depth extraction."""
        source = inputs.get("Source")
        if source is None:
            return np.zeros((render_window[3] - render_window[1],
                           render_window[2] - render_window[0], 4))

        try:
            from ultimate_rotoscopy.models.depth_anything import DepthAnythingV3

            depth_model = DepthAnythingV3()
            result = depth_model.estimate(source)

            output_type = self.get_param("outputType")
            depth_scale = self.get_param("depthScale")

            if output_type == 0:  # Depth
                depth = result.depth * depth_scale
                depth = np.clip(depth, 0, 1)
                return np.stack([depth, depth, depth, np.ones_like(depth)], axis=-1)

            elif output_type == 1:  # Normals
                normals = result.normals
                # Convert from [-1,1] to [0,1]
                normals = (normals + 1) / 2
                alpha = np.ones(normals.shape[:2])
                return np.concatenate([normals, alpha[..., np.newaxis]], axis=-1)

            elif output_type == 2:  # Point cloud preview
                # Simple depth-based shading
                depth = result.depth
                shaded = depth * 0.8 + 0.2
                return np.stack([shaded, shaded, shaded, np.ones_like(depth)], axis=-1)

            else:  # AO
                from ultimate_rotoscopy.depth.shadow_ao import generate_ao
                ao = generate_ao(result.depth)
                return np.stack([ao, ao, ao, np.ones_like(ao)], axis=-1)

        except Exception as e:
            print(f"Depth Extract error: {e}")
            return source


class OFXPluginBundle:
    """
    Bundle of OpenFX plugins for Ultimate Rotoscopy.

    Manages registration and hosting of all OFX plugins.
    """

    def __init__(self):
        self.plugins: Dict[str, OFXImageEffect] = {}
        self._register_plugins()

    def _register_plugins(self):
        """Register all available plugins."""
        self.plugins["UltimateMatte"] = UltimateMatteOFX()
        self.plugins["DepthExtract"] = DepthExtractOFX()

    def get_plugin(self, identifier: str) -> Optional[OFXImageEffect]:
        """Get plugin by identifier."""
        return self.plugins.get(identifier)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all available plugins."""
        return [p.describe() for p in self.plugins.values()]

    def export_bundle(self, output_path: Path) -> Path:
        """
        Export plugins as OFX bundle.

        Creates the standard OFX bundle structure:
        - PluginName.ofx.bundle/
          - Contents/
            - Info.plist
            - Linux-x86-64/
              - PluginName.ofx
            - MacOS/
              - PluginName.ofx
            - Win64/
              - PluginName.ofx

        Args:
            output_path: Directory to create bundle in

        Returns:
            Path to created bundle
        """
        bundle_name = "UltimateRotoscopy.ofx.bundle"
        bundle_path = output_path / bundle_name

        # Create directory structure
        contents = bundle_path / "Contents"
        contents.mkdir(parents=True, exist_ok=True)

        # Write Info.plist
        plist_content = self._generate_plist()
        (contents / "Info.plist").write_text(plist_content)

        # Create platform directories
        for platform in ["Linux-x86-64", "MacOS", "Win64"]:
            (contents / platform).mkdir(exist_ok=True)

        return bundle_path

    def _generate_plist(self) -> str:
        """Generate Info.plist content."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleExecutable</key>
    <string>UltimateRotoscopy</string>
    <key>CFBundleIdentifier</key>
    <string>com.ultimaterotoscopy.ofx</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Ultimate Rotoscopy</string>
    <key>CFBundlePackageType</key>
    <string>BNDL</string>
    <key>CFBundleVersion</key>
    <string>2.0.0</string>
</dict>
</plist>'''


# Plugin entry points for OFX hosts
_bundle = None


def get_bundle() -> OFXPluginBundle:
    """Get the plugin bundle singleton."""
    global _bundle
    if _bundle is None:
        _bundle = OFXPluginBundle()
    return _bundle


def OfxGetNumberOfPlugins() -> int:
    """OFX API: Get number of plugins."""
    return len(get_bundle().plugins)


def OfxGetPlugin(index: int) -> Optional[Dict[str, Any]]:
    """OFX API: Get plugin info by index."""
    plugins = list(get_bundle().plugins.values())
    if 0 <= index < len(plugins):
        return plugins[index].describe()
    return None
