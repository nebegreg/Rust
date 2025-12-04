"""
Shadow and Ambient Occlusion from Depth Maps
=============================================

Extract shadows and ambient occlusion from depth maps
for professional compositing.

These effects help integrate CG elements and
green screen footage into backgrounds.

Features:
- Contact shadow generation
- Self-shadowing from depth
- Screen-space ambient occlusion (SSAO)
- Ground shadow projection
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ShadowType(Enum):
    """Types of shadow generation."""
    CONTACT = "contact"         # Contact shadows at object base
    PROJECTED = "projected"     # Projected ground shadow
    SELF = "self"               # Self-shadowing
    COMBINED = "combined"       # All shadow types


class AOType(Enum):
    """Types of ambient occlusion."""
    SSAO = "ssao"               # Screen-space AO
    HBAO = "hbao"               # Horizon-based AO
    RTAO = "rtao"               # Ray-traced AO (approximation)


@dataclass
class ShadowConfig:
    """Shadow generation configuration."""
    shadow_type: ShadowType = ShadowType.COMBINED

    # Light direction (normalized)
    light_direction: Tuple[float, float, float] = (0.5, -0.8, 0.3)
    light_intensity: float = 1.0
    light_softness: float = 0.5      # 0-1, shadow softness

    # Contact shadow
    contact_distance: float = 0.02    # Relative to scene scale
    contact_falloff: float = 2.0      # Exponential falloff
    contact_intensity: float = 0.8

    # Projected shadow
    projection_blur: float = 10.0     # Blur radius
    projection_density: float = 0.7   # Shadow darkness

    # Self shadow
    self_shadow_bias: float = 0.001
    self_shadow_samples: int = 16


@dataclass
class AOConfig:
    """Ambient occlusion configuration."""
    ao_type: AOType = AOType.SSAO

    # SSAO settings
    radius: float = 0.5             # AO radius in scene units
    num_samples: int = 16           # Samples per pixel
    intensity: float = 1.0          # AO strength
    bias: float = 0.025             # Depth bias
    falloff: float = 2.0            # Distance falloff

    # Quality
    half_res: bool = False          # Compute at half resolution
    blur_result: bool = True        # Apply bilateral blur


@dataclass
class ShadowAOResult:
    """Result from shadow/AO generation."""
    shadow: np.ndarray              # Combined shadow map (0=shadow, 1=lit)
    ambient_occlusion: np.ndarray   # AO map (0=occluded, 1=open)
    contact_shadow: Optional[np.ndarray] = None
    projected_shadow: Optional[np.ndarray] = None
    self_shadow: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DepthUtils:
    """Utility functions for depth processing."""

    @staticmethod
    def normalize_depth(depth: np.ndarray) -> np.ndarray:
        """Normalize depth to 0-1 range."""
        d_min = np.min(depth)
        d_max = np.max(depth)
        if d_max - d_min < 1e-6:
            return np.zeros_like(depth)
        return (depth - d_min) / (d_max - d_min)

    @staticmethod
    def depth_to_world_space(
        depth: np.ndarray,
        fov: float = 60.0,
        aspect: float = 1.0,
    ) -> np.ndarray:
        """
        Convert depth map to world-space positions.

        Args:
            depth: Normalized depth map (0=near, 1=far)
            fov: Field of view in degrees
            aspect: Aspect ratio (width/height)

        Returns:
            XYZ positions array (H, W, 3)
        """
        h, w = depth.shape
        fov_rad = np.radians(fov)

        # Create pixel coordinates
        y, x = np.mgrid[0:h, 0:w]
        x = (x - w / 2) / (w / 2)  # -1 to 1
        y = (y - h / 2) / (h / 2)  # -1 to 1

        # Convert to view space
        z = depth
        x = x * z * np.tan(fov_rad / 2) * aspect
        y = y * z * np.tan(fov_rad / 2)

        return np.stack([x, y, z], axis=-1)

    @staticmethod
    def compute_normals(depth: np.ndarray) -> np.ndarray:
        """Compute surface normals from depth."""
        import cv2

        # Compute gradients
        dzdx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        dzdy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)

        # Normal = (-dz/dx, -dz/dy, 1) normalized
        normal = np.stack([-dzdx, -dzdy, np.ones_like(depth)], axis=-1)

        # Normalize
        norm = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6
        normal = normal / norm

        return normal.astype(np.float32)


class ContactShadowGenerator:
    """
    Generate contact shadows from depth maps.

    Contact shadows appear where objects meet surfaces,
    adding important visual grounding.
    """

    def __init__(self, config: ShadowConfig):
        self.config = config

    def generate(
        self,
        depth: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """
        Generate contact shadows.

        Args:
            depth: Normalized depth map
            alpha: Object alpha mask

        Returns:
            Contact shadow map (0=shadow, 1=no shadow)
        """
        import cv2

        h, w = depth.shape[:2]

        # Find object boundaries (bottom edges)
        alpha_blur = cv2.GaussianBlur(alpha.astype(np.float32), (5, 5), 0)
        alpha_grad = cv2.Sobel(alpha_blur, cv2.CV_64F, 0, 1, ksize=3)

        # Bottom edges have positive gradient
        bottom_edge = np.maximum(alpha_grad, 0)
        bottom_edge = bottom_edge / (np.max(bottom_edge) + 1e-6)

        # Get depth at object base
        object_mask = alpha > 0.5
        if np.any(object_mask):
            base_depth = np.percentile(depth[object_mask], 95)  # Near bottom
        else:
            base_depth = np.mean(depth)

        # Create distance field from bottom edge
        edge_mask = (bottom_edge > 0.1).astype(np.uint8)
        dist = cv2.distanceTransform(1 - edge_mask, cv2.DIST_L2, 5)
        dist = dist / (np.max(dist) + 1e-6)

        # Shadow based on distance with depth consideration
        shadow_dist = self.config.contact_distance * w

        # Compute shadow with falloff
        shadow = 1 - np.exp(-self.config.contact_falloff * dist / shadow_dist)

        # Modulate by depth similarity
        depth_diff = np.abs(depth - base_depth)
        depth_weight = np.exp(-depth_diff * 20)  # Sharp falloff

        shadow = 1 - (1 - shadow) * depth_weight * self.config.contact_intensity

        # Remove shadow inside object
        shadow[object_mask] = 1.0

        # Blur for softness
        blur_size = int(self.config.light_softness * 20) * 2 + 1
        shadow = cv2.GaussianBlur(shadow.astype(np.float32), (blur_size, blur_size), 0)

        return np.clip(shadow, 0, 1)


class ProjectedShadowGenerator:
    """
    Generate projected ground shadows.

    Projects object silhouette onto ground plane
    based on light direction.
    """

    def __init__(self, config: ShadowConfig):
        self.config = config

    def generate(
        self,
        alpha: np.ndarray,
        depth: np.ndarray,
        ground_plane: float = 1.0,  # Depth of ground
    ) -> np.ndarray:
        """
        Generate projected shadow.

        Args:
            alpha: Object alpha mask
            depth: Depth map
            ground_plane: Depth value of ground plane

        Returns:
            Projected shadow map
        """
        import cv2

        h, w = alpha.shape[:2]

        # Light direction
        lx, ly, lz = self.config.light_direction
        light_vec = np.array([lx, ly, lz])
        light_vec = light_vec / (np.linalg.norm(light_vec) + 1e-6)

        # Compute shadow offset based on light angle
        shadow_offset_x = -light_vec[0] / (light_vec[1] + 1e-6)
        shadow_offset_y = light_vec[2] / (light_vec[1] + 1e-6)

        # Scale by depth (objects closer cast longer shadows)
        max_offset = int(h * 0.3)  # Maximum shadow length

        # Create shadow map
        shadow = np.ones((h, w), dtype=np.float32)

        # For each pixel in alpha, project shadow
        y_coords, x_coords = np.where(alpha > 0.5)

        for i in range(len(x_coords)):
            x, y = x_coords[i], y_coords[i]
            d = depth[y, x]

            # Distance to ground
            ground_dist = ground_plane - d
            if ground_dist <= 0:
                continue

            # Shadow position
            sx = int(x + shadow_offset_x * ground_dist * max_offset)
            sy = int(y + shadow_offset_y * ground_dist * max_offset)

            if 0 <= sx < w and 0 <= sy < h:
                # Fade shadow with distance
                fade = np.exp(-ground_dist * 3)
                shadow[sy, sx] = min(shadow[sy, sx], 1 - self.config.projection_density * fade)

        # Blur shadow
        blur_size = int(self.config.projection_blur) * 2 + 1
        shadow = cv2.GaussianBlur(shadow, (blur_size, blur_size), 0)

        # Don't shadow the object itself
        shadow[alpha > 0.5] = 1.0

        return shadow


class SelfShadowGenerator:
    """
    Generate self-shadowing from depth.

    Computes shadows cast by the object onto itself
    based on surface geometry.
    """

    def __init__(self, config: ShadowConfig):
        self.config = config

    def generate(
        self,
        depth: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate self-shadow map.

        Args:
            depth: Normalized depth map
            normals: Surface normals (computed if not provided)

        Returns:
            Self-shadow map
        """
        import cv2

        if normals is None:
            normals = DepthUtils.compute_normals(depth)

        h, w = depth.shape[:2]

        # Light direction
        light_dir = np.array(self.config.light_direction)
        light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-6)

        # Basic N.L shadow
        n_dot_l = np.sum(normals * light_dir, axis=-1)
        direct_light = np.clip(n_dot_l, 0, 1)

        # Ray march for self-shadowing
        shadow = np.ones((h, w), dtype=np.float32)

        # Sample along light direction
        num_samples = self.config.self_shadow_samples
        step_size = 0.02

        for s in range(1, num_samples):
            # Offset position towards light
            offset_x = int(light_dir[0] * s * step_size * w)
            offset_y = int(light_dir[1] * s * step_size * h)
            offset_z = light_dir[2] * s * step_size

            # Check if occluded
            y, x = np.mgrid[0:h, 0:w]
            sample_x = np.clip(x + offset_x, 0, w - 1)
            sample_y = np.clip(y + offset_y, 0, h - 1)

            sample_depth = depth[sample_y, sample_x]
            expected_depth = depth + offset_z

            # Occluded if sample is closer
            occluded = sample_depth < expected_depth - self.config.self_shadow_bias
            shadow[occluded] *= 0.9  # Accumulate occlusion

        # Combine with direct lighting
        shadow = shadow * direct_light

        # Blur for softness
        blur_size = int(self.config.light_softness * 5) * 2 + 1
        shadow = cv2.GaussianBlur(shadow.astype(np.float32), (blur_size, blur_size), 0)

        return np.clip(shadow, 0, 1)


class SSAOGenerator:
    """
    Screen-Space Ambient Occlusion (SSAO).

    Approximates ambient occlusion using depth buffer
    for soft shadows in concave regions.
    """

    def __init__(self, config: AOConfig):
        self.config = config

    def generate(
        self,
        depth: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate SSAO.

        Args:
            depth: Normalized depth map
            normals: Surface normals (optional, improves quality)

        Returns:
            Ambient occlusion map (0=occluded, 1=open)
        """
        import cv2

        h, w = depth.shape[:2]

        # Work at half resolution if specified
        if self.config.half_res:
            depth_work = cv2.resize(depth, (w // 2, h // 2))
            if normals is not None:
                normals_work = cv2.resize(normals, (w // 2, h // 2))
            else:
                normals_work = None
            h_work, w_work = depth_work.shape[:2]
        else:
            depth_work = depth
            normals_work = normals
            h_work, w_work = h, w

        if normals_work is None:
            normals_work = DepthUtils.compute_normals(depth_work)

        # Generate sample kernel (hemisphere)
        kernel = self._generate_kernel()

        # SSAO computation
        ao = np.zeros((h_work, w_work), dtype=np.float32)

        radius_pixels = int(self.config.radius * w_work)

        for i in range(self.config.num_samples):
            # Random offset in hemisphere
            offset = kernel[i]

            # Sample position
            sample_x = np.clip(
                np.arange(w_work) + int(offset[0] * radius_pixels),
                0, w_work - 1
            )
            sample_y = np.clip(
                np.arange(h_work) + int(offset[1] * radius_pixels),
                0, h_work - 1
            )

            # Create 2D index arrays
            yy, xx = np.meshgrid(sample_y, sample_x, indexing='ij')

            # Sample depth
            sample_depth = depth_work[yy, xx]

            # Expected depth
            expected_depth = depth_work + offset[2] * self.config.radius

            # Check occlusion
            range_check = np.abs(depth_work - sample_depth) < self.config.radius
            occluded = (sample_depth < expected_depth - self.config.bias) & range_check

            ao += occluded.astype(np.float32)

        # Normalize
        ao = 1 - (ao / self.config.num_samples)

        # Apply intensity and falloff
        ao = np.power(ao, self.config.falloff) * self.config.intensity
        ao = 1 - (1 - ao) * self.config.intensity

        # Upscale if needed
        if self.config.half_res:
            ao = cv2.resize(ao, (w, h))

        # Bilateral blur for edge preservation
        if self.config.blur_result:
            ao = cv2.bilateralFilter(ao.astype(np.float32), 9, 75, 75)

        return np.clip(ao, 0, 1)

    def _generate_kernel(self) -> np.ndarray:
        """Generate hemisphere sample kernel."""
        kernel = []

        for i in range(self.config.num_samples):
            # Random direction in hemisphere
            theta = 2 * np.pi * np.random.random()
            phi = np.arccos(np.random.random())

            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            # Scale by random length (more samples near center)
            scale = (i + 1) / self.config.num_samples
            scale = 0.1 + scale * scale * 0.9

            kernel.append([x * scale, y * scale, z * scale])

        return np.array(kernel)


class ShadowAOGenerator:
    """
    Combined shadow and ambient occlusion generator.

    Provides complete shading information from depth maps
    for professional compositing.

    Example:
        >>> generator = ShadowAOGenerator(
        ...     ShadowConfig(light_direction=(0.5, -0.8, 0.3)),
        ...     AOConfig(radius=0.5, num_samples=32),
        ... )
        >>>
        >>> result = generator.generate(depth_map, alpha_mask)
        >>> shadow = result.shadow
        >>> ao = result.ambient_occlusion
    """

    def __init__(
        self,
        shadow_config: Optional[ShadowConfig] = None,
        ao_config: Optional[AOConfig] = None,
    ):
        self.shadow_config = shadow_config or ShadowConfig()
        self.ao_config = ao_config or AOConfig()

        self.contact_generator = ContactShadowGenerator(self.shadow_config)
        self.projected_generator = ProjectedShadowGenerator(self.shadow_config)
        self.self_shadow_generator = SelfShadowGenerator(self.shadow_config)
        self.ssao_generator = SSAOGenerator(self.ao_config)

    def generate(
        self,
        depth: np.ndarray,
        alpha: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
    ) -> ShadowAOResult:
        """
        Generate shadows and ambient occlusion.

        Args:
            depth: Depth map (normalized 0-1)
            alpha: Object alpha mask (optional)
            normals: Surface normals (optional)

        Returns:
            ShadowAOResult with all shadow/AO maps
        """
        # Normalize depth
        depth = DepthUtils.normalize_depth(depth)

        if alpha is None:
            alpha = np.ones_like(depth)

        # Generate AO
        ao = self.ssao_generator.generate(depth, normals)

        # Generate shadows based on config
        contact = None
        projected = None
        self_shadow = None

        if self.shadow_config.shadow_type in (ShadowType.CONTACT, ShadowType.COMBINED):
            contact = self.contact_generator.generate(depth, alpha)

        if self.shadow_config.shadow_type in (ShadowType.PROJECTED, ShadowType.COMBINED):
            projected = self.projected_generator.generate(alpha, depth)

        if self.shadow_config.shadow_type in (ShadowType.SELF, ShadowType.COMBINED):
            self_shadow = self.self_shadow_generator.generate(depth, normals)

        # Combine shadows
        combined_shadow = np.ones_like(depth)
        if contact is not None:
            combined_shadow *= contact
        if projected is not None:
            combined_shadow *= projected
        if self_shadow is not None:
            combined_shadow *= self_shadow

        return ShadowAOResult(
            shadow=combined_shadow,
            ambient_occlusion=ao,
            contact_shadow=contact,
            projected_shadow=projected,
            self_shadow=self_shadow,
            metadata={
                "shadow_type": self.shadow_config.shadow_type.value,
                "ao_type": self.ao_config.ao_type.value,
            }
        )


def generate_shadow(
    depth: np.ndarray,
    alpha: np.ndarray,
    light_direction: Tuple[float, float, float] = (0.5, -0.8, 0.3),
) -> np.ndarray:
    """
    Quick shadow generation.

    Args:
        depth: Depth map
        alpha: Object alpha mask
        light_direction: Light direction vector

    Returns:
        Combined shadow map
    """
    config = ShadowConfig(
        shadow_type=ShadowType.COMBINED,
        light_direction=light_direction,
    )
    generator = ShadowAOGenerator(config)
    result = generator.generate(depth, alpha)
    return result.shadow


def generate_ao(
    depth: np.ndarray,
    radius: float = 0.5,
    intensity: float = 1.0,
) -> np.ndarray:
    """
    Quick AO generation.

    Args:
        depth: Depth map
        radius: AO sampling radius
        intensity: AO intensity

    Returns:
        Ambient occlusion map
    """
    config = AOConfig(radius=radius, intensity=intensity)
    generator = SSAOGenerator(config)
    return generator.generate(depth)
