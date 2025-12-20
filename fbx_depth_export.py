"""Lightweight FBX mesh export from depth maps for Autodesk Flame.

This writer generates a simple ASCII FBX 7.4 mesh built from a sampled
regular grid. It avoids external dependencies so the GUI can export a
quick depth proxy even in constrained environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass
class DepthMeshConfig:
    sample: int = 4
    depth_scale: float = 1.0
    mask_threshold: float = 0.5
    invert_depth: bool = False


class DepthToFBX:
    """Convert a depth map (and optional mask) into an FBX mesh."""

    def __init__(self, config: Optional[DepthMeshConfig] = None) -> None:
        self.config = config or DepthMeshConfig()

    def export_mesh(
        self,
        depth: np.ndarray,
        mask: Optional[np.ndarray],
        output_path: Path | str,
    ) -> Path:
        output_path = Path(output_path)
        vertices, faces = self._build_mesh(depth, mask)
        payload = self._build_fbx(vertices, faces)
        output_path.write_text(payload)
        return output_path

    def _build_mesh(
        self, depth: np.ndarray, mask: Optional[np.ndarray]
    ) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
        step = max(1, int(self.config.sample))
        depth = np.nan_to_num(depth.astype(np.float32))

        if mask is not None:
            mask = mask.astype(np.float32)
            if mask.max() > 1.0:
                mask = mask / 255.0

        h, w = depth.shape
        vertices: list[tuple[float, float, float]] = []
        indices_map: dict[tuple[int, int], int] = {}

        for y in range(0, h, step):
            for x in range(0, w, step):
                if mask is not None and mask[y, x] < self.config.mask_threshold:
                    continue
                z = depth[y, x] * self.config.depth_scale
                if self.config.invert_depth:
                    z = -z
                vert = (float(x), float(h - y), float(z))
                indices_map[(x, y)] = len(vertices)
                vertices.append(vert)

        faces: list[tuple[int, int, int]] = []
        for y in range(0, h - step, step):
            for x in range(0, w - step, step):
                if mask is not None:
                    if mask[y, x] < self.config.mask_threshold:
                        continue
                    if mask[y + step, x] < self.config.mask_threshold:
                        continue
                    if mask[y, x + step] < self.config.mask_threshold:
                        continue
                    if mask[y + step, x + step] < self.config.mask_threshold:
                        continue

                v00 = indices_map.get((x, y))
                v10 = indices_map.get((x + step, y))
                v01 = indices_map.get((x, y + step))
                v11 = indices_map.get((x + step, y + step))
                if None in (v00, v10, v01, v11):
                    continue

                faces.append((v00, v10, v11))
                faces.append((v00, v11, v01))

        return vertices, faces

    def _build_fbx(
        self,
        vertices: Iterable[tuple[float, float, float]],
        faces: Iterable[tuple[int, int, int]],
    ) -> str:
        verts = list(vertices)
        polys = list(faces)

        def serialize_floats(values: Iterable[float]) -> str:
            return ",".join(f"{v:.6f}" for v in values)

        def serialize_ints(values: Iterable[int]) -> str:
            return ",".join(str(v) for v in values)

        vertex_blob = serialize_floats(v for triplet in verts for v in triplet)

        # FBX requires the last index of each polygon to be negative-1
        polygon_indices: list[int] = []
        for tri in polys:
            polygon_indices.extend([tri[0], tri[1], -(tri[2] + 1)])
        poly_blob = serialize_ints(polygon_indices)

        now = datetime.now()
        header = f"""; FBX 7.4.0 project file
FBXHeaderExtension:  {{
    FBXHeaderVersion: 1003
    FBXVersion: 7400
    CreationTimeStamp:  {{
        Version: 1000
        Year: {now.year}
        Month: {now.month}
        Day: {now.day}
        Hour: {now.hour}
        Minute: {now.minute}
        Second: {now.second}
        Millisecond: {int(now.microsecond/1000)}
    }}
    Creator: "Ultimate Rotoscopy"
}}
Definitions:  {{
    Version: 100
    Count: 2
    ObjectType: "Model" {{ Count: 1 }}
    ObjectType: "Geometry" {{ Count: 1 }}
}}
Objects:  {{
    Geometry: 1, "Geometry::DepthMesh", "Mesh" {{
        Vertices: *{len(verts) * 3} {{ {vertex_blob} }}
        PolygonVertexIndex: *{len(polygon_indices)} {{ {poly_blob} }}
        GeometryVersion: 124
    }}
    Model: 2, "Model::DepthMesh", "Mesh" {{
        Properties70:  {{
            P: "Lcl Translation", "Lcl Translation", "", "A",0,0,0
        }}
        Shading: T
        Culling: "CullingOff"
    }}
}}
Connections:  {{
    C: "OO",1,2
}}
"""
        return header
