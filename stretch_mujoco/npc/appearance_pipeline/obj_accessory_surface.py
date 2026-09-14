"""Recipe-driven UV surface fallbacks for fused OBJ accessories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .accessory_recipe import SmplxHeadSurfaceFallback


@dataclass(frozen=True)
class SurfaceFallbackArtifacts:
    texture: Path
    mask: Path
    pixels: int


def bake_smplx_head_surface_fallback(
    *,
    source_atlas: Path,
    body_frame: Path,
    destination_texture: Path,
    destination_mask: Path,
    fallback: SmplxHeadSurfaceFallback,
) -> SurfaceFallbackArtifacts:
    """Bake an opaque, recipe-defined scalp fallback onto an SMPL-X atlas copy."""
    atlas = cv2.imread(str(source_atlas), cv2.IMREAD_UNCHANGED)
    if atlas is None or atlas.ndim != 3 or atlas.shape[2] not in {3, 4}:
        raise ValueError(f"Could not decode RGB/RGBA source atlas: {source_atlas}")
    vertices, texcoords, faces = _read_body_uv_mesh(body_frame)
    height, width = atlas.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for vertex_indices, uv_indices in faces:
        x, y, z = vertices[vertex_indices].mean(axis=0)
        on_head = z > fallback.head_min_z_m and np.hypot(x, y) < fallback.head_radius_m
        selected = (
            z >= fallback.front_hairline_z_m
            or (z >= fallback.temple_min_z_m and y >= fallback.temple_min_y_m)
            or (z >= fallback.rear_min_z_m and y >= fallback.rear_min_y_m)
        )
        if not on_head or not selected:
            continue
        polygon = np.asarray(
            [
                (
                    round(texcoords[index][0] * (width - 1)),
                    round((1.0 - texcoords[index][1]) * (height - 1)),
                )
                for index in uv_indices
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_AA)
    pixels = int(cv2.countNonZero(mask))
    if pixels < 100:
        raise ValueError("SMPL-X head surface fallback mask is unexpectedly empty")

    hair = np.empty_like(atlas[:, :, :3])
    hair[:, :] = fallback.color_bgr
    rows = np.arange(height, dtype=np.float32)[:, None]
    variation = np.rint(3.0 * np.sin(rows * 0.23)).astype(np.int16)
    hair = np.clip(hair.astype(np.int16) + variation[:, :, None], 0, 255).astype(np.uint8)
    atlas[mask > 0, :3] = hair[mask > 0]
    if atlas.shape[2] == 4:
        atlas[mask > 0, 3] = 255
    destination_texture.parent.mkdir(parents=True, exist_ok=True)
    destination_mask.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination_texture), atlas):
        raise OSError(f"Could not write OBJ accessory fallback texture: {destination_texture}")
    if not cv2.imwrite(str(destination_mask), mask):
        raise OSError(f"Could not write OBJ accessory fallback mask: {destination_mask}")
    return SurfaceFallbackArtifacts(destination_texture, destination_mask, pixels)


def _read_body_uv_mesh(
    path: Path,
) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[np.ndarray, list[int]]]]:
    vertices: list[list[float]] = []
    texcoords: list[tuple[float, float]] = []
    faces: list[tuple[np.ndarray, list[int]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("vt "):
            _, u, v, *_ = line.split()
            texcoords.append((float(u), float(v)))
        elif line.startswith("f "):
            vertex_indices: list[int] = []
            uv_indices: list[int] = []
            for token in line.split()[1:]:
                vertex, uv, *_ = token.split("/")
                if not uv:
                    raise ValueError(f"Body frame face has no UV coordinate: {token}")
                vertex_indices.append(int(vertex) - 1)
                uv_indices.append(int(uv) - 1)
            faces.append((np.asarray(vertex_indices, dtype=np.int64), uv_indices))
    if not vertices or not texcoords or not faces:
        raise ValueError(f"Body frame needs vertices, UVs, and faces: {path}")
    vertex_array = np.asarray(vertices, dtype=np.float64)
    if any(
        np.any(indices < 0)
        or np.any(indices >= len(vertex_array))
        or any(uv < 0 or uv >= len(texcoords) for uv in uvs)
        for indices, uvs in faces
    ):
        raise ValueError(f"Body frame has invalid vertex or UV indices: {path}")
    return vertex_array, texcoords, faces
