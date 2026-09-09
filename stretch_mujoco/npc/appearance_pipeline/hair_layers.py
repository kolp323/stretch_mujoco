"""Create UV-aligned, texture-only short-hair layers from an NPC mesh."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class _ObjMesh:
    vertices: np.ndarray
    uvs: np.ndarray
    faces: tuple[tuple[tuple[int, int], ...], ...]


def generate_short_hair_layers(spec_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Rasterize high head faces into one colored RGBA short-hair layer per style.

    The reference mesh must use the same UV topology as the texture being
    baked. This creates a cap-like texture effect only; it cannot add hair
    silhouette, strands, or animation-driven movement.
    """
    source = Path(spec_path).resolve()
    spec = _mapping(json.loads(source.read_text(encoding="utf-8")), "Hair layer spec")
    reference_mesh = source.parent / _required_string(spec, "reference_mesh", "Hair layer spec")
    mesh = _read_obj(reference_mesh)
    min_height = float(spec.get("min_height_m", 1.57))
    horizontal_radius = float(spec.get("horizontal_radius_m", 0.18))
    normal_z_min = spec.get("normal_z_min")
    if normal_z_min is not None and not isinstance(normal_z_min, int | float):
        raise ValueError("Hair layer spec normal_z_min must be a number")
    if min_height <= 0 or horizontal_radius <= 0:
        raise ValueError("Hair layer thresholds must be positive")
    styles = _mapping(spec.get("styles"), "Hair layer spec styles")
    if not styles:
        raise ValueError("Hair layer spec must define at least one style")
    mask = _head_mask(
        mesh,
        min_height,
        horizontal_radius,
        float(normal_z_min) if normal_z_min is not None else None,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for style_id, raw_style in styles.items():
        color = _rgb(_mapping(raw_style, f"Hair style '{style_id}'").get("rgb"), style_id)
        layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
        layer[..., :3] = color[::-1]
        layer[..., 3] = mask
        destination = output / f"{style_id}.png"
        if not cv2.imwrite(str(destination), layer, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
            raise OSError(f"Could not write short-hair layer '{destination}'")
        written[str(style_id)] = destination
    return written


def _head_mask(
    mesh: _ObjMesh, min_height: float, horizontal_radius: float, normal_z_min: float | None = None
) -> np.ndarray:
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    for face in mesh.faces:
        vertex_ids = np.asarray([vertex_id for vertex_id, _ in face])
        centroid = mesh.vertices[vertex_ids].mean(axis=0)
        if centroid[2] < min_height or np.linalg.norm(centroid[:2]) > horizontal_radius:
            continue
        if normal_z_min is not None:
            edges = mesh.vertices[vertex_ids[1:]] - mesh.vertices[vertex_ids[0]]
            normal = np.cross(edges[0], edges[1])
            length = np.linalg.norm(normal)
            if length == 0 or normal[2] / length < normal_z_min:
                continue
        uv_ids = np.asarray([uv_id for _, uv_id in face])
        pixels = np.rint(
            np.column_stack((mesh.uvs[uv_ids, 0], 1.0 - mesh.uvs[uv_ids, 1])) * 1023
        ).astype(np.int32)
        cv2.fillConvexPoly(mask, pixels, 255, lineType=cv2.LINE_AA)
    return mask


def _read_obj(path: Path) -> _ObjMesh:
    vertices: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: list[tuple[tuple[int, int], ...]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values:
            continue
        if values[0] == "v":
            vertices.append([float(value) for value in values[1:4]])
        elif values[0] == "vt":
            uvs.append([float(value) for value in values[1:3]])
        elif values[0] == "f":
            face: list[tuple[int, int]] = []
            for value in values[1:]:
                indices = value.split("/")
                if len(indices) < 2 or not indices[1]:
                    raise ValueError(f"OBJ '{path}' face lacks UV indices")
                face.append((int(indices[0]) - 1, int(indices[1]) - 1))
            faces.append(tuple(face))
    if not vertices or not uvs or not faces:
        raise ValueError(f"OBJ '{path}' must contain vertices, UVs, and faces")
    return _ObjMesh(np.asarray(vertices), np.asarray(uvs), tuple(faces))


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _required_string(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must define non-empty string '{key}'")
    return value


def _rgb(value: object, style_id: object) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(item, int) for item in value)
    ):
        raise ValueError(f"Hair style '{style_id}' rgb must be a three-item integer array")
    if not all(0 <= item <= 255 for item in value):
        raise ValueError(f"Hair style '{style_id}' rgb must be in [0, 255]")
    return value[0], value[1], value[2]


def main() -> None:
    """Console entry point for UV-aligned short-hair layer generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for path in generate_short_hair_layers(args.spec, args.output_dir).values():
        print(path)
