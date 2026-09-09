"""Generate hash-verified semantic masks for the flat SMPL-X NPC atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .hair_layers import _ObjMesh, _read_obj

SEMANTIC_MASK_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SemanticMaskSet:
    masks: dict[str, Path]
    manifest_path: Path


def generate_semantic_masks(spec_path: str | Path, output_dir: str | Path) -> SemanticMaskSet:
    """Generate reusable grayscale masks from the specified base atlas and mesh."""
    source = Path(spec_path).resolve()
    spec = _mapping(json.loads(source.read_text(encoding="utf-8")), "Semantic mask spec")
    if spec.get("schema_version") != SEMANTIC_MASK_SCHEMA_VERSION:
        raise ValueError("Unsupported semantic mask spec schema_version")
    topology_id = _string(spec.get("texture_topology_id"), "Semantic mask topology")
    base_path = source.parent / _string(spec.get("base"), "Semantic mask base")
    base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    if base is None:
        raise ValueError(f"Could not decode semantic mask base '{base_path}'")
    mesh_path = source.parent / _string(spec.get("reference_mesh"), "Semantic mask mesh")
    mesh = _read_obj(mesh_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    regions = _mapping(spec.get("flat_regions"), "Semantic mask flat_regions")
    masks: dict[str, Path] = {}
    for mask_id, raw_region in regions.items():
        region = _mapping(raw_region, f"Semantic region '{mask_id}'")
        color = _rgb(region.get("source_rgb"), f"Semantic region '{mask_id}' source_rgb")
        tolerance = int(region.get("tolerance", 8))
        if not 0 <= tolerance <= 255:
            raise ValueError(f"Semantic region '{mask_id}' tolerance must be in [0, 255]")
        distance = np.max(
            np.abs(base.astype(np.int16) - np.array(color[::-1], dtype=np.int16)), axis=2
        )
        masks[str(mask_id)] = _write_mask(output / f"{mask_id}.png", (distance <= tolerance) * 255)
    head = _mapping(spec.get("head"), "Semantic mask head")
    min_height = float(head.get("min_height_m", 1.45))
    radius = float(head.get("horizontal_radius_m", 0.18))
    hair_height = float(head.get("hair_min_height_m", 1.57))
    face_min_y = _optional_float(head, "face_min_y_m")
    face_max_y = _optional_float(head, "face_max_y_m")
    face_normal_y_max = _optional_float(head, "face_normal_y_max")
    hair_normal_z_min = _optional_float(head, "hair_normal_z_min")
    if min_height <= 0 or radius <= 0 or hair_height < min_height:
        raise ValueError("Semantic head thresholds are invalid")
    masks["hair_cap"] = _write_mask(
        output / "hair_cap.png",
        _mesh_mask(
            mesh,
            min_height=hair_height,
            radius=radius,
            normal_axis=2 if hair_normal_z_min is not None else None,
            normal_min=hair_normal_z_min,
        ),
    )
    masks["face"] = _write_mask(
        output / "face.png",
        _mesh_mask(
            mesh,
            min_height=min_height,
            radius=radius,
            min_y=face_min_y,
            max_y=face_max_y,
            normal_axis=1 if face_normal_y_max is not None else None,
            normal_max=face_normal_y_max,
        ),
    )
    neck = spec.get("neck")
    if neck is not None:
        neck_region = _mapping(neck, "Semantic mask neck")
        neck_min_height = float(neck_region.get("min_height_m", 1.30))
        neck_max_height = float(neck_region.get("max_height_m", min_height))
        neck_radius = float(neck_region.get("horizontal_radius_m", radius))
        front_max_y = _optional_float(neck_region, "front_max_y_m")
        front_radius = _optional_float(neck_region, "front_horizontal_radius_m")
        front_normal_y_max = _optional_float(neck_region, "front_normal_y_max")
        if (
            neck_min_height < 0
            or neck_max_height <= neck_min_height
            or neck_radius <= 0
            or (front_radius is not None and front_radius <= 0)
            or (front_radius is None) != (front_max_y is None)
        ):
            raise ValueError("Semantic neck thresholds are invalid")
        masks["neck"] = _write_mask(
            output / "neck.png",
            _mesh_mask(
                mesh,
                min_height=neck_min_height,
                max_height=neck_max_height,
                radius=neck_radius,
                front_max_y=front_max_y,
                front_radius=front_radius,
                front_normal_y_max=front_normal_y_max,
            ),
        )
    sideburns = spec.get("sideburns")
    if sideburns is not None:
        sideburn_region = _mapping(sideburns, "Semantic mask sideburns")
        sideburn_min_height = float(sideburn_region.get("min_height_m", 1.47))
        sideburn_max_height = float(sideburn_region.get("max_height_m", 1.66))
        sideburn_radius = float(sideburn_region.get("horizontal_radius_m", radius))
        sideburn_min_abs_x = float(sideburn_region.get("min_abs_x_m", 0.085))
        sideburn_max_abs_x = float(sideburn_region.get("max_abs_x_m", sideburn_radius))
        sideburn_max_y = _optional_float(sideburn_region, "max_y_m")
        if (
            sideburn_min_height < 0
            or sideburn_max_height <= sideburn_min_height
            or sideburn_radius <= 0
            or sideburn_min_abs_x < 0
            or sideburn_max_abs_x < sideburn_min_abs_x
        ):
            raise ValueError("Semantic sideburn thresholds are invalid")
        masks["sideburns"] = _write_mask(
            output / "sideburns.png",
            _mesh_mask(
                mesh,
                min_height=sideburn_min_height,
                max_height=sideburn_max_height,
                radius=sideburn_radius,
                min_abs_x=sideburn_min_abs_x,
                max_abs_x=sideburn_max_abs_x,
                max_y=sideburn_max_y,
            ),
        )
    manifest_path = output / "semantic_masks.json"
    manifest = {
        "schema_version": SEMANTIC_MASK_SCHEMA_VERSION,
        "texture_topology_id": topology_id,
        "source": {
            "base": str(base_path),
            "base_sha256": _sha256(base_path),
            "reference_mesh": str(mesh_path),
            "reference_mesh_sha256": _sha256(mesh_path),
        },
        "masks": {
            mask_id: {"file": path.name, "sha256": _sha256(path)}
            for mask_id, path in sorted(masks.items())
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return SemanticMaskSet(masks, manifest_path)


def _mesh_mask(
    mesh: _ObjMesh,
    *,
    min_height: float,
    max_height: float | None = None,
    radius: float,
    min_y: float | None = None,
    max_y: float | None = None,
    normal_axis: int | None = None,
    normal_min: float | None = None,
    normal_max: float | None = None,
    front_max_y: float | None = None,
    front_radius: float | None = None,
    front_normal_y_max: float | None = None,
    min_abs_x: float | None = None,
    max_abs_x: float | None = None,
) -> np.ndarray:
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    for face in mesh.faces:
        vertex_ids = np.asarray([vertex_id for vertex_id, _ in face])
        centroid = mesh.vertices[vertex_ids].mean(axis=0)
        normal: np.ndarray | None = None
        if front_normal_y_max is not None or normal_axis is not None:
            edges = mesh.vertices[vertex_ids[1:]] - mesh.vertices[vertex_ids[0]]
            normal = np.cross(edges[0], edges[1])
            length = np.linalg.norm(normal)
            if length == 0:
                continue
            normal = normal / length
        front_surface = front_radius is not None and (
            (front_max_y is not None and centroid[1] <= front_max_y)
            or (
                normal is not None
                and front_normal_y_max is not None
                and normal[1] <= front_normal_y_max
            )
        )
        radius_limit = front_radius if front_surface else radius
        if (
            centroid[2] < min_height
            or (max_height is not None and centroid[2] > max_height)
            or np.linalg.norm(centroid[:2]) > radius_limit
            or (min_abs_x is not None and abs(centroid[0]) < min_abs_x)
            or (max_abs_x is not None and abs(centroid[0]) > max_abs_x)
        ):
            continue
        if min_y is not None and centroid[1] < min_y:
            continue
        if max_y is not None and centroid[1] > max_y:
            continue
        if normal_axis is not None:
            if normal is None:
                raise RuntimeError("Normal-dependent mesh mask did not compute a face normal")
            component = normal[normal_axis]
            if normal_min is not None and component < normal_min:
                continue
            if normal_max is not None and component > normal_max:
                continue
        uv_ids = np.asarray([uv_id for _, uv_id in face])
        pixels = np.rint(
            np.column_stack((mesh.uvs[uv_ids, 0], 1.0 - mesh.uvs[uv_ids, 1])) * 1023
        ).astype(np.int32)
        cv2.fillConvexPoly(mask, pixels, 255, lineType=cv2.LINE_AA)
    return mask


def _write_mask(path: Path, image: np.ndarray) -> Path:
    if not cv2.imwrite(str(path), image.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise OSError(f"Could not write semantic mask '{path}'")
    return path


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ValueError(f"Semantic head '{key}' must be a number")
    return float(value)


def _rgb(value: object, context: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{context} must be a three-item integer RGB array")
    if not all(0 <= item <= 255 for item in value):
        raise ValueError(f"{context} components must be in [0, 255]")
    return value[0], value[1], value[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Console entry point for formal semantic mask generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(generate_semantic_masks(args.spec, args.output_dir).manifest_path)
