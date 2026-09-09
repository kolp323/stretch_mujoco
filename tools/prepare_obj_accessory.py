#!/usr/bin/env python3
"""Prepare a source OBJ or GLB accessory for the NPC frame-sequence attachment path.

The input archive is retained as provenance only: MuJoCo loads the normalized
OBJ written to ``--output-dir``.  The current source-cap convention is
centimetres with Y up; it is converted to metre, MuJoCo Z-up coordinates and
centred horizontally at the brim.  Its lowest point becomes the per-frame
head-top attachment origin.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from stretch_mujoco.npc.appearance_pipeline.hair_layers import _read_obj


SOURCE_ARCHIVE_MEMBER = "source/cap.zip"
SOURCE_OBJ_MEMBER = "cap.obj"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_obj(
    source_asset: Path,
    *,
    source_format: str = "nested_zip_obj",
    nested_archive_member: str | None = SOURCE_ARCHIVE_MEMBER,
    obj_member: str | None = SOURCE_OBJ_MEMBER,
) -> tuple[bytes, dict[str, str]]:
    source_bytes = source_asset.read_bytes()
    if source_format == "nested_zip_obj":
        assert nested_archive_member is not None and obj_member is not None
        with zipfile.ZipFile(io.BytesIO(source_bytes)) as outer:
            nested_bytes = outer.read(nested_archive_member)
        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
            obj_bytes = nested.read(obj_member)
        return obj_bytes, {
            "source_asset_sha256": _sha256(source_bytes),
            "nested_archive_member": nested_archive_member,
            "nested_archive_sha256": _sha256(nested_bytes),
            "source_obj_member": obj_member,
            "source_obj_sha256": _sha256(obj_bytes),
        }
    if source_format == "glb":
        obj_bytes = _glb_obj(source_bytes)
        return obj_bytes, {
            "source_asset_sha256": _sha256(source_bytes),
            "source_glb_sha256": _sha256(source_bytes),
            "source_obj_sha256": _sha256(obj_bytes),
        }
    raise ValueError(f"Unsupported source_format: {source_format}")


def _glb_obj(source: bytes) -> bytes:
    """Convert a self-contained triangle GLB into a world-space OBJ in memory."""
    if len(source) < 20 or source[:4] != b"glTF":
        raise ValueError("GLB source has an invalid header")
    _, version, total_length = np.frombuffer(source[:12], dtype="<u4")
    if version != 2 or total_length != len(source):
        raise ValueError("GLB source must use version 2 and have a valid length")
    offset, document, binary = 12, None, None
    while offset < len(source):
        length, chunk_type = np.frombuffer(source[offset : offset + 8], dtype="<u4")
        offset += 8
        chunk = source[offset : offset + int(length)]
        offset += int(length)
        if chunk_type == 0x4E4F534A:
            document = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary = chunk
    if not isinstance(document, dict) or binary is None:
        raise ValueError("GLB source needs JSON and binary chunks")
    nodes = document.get("nodes")
    meshes = document.get("meshes")
    scenes = document.get("scenes")
    if not isinstance(nodes, list) or not isinstance(meshes, list) or not isinstance(scenes, list):
        raise ValueError("GLB source needs nodes, meshes, and scenes")
    scene_index = int(document.get("scene", 0))
    if not 0 <= scene_index < len(scenes) or not isinstance(scenes[scene_index], dict):
        raise ValueError("GLB source scene is invalid")
    world: dict[int, np.ndarray] = {}

    def visit(node_index: int, parent: np.ndarray) -> None:
        if not 0 <= node_index < len(nodes) or not isinstance(nodes[node_index], dict):
            raise ValueError("GLB source node is invalid")
        node = nodes[node_index]
        transform = parent @ _gltf_node_matrix(node)
        world[node_index] = transform
        for child in node.get("children", []):
            if not isinstance(child, int):
                raise ValueError("GLB source child node index is invalid")
            visit(child, transform)

    for root_node in scenes[scene_index].get("nodes", []):
        if not isinstance(root_node, int):
            raise ValueError("GLB source scene root node index is invalid")
        visit(root_node, np.eye(4))
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    for node_index, transform in world.items():
        node = nodes[node_index]
        mesh_index = node.get("mesh")
        if mesh_index is None:
            continue
        if not isinstance(mesh_index, int) or not 0 <= mesh_index < len(meshes):
            raise ValueError("GLB source mesh index is invalid")
        mesh = meshes[mesh_index]
        if not isinstance(mesh, dict) or not isinstance(mesh.get("primitives"), list):
            raise ValueError("GLB source mesh primitives are invalid")
        for primitive in mesh["primitives"]:
            if not isinstance(primitive, dict) or primitive.get("mode", 4) != 4:
                raise ValueError("GLB source only supports triangle primitives")
            attributes = primitive.get("attributes")
            if not isinstance(attributes, dict) or not isinstance(attributes.get("POSITION"), int):
                raise ValueError("GLB source primitive needs POSITION")
            positions = _gltf_accessor(document, binary, attributes["POSITION"])
            if positions.shape[1] != 3:
                raise ValueError("GLB POSITION accessor must be VEC3")
            indices = (
                _gltf_accessor(document, binary, primitive["indices"]).reshape(-1)
                if isinstance(primitive.get("indices"), int)
                else np.arange(len(positions), dtype=np.uint32)
            )
            if len(indices) % 3:
                raise ValueError("GLB triangle index count must be divisible by three")
            homogeneous = np.column_stack((positions, np.ones(len(positions))))
            vertex_offset = sum(len(item) for item in vertices)
            vertices.append((transform @ homogeneous.T).T[:, :3])
            faces.append(indices.reshape(-1, 3).astype(np.uint32) + vertex_offset + 1)
    if not vertices or not faces:
        raise ValueError("GLB source has no mesh triangles")
    return (
        "# Converted from a self-contained GLB source.\n"
        + "\n".join(f"v {x:.9g} {y:.9g} {z:.9g}" for item in vertices for x, y, z in item)
        + "\n"
        + "\n".join(f"f {a} {b} {c}" for item in faces for a, b, c in item)
        + "\n"
    ).encode("utf-8")


def _gltf_node_matrix(node: dict[str, object]) -> np.ndarray:
    matrix = node.get("matrix")
    if isinstance(matrix, list):
        if len(matrix) != 16:
            raise ValueError("GLB node matrix must have 16 components")
        return np.asarray(matrix, dtype=np.float64).reshape(4, 4).T
    translation = np.asarray(node.get("translation", [0, 0, 0]), dtype=np.float64)
    scale = np.asarray(node.get("scale", [1, 1, 1]), dtype=np.float64)
    rotation = np.asarray(node.get("rotation", [0, 0, 0, 1]), dtype=np.float64)
    if translation.shape != (3,) or scale.shape != (3,) or rotation.shape != (4,):
        raise ValueError("GLB node TRS transform is invalid")
    x, y, z, w = rotation
    rotation_matrix = np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    result = np.eye(4)
    result[:3, :3] = rotation_matrix @ np.diag(scale)
    result[:3, 3] = translation
    return result


def _gltf_accessor(document: dict[str, object], binary: bytes, accessor_index: int) -> np.ndarray:
    accessors = document.get("accessors")
    buffer_views = document.get("bufferViews")
    if not isinstance(accessors, list) or not isinstance(buffer_views, list):
        raise ValueError("GLB source accessors or buffer views are invalid")
    if not 0 <= accessor_index < len(accessors) or not isinstance(accessors[accessor_index], dict):
        raise ValueError("GLB accessor index is invalid")
    accessor = accessors[accessor_index]
    view_index = accessor.get("bufferView")
    component_type = accessor.get("componentType")
    count = accessor.get("count")
    component_count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}.get(accessor.get("type"))
    dtype = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}.get(component_type)
    if (
        not isinstance(view_index, int)
        or not 0 <= view_index < len(buffer_views)
        or not isinstance(buffer_views[view_index], dict)
        or not isinstance(count, int)
        or component_count is None
        or dtype is None
    ):
        raise ValueError("GLB accessor declaration is invalid")
    view = buffer_views[view_index]
    item_size = np.dtype(dtype).itemsize
    stride = int(view.get("byteStride", item_size * component_count))
    byte_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    required = byte_offset + max(0, count - 1) * stride + item_size * component_count
    if byte_offset < 0 or required > len(binary):
        raise ValueError("GLB accessor exceeds binary chunk")
    return np.ndarray(
        (count, component_count),
        dtype=np.dtype(dtype),
        buffer=binary,
        offset=byte_offset,
        strides=(stride, item_size),
    ).copy()


def normalized_obj(
    source_obj: bytes,
    *,
    mesh_scale: float = 1.3,
    back_tilt_degrees: float = 13.0,
    roll_degrees: float = 0.0,
    yaw_degrees: float = 0.0,
    source_unit_scale: float = 0.01,
    source_vertical_anchor: float | None = None,
) -> bytes:
    """Convert a Y-up source OBJ into a metre, Z-up, head-anchored OBJ."""
    lines = source_obj.decode("utf-8").splitlines()
    vertices = np.array(
        [[float(value) for value in line.split()[1:4]] for line in lines if line.startswith("v ")],
        dtype=np.float64,
    )
    converted = transform_accessory_vertices(
        vertices,
        mesh_scale=mesh_scale,
        back_tilt_degrees=back_tilt_degrees,
        roll_degrees=roll_degrees,
        yaw_degrees=yaw_degrees,
        source_unit_scale=source_unit_scale,
        source_vertical_anchor=source_vertical_anchor,
    )
    passthrough = [line for line in lines if not line.startswith(("v ", "mtllib ", "usemtl "))]
    return (
        "# Prepared from a recipe-bound Y-up source -> metres/MuJoCo Z-up.\n"
        + "\n".join([*(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in converted), *passthrough])
        + "\n"
    ).encode("utf-8")


def transform_accessory_vertices(
    vertices: np.ndarray,
    *,
    mesh_scale: float,
    back_tilt_degrees: float,
    roll_degrees: float,
    yaw_degrees: float,
    source_unit_scale: float,
    source_vertical_anchor: float | None,
) -> np.ndarray:
    """Apply the authoritative source-to-head-local accessory transform."""
    if not np.isfinite(mesh_scale) or mesh_scale <= 0 or not np.isfinite(source_unit_scale):
        raise ValueError("mesh_scale and source_unit_scale must be positive finite numbers")
    if source_unit_scale <= 0:
        raise ValueError("source_unit_scale must be positive")
    if not all(np.isfinite(value) for value in (back_tilt_degrees, roll_degrees, yaw_degrees)):
        raise ValueError("back_tilt_degrees, roll_degrees and yaw_degrees must be finite")
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
        raise ValueError("Source OBJ has no vertices")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Source OBJ vertices must be finite")
    minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
    centre = (minimum + maximum) / 2.0
    vertical_origin = minimum[1] if source_vertical_anchor is None else source_vertical_anchor
    # (source X, source Z, source Y) makes Y-up sources compatible with
    # MuJoCo's Z-up convention. Long hair anchors at an authored source head
    # height, while hats use their lowest point.
    converted = np.column_stack(
        (
            vertices[:, 0] - centre[0],
            vertices[:, 2] - centre[2],
            vertices[:, 1] - vertical_origin,
        )
    ) * (source_unit_scale * mesh_scale)
    # Local -Y is NPC forward. Negative X rotation carries the crown toward
    # +Y, yielding a backward cap tilt around the head's lateral axis.
    angle = -np.deg2rad(back_tilt_degrees)
    tilt = np.array(
        ((1.0, 0.0, 0.0), (0.0, np.cos(angle), -np.sin(angle)), (0.0, np.sin(angle), np.cos(angle)))
    )
    converted = converted @ tilt.T
    roll = np.deg2rad(roll_degrees)
    roll_rotation = np.array(
        ((np.cos(roll), 0.0, np.sin(roll)), (0.0, 1.0, 0.0), (-np.sin(roll), 0.0, np.cos(roll)))
    )
    converted = converted @ roll_rotation.T
    yaw = np.deg2rad(yaw_degrees)
    yaw_rotation = np.array(
        ((np.cos(yaw), -np.sin(yaw), 0.0), (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0))
    )
    converted = converted @ yaw_rotation.T
    return converted


def head_top_position(
    points: np.ndarray,
    *,
    lateral_offset_m: float = 0.0,
    head_clearance_m: float = -0.026,
    back_offset_m: float = 0.098,
) -> list[float]:
    """Resolve one accessory anchor from a body frame and recipe offsets."""
    if not all(np.isfinite(value) for value in (lateral_offset_m, head_clearance_m, back_offset_m)):
        raise ValueError("Accessory position offsets must be finite")
    head = points[(points[:, 2] > 1.45) & (np.hypot(points[:, 0], points[:, 1]) < 0.19)]
    if not len(head):
        raise ValueError("Could not identify head vertices")
    return [
        float(np.median(head[:, 0]) + lateral_offset_m),
        # NPC forward is local -Y, so +Y moves an accessory backward.
        float(np.median(head[:, 1]) + back_offset_m),
        float(np.max(head[:, 2]) + head_clearance_m),
    ]


def head_top_anchors(
    manifest_path: Path,
    *,
    lateral_offset_m: float = 0.0,
    head_clearance_m: float = -0.026,
    back_offset_m: float = 0.098,
) -> dict[str, list[dict[str, list[float]]]]:
    """Place the normalized mesh above and behind the animated crown."""
    if not all(np.isfinite(value) for value in (lateral_offset_m, head_clearance_m, back_offset_m)):
        raise ValueError("Accessory position offsets must be finite")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = payload["bundles"]["smplx_office_neutral_v1"]
    anchors: dict[str, list[dict[str, list[float]]]] = {}
    for clip, item in bundle["clips"].items():
        anchors[clip] = []
        for relative in item["frames"]:
            points = _read_obj(manifest_path.parent / relative).vertices
            position = head_top_position(
                points,
                lateral_offset_m=lateral_offset_m,
                head_clearance_m=head_clearance_m,
                back_offset_m=back_offset_m,
            )
            anchors[clip].append({"position": position})
    return anchors


def prepare(
    source_asset: Path,
    manifest_path: Path,
    output_dir: Path,
    accessory_id: str,
    *,
    lateral_offset_m: float = 0.0,
    head_clearance_m: float = -0.026,
    back_offset_m: float = 0.098,
    mesh_scale: float = 1.3,
    back_tilt_degrees: float = 13.0,
    roll_degrees: float = 0.0,
    yaw_degrees: float = 0.0,
    source_format: str = "nested_zip_obj",
    nested_archive_member: str | None = SOURCE_ARCHIVE_MEMBER,
    obj_member: str | None = SOURCE_OBJ_MEMBER,
    source_unit_scale: float = 0.01,
    source_vertical_anchor: float | None = None,
) -> dict[str, str]:
    """Write a normalized OBJ, all-clip anchors, and an auditable receipt."""
    source_obj, provenance = _source_obj(
        source_asset,
        source_format=source_format,
        nested_archive_member=nested_archive_member,
        obj_member=obj_member,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / f"{accessory_id}.obj"
    mesh_path.write_bytes(
        normalized_obj(
            source_obj,
            mesh_scale=mesh_scale,
            back_tilt_degrees=back_tilt_degrees,
            roll_degrees=roll_degrees,
            yaw_degrees=yaw_degrees,
            source_unit_scale=source_unit_scale,
            source_vertical_anchor=source_vertical_anchor,
        )
    )
    anchors_path = output_dir / f"{accessory_id}.anchors.json"
    anchors_path.write_text(
        json.dumps(
            head_top_anchors(
                manifest_path,
                lateral_offset_m=lateral_offset_m,
                head_clearance_m=head_clearance_m,
                back_offset_m=back_offset_m,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = {
        "asset_id": accessory_id,
        "asset_quality": "production",
        "source_format": source_format,
        "coordinate_transform": "source_y_up_to_mujoco_m_z_up",
        "source_unit_scale": source_unit_scale,
        "source_vertical_anchor": source_vertical_anchor,
        "anchor_contract": "local_z_zero_at_head_top_per_animation_frame",
        "lateral_offset_m": lateral_offset_m,
        "head_clearance_m": head_clearance_m,
        "back_offset_m": back_offset_m,
        "mesh_scale": mesh_scale,
        "back_tilt_degrees": back_tilt_degrees,
        "roll_degrees": roll_degrees,
        "yaw_degrees": yaw_degrees,
        **provenance,
        "outputs": {
            "mesh": mesh_path.name,
            "mesh_sha256": _sha256(mesh_path.read_bytes()),
            "anchors": anchors_path.name,
            "anchors_sha256": _sha256(anchors_path.read_bytes()),
        },
    }
    receipt_path = output_dir / f"{accessory_id}.receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return {"mesh": str(mesh_path), "anchors": str(anchors_path), "receipt": str(receipt_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accessory-id", default="baseball_cap_v1")
    parser.add_argument(
        "--head-clearance-m",
        type=float,
        default=-0.026,
        help="Vertical distance above each animated crown (default: -0.026 m)",
    )
    parser.add_argument(
        "--back-offset-m",
        type=float,
        default=0.098,
        help="Distance behind each animated crown along local +Y (default: 0.098 m)",
    )
    parser.add_argument(
        "--mesh-scale", type=float, default=1.3, help="Uniform local mesh scale (default: 1.3)"
    )
    parser.add_argument(
        "--back-tilt-degrees",
        type=float,
        default=13.0,
        help="Backward tilt around local X (default: 13 degrees)",
    )
    args = parser.parse_args()
    paths = prepare(
        args.source_archive.resolve(),
        args.manifest.resolve(),
        args.output_dir.resolve(),
        args.accessory_id,
        head_clearance_m=args.head_clearance_m,
        back_offset_m=args.back_offset_m,
        mesh_scale=args.mesh_scale,
        back_tilt_degrees=args.back_tilt_degrees,
    )
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
