#!/usr/bin/env python3
"""Fuse a static OBJ accessory into every frame of an NPC mesh sequence.

MuJoCo recentres each mesh asset independently.  A separate accessory geom
therefore cannot reliably share a mesh-sequence body's local frame across
clips.  This tool writes one combined OBJ per source frame: body and accessory
vertices are in the same asset, so alpha frame selection cannot move them
relative to each other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


BODY_OCCLUSION_MODES = frozenset({"preserve", "cull_covered_head_faces"})
_CAP_OCCLUSION_GRID_CELL_M = 0.02


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vertices_and_faces(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append(_face_indices(line))
    if not vertices or not faces:
        raise ValueError(f"OBJ '{path}' needs vertices and faces")
    return np.asarray(vertices, dtype=np.float64), faces


def _face_indices(line: str) -> list[int]:
    return [int(token.split("/")[0]) for token in line.split()[1:]]


def _head_vertex_mask(vertices: np.ndarray) -> np.ndarray:
    return (vertices[:, 2] > 1.45) & (np.hypot(vertices[:, 0], vertices[:, 1]) < 0.19)


def _require_watertight_occlusion_mesh(accessory_faces: list[list[int]]) -> None:
    """Reject open shells because they cannot define an unambiguous culling volume."""
    edge_counts: Counter[tuple[int, int]] = Counter()
    for face in accessory_faces:
        for index, vertex in enumerate(face):
            neighbour = face[(index + 1) % len(face)]
            edge = (min(vertex, neighbour), max(vertex, neighbour))
            edge_counts[edge] += 1
    boundary_edges = sum(count == 1 for count in edge_counts.values())
    nonmanifold_edges = sum(count > 2 for count in edge_counts.values())
    if boundary_edges or nonmanifold_edges:
        raise ValueError(
            "Head-face culling requires a watertight accessory occlusion mesh: "
            f"boundary_edges={boundary_edges}, nonmanifold_edges={nonmanifold_edges}"
        )


def _point_is_covered_by_triangles(point: np.ndarray, triangles: np.ndarray) -> bool:
    """Return whether any cap triangle reaches ``point`` along the world Z axis."""
    xy = triangles[:, :, :2]
    denominator = (xy[:, 1, 1] - xy[:, 2, 1]) * (xy[:, 0, 0] - xy[:, 2, 0]) + (
        xy[:, 2, 0] - xy[:, 1, 0]
    ) * (xy[:, 0, 1] - xy[:, 2, 1])
    valid = np.abs(denominator) >= 1e-10
    first_weight = np.divide(
        (xy[:, 1, 1] - xy[:, 2, 1]) * (point[0] - xy[:, 2, 0])
        + (xy[:, 2, 0] - xy[:, 1, 0]) * (point[1] - xy[:, 2, 1]),
        denominator,
        out=np.zeros_like(denominator),
        where=valid,
    )
    second_weight = np.divide(
        (xy[:, 2, 1] - xy[:, 0, 1]) * (point[0] - xy[:, 2, 0])
        + (xy[:, 0, 0] - xy[:, 2, 0]) * (point[1] - xy[:, 2, 1]),
        denominator,
        out=np.zeros_like(denominator),
        where=valid,
    )
    third_weight = 1.0 - first_weight - second_weight
    inside = (np.minimum(np.minimum(first_weight, second_weight), third_weight) >= -1e-7) & valid
    cap_z = (
        first_weight * triangles[:, 0, 2]
        + second_weight * triangles[:, 1, 2]
        + third_weight * triangles[:, 2, 2]
    )
    return bool(np.any(inside & (cap_z >= point[2] - 1e-3)))


def _covered_head_face_indices(
    body_vertices: np.ndarray,
    body_faces: list[list[int]],
    accessory_vertices: np.ndarray,
    accessory_faces: list[list[int]],
) -> set[int]:
    """Find fully-head faces hidden behind a cap surface in the current frame."""
    head_vertices = _head_vertex_mask(body_vertices)
    cap_triangles = [
        accessory_vertices[[face[0] - 1, face[index] - 1, face[index + 1] - 1]]
        for face in accessory_faces
        for index in range(1, len(face) - 1)
    ]
    cap_bottom_z = float(accessory_vertices[:, 2].min())
    cap_grid: dict[tuple[int, int], list[np.ndarray]] = {}
    for triangle in cap_triangles:
        lower = np.floor(triangle[:, :2].min(axis=0) / _CAP_OCCLUSION_GRID_CELL_M).astype(int)
        upper = np.floor(triangle[:, :2].max(axis=0) / _CAP_OCCLUSION_GRID_CELL_M).astype(int)
        for x_cell in range(lower[0], upper[0] + 1):
            for y_cell in range(lower[1], upper[1] + 1):
                cap_grid.setdefault((x_cell, y_cell), []).append(triangle)
    triangle_grid = {cell: np.asarray(triangles) for cell, triangles in cap_grid.items()}
    covered: set[int] = set()
    for face_index, face in enumerate(body_faces):
        indices = np.asarray([vertex - 1 for vertex in face])
        if len(indices) < 3 or not np.all(head_vertices[indices]):
            continue
        centroid = body_vertices[indices].mean(axis=0)
        if centroid[2] < cap_bottom_z - 1e-3:
            continue
        cell = tuple(np.floor(centroid[:2] / _CAP_OCCLUSION_GRID_CELL_M).astype(int))
        if cell in triangle_grid and _point_is_covered_by_triangles(centroid, triangle_grid[cell]):
            covered.add(face_index)
    return covered


def _maskable_body_geometry(
    body_vertices: list[str],
    body_faces: list[str],
    *,
    maskable_body_face_indices: set[int],
    covered_body_face_indices: set[int],
) -> tuple[list[str], list[str]]:
    """Duplicate maskable faces so each frame can hide them without topology drift."""
    output_vertices = list(body_vertices)
    output_faces = list(body_faces)
    for face_index in sorted(maskable_body_face_indices):
        tokens = body_faces[face_index].split()[1:]
        vertex_indices = _face_indices(body_faces[face_index])
        duplicate_start = len(output_vertices) + 1
        if face_index in covered_body_face_indices:
            duplicated = [body_vertices[vertex_indices[0] - 1]] * len(vertex_indices)
        else:
            duplicated = [body_vertices[vertex_index - 1] for vertex_index in vertex_indices]
        output_vertices.extend(duplicated)
        output_faces[face_index] = "f " + " ".join(
            f"{duplicate_start + token_index}{token[token.find('/'):] if '/' in token else ''}"
            for token_index, token in enumerate(tokens)
        )
    return output_vertices, output_faces


def _fused_obj_with_vertices(
    body_path: Path,
    accessory_path: Path,
    accessory_vertices: np.ndarray,
    *,
    body_occlusion_mode: str = "preserve",
    covered_body_face_indices: set[int] | None = None,
    maskable_body_face_indices: set[int] | None = None,
) -> bytes:
    """Append already-positioned accessory vertices to one body OBJ."""
    if body_occlusion_mode not in BODY_OCCLUSION_MODES:
        raise ValueError(f"Unsupported body occlusion mode: {body_occlusion_mode}")
    body_lines = body_path.read_text(encoding="utf-8").splitlines()
    body_vertices = [line for line in body_lines if line.startswith("v ")]
    body_uvs = [line for line in body_lines if line.startswith("vt ")]
    body_normals = [line for line in body_lines if line.startswith("vn ")]
    body_faces = [line for line in body_lines if line.startswith("f ")]
    if not body_vertices or not body_faces:
        raise ValueError(f"Body OBJ '{body_path}' needs vertices and faces")
    _, accessory_faces = _vertices_and_faces(accessory_path)
    if body_occlusion_mode == "cull_covered_head_faces":
        if covered_body_face_indices is None:
            body_vertex_array = np.asarray(
                [[float(value) for value in line.split()[1:4]] for line in body_vertices],
                dtype=np.float64,
            )
            covered_body_face_indices = _covered_head_face_indices(
                body_vertex_array,
                [_face_indices(line) for line in body_faces],
                accessory_vertices,
                accessory_faces,
            )
        body_vertices, body_faces = _maskable_body_geometry(
            body_vertices,
            body_faces,
            maskable_body_face_indices=maskable_body_face_indices or covered_body_face_indices,
            covered_body_face_indices=covered_body_face_indices,
        )
    # Cap faces use a single UV at the atlas origin. This leaves every body UV
    # untouched while keeping the fused mesh valid for the existing body
    # material path. A future authored cap atlas may replace this UV choice.
    cap_uv = len(body_uvs) + 1
    cap_vertex_offset = len(body_vertices)
    output = [
        "# Fused NPC body + accessory; do not edit generated projection.",
        *body_vertices,
        *(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in accessory_vertices),
        *body_uvs,
        "vt 0 0",
        *body_normals,
        *body_faces,
        *(
            "f " + " ".join(f"{cap_vertex_offset + index}/{cap_uv}" for index in face)
            for face in accessory_faces
        ),
    ]
    return ("\n".join(output) + "\n").encode("utf-8")


def fused_obj(body_path: Path, accessory_path: Path, position: list[float]) -> bytes:
    """Append a translated accessory to a body OBJ with one stable cap UV."""
    accessory_vertices, _ = _vertices_and_faces(accessory_path)
    return _fused_obj_with_vertices(
        body_path, accessory_path, accessory_vertices + np.asarray(position, dtype=np.float64)
    )


def _head_indices(vertices: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(_head_vertex_mask(vertices))
    if len(indices) < 16:
        raise ValueError("Reference body frame has too few head vertices for head-follow fusion")
    return indices


def head_follow_vertices(
    reference_head: np.ndarray, target_head: np.ndarray, reference_accessory: np.ndarray
) -> np.ndarray:
    """Apply the rigid head transform from a reference body frame to a target frame."""
    if (
        reference_head.shape != target_head.shape
        or reference_head.ndim != 2
        or reference_head.shape[1] != 3
    ):
        raise ValueError("Head-follow frames need corresponding Nx3 vertices")
    reference_centre = reference_head.mean(axis=0)
    target_centre = target_head.mean(axis=0)
    covariance = (reference_head - reference_centre).T @ (target_head - target_centre)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    return (reference_accessory - reference_centre) @ rotation.T + target_centre


def fuse_manifest(
    manifest_path: Path,
    accessory_path: Path,
    anchors_path: Path,
    output_dir: Path,
    output_manifest: Path,
    *,
    bundle_id: str,
    body_occlusion_mode: str = "preserve",
) -> dict[str, object]:
    """Create fused frame projections and a manifest copy that references them."""
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = source["bundles"][bundle_id]
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    # The preview manifest retains the source accessory declaration for
    # provenance even though the population no longer instantiates it. Keep
    # those hashes current so strict manifest validation cannot be bypassed by
    # a stale, unused declaration.
    for accessory in bundle.get("accessories", {}).values():
        for field, path in (("mesh", accessory_path), ("anchors", anchors_path)):
            relative = accessory.get(field)
            if not isinstance(relative, str):
                continue
            if (manifest_path.parent / relative).resolve() == path.resolve():
                bundle["sha256"][relative] = _sha256(path)
    if "idle" not in bundle["clips"] or not bundle["clips"]["idle"]["frames"]:
        raise ValueError("Head-follow fusion requires an idle reference frame")
    reference_relative = bundle["clips"]["idle"]["frames"][0]
    reference_body, reference_body_faces = _vertices_and_faces(
        manifest_path.parent / reference_relative
    )
    head_indices = _head_indices(reference_body)
    reference_anchor = anchors.get("idle", [{}])[0].get("position")
    if not isinstance(reference_anchor, list) or len(reference_anchor) != 3:
        raise ValueError("Head-follow fusion requires idle/0 accessory position")
    accessory_vertices, accessory_faces = _vertices_and_faces(accessory_path)
    reference_accessory = accessory_vertices + np.asarray(reference_anchor, dtype=np.float64)
    frame_projections: dict[tuple[str, int], tuple[np.ndarray, set[int]]] = {}
    maskable_body_face_indices: set[int] = set()
    if body_occlusion_mode == "cull_covered_head_faces":
        _require_watertight_occlusion_mesh(accessory_faces)
    for clip_id, clip in bundle["clips"].items():
        frames = clip["frames"]
        clip_anchors = anchors.get(clip_id)
        if not isinstance(clip_anchors, list) or len(clip_anchors) != len(frames):
            raise ValueError(f"Accessory anchors do not match {clip_id} frame count")
        for frame_index, relative in enumerate(frames):
            position = clip_anchors[frame_index].get("position")
            if not isinstance(position, list) or len(position) != 3:
                raise ValueError(f"Accessory anchor {clip_id}/{frame_index} needs position")
            target_body, target_body_faces = _vertices_and_faces(manifest_path.parent / relative)
            if len(target_body) != len(reference_body):
                raise ValueError(f"Body frame {clip_id}/{frame_index} has incompatible topology")
            followed_accessory = head_follow_vertices(
                reference_body[head_indices], target_body[head_indices], reference_accessory
            )
            covered = set()
            if body_occlusion_mode == "cull_covered_head_faces":
                covered = _covered_head_face_indices(
                    target_body, target_body_faces, followed_accessory, accessory_faces
                )
                maskable_body_face_indices.update(covered)
            frame_projections[(clip_id, frame_index)] = (followed_accessory, covered)
    output_dir.mkdir(parents=True, exist_ok=True)
    fused_paths: dict[str, list[str]] = {}
    hashes = bundle["sha256"]
    for clip_id, clip in bundle["clips"].items():
        frames = clip["frames"]
        clip_anchors = anchors.get(clip_id)
        if not isinstance(clip_anchors, list) or len(clip_anchors) != len(frames):
            raise ValueError(f"Accessory anchors do not match {clip_id} frame count")
        fused_paths[clip_id] = []
        for frame_index, relative in enumerate(frames):
            destination = output_dir / clip_id / f"frame_{frame_index:03d}.obj"
            destination.parent.mkdir(parents=True, exist_ok=True)
            followed_accessory, covered_in_frame = frame_projections[(clip_id, frame_index)]
            destination.write_bytes(
                _fused_obj_with_vertices(
                    manifest_path.parent / relative,
                    accessory_path,
                    followed_accessory,
                    body_occlusion_mode=body_occlusion_mode,
                    covered_body_face_indices=covered_in_frame,
                    maskable_body_face_indices=maskable_body_face_indices,
                )
            )
            try:
                output_relative = destination.relative_to(output_manifest.parent).as_posix()
            except ValueError as error:
                raise ValueError("Fused output must be below output manifest directory") from error
            fused_paths[clip_id].append(output_relative)
            hashes[output_relative] = _sha256(destination)
        clip["frames"] = fused_paths[clip_id]
    source["fused_accessory"] = {
        "mode": "body_frame_projection",
        "head_follow": "rigid_head_alignment_v1",
        "reference_frame": "idle/0",
        "mesh": str(accessory_path),
        "mesh_sha256": _sha256(accessory_path),
        "anchors": str(anchors_path),
        "anchors_sha256": _sha256(anchors_path),
        "body_occlusion_mode": body_occlusion_mode,
        "maskable_head_face_count": len(maskable_body_face_indices),
    }
    output_manifest.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "mode": "body_frame_projection",
        "head_follow": "rigid_head_alignment_v1",
        "reference_frame": "idle/0",
        "source_manifest_sha256": _sha256(manifest_path),
        "accessory_sha256": _sha256(accessory_path),
        "anchors_sha256": _sha256(anchors_path),
        "body_occlusion_mode": body_occlusion_mode,
        "maskable_head_face_count": len(maskable_body_face_indices),
        "clips": fused_paths,
    }
    receipt_path = output_dir / "fused_accessory.receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(output_manifest), "receipt": str(receipt_path), "clips": fused_paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--cap-mesh", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--bundle", default="smplx_office_neutral_v1")
    parser.add_argument(
        "--body-occlusion-mode", choices=sorted(BODY_OCCLUSION_MODES), default="preserve"
    )
    args = parser.parse_args()
    result = fuse_manifest(
        args.asset_manifest.resolve(),
        args.cap_mesh.resolve(),
        args.anchors.resolve(),
        args.output_dir.resolve(),
        args.output_manifest.resolve(),
        bundle_id=args.bundle,
        body_occlusion_mode=args.body_occlusion_mode,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
