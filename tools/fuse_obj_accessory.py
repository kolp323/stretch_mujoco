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
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vertices_and_faces(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(token.split("/")[0]) for token in line.split()[1:]])
    if not vertices or not faces:
        raise ValueError(f"OBJ '{path}' needs vertices and faces")
    return np.asarray(vertices, dtype=np.float64), faces


def _fused_obj_with_vertices(
    body_path: Path, accessory_path: Path, accessory_vertices: np.ndarray
) -> bytes:
    """Append already-positioned accessory vertices to one body OBJ."""
    body_lines = body_path.read_text(encoding="utf-8").splitlines()
    body_vertices = [line for line in body_lines if line.startswith("v ")]
    body_uvs = [line for line in body_lines if line.startswith("vt ")]
    body_normals = [line for line in body_lines if line.startswith("vn ")]
    body_faces = [line for line in body_lines if line.startswith("f ")]
    if not body_vertices or not body_faces:
        raise ValueError(f"Body OBJ '{body_path}' needs vertices and faces")
    _, accessory_faces = _vertices_and_faces(accessory_path)
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
    indices = np.flatnonzero((vertices[:, 2] > 1.45) & (np.hypot(vertices[:, 0], vertices[:, 1]) < 0.19))
    if len(indices) < 16:
        raise ValueError("Reference body frame has too few head vertices for head-follow fusion")
    return indices


def head_follow_vertices(
    reference_head: np.ndarray, target_head: np.ndarray, reference_accessory: np.ndarray
) -> np.ndarray:
    """Apply the rigid head transform from a reference body frame to a target frame."""
    if reference_head.shape != target_head.shape or reference_head.ndim != 2 or reference_head.shape[1] != 3:
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
) -> dict[str, object]:
    """Create fused frame projections and a manifest copy that references them."""
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = source["bundles"][bundle_id]
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    if "idle" not in bundle["clips"] or not bundle["clips"]["idle"]["frames"]:
        raise ValueError("Head-follow fusion requires an idle reference frame")
    reference_relative = bundle["clips"]["idle"]["frames"][0]
    reference_body, _ = _vertices_and_faces(manifest_path.parent / reference_relative)
    head_indices = _head_indices(reference_body)
    reference_anchor = anchors.get("idle", [{}])[0].get("position")
    if not isinstance(reference_anchor, list) or len(reference_anchor) != 3:
        raise ValueError("Head-follow fusion requires idle/0 accessory position")
    accessory_vertices, _ = _vertices_and_faces(accessory_path)
    reference_accessory = accessory_vertices + np.asarray(reference_anchor, dtype=np.float64)
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
            position = clip_anchors[frame_index].get("position")
            if not isinstance(position, list) or len(position) != 3:
                raise ValueError(f"Accessory anchor {clip_id}/{frame_index} needs position")
            destination = output_dir / clip_id / f"frame_{frame_index:03d}.obj"
            destination.parent.mkdir(parents=True, exist_ok=True)
            target_body, _ = _vertices_and_faces(manifest_path.parent / relative)
            if len(target_body) != len(reference_body):
                raise ValueError(f"Body frame {clip_id}/{frame_index} has incompatible topology")
            followed_accessory = head_follow_vertices(
                reference_body[head_indices], target_body[head_indices], reference_accessory
            )
            destination.write_bytes(
                _fused_obj_with_vertices(manifest_path.parent / relative, accessory_path, followed_accessory)
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
    }
    output_manifest.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "mode": "body_frame_projection",
        "head_follow": "rigid_head_alignment_v1",
        "reference_frame": "idle/0",
        "source_manifest_sha256": _sha256(manifest_path),
        "accessory_sha256": _sha256(accessory_path),
        "anchors_sha256": _sha256(anchors_path),
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
    args = parser.parse_args()
    result = fuse_manifest(
        args.asset_manifest.resolve(),
        args.cap_mesh.resolve(),
        args.anchors.resolve(),
        args.output_dir.resolve(),
        args.output_manifest.resolve(),
        bundle_id=args.bundle,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
