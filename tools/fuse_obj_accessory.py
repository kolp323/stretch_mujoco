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
import copy
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
    body_path: Path,
    accessory_path: Path,
    accessory_vertices: np.ndarray,
    accessory_uv: tuple[float, float] = (0.0, 0.0),
    double_sided: bool = False,
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
    if double_sided:
        accessory_faces = [*accessory_faces, *(list(reversed(face)) for face in accessory_faces)]
    # Accessory faces use a single identity-selected UV. This leaves every body
    # UV untouched while keeping each fused mesh valid for the body material
    # path until an authored accessory material pipeline is introduced.
    accessory_uv_index = len(body_uvs) + 1
    accessory_vertex_offset = len(body_vertices)
    output = [
        "# Fused NPC body + accessory; do not edit generated projection.",
        *body_vertices,
        *(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in accessory_vertices),
        *body_uvs,
        f"vt {accessory_uv[0]:.9g} {accessory_uv[1]:.9g}",
        *body_normals,
        *body_faces,
        *(
            "f "
            + " ".join(f"{accessory_vertex_offset + index}/{accessory_uv_index}" for index in face)
            for face in accessory_faces
        ),
    ]
    return ("\n".join(output) + "\n").encode("utf-8")


def fused_obj(
    body_path: Path,
    accessory_path: Path,
    position: list[float],
    accessory_uv: tuple[float, float] = (0.0, 0.0),
    double_sided: bool = False,
) -> bytes:
    """Append a translated accessory to a body OBJ with one stable cap UV."""
    accessory_vertices, _ = _vertices_and_faces(accessory_path)
    return _fused_obj_with_vertices(
        body_path,
        accessory_path,
        accessory_vertices + np.asarray(position, dtype=np.float64),
        accessory_uv,
        double_sided,
    )


def _head_indices(vertices: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(
        (vertices[:, 2] > 1.45) & (np.hypot(vertices[:, 0], vertices[:, 1]) < 0.19)
    )
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
    fused_bundle_id: str | None = None,
    accessory_uv: tuple[float, float] = (0.0, 0.0),
    double_sided: bool = False,
) -> dict[str, object]:
    """Create a target-only fused bundle without modifying the source bundle."""
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_bundle = source["bundles"][bundle_id]
    source_sit_frames = source_bundle.get("clips", {}).get("sit", {}).get("frames")
    source_stand_up_frames = source_bundle.get("clips", {}).get("stand_up", {}).get("frames")
    if isinstance(source_sit_frames, list):
        source_sit_frames = list(source_sit_frames)
    if isinstance(source_stand_up_frames, list):
        source_stand_up_frames = list(source_stand_up_frames)
    fused_bundle_id = fused_bundle_id or bundle_id
    if fused_bundle_id != bundle_id and fused_bundle_id in source["bundles"]:
        raise ValueError(f"Fused bundle already exists: {fused_bundle_id}")
    bundle = copy.deepcopy(source_bundle) if fused_bundle_id != bundle_id else source_bundle
    source["bundles"][fused_bundle_id] = bundle
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
                _fused_obj_with_vertices(
                    manifest_path.parent / relative,
                    accessory_path,
                    followed_accessory,
                    accessory_uv,
                    double_sided,
                )
            )
            try:
                output_relative = destination.relative_to(output_manifest.parent).as_posix()
            except ValueError as error:
                raise ValueError("Fused output must be below output manifest directory") from error
            fused_paths[clip_id].append(output_relative)
            hashes[output_relative] = _sha256(destination)
        clip["frames"] = fused_paths[clip_id]
    if (
        isinstance(source_sit_frames, list)
        and isinstance(source_stand_up_frames, list)
        and source_stand_up_frames == list(reversed(source_sit_frames))
    ):
        # The asset contract represents stand_up as the same physical frames
        # played in reverse. Preserve that identity after fusion rather than
        # emitting duplicate paths which strict manifest validation rejects.
        fused_paths["stand_up"] = list(reversed(fused_paths["sit"]))
        bundle["clips"]["stand_up"]["frames"] = fused_paths["stand_up"]
    source.setdefault("fused_accessories", {})[fused_bundle_id] = {
        "mode": "body_frame_projection",
        "head_follow": "rigid_head_alignment_v1",
        "reference_frame": "idle/0",
        "mesh": str(accessory_path),
        "mesh_sha256": _sha256(accessory_path),
        "anchors": str(anchors_path),
        "anchors_sha256": _sha256(anchors_path),
        "double_sided": double_sided,
    }
    output_manifest.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "mode": "body_frame_projection",
        "head_follow": "rigid_head_alignment_v1",
        "reference_frame": "idle/0",
        "source_manifest_sha256": _sha256(manifest_path),
        "accessory_sha256": _sha256(accessory_path),
        "anchors_sha256": _sha256(anchors_path),
        "double_sided": double_sided,
        "clips": fused_paths,
    }
    receipt_path = output_dir / "fused_accessory.receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest": str(output_manifest),
        "receipt": str(receipt_path),
        "clips": fused_paths,
        "bundle": fused_bundle_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--cap-mesh", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--bundle", default="smplx_office_neutral_v1")
    parser.add_argument("--fused-bundle", required=True)
    parser.add_argument("--accessory-uv", type=float, nargs=2, required=True)
    parser.add_argument("--double-sided", action="store_true")
    args = parser.parse_args()
    result = fuse_manifest(
        args.asset_manifest.resolve(),
        args.cap_mesh.resolve(),
        args.anchors.resolve(),
        args.output_dir.resolve(),
        args.output_manifest.resolve(),
        bundle_id=args.bundle,
        fused_bundle_id=args.fused_bundle,
        accessory_uv=tuple(args.accessory_uv),
        double_sided=args.double_sided,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
