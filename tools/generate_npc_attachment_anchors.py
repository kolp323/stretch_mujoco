#!/usr/bin/env python3
"""Generate per-frame hand anchors for a topology-stable NPC mesh bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _vertices(path: Path) -> np.ndarray:
    values: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                values.append((float(x), float(y), float(z)))
    if not values:
        raise ValueError(f"OBJ contains no vertices: {path}")
    return np.asarray(values, dtype=np.float64)


def generate(
    manifest_path: Path,
    bundle_id: str,
    reference_obj: Path,
    reference_metadata: Path,
    output_path: Path,
    *,
    role: str = "handover",
    joint_index: int = 21,
    radius_m: float = 0.055,
) -> None:
    """Project a reference wrist neighbourhood through every baked mesh frame."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = manifest["bundles"][bundle_id]
    metadata = json.loads(reference_metadata.read_text(encoding="utf-8"))
    joint = np.asarray(metadata["joints_m"][joint_index], dtype=np.float64)
    reference = _vertices(reference_obj)
    distances = np.linalg.norm(reference - joint, axis=1)
    vertex_ids = np.flatnonzero(distances <= radius_m)
    if vertex_ids.size < 8:
        raise ValueError(
            f"Reference wrist neighbourhood is too small: {vertex_ids.size} vertices"
        )

    clips: dict[str, list[list[float]]] = {}
    for clip_id, clip in bundle["clips"].items():
        anchors: list[list[float]] = []
        for relative_frame in clip["frames"]:
            vertices = _vertices(manifest_path.parent / relative_frame)
            if len(vertices) != len(reference):
                raise ValueError(f"Mesh topology differs from reference: {relative_frame}")
            anchors.append([round(float(value), 7) for value in vertices[vertex_ids].mean(axis=0)])
        clips[clip_id] = anchors

    payload = {
        "schema_version": 1,
        "coordinate_system": bundle["coordinate_system"],
        "role": role,
        "source": {
            "reference_obj": os.path.relpath(reference_obj, manifest_path.parent),
            "reference_metadata": os.path.relpath(reference_metadata, manifest_path.parent),
            "joint_index": joint_index,
            "radius_m": radius_m,
            "vertex_count": int(vertex_ids.size),
        },
        "clips": clips,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = output_path.with_suffix(output_path.suffix + ".tmp")
    candidate.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(candidate, output_path)

    relative_output = str(output_path.relative_to(manifest_path.parent))
    bundle.setdefault("attachment_anchors", {})[role] = relative_output
    bundle["sha256"][relative_output] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    candidate_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    candidate_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(candidate_manifest, manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--reference-obj", type=Path, required=True)
    parser.add_argument("--reference-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", default="handover")
    parser.add_argument("--joint-index", type=int, default=21)
    parser.add_argument("--radius-m", type=float, default=0.055)
    args = parser.parse_args()
    generate(
        args.manifest.resolve(),
        args.bundle,
        args.reference_obj.resolve(),
        args.reference_metadata.resolve(),
        args.output.resolve(),
        role=args.role,
        joint_index=args.joint_index,
        radius_m=args.radius_m,
    )


if __name__ == "__main__":
    main()
