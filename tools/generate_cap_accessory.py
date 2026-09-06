#!/usr/bin/env python3
"""Generate a simple round-dome hat OBJ and per-frame head-top anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stretch_mujoco.npc.appearance_pipeline.hair_layers import _read_obj


def _cylinder(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, ...]],
    *,
    radius: float,
    height: float,
    z_center: float,
    segments: int,
) -> None:
    start = len(vertices) + 1
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    for z in (z_center - height / 2.0, z_center + height / 2.0):
        vertices.extend((radius * np.cos(angle), radius * np.sin(angle), z) for angle in angles)
    bottom_center = len(vertices) + 1
    vertices.append((0.0, 0.0, z_center - height / 2.0))
    top_center = len(vertices) + 1
    vertices.append((0.0, 0.0, z_center + height / 2.0))
    for index in range(segments):
        next_index = (index + 1) % segments
        faces.append(
            (
                start + index,
                start + next_index,
                start + segments + next_index,
                start + segments + index,
            )
        )
        faces.append((bottom_center, start + next_index, start + index))
        faces.append((top_center, start + segments + index, start + segments + next_index))


def _dome(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, ...]],
    *,
    radius: float,
    height: float,
    z_base: float,
    segments: int,
    rings: int,
) -> None:
    """Append a low-poly ellipsoidal dome with its base at ``z_base``."""
    start = len(vertices) + 1
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    for ring in range(rings):
        fraction = ring / rings
        radial = radius * np.cos(fraction * np.pi / 2.0)
        z = z_base + height * np.sin(fraction * np.pi / 2.0)
        vertices.extend((radial * np.cos(angle), radial * np.sin(angle), z) for angle in angles)
    peak = len(vertices) + 1
    vertices.append((0.0, 0.0, z_base + height))
    for ring in range(rings - 1):
        lower = start + ring * segments
        upper = lower + segments
        for index in range(segments):
            next_index = (index + 1) % segments
            faces.append((lower + index, lower + next_index, upper + next_index, upper + index))
    top_ring = start + (rings - 1) * segments
    for index in range(segments):
        faces.append((top_ring + index, top_ring + (index + 1) % segments, peak))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    bundle = payload["bundles"]["smplx_office_neutral_v1"]
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    # A low-poly circular brim plus a shallow dome is intentionally simpler
    # than a stitched cap, while preserving a recognizable hat silhouette.
    _cylinder(vertices, faces, radius=0.095, height=0.010, z_center=-0.014, segments=12)
    _dome(vertices, faces, radius=0.078, height=0.052, z_base=-0.009, segments=12, rings=3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cap_simple_v1.obj").write_text(
        "\n".join(
            [
                *(f"v {x} {y} {z}" for x, y, z in vertices),
                *("f " + " ".join(map(str, face)) for face in faces),
            ]
        )
        + "\n"
    )
    anchors: dict[str, list[dict[str, list[float]]]] = {}
    for clip, item in bundle["clips"].items():
        anchors[clip] = []
        for relative in item["frames"]:
            points = _read_obj(args.manifest.parent / relative).vertices
            head = points[(points[:, 2] > 1.45) & (np.hypot(points[:, 0], points[:, 1]) < 0.19)]
            anchors[clip].append(
                {
                    "position": [
                        float(np.median(head[:, 0])),
                        float(np.median(head[:, 1])),
                        # The mesh's local bounds are recentered by MuJoCo.
                        # A small inset produces a visually seated brim rather
                        # than a visibly floating cap, while avoiding any
                        # exposed penetration at the scalp silhouette.
                        float(np.max(head[:, 2]) - 0.010),
                    ]
                }
            )
    (args.output_dir / "cap_simple_v1.anchors.json").write_text(
        json.dumps(anchors, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
