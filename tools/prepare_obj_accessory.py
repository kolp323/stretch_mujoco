#!/usr/bin/env python3
"""Prepare a source OBJ accessory for the NPC frame-sequence attachment path.

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


def _source_obj(source_archive: Path) -> tuple[bytes, dict[str, str]]:
    outer_bytes = source_archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer:
        nested_bytes = outer.read(SOURCE_ARCHIVE_MEMBER)
    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
        obj_bytes = nested.read(SOURCE_OBJ_MEMBER)
    return obj_bytes, {
        "source_archive_sha256": _sha256(outer_bytes),
        "nested_archive_member": SOURCE_ARCHIVE_MEMBER,
        "nested_archive_sha256": _sha256(nested_bytes),
        "source_obj_member": SOURCE_OBJ_MEMBER,
        "source_obj_sha256": _sha256(obj_bytes),
    }


def normalized_obj(source_obj: bytes) -> bytes:
    """Convert the source cap's centimetre Y-up vertices to metre Z-up OBJ."""
    lines = source_obj.decode("utf-8").splitlines()
    vertices = np.array(
        [[float(value) for value in line.split()[1:4]] for line in lines if line.startswith("v ")],
        dtype=np.float64,
    )
    if not len(vertices):
        raise ValueError("Source OBJ has no vertices")
    minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
    centre = (minimum + maximum) / 2.0
    # (source X, source Z, source Y) makes the source's Y-up convention
    # compatible with MuJoCo's Z-up convention.  Bottom is Z=0 so anchors
    # can consistently use the animated head-top position.
    converted = np.column_stack(
        (
            vertices[:, 0] - centre[0],
            vertices[:, 2] - centre[2],
            vertices[:, 1] - minimum[1],
        )
    ) * 0.01
    passthrough = [line for line in lines if not line.startswith(("v ", "mtllib ", "usemtl "))]
    return (
        "# Prepared from cap.zip; source centimetres/Y-up -> metres/MuJoCo Z-up.\n"
        + "\n".join([*(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in converted), *passthrough])
        + "\n"
    ).encode("utf-8")


def head_top_anchors(
    manifest_path: Path, *, head_clearance_m: float = 0.005, back_offset_m: float = 0.07
) -> dict[str, list[dict[str, list[float]]]]:
    """Place the normalized mesh above and behind the animated crown."""
    if head_clearance_m < 0:
        raise ValueError("head_clearance_m must not be negative")
    if back_offset_m < 0:
        raise ValueError("back_offset_m must not be negative")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = payload["bundles"]["smplx_office_neutral_v1"]
    anchors: dict[str, list[dict[str, list[float]]]] = {}
    for clip, item in bundle["clips"].items():
        anchors[clip] = []
        for relative in item["frames"]:
            points = _read_obj(manifest_path.parent / relative).vertices
            head = points[(points[:, 2] > 1.45) & (np.hypot(points[:, 0], points[:, 1]) < 0.19)]
            if not len(head):
                raise ValueError(f"Could not identify head vertices in {relative}")
            anchors[clip].append(
                {
                    "position": [
                        float(np.median(head[:, 0])),
                        # NPC forward is local -Y, so +Y moves a hat backward.
                        float(np.median(head[:, 1]) + back_offset_m),
                        float(np.max(head[:, 2]) + head_clearance_m),
                    ]
                }
            )
    return anchors


def prepare(
    source_archive: Path,
    manifest_path: Path,
    output_dir: Path,
    accessory_id: str,
    *,
    head_clearance_m: float = 0.005,
    back_offset_m: float = 0.07,
) -> dict[str, str]:
    """Write a normalized OBJ, all-clip anchors, and an auditable receipt."""
    source_obj, provenance = _source_obj(source_archive)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / f"{accessory_id}.obj"
    mesh_path.write_bytes(normalized_obj(source_obj))
    anchors_path = output_dir / f"{accessory_id}.anchors.json"
    anchors_path.write_text(
        json.dumps(
            head_top_anchors(
                manifest_path,
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
        "asset_quality": "preview",
        "source_format": "nested_zip_obj",
        "coordinate_transform": "source_cm_y_up_to_mujoco_m_z_up",
        "anchor_contract": "local_z_zero_at_head_top_per_animation_frame",
        "head_clearance_m": head_clearance_m,
        "back_offset_m": back_offset_m,
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
    parser.add_argument("--accessory-id", default="cap_source_v1")
    parser.add_argument(
        "--head-clearance-m",
        type=float,
        default=0.005,
        help="Vertical distance above each animated crown (default: 0.005 m)",
    )
    parser.add_argument(
        "--back-offset-m",
        type=float,
        default=0.07,
        help="Distance behind each animated crown along local +Y (default: 0.07 m)",
    )
    args = parser.parse_args()
    paths = prepare(
        args.source_archive.resolve(),
        args.manifest.resolve(),
        args.output_dir.resolve(),
        args.accessory_id,
        head_clearance_m=args.head_clearance_m,
        back_offset_m=args.back_offset_m,
    )
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
