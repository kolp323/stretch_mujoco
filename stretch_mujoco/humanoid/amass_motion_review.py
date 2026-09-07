"""Render local AMASS SMPL-X motion crops for human action review."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import click
import numpy as np

from .smplx_converter import SmplxAssetError, _load_runtime, _to_mujoco_coordinates, find_model_file


class AmassReviewError(ValueError):
    """Raised when a local AMASS crop cannot be reviewed safely."""


@dataclass(frozen=True)
class AmassMotion:
    source_id: str
    source_sha256: str
    body_pose: np.ndarray
    global_orient: np.ndarray
    transl: np.ndarray
    mocap_frame_rate: float


def _payload_for_source(source: Path, source_id: str) -> bytes:
    if source.is_file() and source.suffix == ".npz":
        if source_id != source.name:
            raise AmassReviewError(f"Direct NPZ source_id must be '{source.name}'")
        return source.read_bytes()
    if source.is_dir():
        candidate = source / source_id
        if not candidate.is_file() or candidate.suffix != ".npz":
            raise AmassReviewError(f"AMASS source_id is unavailable: {source_id!r}")
        return candidate.read_bytes()
    if source.is_file() and source.name.endswith(".tar.bz2"):
        with tarfile.open(source, "r:bz2") as archive:
            for member in archive:
                if member.isfile() and member.name == source_id:
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        break
                    return extracted.read()
        raise AmassReviewError(f"AMASS source_id is unavailable: {source_id!r}")
    raise AmassReviewError("AMASS source must be an NPZ file, directory, or .tar.bz2 archive")


def _scalar_fps(payload: np.lib.npyio.NpzFile) -> float:
    if "mocap_frame_rate" not in payload:
        raise AmassReviewError("AMASS motion is missing mocap_frame_rate")
    value = payload["mocap_frame_rate"]
    if value.shape != () or not np.issubdtype(value.dtype, np.number) or float(value) <= 0:
        raise AmassReviewError("AMASS motion has an invalid mocap_frame_rate")
    return float(value)


def load_amass_motion(source: Path, source_id: str) -> AmassMotion:
    """Load one local SMPL-X motion member without NumPy pickle support."""
    raw = _payload_for_source(source, source_id)
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
            if "pose_body" not in payload:
                raise AmassReviewError("AMASS motion is missing pose_body")
            body_pose = np.asarray(payload["pose_body"], dtype=np.float32)
            if body_pose.ndim != 2 or body_pose.shape[1] != 63 or not np.isfinite(body_pose).all():
                raise AmassReviewError("AMASS pose_body must be finite with shape (frames, 63)")
            frame_count = body_pose.shape[0]
            global_orient = np.asarray(
                payload["root_orient"] if "root_orient" in payload else np.zeros((frame_count, 3)),
                dtype=np.float32,
            )
            transl = np.asarray(
                payload["trans"] if "trans" in payload else np.zeros((frame_count, 3)),
                dtype=np.float32,
            )
            if global_orient.shape != (frame_count, 3) or transl.shape != (frame_count, 3):
                raise AmassReviewError(
                    "AMASS root_orient and trans must match pose_body frame count"
                )
            if not np.isfinite(global_orient).all() or not np.isfinite(transl).all():
                raise AmassReviewError("AMASS root_orient and trans must be finite")
            fps = _scalar_fps(payload)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, AmassReviewError):
            raise
        raise AmassReviewError(f"Could not read AMASS motion '{source_id}': {error}") from error
    return AmassMotion(
        source_id=source_id,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        mocap_frame_rate=fps,
    )


def review_frame_indices(
    start_frame: int, end_frame: int, frame_count: int, samples: int
) -> np.ndarray:
    if start_frame < 0 or end_frame <= start_frame or end_frame > frame_count:
        raise AmassReviewError("Review frame range must be non-empty and within the source motion")
    if samples < 2:
        raise AmassReviewError("Review requires at least two samples")
    return np.linspace(
        start_frame, end_frame - 1, num=min(samples, end_frame - start_frame), dtype=int
    )


def render_motion_review(
    source: Path,
    source_id: str,
    start_frame: int,
    end_frame: int,
    model_root: Path,
    output_dir: Path,
    samples: int = 9,
) -> tuple[Path, Path]:
    """Render a local contact sheet and an adjacent provenance receipt."""
    motion = load_amass_motion(source, source_id)
    indices = review_frame_indices(start_frame, end_frame, motion.body_pose.shape[0], samples)
    model_file = find_model_file(model_root, "smplx", "neutral")
    try:
        smplx, torch = _load_runtime()
    except SmplxAssetError as error:
        raise AmassReviewError(str(error)) from error
    model = smplx.create(
        str(model_file),
        model_type="smplx",
        gender="neutral",
        ext=model_file.suffix.lstrip("."),
        use_pca=False,
        batch_size=len(indices),
    )
    with torch.no_grad():
        result = model(
            body_pose=torch.from_numpy(motion.body_pose[indices]),
            global_orient=torch.from_numpy(motion.global_orient[indices]),
            transl=torch.from_numpy(motion.transl[indices]),
            return_verts=True,
        )
    vertices = _to_mujoco_coordinates(
        result.vertices.detach().cpu().numpy().reshape(-1, 3)
    ).reshape(len(indices), -1, 3)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(source_id).stem}_{start_frame}_{end_frame}"
    image_path = output_dir / f"{stem}.png"
    receipt_path = output_dir / f"{stem}.review.json"
    if image_path.exists() or receipt_path.exists():
        raise AmassReviewError(f"Refusing to overwrite existing review output '{stem}'")
    import matplotlib

    matplotlib.use("Agg")
    plt: Any = import_module("matplotlib.pyplot")
    poly3d: Any = import_module("mpl_toolkits.mplot3d.art3d")
    poly_3d_collection: Any = poly3d.Poly3DCollection

    columns = min(3, len(indices))
    rows = int(np.ceil(len(indices) / columns))
    figure = plt.figure(figsize=(4 * columns, 5 * rows))
    minima, maxima = vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))
    span = float((maxima - minima).max())
    center = (minima + maxima) / 2.0
    faces = np.asarray(model.faces, dtype=np.int64)
    for panel, (frame_index, frame_vertices) in enumerate(zip(indices, vertices), start=1):
        axis: Any = figure.add_subplot(rows, columns, panel, projection="3d")
        mesh = poly_3d_collection(frame_vertices[faces], facecolor="#759cc9", edgecolor="none")
        axis.add_collection3d(mesh)
        axis.set_xlim(center[0] - span / 2, center[0] + span / 2)
        axis.set_ylim(center[1] - span / 2, center[1] + span / 2)
        axis.set_zlim(center[2] - span / 2, center[2] + span / 2)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=12, azim=-68)
        axis.set_axis_off()
        axis.set_title(f"frame {frame_index} ({frame_index / motion.mocap_frame_rate:.2f}s)")
    figure.tight_layout()
    figure.savefig(image_path, dpi=150)
    plt.close(figure)
    receipt = {
        "schema_version": 1,
        "review_status": "human_review_required",
        "source_id": source_id,
        "source_sha256": motion.source_sha256,
        "source_frame_range": [start_frame, end_frame],
        "source_fps": motion.mocap_frame_rate,
        "reviewed_frame_indices": indices.tolist(),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "note": "This receipt is visual evidence only and does not approve an action clip.",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return image_path, receipt_path


@click.command()
@click.option("--source", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--source-id", required=True)
@click.option("--start-frame", type=int, required=True)
@click.option("--end-frame", type=int, required=True)
@click.option("--model-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--samples", type=click.IntRange(min=2), default=9, show_default=True)
def main(
    source: Path,
    source_id: str,
    start_frame: int,
    end_frame: int,
    model_root: Path,
    output_dir: Path,
    samples: int,
) -> None:
    """Render one AMASS crop for local human review."""
    try:
        image_path, receipt_path = render_motion_review(
            source, source_id, start_frame, end_frame, model_root, output_dir, samples
        )
    except (AmassReviewError, SmplxAssetError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Wrote {image_path} and {receipt_path}")
