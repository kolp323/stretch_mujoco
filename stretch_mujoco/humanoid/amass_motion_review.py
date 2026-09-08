"""Render local AMASS SMPL-X motion crops for human action review."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import zipfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import click
import numpy as np

from .amass_intake import canonicalize_vertical_translation
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


@dataclass(frozen=True)
class AmassReviewQueueItem:
    """One pending, explicitly bounded visual review."""

    source_id: str
    start_frame: int
    end_frame: int


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


def _payloads_for_sources(source: Path, source_ids: set[str]) -> dict[str, bytes]:
    """Read named local members, scanning an archive at most once."""
    if not source_ids:
        return {}
    if source.is_file() and source.suffix == ".npz":
        if source.name not in source_ids or len(source_ids) != 1:
            raise AmassReviewError("Direct NPZ review source_id must match the file name")
        return {source.name: source.read_bytes()}
    if source.is_dir():
        payloads = {
            source_id: (source / source_id).read_bytes()
            for source_id in source_ids
            if (source / source_id).is_file() and (source / source_id).suffix == ".npz"
        }
    elif source.is_file() and source.name.endswith(".tar.bz2"):
        payloads = {}
        with tarfile.open(source, "r:bz2") as archive:
            for member in archive:
                if not member.isfile() or member.name not in source_ids:
                    continue
                extracted = archive.extractfile(member)
                if extracted is not None:
                    payloads[member.name] = extracted.read()
                if len(payloads) == len(source_ids):
                    break
    else:
        raise AmassReviewError("AMASS source must be an NPZ file, directory, or .tar.bz2 archive")
    missing = sorted(source_ids - payloads.keys())
    if missing:
        raise AmassReviewError(f"AMASS source_id is unavailable: {missing[0]!r}")
    return payloads


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
    return _motion_from_payload(source_id, raw)


def review_output_stem(source_id: str, start_frame: int, end_frame: int) -> str:
    """Return a filesystem-safe, source-unique review artifact name."""
    safe_source_id = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source_id).stem).strip("._")
    safe_source_id = safe_source_id or "motion"
    source_token = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe_source_id}--{source_token}--{start_frame}-{end_frame}"


def _motion_from_payload(source_id: str, raw: bytes) -> AmassMotion:
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


def _canonical_review_root_parameters(frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Match restricted baking's identity root orientation and anchored translation."""
    if frame_count < 1:
        raise AmassReviewError("Canonical review requires at least one frame")
    root = np.zeros((frame_count, 3), dtype=np.float32)
    return root, root.copy()


def build_review_queue(
    candidate_index: Path,
    batch_size: int,
    min_duration_seconds: float,
    max_duration_seconds: float,
    crop_seconds: float,
) -> tuple[AmassReviewQueueItem, ...]:
    """Select a deterministic, source-balanced batch from an AMASS candidate index."""
    if batch_size < 1 or min_duration_seconds <= 0 or max_duration_seconds < min_duration_seconds:
        raise AmassReviewError("Review queue requires valid batch size and duration bounds")
    if crop_seconds <= 0:
        raise AmassReviewError("Review queue crop_seconds must be positive")
    try:
        records = [
            json.loads(line)
            for line in candidate_index.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise AmassReviewError(
            f"Could not read candidate index '{candidate_index}': {error}"
        ) from error
    groups: dict[str, list[AmassReviewQueueItem]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("usable") is not True:
            continue
        source_id = record.get("source_id")
        frame_count = record.get("frame_count")
        fps = record.get("mocap_frame_rate")
        duration = record.get("duration_seconds")
        if (
            not isinstance(source_id, str)
            or not isinstance(frame_count, int)
            or isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not min_duration_seconds <= duration <= max_duration_seconds
        ):
            continue
        end_frame = min(frame_count, max(2, int(float(fps) * crop_seconds)))
        if end_frame < 2:
            continue
        groups.setdefault(Path(source_id).parent.as_posix(), []).append(
            AmassReviewQueueItem(source_id, 0, end_frame)
        )
    if not groups:
        raise AmassReviewError(
            "Candidate index has no usable entries within the requested duration range"
        )
    for group in groups.values():
        group.sort(key=lambda item: item.source_id)
    queue: list[AmassReviewQueueItem] = []
    while groups and len(queue) < batch_size:
        for group_id in sorted(tuple(groups)):
            queue.append(groups[group_id].pop(0))
            if not groups[group_id]:
                del groups[group_id]
            if len(queue) == batch_size:
                break
    return tuple(queue)


def write_review_queue(
    candidate_index: Path, output_path: Path, items: tuple[AmassReviewQueueItem, ...]
) -> Path:
    """Write an ignored pending-review queue without replacing a prior decision record."""
    if output_path.exists():
        raise AmassReviewError(f"Refusing to overwrite existing review queue '{output_path}'")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_status": "pending_human_review",
                "candidate_index_sha256": hashlib.sha256(candidate_index.read_bytes()).hexdigest(),
                "items": [
                    {
                        "source_id": item.source_id,
                        "source_frame_range": [item.start_frame, item.end_frame],
                        "output_stem": review_output_stem(
                            item.source_id, item.start_frame, item.end_frame
                        ),
                    }
                    for item in items
                ],
                "note": "Queue items are not action approvals or baker inputs.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def load_review_queue(path: Path) -> tuple[AmassReviewQueueItem, ...]:
    """Load a pending queue and reject malformed or duplicate review items."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AmassReviewError(f"Could not read review queue '{path}': {error}") from error
    items = document.get("items") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or not isinstance(items, list)
        or not items
    ):
        raise AmassReviewError("Review queue must be a schema_version 1 document with items")
    parsed: list[AmassReviewQueueItem] = []
    for item in items:
        frame_range = item.get("source_frame_range") if isinstance(item, dict) else None
        source_id = item.get("source_id") if isinstance(item, dict) else None
        if (
            not isinstance(source_id, str)
            or not isinstance(frame_range, list)
            or len(frame_range) != 2
            or any(isinstance(frame, bool) or not isinstance(frame, int) for frame in frame_range)
            or frame_range[0] < 0
            or frame_range[1] <= frame_range[0]
        ):
            raise AmassReviewError("Review queue has an invalid source_id or source_frame_range")
        parsed.append(AmassReviewQueueItem(source_id, frame_range[0], frame_range[1]))
    if len({(item.source_id, item.start_frame, item.end_frame) for item in parsed}) != len(parsed):
        raise AmassReviewError("Review queue contains duplicate items")
    return tuple(parsed)


def _render_loaded_motion(
    motion: AmassMotion,
    start_frame: int,
    end_frame: int,
    model_root: Path,
    output_dir: Path,
    samples: int = 9,
) -> tuple[Path, Path]:
    """Render a local contact sheet and an adjacent provenance receipt."""
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
    canonical_orient, _ = _canonical_review_root_parameters(len(indices))
    canonical_transl = canonicalize_vertical_translation(motion.transl[indices])
    with torch.no_grad():
        result = model(
            body_pose=torch.from_numpy(motion.body_pose[indices]),
            global_orient=torch.from_numpy(canonical_orient),
            transl=torch.from_numpy(canonical_transl),
            return_verts=True,
        )
    vertices = _to_mujoco_coordinates(
        result.vertices.detach().cpu().numpy().reshape(-1, 3)
    ).reshape(len(indices), -1, 3)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = review_output_stem(motion.source_id, start_frame, end_frame)
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
        "source_id": motion.source_id,
        "source_sha256": motion.source_sha256,
        "source_frame_range": [start_frame, end_frame],
        "source_fps": motion.mocap_frame_rate,
        "reviewed_frame_indices": indices.tolist(),
        "root_policy": "canonical_identity_with_relative_smpl_y_translation",
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "note": (
            "This receipt is visual evidence only and does not approve an action clip. "
            "Source root_orient and horizontal trans are deliberately excluded; the restricted "
            "baker retains only relative SMPL-Y translation for vertical action transitions."
        ),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return image_path, receipt_path


def render_motion_review(
    source: Path,
    source_id: str,
    start_frame: int,
    end_frame: int,
    model_root: Path,
    output_dir: Path,
    samples: int = 9,
) -> tuple[Path, Path]:
    """Render one local contact sheet and its adjacent provenance receipt."""
    return _render_loaded_motion(
        load_amass_motion(source, source_id),
        start_frame,
        end_frame,
        model_root,
        output_dir,
        samples,
    )


def render_review_queue(
    source: Path, queue_path: Path, model_root: Path, output_dir: Path, samples: int = 9
) -> tuple[tuple[Path, Path], ...]:
    """Render a queued batch after preflighting all artifacts and source members."""
    items = load_review_queue(queue_path)
    outputs = [
        (
            output_dir
            / f"{review_output_stem(item.source_id, item.start_frame, item.end_frame)}.png",
            output_dir
            / f"{review_output_stem(item.source_id, item.start_frame, item.end_frame)}.review.json",
        )
        for item in items
    ]
    if any(path.exists() for output in outputs for path in output):
        raise AmassReviewError("Refusing to overwrite an existing queued review artifact")
    # This validates every requested member before the first output is written and streams a
    # compressed archive only once for the complete batch.
    payloads = _payloads_for_sources(source, {item.source_id for item in items})
    return tuple(
        _render_loaded_motion(
            _motion_from_payload(item.source_id, payloads[item.source_id]),
            item.start_frame,
            item.end_frame,
            model_root,
            output_dir,
            samples,
        )
        for item in items
    )


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


@click.group()
def queue_main() -> None:
    """Create and render pending local AMASS human-review queues."""


@queue_main.command("create")
@click.option("--candidate-index", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=20, show_default=True)
@click.option("--min-duration-seconds", type=click.FloatRange(min=0, min_open=True), default=2.0)
@click.option("--max-duration-seconds", type=click.FloatRange(min=0, min_open=True), default=10.0)
@click.option("--crop-seconds", type=click.FloatRange(min=0, min_open=True), default=4.0)
def create_queue_command(
    candidate_index: Path,
    output: Path,
    batch_size: int,
    min_duration_seconds: float,
    max_duration_seconds: float,
    crop_seconds: float,
) -> None:
    """Write one deterministic, source-balanced pending-review queue."""
    try:
        items = build_review_queue(
            candidate_index,
            batch_size,
            min_duration_seconds,
            max_duration_seconds,
            crop_seconds,
        )
        click.echo(write_review_queue(candidate_index, output, items))
    except AmassReviewError as error:
        raise click.ClickException(str(error)) from error


@queue_main.command("render")
@click.option("--source", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--queue", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--model-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--samples", type=click.IntRange(min=2), default=9, show_default=True)
def render_queue_command(
    source: Path, queue: Path, model_root: Path, output_dir: Path, samples: int
) -> None:
    """Render every pending item in a queue without overwriting prior evidence."""
    try:
        for image_path, receipt_path in render_review_queue(
            source, queue, model_root, output_dir, samples
        ):
            click.echo(f"Wrote {image_path} and {receipt_path}")
    except (AmassReviewError, SmplxAssetError) as error:
        raise click.ClickException(str(error)) from error
