"""Select local AMASS SMPL-X motions for the restricted NPC animation baker."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, cast

import click
import numpy as np


class AmassIntakeError(ValueError):
    """Raised when local AMASS data cannot satisfy an explicit selection."""


@dataclass(frozen=True)
class AmassCandidate:
    source_id: str
    frame_count: int | None
    mocap_frame_rate: float | None
    duration_seconds: float | None
    surface_model_type: str | None
    gender: str | None
    usable: bool
    reason: str | None = None


@dataclass(frozen=True)
class _SourcePayload:
    source_id: str
    payload: bytes


def _source_payloads(source: Path) -> Iterator[_SourcePayload]:
    if source.is_file() and source.suffix == ".npz":
        yield _SourcePayload(source.name, source.read_bytes())
        return
    if source.is_dir():
        for path in sorted(source.rglob("*.npz")):
            yield _SourcePayload(path.relative_to(source).as_posix(), path.read_bytes())
        return
    if source.is_file() and source.name.endswith(".tar.bz2"):
        with tarfile.open(source, "r:bz2") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".npz"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                yield _SourcePayload(member.name, extracted.read())
        return
    raise AmassIntakeError("AMASS source must be an NPZ file, directory, or .tar.bz2 archive")


def _scalar_string(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if value.shape != ():
        return None
    return str(value.item())


def _scalar_float(data: np.lib.npyio.NpzFile, key: str) -> float | None:
    if key not in data:
        return None
    value = data[key]
    if value.shape != () or not np.issubdtype(value.dtype, np.number):
        return None
    return float(value.item())


def _candidate(payload: _SourcePayload) -> AmassCandidate:
    try:
        with np.load(io.BytesIO(payload.payload), allow_pickle=False) as data:
            if "pose_body" not in data:
                return AmassCandidate(
                    payload.source_id, None, None, None, None, None, False, "missing_pose_body"
                )
            pose_body = data["pose_body"]
            frame_rate = _scalar_float(data, "mocap_frame_rate")
            surface_model_type = _scalar_string(data, "surface_model_type")
            gender = _scalar_string(data, "gender")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        return AmassCandidate(
            payload.source_id, None, None, None, None, None, False, f"unreadable:{error}"
        )
    if pose_body.ndim != 2 or pose_body.shape[1] != 63 or pose_body.shape[0] == 0:
        return AmassCandidate(
            payload.source_id,
            int(pose_body.shape[0]) if pose_body.ndim else None,
            frame_rate,
            None,
            surface_model_type,
            gender,
            False,
            "invalid_pose_body_shape",
        )
    if not np.issubdtype(pose_body.dtype, np.number) or not np.isfinite(pose_body).all():
        return AmassCandidate(
            payload.source_id,
            int(pose_body.shape[0]),
            frame_rate,
            None,
            surface_model_type,
            gender,
            False,
            "non_finite_pose_body",
        )
    if frame_rate is None or frame_rate <= 0:
        return AmassCandidate(
            payload.source_id,
            int(pose_body.shape[0]),
            frame_rate,
            None,
            surface_model_type,
            gender,
            False,
            "invalid_mocap_frame_rate",
        )
    if surface_model_type is None or surface_model_type.lower() != "smplx":
        return AmassCandidate(
            payload.source_id,
            int(pose_body.shape[0]),
            frame_rate,
            float(pose_body.shape[0]) / frame_rate,
            surface_model_type,
            gender,
            False,
            "unsupported_surface_model_type",
        )
    return AmassCandidate(
        payload.source_id,
        int(pose_body.shape[0]),
        frame_rate,
        float(pose_body.shape[0]) / frame_rate,
        surface_model_type,
        gender,
        True,
    )


def list_candidates(source: Path) -> tuple[AmassCandidate, ...]:
    """Inspect local candidate files without extracting or writing source data."""
    return tuple(_candidate(payload) for payload in _source_payloads(source))


def _load_selection(path: Path) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AmassIntakeError(f"Could not read selection file '{path}': {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise AmassIntakeError("Selection file must be a schema_version 1 object")
    license_info = document.get("license")
    if not isinstance(license_info, dict) or not all(
        isinstance(license_info.get(field), str) and license_info[field]
        for field in ("name", "url")
    ):
        raise AmassIntakeError("Selection file requires license.name and license.url")
    selections = document.get("selections")
    if (
        not isinstance(selections, list)
        or not selections
        or not all(isinstance(item, dict) for item in selections)
    ):
        raise AmassIntakeError("Selection file requires a non-empty selections list")
    return license_info, tuple(selections)


def _sample_indices(
    start_frame: int, end_frame: int, source_fps: float, target_fps: float
) -> np.ndarray:
    if start_frame < 0 or end_frame <= start_frame:
        raise AmassIntakeError("Selection start_frame/end_frame must define a non-empty range")
    if target_fps <= 0 or target_fps > source_fps:
        raise AmassIntakeError(
            "Selection target_fps must be positive and no greater than source FPS"
        )
    count = int(np.floor((end_frame - start_frame) * target_fps / source_fps))
    if count < 2:
        raise AmassIntakeError("Selection becomes fewer than two frames after resampling")
    return start_frame + np.floor(np.arange(count) * source_fps / target_fps).astype(np.int64)


def prepare_motions(
    source: Path, selection_path: Path, output_dir: Path, receipt_dir: Path
) -> tuple[Path, ...]:
    """Create ignored baker inputs and receipts from explicitly selected local motions."""
    license_info, selections = _load_selection(selection_path)
    selected_source_ids = {
        source_id
        for selection in selections
        if isinstance((source_id := selection.get("source_id")), str)
    }
    payloads = {
        payload.source_id: payload
        for payload in _source_payloads(source)
        if payload.source_id in selected_source_ids
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    seen_targets: set[str] = set()
    for selection in selections:
        required = ("target_clip", "source_id", "start_frame", "end_frame", "target_fps")
        if any(field not in selection for field in required):
            raise AmassIntakeError(f"Selection is missing fields: {', '.join(required)}")
        target_clip = selection["target_clip"]
        source_id = selection["source_id"]
        if not isinstance(target_clip, str) or not target_clip or "/" in target_clip:
            raise AmassIntakeError("Selection target_clip must be a simple file name")
        if target_clip in seen_targets:
            raise AmassIntakeError(f"Selection duplicates target_clip '{target_clip}'")
        seen_targets.add(target_clip)
        if not isinstance(source_id, str) or source_id not in payloads:
            raise AmassIntakeError(f"Selection source_id is unavailable: {source_id!r}")
        start_frame = selection["start_frame"]
        end_frame = selection["end_frame"]
        target_fps = selection["target_fps"]
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (start_frame, end_frame, target_fps)
        ):
            raise AmassIntakeError("Selection frame bounds and target_fps must be numeric")
        start_frame = int(cast(float, start_frame))
        end_frame = int(cast(float, end_frame))
        target_fps = float(cast(float, target_fps))
        reverse = bool(selection.get("reverse", False))
        payload = payloads[source_id]
        with np.load(io.BytesIO(payload.payload), allow_pickle=False) as data:
            if "pose_body" not in data:
                raise AmassIntakeError(f"Selection source '{source_id}' has no pose_body")
            source_fps = _scalar_float(data, "mocap_frame_rate")
            surface_model_type = _scalar_string(data, "surface_model_type")
            gender = _scalar_string(data, "gender")
            poses = np.asarray(data["pose_body"], dtype=np.float32)
        if source_fps is None or source_fps <= 0:
            raise AmassIntakeError(f"Selection source '{source_id}' has invalid mocap_frame_rate")
        if surface_model_type is None or surface_model_type.lower() != "smplx":
            raise AmassIntakeError(
                f"Selection source '{source_id}' must declare surface_model_type 'smplx'"
            )
        if poses.ndim != 2 or poses.shape[1] != 63 or not np.isfinite(poses).all():
            raise AmassIntakeError(f"Selection source '{source_id}' has invalid pose_body")
        if end_frame > poses.shape[0]:
            raise AmassIntakeError(f"Selection source '{source_id}' end_frame exceeds frame count")
        indexes = _sample_indices(start_frame, end_frame, source_fps, target_fps)
        selected = poses[indexes]
        if reverse:
            selected = selected[::-1].copy()
        output_path = output_dir / f"{target_clip}.npz"
        receipt_path = receipt_dir / f"{target_clip}.receipt.json"
        if output_path.exists() or receipt_path.exists():
            raise AmassIntakeError(f"Refusing to overwrite existing output for '{target_clip}'")
        np.savez(output_path, body_pose=selected)
        receipt = {
            "schema_version": 1,
            "license": license_info,
            "target_clip": target_clip,
            "source_id": source_id,
            "source_sha256": hashlib.sha256(payload.payload).hexdigest(),
            "surface_model_type": surface_model_type,
            "gender": gender,
            "source_fps": source_fps,
            "source_frame_range": [start_frame, end_frame],
            "target_fps": target_fps,
            "target_frame_count": int(selected.shape[0]),
            "reverse": reverse,
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        outputs.append(output_path)
    return tuple(outputs)


@click.group()
def main() -> None:
    """Inspect and prepare local, licensed AMASS motion inputs."""


@main.command("list")
@click.option("--source", type=click.Path(path_type=Path, exists=True), required=True)
def list_command(source: Path) -> None:
    """List compatible and rejected AMASS candidates without extracting them."""
    for candidate in list_candidates(source):
        click.echo(json.dumps(asdict(candidate), sort_keys=True))


@main.command("prepare")
@click.option("--source", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--selection", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--receipt-dir", type=click.Path(path_type=Path), required=True)
def prepare_command(source: Path, selection: Path, output_dir: Path, receipt_dir: Path) -> None:
    """Write selected, resampled baker inputs and local provenance receipts."""
    try:
        outputs = prepare_motions(source, selection, output_dir, receipt_dir)
    except AmassIntakeError as error:
        raise click.ClickException(str(error)) from error
    for output in outputs:
        click.echo(str(output))


if __name__ == "__main__":
    main()
