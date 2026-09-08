"""Select semantically labeled local AMASS candidates from a BABEL release archive."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import click

from stretch_mujoco.npc.animation.graph import OFFICE_CLIPS


class BabelIntakeError(ValueError):
    """Raised when BABEL annotations cannot be safely linked to local AMASS motions."""


# These are deliberately exact normalized BABEL labels. Broad categories such as
# ``gesture`` or ``interact with/use object`` require human interpretation and
# must not silently become office animation clips.
OFFICE_BABEL_LABELS: dict[str, frozenset[str]] = {
    "idle": frozenset({"stand", "stand in place", "stand still", "stand with arms down"}),
    "walk": frozenset(
        {
            "walk",
            "walk forward",
            "walk backward",
            "walk left",
            "walk right",
            "walk to the left",
            "walk to the right",
            "walk back",
        }
    ),
    "sit_down": frozenset({"sit down", "sit down in chair", "sit on a chair"}),
    "stand_up": frozenset({"stand up", "standup"}),
    "gesture_wave": frozenset(
        {"wave right hand", "wave with right hand", "wave left hand", "wave with left hand"}
    ),
    "gesture_point": frozenset(
        {
            "point to the right",
            "point left",
            "point straight ahead",
            "point straight ahead with right hand",
            "point in front with right hand",
            "point right hand forward",
            "point left hand forward",
        }
    ),
    "pick_up": frozenset(
        {
            "pick up phone",
            "pickup phone",
            "pick up phone with right hand",
            "pickup object with left hand",
            "pick an object on the right",
        }
    ),
    "place": frozenset({"place", "place down"}),
    "talk": frozenset({"talk", "talk on phone", "talk on cell phone"}),
    "receive": frozenset({"receive"}),
    "seated_idle": frozenset({"sit"}),
    # BABEL's ``type motion`` is an exact annotation, not a generic interaction
    # category.  It remains a candidate for both office roles until visual review.
    "use_computer": frozenset({"type motion"}),
    "work": frozenset({"type motion"}),
    "eat": frozenset({"eat with right hand"}),
    "give": frozenset({"give deck cards with right hand"}),
}


@dataclass(frozen=True)
class BabelCandidate:
    """A technically present local motion with an exact BABEL frame label."""

    target_clip: str
    source_id: str
    source_frame_range: tuple[int, int]
    source_fps: float
    babel_split: str
    babel_sequence_id: str
    babel_raw_label: str
    babel_proc_label: str
    babel_categories: tuple[str, ...]


def _read_candidate_index(path: Path) -> dict[str, tuple[int, float]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise BabelIntakeError(f"Could not read AMASS candidate index '{path}': {error}") from error
    available: dict[str, tuple[int, float]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("usable") is not True:
            continue
        source_id = row.get("source_id")
        frame_count = row.get("frame_count")
        fps = row.get("mocap_frame_rate")
        if (
            isinstance(source_id, str)
            and isinstance(frame_count, int)
            and not isinstance(fps, bool)
            and isinstance(fps, (int, float))
            and frame_count > 1
            and fps > 0
        ):
            available[source_id] = (frame_count, float(fps))
    if not available:
        raise BabelIntakeError("AMASS candidate index has no usable motions")
    return available


def babel_feature_to_amass_stageii(feature_path: str) -> str | None:
    """Map a supported BABEL ``*_poses`` path to its local AMASS stage-II member."""
    path = PurePosixPath(feature_path)
    parts = path.parts
    source_roots = {
        ("BMLmovi", "BMLmovi"): "BMLmovi",
        ("CMU", "CMU"): "CMU",
        ("Transitionsmocap", "Transitions_mocap"): "Transitions",
    }
    if len(parts) < 4 or any(part in {".", ".."} for part in parts) or path.suffix != ".npz":
        return None
    dataset_key = (parts[0], parts[1])
    if dataset_key not in source_roots:
        return None
    if not path.stem.endswith("_poses"):
        return None
    stem = path.stem.removesuffix("_poses") + "_stageii"
    return PurePosixPath(source_roots[dataset_key], *parts[2:-1], f"{stem}.npz").as_posix()


# Kept as a compatibility alias for callers created before CMU support.
babel_feature_to_bmlmovi_stageii = babel_feature_to_amass_stageii


def _normalize_label(label: str) -> str:
    """Normalize BABEL's incidental spacing without broadening label matching."""
    return " ".join(label.split())


def _frame_range(
    start_seconds: Any, end_seconds: Any, fps: float, frame_count: int
) -> tuple[int, int] | None:
    if (
        isinstance(start_seconds, bool)
        or isinstance(end_seconds, bool)
        or not isinstance(start_seconds, (int, float))
        or not isinstance(end_seconds, (int, float))
        or not math.isfinite(start_seconds)
        or not math.isfinite(end_seconds)
        or end_seconds <= start_seconds
    ):
        return None
    start_frame = max(0, math.floor(start_seconds * fps))
    end_frame = min(frame_count, math.ceil(end_seconds * fps))
    return (start_frame, end_frame) if end_frame > start_frame else None


def find_babel_candidates(
    babel_archive: Path, candidate_index: Path, per_clip: int, min_duration_seconds: float
) -> tuple[BabelCandidate, ...]:
    """Return exact, local supported-AMASS candidates from BABEL frame annotations."""
    if per_clip < 1 or min_duration_seconds <= 0:
        raise BabelIntakeError("per_clip and min_duration_seconds must be positive")
    local_motions = _read_candidate_index(candidate_index)
    candidates: dict[str, list[BabelCandidate]] = {clip: [] for clip in OFFICE_BABEL_LABELS}
    try:
        with zipfile.ZipFile(babel_archive) as archive:
            for split in ("train", "val", "test"):
                member = f"babel_v1.0_release/{split}.json"
                records = json.loads(archive.read(member))
                if not isinstance(records, dict):
                    raise BabelIntakeError(f"BABEL member '{member}' is not an object")
                for sequence_id, sequence in records.items():
                    if not isinstance(sequence_id, str) or not isinstance(sequence, dict):
                        continue
                    source_id = babel_feature_to_amass_stageii(str(sequence.get("feat_p", "")))
                    if source_id not in local_motions:
                        continue
                    frame_annotation = sequence.get("frame_ann")
                    if not isinstance(frame_annotation, dict):
                        continue
                    labels = frame_annotation.get("labels")
                    if not isinstance(labels, list):
                        continue
                    frame_count, fps = local_motions[source_id]
                    for label in labels:
                        if not isinstance(label, dict):
                            continue
                        raw_proc_label = label.get("proc_label")
                        if not isinstance(raw_proc_label, str):
                            continue
                        proc_label = _normalize_label(raw_proc_label)
                        frame_range = _frame_range(
                            label.get("start_t"), label.get("end_t"), fps, frame_count
                        )
                        if (
                            frame_range is None
                            or (frame_range[1] - frame_range[0]) / fps < min_duration_seconds
                        ):
                            continue
                        categories = label.get("act_cat")
                        category_names = (
                            tuple(category for category in categories if isinstance(category, str))
                            if isinstance(categories, list)
                            else ()
                        )
                        for target_clip, exact_labels in OFFICE_BABEL_LABELS.items():
                            if proc_label in exact_labels:
                                candidates[target_clip].append(
                                    BabelCandidate(
                                        target_clip=target_clip,
                                        source_id=source_id,
                                        source_frame_range=frame_range,
                                        source_fps=fps,
                                        babel_split=split,
                                        babel_sequence_id=sequence_id,
                                        babel_raw_label=str(label.get("raw_label", "")),
                                        babel_proc_label=proc_label,
                                        babel_categories=category_names,
                                    )
                                )
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        if isinstance(error, BabelIntakeError):
            raise
        raise BabelIntakeError(
            f"Could not read BABEL archive '{babel_archive}': {error}"
        ) from error
    selected: list[BabelCandidate] = []
    for target_clip, entries in candidates.items():
        entries.sort(
            key=lambda candidate: (
                -(candidate.source_frame_range[1] - candidate.source_frame_range[0]),
                candidate.source_id,
                candidate.source_frame_range,
            )
        )
        selected.extend(entries[:per_clip])
    return tuple(selected)


def write_babel_candidates(
    output_path: Path,
    babel_archive: Path,
    candidate_index: Path,
    candidates: tuple[BabelCandidate, ...],
) -> Path:
    """Write a non-selection candidate record and refuse to replace review evidence."""
    if output_path.exists():
        raise BabelIntakeError(
            f"Refusing to overwrite existing BABEL candidate file '{output_path}'"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    covered = {candidate.target_clip for candidate in candidates}
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_status": "pending_visual_review",
                "babel_archive_sha256": hashlib.sha256(babel_archive.read_bytes()).hexdigest(),
                "candidate_index_sha256": hashlib.sha256(candidate_index.read_bytes()).hexdigest(),
                "selection_policy": "exact_babel_frame_label_only",
                "candidates": [asdict(candidate) for candidate in candidates],
                "missing_office_clips": sorted(set(OFFICE_CLIPS) - covered),
                "note": "Candidates are not AMASS intake selections or baker inputs.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


@click.command()
@click.option("--babel-archive", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--candidate-index", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--per-clip", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--min-duration-seconds", type=click.FloatRange(min=0, min_open=True), default=0.5)
def main(
    babel_archive: Path,
    candidate_index: Path,
    output: Path,
    per_clip: int,
    min_duration_seconds: float,
) -> None:
    """Create provenance-backed BABEL candidates for local human review."""
    try:
        candidates = find_babel_candidates(
            babel_archive, candidate_index, per_clip, min_duration_seconds
        )
        click.echo(write_babel_candidates(output, babel_archive, candidate_index, candidates))
    except BabelIntakeError as error:
        raise click.ClickException(str(error)) from error
