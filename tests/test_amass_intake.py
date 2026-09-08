import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from stretch_mujoco.humanoid.amass_intake import (
    AmassIntakeError,
    canonicalize_vertical_translation,
    list_candidates,
    prepare_motions,
)


def _write_motion(path: Path, frames: int = 30, surface_model_type: str = "smplx") -> None:
    poses = np.arange(frames * 63, dtype=np.float32).reshape(frames, 63)
    np.savez(
        path,
        pose_body=poses,
        mocap_frame_rate=np.array(30.0),
        surface_model_type=np.array(surface_model_type),
        gender=np.array("neutral"),
    )


def _write_selection(path: Path, **overrides: object) -> None:
    selection = {
        "target_clip": "stand_up",
        "source_id": "Transitions/example.npz",
        "start_frame": 0,
        "end_frame": 30,
        "target_fps": 10,
    }
    selection.update(overrides)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "license": {"name": "AMASS research", "url": "https://amass.is.tue.mpg.de"},
                "selections": [selection],
            }
        )
    )


def test_list_candidates_reports_valid_and_rejected_local_npz(tmp_path: Path) -> None:
    _write_motion(tmp_path / "valid.npz")
    np.savez(tmp_path / "missing_pose.npz", markers_latent=np.zeros((2, 3)))

    candidates = {candidate.source_id: candidate for candidate in list_candidates(tmp_path)}

    assert candidates["valid.npz"].usable
    assert candidates["valid.npz"].frame_count == 30
    assert candidates["valid.npz"].duration_seconds == 1.0
    assert candidates["missing_pose.npz"].reason == "missing_pose_body"


def test_list_candidates_rejects_non_smplx_pose_body(tmp_path: Path) -> None:
    _write_motion(tmp_path / "smplh.npz", surface_model_type="smplh")

    (candidate,) = list_candidates(tmp_path / "smplh.npz")

    assert not candidate.usable
    assert candidate.reason == "unsupported_surface_model_type"


def test_list_candidates_reads_tar_bz2_without_extracting(tmp_path: Path) -> None:
    source_dir = tmp_path / "Transitions"
    source_dir.mkdir()
    _write_motion(source_dir / "example.npz")
    archive = tmp_path / "Transitions.tar.bz2"
    with tarfile.open(archive, "w:bz2") as output:
        output.add(source_dir / "example.npz", arcname="Transitions/example.npz")

    candidates = list_candidates(archive)

    assert len(candidates) == 1
    assert candidates[0].source_id == "Transitions/example.npz"
    assert candidates[0].surface_model_type == "smplx"


def test_prepare_motions_resamples_and_writes_provenance_receipt(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "Transitions"
    source_dir.mkdir(parents=True)
    _write_motion(source_dir / "example.npz")
    selection = tmp_path / "selection.json"
    _write_selection(selection)

    outputs = prepare_motions(
        tmp_path / "source", selection, tmp_path / "motions", tmp_path / "receipts"
    )

    assert outputs == (tmp_path / "motions" / "stand_up.npz",)
    with np.load(outputs[0], allow_pickle=False) as prepared:
        assert prepared["body_pose"].shape == (10, 63)
        assert prepared["body_pose"][:, 0].tolist() == list(range(0, 30 * 63, 3 * 63))
        assert np.array_equal(prepared["transl"], np.zeros((10, 3), dtype=np.float32))
    receipt = json.loads((tmp_path / "receipts" / "stand_up.receipt.json").read_text())
    assert receipt["source_frame_range"] == [0, 30]
    assert receipt["target_fps"] == 10.0
    assert receipt["target_frame_count"] == 10
    assert len(receipt["source_sha256"]) == 64
    assert len(receipt["output_sha256"]) == 64
    assert receipt["root_translation_policy"] == "relative_smpl_y_only_runtime_anchor"


def test_canonicalize_vertical_translation_pins_horizontal_motion() -> None:
    source = np.array([[5.0, 2.0, -3.0], [7.0, 2.5, 9.0]], dtype=np.float32)

    canonical = canonicalize_vertical_translation(source)

    assert np.array_equal(canonical, np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]]))


def test_prepare_motions_reads_only_selected_tar_member(tmp_path: Path) -> None:
    source_dir = tmp_path / "Transitions"
    source_dir.mkdir()
    _write_motion(source_dir / "example.npz")
    _write_motion(source_dir / "unselected.npz")
    archive = tmp_path / "Transitions.tar.bz2"
    with tarfile.open(archive, "w:bz2") as output:
        output.add(source_dir, arcname="Transitions")
    selection = tmp_path / "selection.json"
    _write_selection(selection)

    outputs = prepare_motions(archive, selection, tmp_path / "motions", tmp_path / "receipts")

    assert outputs == (tmp_path / "motions" / "stand_up.npz",)


def test_prepare_motions_refuses_to_overwrite_existing_outputs(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "Transitions"
    source_dir.mkdir(parents=True)
    _write_motion(source_dir / "example.npz")
    selection = tmp_path / "selection.json"
    _write_selection(selection)
    (tmp_path / "motions").mkdir()
    (tmp_path / "motions" / "stand_up.npz").write_bytes(b"user-owned")

    with pytest.raises(AmassIntakeError, match="Refusing to overwrite"):
        prepare_motions(tmp_path / "source", selection, tmp_path / "motions", tmp_path / "receipts")
