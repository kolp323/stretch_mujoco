import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from stretch_mujoco.humanoid.amass_motion_review import (
    AmassReviewError,
    _canonical_review_root_parameters,
    build_review_queue,
    load_amass_motion,
    load_review_queue,
    review_frame_indices,
    review_output_stem,
    write_review_queue,
)


def _write_motion(path: Path) -> None:
    np.savez(
        path,
        pose_body=np.arange(30 * 63, dtype=np.float32).reshape(30, 63),
        root_orient=np.zeros((30, 3), dtype=np.float32),
        trans=np.ones((30, 3), dtype=np.float32),
        mocap_frame_rate=np.array(30.0),
    )


def test_load_amass_motion_reads_a_named_tar_member(tmp_path: Path) -> None:
    source_dir = tmp_path / "BMLmovi"
    source_dir.mkdir()
    _write_motion(source_dir / "candidate.npz")
    archive = tmp_path / "BMLmovi.tar.bz2"
    with tarfile.open(archive, "w:bz2") as output:
        output.add(source_dir / "candidate.npz", arcname="BMLmovi/candidate.npz")

    motion = load_amass_motion(archive, "BMLmovi/candidate.npz")

    assert motion.body_pose.shape == (30, 63)
    assert motion.global_orient.shape == (30, 3)
    assert motion.transl[0].tolist() == [1.0, 1.0, 1.0]
    assert motion.mocap_frame_rate == 30.0
    assert len(motion.source_sha256) == 64


def test_review_frame_indices_include_both_crop_endpoints() -> None:
    assert review_frame_indices(3, 12, 30, 4).tolist() == [3, 5, 8, 11]


def test_review_frame_indices_reject_invalid_crop() -> None:
    with pytest.raises(AmassReviewError, match="within the source motion"):
        review_frame_indices(5, 31, 30, 3)


def test_canonical_review_root_matches_restricted_baker_orientation_contract() -> None:
    orient, transl = _canonical_review_root_parameters(3)

    assert orient.dtype == np.float32
    assert transl.dtype == np.float32
    assert orient.shape == (3, 3)
    assert np.array_equal(orient, np.zeros((3, 3), dtype=np.float32))
    assert np.array_equal(transl, orient)
    assert orient is not transl


def test_review_output_stem_is_unique_for_same_named_members() -> None:
    first = review_output_stem("BMLmovi/Subject_1/action_stageii.npz", 0, 120)
    second = review_output_stem("BMLmovi/Subject_2/action_stageii.npz", 0, 120)

    assert first != second
    assert first.startswith("action_stageii--")
    assert "/" not in first


def test_review_queue_balances_source_groups_and_writes_pending_record(tmp_path: Path) -> None:
    index = tmp_path / "candidates.jsonl"
    records = [
        {
            "source_id": "BMLmovi/Subject_2/a.npz",
            "frame_count": 720,
            "mocap_frame_rate": 120.0,
            "duration_seconds": 6.0,
            "usable": True,
        },
        {
            "source_id": "BMLmovi/Subject_1/b.npz",
            "frame_count": 720,
            "mocap_frame_rate": 120.0,
            "duration_seconds": 6.0,
            "usable": True,
        },
        {
            "source_id": "BMLmovi/Subject_1/c.npz",
            "frame_count": 720,
            "mocap_frame_rate": 120.0,
            "duration_seconds": 6.0,
            "usable": True,
        },
        {"source_id": "BMLmovi/Subject_3/d.npz", "usable": False},
    ]
    index.write_text("".join(json.dumps(record) + "\n" for record in records))

    items = build_review_queue(index, 3, 2.0, 10.0, 4.0)
    queue_path = write_review_queue(index, tmp_path / "queue.json", items)

    assert [item.source_id for item in items] == [
        "BMLmovi/Subject_1/b.npz",
        "BMLmovi/Subject_2/a.npz",
        "BMLmovi/Subject_1/c.npz",
    ]
    assert all(item.end_frame == 480 for item in items)
    assert load_review_queue(queue_path) == items
    assert json.loads(queue_path.read_text())["review_status"] == "pending_human_review"
