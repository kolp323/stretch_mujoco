import tarfile
from pathlib import Path

import numpy as np
import pytest

from stretch_mujoco.humanoid.amass_motion_review import (
    AmassReviewError,
    load_amass_motion,
    review_frame_indices,
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
