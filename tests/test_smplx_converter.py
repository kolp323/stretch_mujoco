from pathlib import Path

import pytest

from stretch_mujoco.humanoid.smplx_converter import SmplxAssetError, find_model_file


def test_find_model_file_accepts_nested_official_layout(tmp_path: Path) -> None:
    model_file = tmp_path / "smplx" / "SMPLX_NEUTRAL.npz"
    model_file.parent.mkdir()
    model_file.touch()

    assert find_model_file(tmp_path, "smplx", "neutral") == model_file


def test_find_model_file_accepts_a_shared_symlinked_layout(tmp_path: Path) -> None:
    shared = tmp_path / "shared" / "smplx"
    shared.mkdir(parents=True)
    model_file = shared / "SMPLX_NEUTRAL.npz"
    model_file.touch()
    root = tmp_path / "worktree" / "private"
    root.mkdir(parents=True)
    (root / "smplx").symlink_to(shared, target_is_directory=True)

    assert find_model_file(root, "smplx", "neutral") == root / "smplx" / "SMPLX_NEUTRAL.npz"


def test_find_model_file_reports_missing_licensed_asset(tmp_path: Path) -> None:
    with pytest.raises(SmplxAssetError, match="No licensed SMPLX parameters"):
        find_model_file(tmp_path, "smplx", "neutral")
