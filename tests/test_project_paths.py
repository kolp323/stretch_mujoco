from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from stretch_mujoco.paths import cache_root, output_root, require_external_directory
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


def test_runtime_roots_honor_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "runs"
    cache = tmp_path / "cache"
    monkeypatch.setenv("STRETCH_MUJOCO_OUTPUT_DIR", str(output))
    monkeypatch.setenv("STRETCH_MUJOCO_CACHE_DIR", str(cache))

    assert output_root() == output.resolve()
    assert cache_root() == cache.resolve()


def test_external_directory_requires_explicit_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STRETCH_MUJOCO_TEST_DATA", raising=False)

    with pytest.raises(ValueError, match="STRETCH_MUJOCO_TEST_DATA"):
        require_external_directory(
            None,
            environment_variable="STRETCH_MUJOCO_TEST_DATA",
            description="test data",
        )


def test_simulator_camera_default_is_not_mutable() -> None:
    parameter = inspect.signature(StretchMujocoSimulator).parameters["cameras_to_use"]
    assert parameter.default is None
