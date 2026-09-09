import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import validate_no_hair_npc_lock as no_hair_lock


def test_no_hair_lock_pins_population_textures_and_acceptance_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    population = {
        "npcs": {
            "npc_one": {
                "embodiment": {
                    "appearance": "office_one_v1",
                    "visual_identity": "office_one_v1",
                    "appearance_config": {"top": "top_sage_v2"},
                }
            }
        }
    }
    _write_json(tmp_path / "population.json", population)
    (tmp_path / "body.png").write_bytes(b"body")
    (tmp_path / "review.mp4").write_bytes(b"video")
    _write_json(
        tmp_path / "review.acceptance.json",
        {
            "npc_id": "npc_one",
            "passed": True,
            "shots": [
                {"name": name, "clip": clip}
                for name, clip in no_hair_lock.REQUIRED_ACCEPTANCE_SHOTS
            ],
        },
    )
    (tmp_path / "scene.xml").write_text("<mujoco>baseball_cap_v1</mujoco>")
    (tmp_path / "source.json").write_text("source")
    lock = {
        "schema_version": 1,
        "population": "population.json",
        "source_hashes": {"source.json": _sha(tmp_path / "source.json")},
        "build": {
            "tool": "tool.py",
            "include_base_scene": True,
            "accessory_runtime_configs": ["runtime.json"],
            "excluded_runtime_accessory_ids": ["beautiful_hair_v1", "ponytail_hair_v1"],
        },
        "scene": {
            "path": "scene.xml",
            "sha256": _sha(tmp_path / "scene.xml"),
            "required_accessory_ids": ["baseball_cap_v1"],
            "excluded_accessory_ids": ["beautiful_hair_v1", "ponytail_hair_v1"],
        },
        "npcs": {
            "npc_one": {
                "appearance": "office_one_v1",
                "visual_identity": "office_one_v1",
                "appearance_config": {"top": "top_sage_v2"},
                "body_path": "body.png",
                "body_sha256": _sha(tmp_path / "body.png"),
                "video_path": "review.mp4",
                "video_sha256": _sha(tmp_path / "review.mp4"),
                "report_path": "review.acceptance.json",
                "report_sha256": _sha(tmp_path / "review.acceptance.json"),
            }
        },
    }
    (tmp_path / "tool.py").write_text("# builder\n")
    (tmp_path / "runtime.json").write_text("runtime\n")
    lock_path = tmp_path / "lock.json"
    _write_json(lock_path, lock)
    monkeypatch.setattr(
        no_hair_lock,
        "NpcPopulation",
        SimpleNamespace(from_json=lambda _path: SimpleNamespace(npcs={"npc_one": object()})),
    )

    assert no_hair_lock.validate_no_hair_npc_lock(lock_path, tmp_path)["passed"] is True

    (tmp_path / "body.png").write_bytes(b"different")
    with pytest.raises(ValueError, match="body texture sha256 does not match"):
        no_hair_lock.validate_no_hair_npc_lock(lock_path, tmp_path)


def test_no_hair_lock_rejects_report_without_all_seated_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {
        "npc_id": "npc_one",
        "passed": True,
        "shots": [
            {"name": name, "clip": clip}
            for name, clip in no_hair_lock.REQUIRED_ACCEPTANCE_SHOTS[:-1]
        ],
    }
    _write_json(tmp_path / "review.acceptance.json", report)
    _write_json(
        tmp_path / "population.json",
        {
            "npcs": {
                "npc_one": {
                    "embodiment": {
                        "appearance": "office_one_v1",
                        "visual_identity": "office_one_v1",
                        "appearance_config": {"top": "top_sage_v2"},
                    }
                }
            }
        },
    )
    (tmp_path / "body.png").write_bytes(b"body")
    (tmp_path / "review.mp4").write_bytes(b"video")
    (tmp_path / "scene.xml").write_text("<mujoco>baseball_cap_v1</mujoco>")
    (tmp_path / "source.json").write_text("source")
    (tmp_path / "tool.py").write_text("# builder\n")
    (tmp_path / "runtime.json").write_text("runtime\n")
    lock = {
        "schema_version": 1,
        "population": "population.json",
        "source_hashes": {"source.json": _sha(tmp_path / "source.json")},
        "build": {
            "tool": "tool.py",
            "include_base_scene": True,
            "accessory_runtime_configs": ["runtime.json"],
            "excluded_runtime_accessory_ids": ["beautiful_hair_v1", "ponytail_hair_v1"],
        },
        "scene": {
            "path": "scene.xml",
            "sha256": _sha(tmp_path / "scene.xml"),
            "required_accessory_ids": ["baseball_cap_v1"],
            "excluded_accessory_ids": ["beautiful_hair_v1", "ponytail_hair_v1"],
        },
        "npcs": {
            "npc_one": {
                "appearance": "office_one_v1",
                "visual_identity": "office_one_v1",
                "appearance_config": {"top": "top_sage_v2"},
                "body_path": "body.png",
                "body_sha256": _sha(tmp_path / "body.png"),
                "video_path": "review.mp4",
                "video_sha256": _sha(tmp_path / "review.mp4"),
                "report_path": "review.acceptance.json",
                "report_sha256": _sha(tmp_path / "review.acceptance.json"),
            }
        },
    }
    lock_path = tmp_path / "lock.json"
    _write_json(lock_path, lock)
    monkeypatch.setattr(
        no_hair_lock,
        "NpcPopulation",
        SimpleNamespace(from_json=lambda _path: SimpleNamespace(npcs={"npc_one": object()})),
    )

    with pytest.raises(ValueError, match="do not include the required walk and seated sit views"):
        no_hair_lock.validate_no_hair_npc_lock(lock_path, tmp_path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
