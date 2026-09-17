from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stretch_mujoco.npc.interaction_station import load_interaction_site_plan
from stretch_mujoco.npc.scene_compiler import compile_scene_npc_config
from stretch_mujoco.npc.scene_config import load_scene_npc_config
from tools.propose_interaction_sites import NATIVE_HOME_COMPUTERS, SEAT_CATEGORIES, propose_plan
from tools.validate_interaction_site_plan import validate_config


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "stretch_mujoco/models"
CONFIGS = tuple(sorted((MODELS / "scene_npc_configs/office").glob("*.json"))) + tuple(
    sorted((MODELS / "scene_npc_configs/home").glob("*.json"))
)
PLANS = tuple(sorted((MODELS / "scene_interaction_plans/office").glob("*.json"))) + tuple(
    sorted((MODELS / "scene_interaction_plans/home").glob("*.json"))
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exactly_twenty_v2_configs_and_candidate_plans_have_minimum_resources() -> None:
    assert len(CONFIGS) == len(PLANS) == 20
    assert {path.stem for path in CONFIGS} == {path.stem for path in PLANS}
    for config_path in CONFIGS:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert raw_config["schema"] == "scene_npc_config/v2"
        assert "interactions" not in raw_config
        assert "unsupported" not in config_path.read_text(encoding="utf-8")
        config = load_scene_npc_config(config_path)
        assert config.interaction_plan is not None
        assert config.interaction_plan.required_capabilities == frozenset(
            {"conversation", "handover", "sit", "work"}
        )
        plan = load_interaction_site_plan(
            config.interaction_plan.path,
            source_manifest=config.source_manifest,
            source_mjcf=config.source_mjcf,
            expected_scene_id=config.scene_id,
        )
        assert len(plan.conversation_stations) >= 3
        assert len(plan.handover_stations) >= 3
        assert len(plan.workstations) >= 1
        assert plan.seat_slots
        assert all(station.modes == ("npc_to_npc",) for station in plan.handover_stations)


def test_seat_coverage_primitive_provenance_and_static_validation_are_closed() -> None:
    for config_path in CONFIGS:
        config = load_scene_npc_config(config_path)
        plan = load_interaction_site_plan(
            config.interaction_plan.path,
            source_manifest=config.source_manifest,
            source_mjcf=config.source_mjcf,
            expected_scene_id=config.scene_id,
        )
        manifest = json.loads(config.source_manifest.read_text(encoding="utf-8"))
        seat_count = sum(asset.get("category") in SEAT_CATEGORIES for asset in manifest["assets"])
        assert len(plan.seat_slots) + len(plan.seat_exemptions) == seat_count
        record = validate_config(config_path)
        assert record["candidate_validated"] is True
        assert record["production_evidence"] is False
        assert record["validators"]["npc_navigation"] == "passed"
        assert record["validators"]["robot_navigation"] == "deferred_out_of_scope"
        if config.kind == "home":
            kits = {
                asset.get("provenance")
                for asset in manifest["assets"]
                if asset.get("provenance") == "primitive_workstation_kit/v1"
            }
            assert bool(kits) is (config.scene_id not in NATIVE_HOME_COMPUTERS)


def test_plan_authoring_is_deterministic_without_mutating_stable_sources() -> None:
    before = {_path.stem: _sha(_path) for _path in PLANS}
    for config_path in CONFIGS:
        config = load_scene_npc_config(config_path)
        proposed = propose_plan(config.source_mjcf, config.source_manifest, config.kind)
        encoded = (json.dumps(proposed, indent=2, sort_keys=True) + "\n").encode()
        assert hashlib.sha256(encoded).hexdigest() == before[config.scene_id]


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda path: path.stem)
def test_each_candidate_compiles_config_only_deterministically(tmp_path: Path, config_path: Path) -> None:
    first = compile_scene_npc_config(
        config_path, tmp_path / config_path.stem / "first.xml", check_navigation=False
    )
    second = compile_scene_npc_config(
        config_path, tmp_path / config_path.stem / "second.xml", check_navigation=False
    )
    assert {name: _sha(path) for name, path in first.items()} == {
        name: _sha(path) for name, path in second.items()
    }
