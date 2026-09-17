from __future__ import annotations

import json
from pathlib import Path

import pytest

from stretch_mujoco.npc.scene_compiler import compile_scene_npc_config
from stretch_mujoco.npc.scene_config import SceneConfigError, load_scene_npc_config
from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile
from stretch_mujoco.semantics import SemanticWorld
from stretch_mujoco.semantics.scene_discovery import SemanticCoverageError


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "stretch_mujoco" / "models" / "scene_npc_configs" / "fixtures"


def _config(tmp_path: Path) -> Path:
    payload = json.loads((FIXTURES / "minimal_scene.json").read_text())
    payload["scene"]["source_mjcf"] = str(FIXTURES / "minimal_scene.xml")
    payload["scene"]["source_manifest"] = str(FIXTURES / "minimal_scene.manifest.json")
    payload["scene"]["semantic_policy"] = str(
        ROOT / "stretch_mujoco" / "models" / "semantic_policies" / "fixture.json"
    )
    payload["scene"]["npc_catalog"] = str(
        ROOT / "stretch_mujoco" / "models" / "office_population.production.example.json"
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def test_compiler_emits_full_registry_v1_projection_coverage_and_receipt(tmp_path: Path) -> None:
    outputs = compile_scene_npc_config(_config(tmp_path), tmp_path / "out.xml")
    assert all(path.is_file() for path in outputs.values())
    v2 = json.loads(outputs["semantic_v2"].read_text())
    assert set(v2["entities"]) == {
        "zone.work",
        "object.chair_01",
        "object.cup_01",
        "object.table_01",
    }
    assert "object.cup_01" not in json.loads(outputs["semantic_v1"].read_text())["objects"]
    cup_action = v2["points"]["point.object.cup_01.action"]
    assert cup_action["site"] == "cup_01_grasp_site"
    assert cup_action["navigation_site"] == "point.object.cup_01.approach.01"
    chair_action = v2["points"]["point.object.chair_01.action.sit"]
    assert chair_action["position"] == [1.0, 0.0, 0.8]
    assert chair_action["position"] != v2["points"]["point.object.chair_01.approach.01"]["position"]
    assert SemanticWorld.from_json(outputs["semantic_v1"]).objects["object.chair_01"]
    population = NpcPopulation.from_json(outputs["population"])
    assert population.npcs["npc_alex_chen"].spawn.site.startswith("npc__minimal_scene__")
    trajectory = NpcTrajectoryProfile.from_json(outputs["trajectory_profile"])
    trajectory.validate_scene(outputs["scene"])
    coverage = json.loads(outputs["coverage"].read_text())
    assert coverage["counts"]["unresolved"] == coverage["counts"]["unbound"] == 0
    assert coverage["reachable_navigation_points"] == coverage["required_navigation_points"]
    first_receipt = outputs["receipt"].read_bytes()
    assert (
        compile_scene_npc_config(_config(tmp_path), tmp_path / "out.xml")["receipt"].read_bytes()
        == first_receipt
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["scene"].update(source_manifest="missing.json"),
            "source_scene_or_manifest_missing",
        ),
        (
            lambda payload: payload["semantic_registration"]["entity_overrides"].update(
                {"object.chair_01": {"navigation_requirement": "none"}}
            ),
            "entity_override_",
        ),
        (
            lambda payload: payload["population"]["members"].__setitem__(
                0, {"npc": "npc_fixture", "spawn": "point.missing", "initial_region": "zone.work"}
            ),
            "population_spawn_unbound",
        ),
    ],
)
def test_compiler_rejects_unbound_sources_exemptions_and_spawns(
    tmp_path: Path, mutate, match: str
) -> None:
    path = _config(tmp_path)
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises((SemanticCoverageError, ValueError), match=match):
        compile_scene_npc_config(path, tmp_path / "out.xml")


def test_compiler_rejects_unknown_categories_and_missing_action_approach(tmp_path: Path) -> None:
    path = _config(tmp_path)
    manifest = json.loads((FIXTURES / "minimal_scene.manifest.json").read_text())
    manifest["assets"][0]["category"] = "unknown"
    changed = tmp_path / "changed.manifest.json"
    changed.write_text(json.dumps(manifest))
    payload = json.loads(path.read_text())
    payload["scene"]["source_manifest"] = str(changed)
    path.write_text(json.dumps(payload))
    with pytest.raises(SemanticCoverageError, match="semantic_coverage_incomplete"):
        compile_scene_npc_config(path, tmp_path / "out.xml")
    manifest["assets"][0]["category"] = "chair"
    changed.write_text(json.dumps(manifest))
    payload["semantic_registration"]["entity_overrides"] = {
        "object.cup_01": {
            "semantic_class": "object.graspable",
            "affordances": ["approach", "grasp"],
            "navigation_requirement": "approach",
            "point_bundle": "graspable_on_support",
            "action_navigation_site": "point.missing",
        }
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(SemanticCoverageError, match="action_navigation_unbound"):
        compile_scene_npc_config(path, tmp_path / "out.xml")


def test_compiler_rejects_manifest_body_mismatch_and_unreachable_required_point(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    manifest = json.loads((FIXTURES / "minimal_scene.manifest.json").read_text())
    manifest["assets"][0]["body"] = "does_not_exist"
    changed = tmp_path / "changed.manifest.json"
    changed.write_text(json.dumps(manifest))
    payload = json.loads(path.read_text())
    payload["scene"]["source_manifest"] = str(changed)
    path.write_text(json.dumps(payload))
    with pytest.raises(SemanticCoverageError, match="manifest_xml_body_unbound"):
        compile_scene_npc_config(path, tmp_path / "out.xml")
    manifest["assets"][0]["body"] = "chair_01"
    changed.write_text(json.dumps(manifest))
    payload["semantic_registration"]["entity_overrides"] = {
        "object.chair_01": {
            "semantic_class": "furniture.seat",
            "affordances": ["approach"],
            "navigation_requirement": "approach",
            "point_bundle": "perimeter_approach",
            "explicit_point": [1.0, 0.0, 0.025],
        }
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(
        SemanticCoverageError,
        match="point_candidate_unavailable|required_navigation_unreachable",
    ):
        compile_scene_npc_config(path, tmp_path / "out.xml")


def test_compiler_accepts_variable_population_on_nonsequential_semantic_points(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    payload = json.loads(path.read_text())
    payload["semantic_registration"]["custom_targets"]["point.zone.work.spawn.jordan"] = {
        "owner": "zone.work",
        "position": [-1.5, 1.5, 0.025],
        "yaw": 0.0,
        "usages": ["navigation", "spawn"],
    }
    payload["population"]["members"] = [
        {
            "npc": "npc_alex_chen",
            "spawn": "point.zone.work.approach.01",
            "initial_region": "zone.work",
        },
        {
            "npc": "npc_jordan_patell",
            "spawn": "point.zone.work.spawn.jordan",
            "initial_region": "zone.work",
        },
    ]
    path.write_text(json.dumps(payload))
    coverage = json.loads(
        compile_scene_npc_config(path, tmp_path / "out.xml")["coverage"].read_text()
    )
    assert coverage["reachable_navigation_points"] == coverage["required_navigation_points"]


def test_compiler_resolves_business_routes_and_rejects_unknown_endpoint(tmp_path: Path) -> None:
    path = _config(tmp_path)
    payload = json.loads(path.read_text())
    payload["routes"] = [
        {
            "id": "work_to_table",
            "from": "zone.work",
            "to": "object.table_01",
            "mode": "fixed_contract",
            "actions": ["move_to", "inspect"],
        }
    ]
    path.write_text(json.dumps(payload))
    outputs = compile_scene_npc_config(path, tmp_path / "out.xml")
    trajectory = json.loads(outputs["trajectory_profile"].read_text())
    route = next(
        value for value in trajectory["routes"] if value["route_id"] == "business_work_to_table"
    )
    assert route["mode"] == "fixed_contract"
    assert route["from"] == "point.zone.work.approach.01"
    payload["routes"][0]["to"] = "object.missing"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="route_endpoint_unbound:object.missing"):
        compile_scene_npc_config(path, tmp_path / "out.xml")


def test_compiler_rejects_unknown_nested_policy_fields(tmp_path: Path) -> None:
    path = _config(tmp_path)
    policy = json.loads(
        (ROOT / "stretch_mujoco" / "models" / "semantic_policies" / "fixture.json").read_text()
    )
    policy["rules"][0]["silent_fallback"] = True
    changed_policy = tmp_path / "policy.json"
    changed_policy.write_text(json.dumps(policy))
    payload = json.loads(path.read_text())
    payload["scene"]["semantic_policy"] = str(changed_policy)
    path.write_text(json.dumps(payload))
    with pytest.raises(SemanticCoverageError, match="policy_rule_fields_invalid"):
        compile_scene_npc_config(path, tmp_path / "out.xml")


def test_all_source_scene_configs_declare_traffic_and_capability_outcomes() -> None:
    catalog = json.loads((ROOT / "stretch_mujoco/models/scene_npc_configs/catalog.json").read_text())
    configs = [ROOT / "stretch_mujoco/models/scene_npc_configs" / item for item in catalog["configs"]]
    assert len(configs) == 20
    for path in configs:
        config = load_scene_npc_config(path)
        assert config.traffic["no_direct_fallback"] is True
        assert set(config.interactions) == {"conversation", "robot_handover"}
        for name, declaration in config.interactions.items():
            assert declaration["status"] in {"supported", "unsupported"}
            if declaration["status"] == "unsupported":
                assert declaration.get("reason")
            else:
                assert declaration.get("sites") or declaration.get("objects")


def test_capability_contract_rejects_missing_unsupported_reason_and_fake_site(tmp_path: Path) -> None:
    path = _config(tmp_path)
    payload = json.loads(path.read_text())
    payload["interactions"]["conversation"] = {"status": "unsupported"}
    path.write_text(json.dumps(payload))
    with pytest.raises(SceneConfigError, match="interaction_unsupported_fields"):
        load_scene_npc_config(path)

    payload["interactions"]["conversation"] = {
        "status": "supported",
        "sites": ["conversation_missing_a", "conversation_missing_b"],
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="interaction_site_unbound:conversation"):
        compile_scene_npc_config(path, tmp_path / "out.xml")


def test_capability_contract_accepts_complete_supported_shape(tmp_path: Path) -> None:
    path = _config(tmp_path)
    payload = json.loads(path.read_text())
    payload["interactions"] = {
        "conversation": {"status": "supported", "sites": ["role_a", "role_b"]},
        "robot_handover": {
            "status": "supported",
            "sites": ["giver", "receiver"],
            "object": "cup_01",
            "robot_approach": "robot_approach",
        },
    }
    path.write_text(json.dumps(payload))
    config = load_scene_npc_config(path)
    assert config.interactions["conversation"]["status"] == "supported"
    assert config.interactions["robot_handover"]["object"] == "cup_01"


def test_compiler_projects_traffic_and_capability_contract_to_population(tmp_path: Path) -> None:
    outputs = compile_scene_npc_config(_config(tmp_path), tmp_path / "out.xml")
    population = json.loads(outputs["population"].read_text())
    assert population["traffic_policy"]["policy"] == "sequential_route_reservation"
    assert population["traffic_policy"]["no_direct_fallback"] is True
    assert population["interaction_capabilities"]["conversation"]["status"] == "unsupported"
