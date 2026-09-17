from __future__ import annotations

import json
from pathlib import Path

import pytest

from stretch_mujoco.npc.scene_config import SceneConfigError, load_scene_npc_config
from stretch_mujoco.npc.schema import NpcPopulation


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "stretch_mujoco/models/scene_npc_configs/fixtures"
MODELS = ROOT / "stretch_mujoco/models"


def _v2_config(tmp_path: Path, mutate=None) -> Path:
    payload = json.loads((FIXTURES / "minimal_scene.json").read_text(encoding="utf-8"))
    payload["schema"] = "scene_npc_config/v2"
    payload.pop("interactions")
    payload["scene"]["source_mjcf"] = str(FIXTURES / "minimal_scene.xml")
    payload["scene"]["source_manifest"] = str(FIXTURES / "minimal_scene.manifest.json")
    payload["scene"]["semantic_policy"] = str(MODELS / "semantic_policies/fixture.json")
    payload["scene"]["npc_catalog"] = str(MODELS / "office_population.production.example.json")
    payload["interaction_plan"] = {
        "path": str(MODELS / "scene_interaction_plans/office/office_02_cross_axis.json"),
        "required_capabilities": ["conversation", "handover", "sit", "work"],
    }
    payload["robot_navigation"] = {
        "footprint_radius": 0.32,
        "clearance": 0.08,
        "resolution": 0.08,
    }
    if mutate is not None:
        mutate(payload)
    path = tmp_path / "scene-v2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _v3_population() -> tuple[dict, set[str]]:
    source = MODELS / "office_population.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["schema_version"] = 3
    payload.pop("interaction_templates", None)
    sites = {
        "conversation_speaker",
        "conversation_listener",
        "handover_giver",
        "handover_receiver",
        "handover_robot",
        "handover_transfer",
        "seat_ingress",
        "seat_sit",
        "desk_work",
    }
    sites.update(
        definition["spawn"]["site"] for definition in payload["npcs"].values()
    )
    payload["interaction_stations"] = {
        "conversation": {
            "conversation.meeting.01": {
                "roles": {
                    "speaker": {"site": "conversation_speaker", "yaw": 1.57},
                    "listener": {"site": "conversation_listener", "yaw": -1.57},
                },
                "allowed_actor_pairs": ["npc_npc", "robot_npc"],
                "distance_m": {"min": 0.55, "max": 1.05},
                "yaw_tolerance_rad": 0.3,
            }
        },
        "handover": {
            "handover.work.01": {
                "roles": {
                    "giver": {"site": "handover_giver", "yaw": 1.57},
                    "receiver": {"site": "handover_receiver", "yaw": -1.57},
                    "robot": {"site": "handover_robot", "yaw": 1.57},
                },
                "modes": ["npc_to_npc", "robot_to_npc"],
                "object_ids": ["cup_01"],
                "distance_m": {"min": 0.75, "max": 1.3},
                "yaw_tolerance_rad": 0.3,
                "transfer_site": "handover_transfer",
            }
        },
        "seat": {
            "seat.chair.01": {
                "roles": {
                    "ingress": {"site": "seat_ingress", "yaw": 0.0},
                    "sit": {"site": "seat_sit", "yaw": 0.0},
                },
                "owner_entity": "object.chair_01",
                "seat_type": "chair",
                "slot_index": 1,
                "clearance_radius_m": 0.28,
            }
        },
        "workstation": {
            "workstation.office.01": {
                "roles": {"work": {"site": "desk_work", "yaw": 0.0}},
                "workstation_entity": "object.table_01",
                "computer_entity": "object.computer_01",
                "seat_slot": "seat.chair.01",
            }
        },
    }
    return payload, sites


def test_scene_config_v2_loads_plan_and_distinct_robot_navigation(tmp_path: Path) -> None:
    config = load_scene_npc_config(_v2_config(tmp_path))
    assert config.schema == "scene_npc_config/v2"
    assert config.interaction_plan is not None
    assert config.interaction_plan.required_capabilities == frozenset(
        {"conversation", "handover", "sit", "work"}
    )
    assert config.robot_navigation is not None
    assert config.robot_navigation.footprint_radius == 0.32
    assert all(value["status"] == "unsupported" for value in config.interactions.values())


def test_scene_config_v1_remains_legacy_without_multistation_evidence() -> None:
    config = load_scene_npc_config(FIXTURES / "minimal_scene.json")
    assert config.schema == "scene_npc_config/v1"
    assert config.interaction_plan is None
    assert config.robot_navigation is None
    assert config.interactions["conversation"]["status"] == "unsupported"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload.update(
                interactions={
                    "conversation": {"status": "supported", "sites": ["a", "b"]},
                    "robot_handover": {"status": "unsupported", "reason": "legacy"},
                }
            ),
            "scene_config_v2_fields_are_not_exact",
        ),
        (
            lambda payload: payload["interaction_plan"].update(status="supported"),
            "interaction_plan_fields_are_not_exact",
        ),
        (
            lambda payload: payload["interaction_plan"].update(sites=["a", "b"]),
            "interaction_plan_fields_are_not_exact",
        ),
        (
            lambda payload: payload["robot_navigation"].update(footprint_radius=0),
            "robot_navigation_footprint_radius_invalid",
        ),
        (
            lambda payload: payload["interaction_plan"].update(
                required_capabilities=["conversation", "unknown"]
            ),
            "interaction_plan_required_capabilities_invalid",
        ),
    ],
)
def test_scene_config_v2_rejects_legacy_or_malformed_contract(
    tmp_path: Path, mutate, match: str
) -> None:
    with pytest.raises(SceneConfigError, match=match):
        load_scene_npc_config(_v2_config(tmp_path, mutate))


def test_population_v3_loads_complete_station_catalog() -> None:
    payload, sites = _v3_population()
    population = NpcPopulation.from_dict(
        payload, source_path=MODELS / "office_population.json", sites=sites
    )
    assert population.schema_version == 3
    assert population.legacy_single_station is False
    assert population.interaction_templates == {}
    assert population.interaction_stations is not None
    assert set(population.interaction_stations) == {
        "conversation",
        "handover",
        "seat",
        "workstation",
    }
    handover = population.interaction_stations["handover"]["handover.work.01"]
    assert handover.roles["robot"].site == "handover_robot"
    assert handover.attributes["object_ids"] == ["cup_01"]
    assert handover.attributes["transfer_site"] == "handover_transfer"
    assert handover.attributes["distance_m"] == {"min": 0.75, "max": 1.3}


def test_population_v2_templates_remain_nonproduction_legacy_adapter() -> None:
    population = NpcPopulation.from_json(MODELS / "office_population.json")
    assert population.schema_version == 2
    assert population.legacy_single_station is True
    assert population.interaction_stations is not None
    legacy = population.interaction_stations["conversation"]["legacy.conversation.01"]
    assert legacy.attributes == {
        "legacy_single_station": True,
        "production_evidence": False,
    }
    payload = json.loads((MODELS / "office_population.json").read_text(encoding="utf-8"))
    payload["spawn_policy"] = "legacy_ignored_value"
    population = NpcPopulation.from_dict(
        payload, source_path=MODELS / "office_population.json"
    )
    assert population.spawn_policy is None


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["interaction_stations"]["conversation"]
            ["conversation.meeting.01"]["roles"]["speaker"].update(site="missing"),
            "unknown site 'missing'",
        ),
        (
            lambda payload: payload["interaction_stations"]["handover"]
            ["handover.work.01"]["roles"].pop("robot"),
            "roles are not exact",
        ),
        (
            lambda payload: payload["interaction_stations"]["handover"]
            ["handover.work.01"].update(modes=["invalid"]),
            "modes invalid",
        ),
        (
            lambda payload: payload["interaction_stations"]["conversation"]
            ["conversation.meeting.01"].pop("distance_m"),
            "fields are not exact",
        ),
        (
            lambda payload: payload["interaction_stations"]["conversation"]
            ["conversation.meeting.01"].update(
                distance_m={"min": "near", "max": 1.05}
            ),
            "distance_m invalid",
        ),
        (
            lambda payload: payload["interaction_stations"]["handover"]
            ["handover.work.01"].update(transfer_site="missing_transfer"),
            "unknown transfer_site 'missing_transfer'",
        ),
        (
            lambda payload: payload["interaction_stations"]["handover"]
            ["handover.work.01"].pop("transfer_site"),
            "fields are not exact",
        ),
        (
            lambda payload: payload["interaction_stations"]["handover"]
            ["handover.work.01"].update(yaw_tolerance_rad=0),
            "yaw_tolerance_rad invalid",
        ),
        (
            lambda payload: payload["interaction_stations"]["seat"]
            ["seat.chair.01"].update(slot_index=True),
            "slot_index invalid",
        ),
        (
            lambda payload: payload["interaction_stations"]["seat"]
            ["seat.chair.01"].update(clearance_radius_m=0),
            "clearance_radius_m invalid",
        ),
        (
            lambda payload: payload.update(interaction_templates={}),
            "rejects legacy interaction_templates",
        ),
        (
            lambda payload: payload.update(unknown_v3_field=True),
            "Population v3 has unknown fields: unknown_v3_field",
        ),
        (
            lambda payload: payload["interaction_stations"]["workstation"]
            ["workstation.office.01"].update(seat_slot="seat.missing"),
            "unbound or not a seat station",
        ),
        (
            lambda payload: payload["interaction_stations"]["workstation"]
            ["workstation.office.01"].update(seat_slot="conversation.meeting.01"),
            "unbound or not a seat station",
        ),
    ],
)
def test_population_v3_rejects_malformed_catalog(mutate, match: str) -> None:
    payload, sites = _v3_population()
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        NpcPopulation.from_dict(
            payload, source_path=MODELS / "office_population.json", sites=sites
        )


def test_population_v3_rejects_duplicate_station_id_across_kinds() -> None:
    payload, sites = _v3_population()
    duplicate = payload["interaction_stations"]["conversation"].pop(
        "conversation.meeting.01"
    )
    payload["interaction_stations"]["conversation"]["seat.chair.01"] = duplicate
    with pytest.raises(ValueError, match="Duplicate or empty interaction station ID"):
        NpcPopulation.from_dict(
            payload, source_path=MODELS / "office_population.json", sites=sites
        )
