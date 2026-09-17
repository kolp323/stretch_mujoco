from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from stretch_mujoco.npc.interaction_projection import (
    PHYSICAL_VALIDATORS,
    project_interaction_plan,
)
from stretch_mujoco.npc.interaction_station import load_interaction_site_plan
from stretch_mujoco.npc.schema import NpcPopulation


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / "stretch_mujoco/models/scene_interaction_plans/office/office_02_cross_axis.json"
)
SCENE_CONFIG_PATH = (
    ROOT / "stretch_mujoco/models/scene_npc_configs/office/office_02_cross_axis.json"
)
POPULATION_PATH = ROOT / "stretch_mujoco/models/office_population.production.example.json"


def _source_paths() -> tuple[Path, Path]:
    config = json.loads(SCENE_CONFIG_PATH.read_text(encoding="utf-8"))
    scene = config["scene"]
    return (
        (SCENE_CONFIG_PATH.parent / scene["source_manifest"]).resolve(),
        (SCENE_CONFIG_PATH.parent / scene["source_mjcf"]).resolve(),
    )


def _load_plan(path: Path = PLAN_PATH):
    manifest, mjcf = _source_paths()
    return load_interaction_site_plan(
        path,
        source_manifest=manifest,
        source_mjcf=mjcf,
        expected_scene_id="office_02_cross_axis",
    )


def test_office_02_projection_preserves_roles_attributes_entities_and_relations() -> None:
    projection = project_interaction_plan(_load_plan())

    assert len(projection.derived_sites) == 11
    assert len(projection.semantic_points) == 11
    assert [site["role"] for site in projection.derived_sites] == [
        "conversation_speaker_site",
        "conversation_listener_site",
        "handover_giver_site",
        "handover_receiver_site",
        "handover_robot_site",
        "handover_transfer_site",
        "seat_ingress_site",
        "chair_sit_site",
        "seat_ingress_site",
        "chair_sit_site",
        "desk_work_site",
    ]
    speaker = projection.semantic_points["point.conversation.meeting.01.speaker"]
    assert speaker["station_id"] == "conversation.meeting.01"
    assert speaker["binding"] == "conversation_role"
    assert speaker["target"] == "zone.meeting"
    assert speaker["yaw"] == -1.57079633
    sit = projection.semantic_points["point.seat.object.asset_013_new_dini.01.sit"]
    assert sit["slot_id"] == "seat.object.asset_013_new_dini.01"
    assert sit["binding"] == "seat_sit"
    transfer = projection.semantic_points["point.handover.work.01.transfer"]
    assert transfer["binding"] == "handover_transfer"
    assert transfer["position"] == [0.24, -4.88, 0.92]

    slot = projection.semantic_entities["seat.object.asset_013_new_dini.01"]
    assert slot["semantic_class"] == "resource.seat_slot"
    assert slot["source"]["xml_binding"] == {
        "kind": "site",
        "name": (
            "interaction__office_02_cross_axis__"
            "seat__object__asset_013_new_dini__01__sit"
        ),
    }
    assert projection.semantic_entity_updates == {
        "object.asset_008_new_team": {
            "semantic_class": "furniture.workstation",
            "points": {"action": ["point.workstation.office_02.01.work"]},
        },
        "object.asset_009_new_imac": {"semantic_class": "device.computer"},
    }
    conversation = projection.interaction_stations["conversation"][
        "conversation.meeting.01"
    ]
    assert conversation["distance_m"] == {"min": 0.55, "max": 1.05}
    assert conversation["yaw_tolerance_rad"] == 0.3
    handover = projection.interaction_stations["handover"]["handover.work.01"]
    assert handover["distance_m"] == {"min": 0.75, "max": 1.3}
    assert handover["yaw_tolerance_rad"] == 0.3
    assert handover["transfer_site"].endswith("__transfer")
    seat = projection.interaction_stations["seat"][
        "seat.object.asset_013_new_dini.01"
    ]
    assert seat["slot_index"] == 1
    assert seat["clearance_radius_m"] == 0.28
    assert {tuple(relation.values()) for relation in projection.semantic_relations} == {
        ("seat.object.asset_013_new_dini.01", "PART_OF", "object.asset_013_new_dini"),
        ("seat.object.asset_011_new_dini.01", "PART_OF", "object.asset_011_new_dini"),
        ("object.asset_009_new_imac", "ON", "object.asset_008_new_team"),
        ("seat.object.asset_013_new_dini.01", "NEAR", "object.asset_008_new_team"),
    }


def test_projection_catalog_is_population_v3_compatible() -> None:
    projection = project_interaction_plan(_load_plan())
    payload = json.loads(POPULATION_PATH.read_text(encoding="utf-8"))
    payload["schema_version"] = 3
    payload.pop("interaction_templates")
    payload["interaction_stations"] = projection.interaction_stations
    all_sites = {site["site"] for site in projection.derived_sites}
    all_sites.update(npc["spawn"]["site"] for npc in payload["npcs"].values())

    population = NpcPopulation.from_dict(
        payload,
        source_path=POPULATION_PATH,
        sites=all_sites,
    )
    assert population.schema_version == 3
    assert set(population.interaction_stations or {}) == {
        "conversation",
        "handover",
        "seat",
        "workstation",
    }


def test_projection_has_deterministic_ids_and_candidate_receipt_skeleton() -> None:
    plan = _load_plan()
    first = project_interaction_plan(plan)
    second = project_interaction_plan(plan)

    assert first == second
    assert [site["id"] for site in first.derived_sites] == list(first.semantic_points)
    receipt = first.candidate_receipt
    assert receipt["validated_for_runtime"] is False
    assert receipt["counts"] == {
        "conversation_stations": 1,
        "handover_stations": 1,
        "seat_slots": 2,
        "workstations": 1,
        "derived_interaction_sites": 11,
    }
    assert all(receipt["validators"][name] == {"status": "not_run"} for name in PHYSICAL_VALIDATORS)
    assert receipt["validators"]["source_bindings"] == {"status": "not_run"}


def test_projection_does_not_truncate_multiple_stations(tmp_path: Path) -> None:
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    second_conversation = copy.deepcopy(payload["conversation_stations"][0])
    second_conversation["id"] = "conversation.meeting.02"
    for role in second_conversation["roles"].values():
        role["position"][0] += 2.0
    payload["conversation_stations"].append(second_conversation)
    second_handover = copy.deepcopy(payload["handover_stations"][0])
    second_handover["id"] = "handover.work.02"
    for role in second_handover["roles"].values():
        role["position"][0] += 3.0
    second_handover["transfer_position"][0] += 3.0
    payload["handover_stations"].append(second_handover)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    projection = project_interaction_plan(_load_plan(plan_path))

    assert len(projection.derived_sites) == 17
    assert set(projection.interaction_stations["conversation"]) == {
        "conversation.meeting.01",
        "conversation.meeting.02",
    }
    assert set(projection.interaction_stations["handover"]) == {
        "handover.work.01",
        "handover.work.02",
    }
    assert {
        site["id"]
        for site in projection.derived_sites
        if site.get("station_id") in {"conversation.meeting.02", "handover.work.02"}
    } == {
        "point.conversation.meeting.02.speaker",
        "point.conversation.meeting.02.listener",
        "point.handover.work.02.giver",
        "point.handover.work.02.receiver",
        "point.handover.work.02.robot",
        "point.handover.work.02.transfer",
    }


def test_projection_rejects_duplicate_physical_pose_across_stations(tmp_path: Path) -> None:
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    duplicate = copy.deepcopy(payload["conversation_stations"][0])
    duplicate["id"] = "conversation.meeting.02"
    payload["conversation_stations"].append(duplicate)
    plan_path = tmp_path / "duplicate-pose-plan.json"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            "duplicate_physical_station_pose:conversation.meeting.01:"
            "conversation.meeting.02"
        ),
    ):
        project_interaction_plan(_load_plan(plan_path))
