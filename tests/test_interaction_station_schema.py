from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from stretch_mujoco.npc.interaction_station import (
    InteractionSitePlanError,
    load_interaction_site_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "stretch_mujoco/models/scene_interaction_plans/office/office_02_cross_axis.json"
MANIFEST = ROOT / "stretch_mujoco/models/assets/office_scenes/office_02_cross_axis.json"
MJCF = ROOT / "stretch_mujoco/models/assets/office_scenes/office_02_cross_axis.xml"


def _load(path: Path = PLAN, manifest: Path = MANIFEST):
    return load_interaction_site_plan(
        path,
        source_manifest=manifest,
        source_mjcf=MJCF,
        expected_scene_id="office_02_cross_axis",
    )


def _changed(tmp_path: Path, mutate) -> Path:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _changed_manifest(tmp_path: Path, mutate) -> Path:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_office_02_candidate_is_strict_authoritative_plan_without_runtime_evidence() -> None:
    plan = _load()
    assert not hasattr(plan, "validated_for_runtime")
    assert len(plan.conversation_stations) == 1
    assert len(plan.handover_stations) == 1
    assert plan.handover_stations[0].modes == ("npc_to_npc", "robot_to_npc")
    assert len(plan.seat_slots) == 2
    assert len(plan.workstations) == 1
    assert plan.conversation_stations[0].roles["speaker"].position[:2] == (-7.26, 1.82)
    assert plan.conversation_stations[0].roles["listener"].position[:2] == (-8.06, 1.82)
    assert {
        slot.owner_entity: round(math.dist(slot.ingress.position[:2], slot.sit.position[:2]), 6)
        for slot in plan.seat_slots
    } == {
        "object.asset_013_new_dini": 0.930215,
        "object.asset_011_new_dini": 0.915447,
    }
    assert plan.workstations[0].workstation_entity == "object.asset_008_new_team"
    assert plan.workstations[0].computer_entity == "object.asset_009_new_imac"
    assert plan.workstations[0].seat_slot == "seat.object.asset_013_new_dini.01"
    assert {item["relation"] for item in plan.semantic_relations()} == {
        "PART_OF",
        "ON",
        "NEAR",
    }
    assert plan.site_name(plan.scene_id, "conversation.meeting.01", "speaker").startswith(
        "interaction__office_02_cross_axis__"
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["seat_slots"][0].update(
                owner_entity="object.body_does_not_exist"
            ),
            "seat_owner_unbound_or_not_seating",
        ),
        (
            lambda payload: payload["conversation_stations"][0]["roles"].update(
                listener=dict(payload["conversation_stations"][0]["roles"]["speaker"])
            ),
            "coincident_roles",
        ),
        (
            lambda payload: payload["conversation_stations"][0]["roles"]["listener"].update(
                yaw=0.0
            ),
            "role_yaw_not_facing",
        ),
        (
            lambda payload: payload["seat_slots"][0].pop("ingress"),
            "seat_slot_0_fields_invalid",
        ),
        (
            lambda payload: payload["workstations"][0].update(
                computer_entity="object.116_keyboard_004"
            ),
            "computer_entity_not_computer",
        ),
        (
            lambda payload: payload.update(validated_for_runtime=False),
            "plan_fields_invalid",
        ),
        (
            lambda payload: payload["handover_stations"][0].update(
                object_ids=["object.asset_009_new_imac"]
            ),
            "handover_object_unbound_or_not_graspable",
        ),
        (
            lambda payload: payload["workstations"][0].update(seat_slot="seat.missing"),
            "workstation_seat_slot_unbound",
        ),
    ],
)
def test_plan_rejects_planted_schema_and_cross_reference_violations(
    tmp_path: Path, mutate, match: str
) -> None:
    with pytest.raises(InteractionSitePlanError, match=match):
        _load(_changed(tmp_path, mutate))


@pytest.mark.parametrize(
    ("collection", "match"),
    [
        ("allowed_actor_pairs", "conversation_actor_pairs_invalid"),
        ("modes", "handover_modes_invalid"),
        ("object_ids", "handover_objects_invalid"),
    ],
)
def test_plan_rejects_non_string_nested_arrays(
    tmp_path: Path, collection: str, match: str
) -> None:
    def mutate(payload) -> None:
        owner = (
            payload["conversation_stations"][0]
            if collection == "allowed_actor_pairs"
            else payload["handover_stations"][0]
        )
        owner[collection] = [{"not": "a string"}]

    with pytest.raises(InteractionSitePlanError, match=match):
        _load(_changed(tmp_path, mutate))


@pytest.mark.parametrize(
    "collection",
    [
        "conversation_stations",
        "handover_stations",
        "seat_slots",
        "workstations",
        "seat_exemptions",
    ],
)
def test_plan_rejects_non_array_root_collections(tmp_path: Path, collection: str) -> None:
    path = _changed(tmp_path, lambda payload: payload.update({collection: {}}))
    with pytest.raises(InteractionSitePlanError, match=f"{collection}_must_be_array"):
        _load(path)


def test_plan_rejects_unbound_station_region(tmp_path: Path) -> None:
    path = _changed(
        tmp_path,
        lambda payload: payload["conversation_stations"][0].update(region="zone.missing"),
    )
    with pytest.raises(InteractionSitePlanError, match="station_region_unbound"):
        _load(path)


def test_plan_rejects_station_pose_outside_declared_region(tmp_path: Path) -> None:
    def mutate(payload) -> None:
        roles = payload["conversation_stations"][0]["roles"]
        roles["speaker"]["position"] = [1.0, 3.0, 0.025]
        roles["listener"]["position"] = [0.2, 3.0, 0.025]

    with pytest.raises(
        InteractionSitePlanError,
        match=r"^station_pose_outside_region:conversation\.meeting\.01:speaker$",
    ):
        _load(_changed(tmp_path, mutate))


def test_plan_rejects_nonlocal_seat_ingress(tmp_path: Path) -> None:
    def mutate(payload) -> None:
        slot = payload["seat_slots"][0]
        slot["ingress"]["position"] = [slot["sit"]["position"][0] + 1.5, 3.0, 0.025]

    with pytest.raises(
        InteractionSitePlanError,
        match=r"^seat_ingress_distance_invalid:",
    ):
        _load(_changed(tmp_path, mutate))


def test_plan_rejects_workstation_computer_support_mismatch(tmp_path: Path) -> None:
    path = _changed(
        tmp_path,
        lambda payload: payload["workstations"][0].update(
            workstation_entity="object.asset_000_new_cb_d"
        ),
    )
    with pytest.raises(
        InteractionSitePlanError,
        match=r"^workstation_computer_support_mismatch:workstation\.office_02\.01$",
    ):
        _load(path)


@pytest.mark.parametrize("instance", [8, 9])
def test_plan_rejects_unverifiable_workstation_support_position(
    tmp_path: Path, instance: int
) -> None:
    def mutate(payload) -> None:
        next(asset for asset in payload["assets"] if asset["instance"] == instance).pop(
            "position"
        )

    manifest = _changed_manifest(tmp_path, mutate)
    with pytest.raises(
        InteractionSitePlanError,
        match=(
            r"^workstation_computer_support_unverifiable:"
            r"workstation\.office_02\.01$"
        ),
    ):
        _load(manifest=manifest)


def test_plan_rejects_unverifiable_workstation_support_radius(tmp_path: Path) -> None:
    def mutate(payload) -> None:
        next(asset for asset in payload["assets"] if asset["instance"] == 8).pop(
            "collision_radius"
        )

    manifest = _changed_manifest(tmp_path, mutate)
    with pytest.raises(
        InteractionSitePlanError,
        match=(
            r"^workstation_computer_support_unverifiable:"
            r"workstation\.office_02\.01$"
        ),
    ):
        _load(manifest=manifest)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["handover_stations"][0]["roles"].update(
                robot=dict(payload["handover_stations"][0]["roles"]["receiver"])
            ),
            "coincident_roles:handover.work.01:robot_to_npc:robot:receiver",
        ),
        (
            lambda payload: payload["handover_stations"][0]["roles"]["robot"].update(
                yaw=-1.57079633
            ),
            "role_yaw_not_facing:handover.work.01:robot_to_npc",
        ),
    ],
)
def test_handover_rejects_bad_robot_pair_pose(
    tmp_path: Path, mutate, match: str
) -> None:
    with pytest.raises(InteractionSitePlanError, match=match):
        _load(_changed(tmp_path, mutate))


def test_handover_validates_npc_to_robot_pair_independently(tmp_path: Path) -> None:
    def mutate(payload) -> None:
        station = payload["handover_stations"][0]
        station["modes"] = ["npc_to_robot"]
        station["roles"]["robot"] = {
            "position": [0.65, -4.75, 0.025],
            "yaw": -1.57079633,
        }

    plan = _load(_changed(tmp_path, mutate))
    assert plan.handover_stations[0].modes == ("npc_to_robot",)


def test_plan_rejects_duplicate_json_key(tmp_path: Path) -> None:
    text = PLAN.read_text(encoding="utf-8").replace(
        '"schema": "interaction_site_plan/v1",',
        '"schema": "interaction_site_plan/v1",\n  "schema": "interaction_site_plan/v1",',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(InteractionSitePlanError, match="duplicate_key:schema"):
        _load(path)
