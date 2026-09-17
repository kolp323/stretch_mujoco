from __future__ import annotations

import copy

import pytest

from stretch_mujoco.semantics import (
    InteractionRole,
    ObjectType,
    RelationType,
    SemanticRelation,
    SemanticValidationError,
    SemanticWorld,
)


def _payload() -> dict:
    return {
        "schema_version": 2,
        "scene_id": "fixture_v2",
        "entities": {
            "zone.work": {
                "semantic_class": "region.work",
                "source": {"xml_binding": {"kind": "site", "name": None}},
                "points": {"navigation": [], "action": []},
            },
            "object.chair": {
                "semantic_class": "furniture.seat",
                "source": {"xml_binding": {"kind": "body", "name": "chair_body"}},
                "points": {"navigation": [], "action": []},
            },
            "seat.object.chair.01": {
                "semantic_class": "resource.seat_slot",
                "source": {"xml_binding": {"kind": "site", "name": "seat_sit"}},
                "points": {"navigation": [], "action": []},
            },
            "object.workstation": {
                "semantic_class": "furniture.workstation",
                "source": {"xml_binding": {"kind": "body", "name": "desk_body"}},
                "points": {"navigation": [], "action": []},
            },
            "object.computer": {
                "semantic_class": "device.computer",
                "source": {"xml_binding": {"kind": "body", "name": "computer_body"}},
                "points": {"navigation": [], "action": []},
            },
        },
        "points": {
            "point.conversation.speaker": {
                "owner": "zone.work",
                "site": "conversation_speaker",
                "role": "conversation_speaker_site",
                "binding": "conversation_role",
                "target": "zone.work",
                "station_id": "conversation.work.01",
                "yaw": 1.57,
            },
            "point.conversation.listener": {
                "owner": "zone.work",
                "site": "conversation_listener",
                "role": "conversation_listener_site",
                "binding": "conversation_role",
                "target": "zone.work",
                "station_id": "conversation.work.01",
                "yaw": -1.57,
            },
            "point.handover.giver": {
                "owner": "zone.work",
                "site": "handover_giver",
                "role": "handover_giver_site",
                "binding": "handover_role",
                "target": "object.computer",
                "station_id": "handover.work.01",
                "yaw": 1.57,
            },
            "point.handover.receiver": {
                "owner": "zone.work",
                "site": "handover_receiver",
                "role": "handover_receiver_site",
                "binding": "handover_role",
                "target": "object.computer",
                "station_id": "handover.work.01",
                "yaw": -1.57,
            },
            "point.handover.robot": {
                "owner": "zone.work",
                "site": "handover_robot",
                "role": "handover_robot_site",
                "binding": "handover_role",
                "target": "object.computer",
                "station_id": "handover.work.01",
                "yaw": 1.57,
            },
            "point.handover.transfer": {
                "owner": "zone.work",
                "site": "handover_transfer",
                "role": "handover_transfer_site",
                "binding": "handover_transfer",
                "target": "object.computer",
                "station_id": "handover.work.01",
                "yaw": 0.0,
            },
            "point.seat.ingress": {
                "owner": "seat.object.chair.01",
                "site": "seat_ingress",
                "role": "seat_ingress_site",
                "binding": "seat_ingress",
                "target": "seat.object.chair.01",
                "slot_id": "seat.object.chair.01",
                "yaw": 0.0,
            },
            "point.seat.sit": {
                "owner": "seat.object.chair.01",
                "site": "seat_sit",
                "role": "chair_sit_site",
                "binding": "seat_sit",
                "target": "seat.object.chair.01",
                "slot_id": "seat.object.chair.01",
                "yaw": 0.0,
            },
            "point.work": {
                "owner": "object.workstation",
                "site": "desk_work",
                "role": "desk_work_site",
                "binding": "workstation",
                "target": "object.workstation",
                "station_id": "workstation.work.01",
                "slot_id": "seat.object.chair.01",
                "yaw": 0.0,
            },
        },
        "relations": [
            {
                "subject": "seat.object.chair.01",
                "relation": "PART_OF",
                "object": "object.chair",
            },
            {
                "subject": "seat.object.chair.01",
                "relation": "NEAR",
                "object": "object.workstation",
            },
            {
                "subject": "object.computer",
                "relation": "ON",
                "object": "object.workstation",
            },
        ],
    }


def test_v2_preserves_roles_relations_and_interaction_attributes() -> None:
    world = SemanticWorld._from_v2_payload(_payload())
    assert world.objects["zone.work"].attributes["topology_only"] is True
    assert world.objects["seat.object.chair.01"].object_type == ObjectType.SEAT_SLOT
    assert world.interaction_points["point.conversation.speaker"].role == (
        InteractionRole.CONVERSATION_SPEAKER
    )
    assert world.interaction_points["point.handover.robot"].role == (
        InteractionRole.HANDOVER_ROBOT
    )
    transfer = world.interaction_points["point.handover.transfer"]
    assert transfer.role == InteractionRole.HANDOVER_TRANSFER
    assert transfer.attributes["binding"] == "handover_transfer"
    assert transfer.attributes["station_id"] == "handover.work.01"
    assert world.interaction_points["point.seat.ingress"].role == InteractionRole.SEAT_INGRESS
    assert world.interaction_points["point.seat.sit"].role == InteractionRole.CHAIR_SIT
    assert world.interaction_points["point.work"].role == InteractionRole.DESK_WORK
    attributes = world.interaction_points["point.work"].attributes
    assert attributes["station_id"] == "workstation.work.01"
    assert attributes["slot_id"] == "seat.object.chair.01"
    assert attributes["binding"] == "workstation"
    assert attributes["target"] == "object.workstation"
    assert attributes["yaw"] == 0.0
    assert SemanticRelation(
        "seat.object.chair.01", RelationType.PART_OF, "object.chair"
    ) in world.relations


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["points"]["point.work"].update(role="unknown_role"),
            "Unknown InteractionRole value 'unknown_role'",
        ),
        (
            lambda payload: payload["relations"][0].update(relation="UNKNOWN"),
            "Unknown RelationType value 'UNKNOWN'",
        ),
        (
            lambda payload: payload["relations"][0].update(subject="seat.missing"),
            "Relation subject 'seat.missing' is not registered",
        ),
        (
            lambda payload: payload["relations"][0].update(object="object.missing"),
            "Relation object 'object.missing' is not registered",
        ),
    ],
)
def test_v2_rejects_unknown_roles_relations_and_unbound_endpoints(mutate, match: str) -> None:
    payload = copy.deepcopy(_payload())
    mutate(payload)
    with pytest.raises(SemanticValidationError, match=match):
        SemanticWorld._from_v2_payload(payload)


def test_v1_role_and_relation_parsing_remains_unchanged() -> None:
    world = SemanticWorld.from_v1_json_payload(
        {
            "scene": "legacy",
            "objects": {
                "chair": {
                    "type": "Chair",
                    "binding": {"kind": "body", "name": "chair_body"},
                },
                "desk": {
                    "type": "Workstation",
                    "binding": {"kind": "body", "name": "desk_body"},
                },
            },
            "relations": [{"subject": "chair", "relation": "NEAR", "object": "desk"}],
            "interaction_points": {
                "sit": {
                    "role": "chair_sit_site",
                    "owner": "chair",
                    "site": "chair_sit",
                    "attributes": {"yaw": 0.5},
                }
            },
        }
    )
    assert world.interaction_points["sit"].role == InteractionRole.CHAIR_SIT
    assert world.interaction_points["sit"].attributes == {"yaw": 0.5}
    assert SemanticRelation("chair", RelationType.NEAR, "desk") in world.relations
