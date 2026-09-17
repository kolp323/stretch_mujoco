"""Pure projections from an authored interaction-site plan.

This module deliberately performs no file writes and no physical validation.
The compiler can merge these deterministic fragments into generated artifacts
after later phases provide navigation, collision, reach, and action evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .interaction_station import SceneInteractionPlan, StationPose


VALIDATOR_VERSION = "interaction_projection_phase1b2b1/v1"
PHYSICAL_VALIDATORS = (
    "npc_navigation",
    "robot_navigation",
    "terminal_collision",
    "handover_reach",
    "sit_action",
    "work_action",
)


@dataclass(frozen=True)
class InteractionPlanProjection:
    derived_sites: tuple[dict[str, Any], ...]
    semantic_entities: dict[str, dict[str, Any]]
    semantic_entity_updates: dict[str, dict[str, Any]]
    semantic_points: dict[str, dict[str, Any]]
    semantic_relations: tuple[dict[str, str], ...]
    interaction_stations: dict[str, dict[str, dict[str, Any]]]
    candidate_receipt: dict[str, Any]


def _site_descriptor(
    plan: SceneInteractionPlan,
    resource_id: str,
    role_name: str,
    pose: StationPose,
    *,
    owner: str,
    role: str,
    binding: str,
    target: str,
    station_id: str | None = None,
    slot_id: str | None = None,
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "id": f"point.{resource_id}.{role_name}",
        "site": plan.site_name(plan.scene_id, resource_id, role_name),
        "position": list(pose.position),
        "yaw": pose.yaw,
        "owner": owner,
        "required": False,
        "kind": "action",
        "role": role,
        "binding": binding,
        "target": target,
    }
    if station_id is not None:
        descriptor["station_id"] = station_id
    if slot_id is not None:
        descriptor["slot_id"] = slot_id
    return descriptor


def project_interaction_plan(plan: SceneInteractionPlan) -> InteractionPlanProjection:
    """Return deterministic candidate projections without claiming runtime support."""
    sites: list[dict[str, Any]] = []
    catalog: dict[str, dict[str, dict[str, Any]]] = {
        "conversation": {},
        "handover": {},
        "seat": {},
        "workstation": {},
    }

    conversation_roles = (
        ("speaker", "conversation_speaker_site"),
        ("listener", "conversation_listener_site"),
    )
    for station in plan.conversation_stations:
        role_catalog: dict[str, Any] = {}
        for role_name, semantic_role in conversation_roles:
            pose = station.roles[role_name]
            descriptor = _site_descriptor(
                plan,
                station.station_id,
                role_name,
                pose,
                owner=station.region,
                role=semantic_role,
                binding="conversation_role",
                target=station.region,
                station_id=station.station_id,
            )
            sites.append(descriptor)
            role_catalog[role_name] = {"site": descriptor["site"], "yaw": pose.yaw}
        catalog["conversation"][station.station_id] = {
            "roles": role_catalog,
            "allowed_actor_pairs": list(station.allowed_actor_pairs),
            "distance_m": {
                "min": station.minimum_distance_m,
                "max": station.maximum_distance_m,
            },
            "yaw_tolerance_rad": station.yaw_tolerance_rad,
        }

    handover_roles = (
        ("giver", "handover_giver_site"),
        ("receiver", "handover_receiver_site"),
        ("robot", "handover_robot_site"),
    )
    for station in plan.handover_stations:
        role_catalog = {}
        for role_name, semantic_role in handover_roles:
            pose = station.roles[role_name]
            descriptor = _site_descriptor(
                plan,
                station.station_id,
                role_name,
                pose,
                owner=station.region,
                role=semantic_role,
                binding="handover_role",
                target=station.object_ids[0],
                station_id=station.station_id,
            )
            sites.append(descriptor)
            role_catalog[role_name] = {"site": descriptor["site"], "yaw": pose.yaw}
        transfer = _site_descriptor(
            plan,
            station.station_id,
            "transfer",
            StationPose(station.transfer_position, 0.0),
            owner=station.region,
            role="handover_transfer_site",
            binding="handover_transfer",
            target=station.object_ids[0],
            station_id=station.station_id,
        )
        sites.append(transfer)
        catalog["handover"][station.station_id] = {
            "roles": role_catalog,
            "modes": list(station.modes),
            "object_ids": list(station.object_ids),
            "distance_m": {
                "min": station.minimum_distance_m,
                "max": station.maximum_distance_m,
            },
            "yaw_tolerance_rad": station.yaw_tolerance_rad,
            "transfer_site": transfer["site"],
        }

    semantic_entities: dict[str, dict[str, Any]] = {}
    for slot in plan.seat_slots:
        role_catalog = {}
        for role_name, pose, semantic_role, binding in (
            ("ingress", slot.ingress, "seat_ingress_site", "seat_ingress"),
            ("sit", slot.sit, "chair_sit_site", "seat_sit"),
        ):
            descriptor = _site_descriptor(
                plan,
                slot.slot_id,
                role_name,
                pose,
                owner=slot.slot_id,
                role=semantic_role,
                binding=binding,
                target=slot.slot_id,
                slot_id=slot.slot_id,
            )
            sites.append(descriptor)
            role_catalog[role_name] = {"site": descriptor["site"], "yaw": pose.yaw}
        sit_site = role_catalog["sit"]["site"]
        semantic_entities[slot.slot_id] = {
            "semantic_class": "resource.seat_slot",
            "source": {"xml_binding": {"kind": "site", "name": sit_site}},
            "labels": {"canonical": slot.slot_id, "source_category": slot.seat_type},
            "navigation_requirement": "approach",
            "affordances": ["sit"],
            "points": {
                "navigation": [f"point.{slot.slot_id}.ingress"],
                "action": [f"point.{slot.slot_id}.sit"],
            },
        }
        catalog["seat"][slot.slot_id] = {
            "roles": role_catalog,
            "owner_entity": slot.owner_entity,
            "seat_type": slot.seat_type,
            "slot_index": slot.slot_index,
            "clearance_radius_m": slot.clearance_radius_m,
        }

    semantic_entity_updates: dict[str, dict[str, Any]] = {}
    for workstation in plan.workstations:
        pose = workstation.work_site
        descriptor = _site_descriptor(
            plan,
            workstation.binding_id,
            "work",
            pose,
            owner=workstation.workstation_entity,
            role="desk_work_site",
            binding="workstation",
            target=workstation.computer_entity,
            station_id=workstation.binding_id,
            slot_id=workstation.seat_slot,
        )
        sites.append(descriptor)
        semantic_entity_updates[workstation.workstation_entity] = {
            "semantic_class": "furniture.workstation",
            "points": {"action": [descriptor["id"]]},
        }
        semantic_entity_updates[workstation.computer_entity] = {
            "semantic_class": "device.computer"
        }
        catalog["workstation"][workstation.binding_id] = {
            "roles": {"work": {"site": descriptor["site"], "yaw": pose.yaw}},
            "workstation_entity": workstation.workstation_entity,
            "computer_entity": workstation.computer_entity,
            "seat_slot": workstation.seat_slot,
        }

    point_ids = [site["id"] for site in sites]
    site_names = [site["site"] for site in sites]
    if len(set(point_ids)) != len(point_ids):
        raise ValueError("duplicate_derived_point_id")
    if len(set(site_names)) != len(site_names):
        raise ValueError("duplicate_derived_site_name")
    poses: dict[tuple[tuple[float, ...], float], str] = {}
    for site in sites:
        station_id = site.get("station_id") or site.get("slot_id")
        if station_id is None:
            continue
        key = (tuple(site["position"]), site["yaw"])
        previous = poses.get(key)
        if previous is not None and previous != station_id:
            raise ValueError(f"duplicate_physical_station_pose:{previous}:{station_id}")
        poses[key] = station_id

    semantic_points = {
        descriptor["id"]: {
            key: value for key, value in descriptor.items() if key != "id"
        }
        for descriptor in sites
    }
    counts = {
        "conversation_stations": len(plan.conversation_stations),
        "handover_stations": len(plan.handover_stations),
        "seat_slots": len(plan.seat_slots),
        "workstations": len(plan.workstations),
        "derived_interaction_sites": len(sites),
    }
    validators = {
        "plan_schema": {"status": "passed"},
        "source_bindings": {"status": "not_run"},
        "derived_site_uniqueness": {"status": "passed"},
        **{name: {"status": "not_run"} for name in PHYSICAL_VALIDATORS},
    }
    receipt = {
        "schema": "scene_interaction_candidate_receipt/v1",
        "validator_version": VALIDATOR_VERSION,
        "scene_id": plan.scene_id,
        "validated_for_runtime": False,
        "counts": counts,
        "validators": validators,
    }
    return InteractionPlanProjection(
        derived_sites=tuple(sites),
        semantic_entities=semantic_entities,
        semantic_entity_updates=semantic_entity_updates,
        semantic_points=semantic_points,
        semantic_relations=plan.semantic_relations(),
        interaction_stations=catalog,
        candidate_receipt=receipt,
    )
