"""Strict authoritative model for candidate scene interaction sites.

This module validates authored facts and source bindings only. Navigation,
collision, and action execution remain compiler/runtime responsibilities.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class InteractionSitePlanError(ValueError):
    """An interaction_site_plan/v1 document violates its contract."""


MAX_SEAT_INGRESS_DISTANCE_M = 1.25
WORKSTATION_SUPPORT_TOLERANCE_M = 0.25


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InteractionSitePlanError(f"duplicate_key:{key}")
        result[key] = value
    return result


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractionSitePlanError(f"{context}_must_be_object")
    return value


def _array(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise InteractionSitePlanError(f"{context}_must_be_array")
    return value


def _string_array(
    value: Any,
    context: str,
    *,
    allowed: frozenset[str] | None = None,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(set(value)) != len(value)
        or (allowed is not None and not set(value) <= allowed)
    ):
        raise InteractionSitePlanError(context)
    return tuple(value)


def _exact(value: Mapping[str, Any], fields: set[str], context: str) -> None:
    if set(value) != fields:
        raise InteractionSitePlanError(f"{context}_fields_invalid")


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InteractionSitePlanError(f"{context}_invalid")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InteractionSitePlanError(f"{context}_must_be_finite")
    result = float(value)
    if not math.isfinite(result):
        raise InteractionSitePlanError(f"{context}_must_be_finite")
    return result


def _position(value: Any, context: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise InteractionSitePlanError(f"{context}_must_have_3_coordinates")
    return tuple(_number(item, context) for item in value)  # type: ignore[return-value]


@dataclass(frozen=True)
class StationPose:
    position: tuple[float, float, float]
    yaw: float

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "StationPose":
        payload = _mapping(value, context)
        _exact(payload, {"position", "yaw"}, context)
        yaw = _number(payload["yaw"], f"{context}_yaw")
        if not -math.pi <= yaw <= math.pi:
            raise InteractionSitePlanError(f"{context}_yaw_out_of_range")
        return cls(_position(payload["position"], f"{context}_position"), yaw)


@dataclass(frozen=True)
class ConversationStation:
    station_id: str
    region: str
    allowed_actor_pairs: tuple[str, ...]
    roles: Mapping[str, StationPose]
    minimum_distance_m: float
    maximum_distance_m: float
    yaw_tolerance_rad: float


@dataclass(frozen=True)
class HandoverStation:
    station_id: str
    region: str
    modes: tuple[str, ...]
    roles: Mapping[str, StationPose]
    transfer_position: tuple[float, float, float]
    object_ids: tuple[str, ...]
    minimum_distance_m: float
    maximum_distance_m: float
    yaw_tolerance_rad: float


@dataclass(frozen=True)
class SeatSlot:
    slot_id: str
    owner_entity: str
    seat_type: str
    slot_index: int
    ingress: StationPose
    sit: StationPose
    clearance_radius_m: float


@dataclass(frozen=True)
class WorkstationBinding:
    binding_id: str
    workstation_entity: str
    computer_entity: str
    seat_slot: str
    work_site: StationPose


@dataclass(frozen=True)
class SceneInteractionPlan:
    path: Path
    scene_id: str
    conversation_stations: tuple[ConversationStation, ...]
    handover_stations: tuple[HandoverStation, ...]
    seat_slots: tuple[SeatSlot, ...]
    workstations: tuple[WorkstationBinding, ...]
    seat_exemptions: tuple[Mapping[str, str], ...]

    @staticmethod
    def site_name(scene_id: str, resource_id: str, role: str) -> str:
        clean = resource_id.replace(".", "__")
        return f"interaction__{scene_id}__{clean}__{role}"

    def semantic_relations(self) -> tuple[dict[str, str], ...]:
        relations: list[dict[str, str]] = []
        for slot in self.seat_slots:
            relations.append(
                {"subject": slot.slot_id, "relation": "PART_OF", "object": slot.owner_entity}
            )
        for binding in self.workstations:
            relations.extend(
                (
                    {
                        "subject": binding.computer_entity,
                        "relation": "ON",
                        "object": binding.workstation_entity,
                    },
                    {
                        "subject": binding.seat_slot,
                        "relation": "NEAR",
                        "object": binding.workstation_entity,
                    },
                )
            )
        return tuple(relations)


def _distance(first: StationPose, second: StationPose) -> float:
    return math.dist(first.position[:2], second.position[:2])


def _heading(yaw: float) -> tuple[float, float]:
    # MuJoCo authored sites in these scenes use yaw=0 for +Y.
    return math.sin(yaw), math.cos(yaw)


def _facing_error(source: StationPose, target: StationPose) -> float:
    direction = (
        target.position[0] - source.position[0],
        target.position[1] - source.position[1],
    )
    length = math.hypot(*direction)
    if length <= 1e-9:
        return math.inf
    heading = _heading(source.yaw)
    cosine = max(-1.0, min(1.0, (heading[0] * direction[0] + heading[1] * direction[1]) / length))
    return math.acos(cosine)


def _distance_contract(value: Any, context: str) -> tuple[float, float]:
    payload = _mapping(value, context)
    _exact(payload, {"min", "max"}, context)
    minimum = _number(payload["min"], f"{context}_min")
    maximum = _number(payload["max"], f"{context}_max")
    if minimum <= 0 or maximum < minimum:
        raise InteractionSitePlanError(f"{context}_invalid")
    return minimum, maximum


def _validate_pair(
    roles: Mapping[str, StationPose],
    first: str,
    second: str,
    minimum: float,
    maximum: float,
    tolerance: float,
    context: str,
) -> None:
    separation = _distance(roles[first], roles[second])
    if separation <= 1e-9:
        raise InteractionSitePlanError(f"coincident_roles:{context}:{first}:{second}")
    if not minimum <= separation <= maximum:
        raise InteractionSitePlanError(f"role_distance_invalid:{context}")
    if _facing_error(roles[first], roles[second]) > tolerance or _facing_error(
        roles[second], roles[first]
    ) > tolerance:
        raise InteractionSitePlanError(f"role_yaw_not_facing:{context}")


def _asset_index(manifest: Mapping[str, Any], xml_bodies: set[str]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in manifest.get("assets", []):
        asset = _mapping(raw, "manifest_asset")
        body = asset.get("body")
        if not isinstance(body, str):
            object_id = asset.get("object_id")
            if isinstance(object_id, str) and object_id in xml_bodies:
                body = object_id
        if not isinstance(body, str):
            instance = asset.get("instance")
            if isinstance(instance, int):
                prefix = f"asset_{instance:03d}_"
                matches = sorted(name for name in xml_bodies if name.startswith(prefix))
                body = matches[0] if len(matches) == 1 else None
        if not isinstance(body, str) or body not in xml_bodies:
            continue
        for key in (body, asset.get("object_id")):
            if isinstance(key, str) and key:
                result[key] = asset
                result[f"object.{key}"] = asset
    return result


def _xml_body_inventory(source: Path) -> set[str]:
    root = ET.parse(source).getroot()
    bodies = {node.get("name") for node in root.findall(".//body") if node.get("name")}
    for include in root.findall("include"):
        raw_file = include.get("file")
        if not raw_file:
            continue
        included = Path(raw_file)
        if not included.is_absolute():
            included = source.parent / included
        if included.is_file():
            bodies.update(_xml_body_inventory(included.resolve()))
    return bodies


def _manifest_regions(
    manifest: Mapping[str, Any],
) -> tuple[frozenset[str], dict[str, tuple[float, float, float, float]]]:
    region_ids: set[str] = set()
    region_bounds: dict[str, tuple[float, float, float, float]] = {}
    for context, raw_items in (
        ("manifest_zones", manifest.get("zones", [])),
        ("manifest_regions", manifest.get("regions", [])),
    ):
        for raw in _array(raw_items, context):
            item = _mapping(raw, context.removesuffix("s"))
            raw_id = item.get("id")
            raw_type = item.get("type")
            aliases: set[str] = set()
            if isinstance(raw_id, str) and raw_id:
                aliases.add(raw_id)
            if isinstance(raw_type, str) and raw_type:
                aliases.add(
                    raw_type
                    if raw_type.startswith(("zone.", "room."))
                    else f"zone.{raw_type}"
                )
            region_ids.update(aliases)
            if "bounds" not in item:
                continue
            raw_bounds = item["bounds"]
            if not isinstance(raw_bounds, list) or len(raw_bounds) != 4:
                raise InteractionSitePlanError("manifest_region_bounds_invalid")
            bounds = tuple(
                _number(value, "manifest_region_bounds") for value in raw_bounds
            )
            if bounds[0] > bounds[1] or bounds[2] > bounds[3]:
                raise InteractionSitePlanError("manifest_region_bounds_invalid")
            for alias in aliases:
                region_bounds[alias] = bounds  # type: ignore[assignment]
    return frozenset(region_ids), region_bounds


def _validate_region_positions(
    region_bounds: Mapping[str, tuple[float, float, float, float]],
    *,
    region: str,
    station_id: str,
    positions: Mapping[str, tuple[float, float, float]],
) -> None:
    bounds = region_bounds.get(region)
    if bounds is None:
        return
    x_min, x_max, y_min, y_max = bounds
    for role, position in positions.items():
        if not x_min <= position[0] <= x_max or not y_min <= position[1] <= y_max:
            raise InteractionSitePlanError(
                f"station_pose_outside_region:{station_id}:{role}"
            )


def _asset_xy(asset: Mapping[str, Any]) -> tuple[float, float] | None:
    position = asset.get("position")
    if (
        not isinstance(position, list)
        or len(position) < 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in position[:2]
        )
    ):
        return None
    return float(position[0]), float(position[1])


def _validate_workstation_support(
    *,
    assets: Mapping[str, Mapping[str, Any]],
    binding_id: str,
    workstation_asset: Mapping[str, Any],
    computer_asset: Mapping[str, Any],
) -> None:
    support = computer_asset.get("support")
    if support is not None:
        if not isinstance(support, str) or assets.get(support) is not workstation_asset:
            raise InteractionSitePlanError(
                f"workstation_computer_support_mismatch:{binding_id}"
            )
        return
    workstation_xy = _asset_xy(workstation_asset)
    computer_xy = _asset_xy(computer_asset)
    radius = workstation_asset.get("collision_radius")
    if (
        workstation_xy is None
        or computer_xy is None
        or isinstance(radius, bool)
        or not isinstance(radius, (int, float))
        or not math.isfinite(float(radius))
        or float(radius) < 0
    ):
        raise InteractionSitePlanError(
            f"workstation_computer_support_unverifiable:{binding_id}"
        )
    if math.dist(workstation_xy, computer_xy) > (
        float(radius) + WORKSTATION_SUPPORT_TOLERANCE_M
    ):
        raise InteractionSitePlanError(
            f"workstation_computer_support_mismatch:{binding_id}"
        )


def _parse_roles(value: Any, expected: set[str], context: str) -> dict[str, StationPose]:
    payload = _mapping(value, f"{context}_roles")
    _exact(payload, expected, f"{context}_roles")
    return {role: StationPose.from_dict(raw, f"{context}_{role}") for role, raw in payload.items()}


def load_interaction_site_plan(
    path: str | Path,
    *,
    source_manifest: str | Path,
    source_mjcf: str | Path,
    expected_scene_id: str | None = None,
) -> SceneInteractionPlan:
    """Load v1 and validate source identities without claiming physical acceptance."""
    plan_path = Path(path).resolve()
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError as error:
        raise InteractionSitePlanError(f"invalid_json:{error.msg}") from error
    root = _mapping(payload, "plan")
    _exact(
        root,
        {
            "schema",
            "scene_id",
            "conversation_stations",
            "handover_stations",
            "seat_slots",
            "workstations",
            "seat_exemptions",
        },
        "plan",
    )
    if root["schema"] != "interaction_site_plan/v1":
        raise InteractionSitePlanError("plan_schema_invalid")
    scene_id = _identifier(root["scene_id"], "scene_id")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise InteractionSitePlanError("scene_id_mismatch")
    conversation_payloads = _array(root["conversation_stations"], "conversation_stations")
    handover_payloads = _array(root["handover_stations"], "handover_stations")
    seat_slot_payloads = _array(root["seat_slots"], "seat_slots")
    workstation_payloads = _array(root["workstations"], "workstations")
    seat_exemption_payloads = _array(root["seat_exemptions"], "seat_exemptions")

    manifest = _mapping(
        json.loads(Path(source_manifest).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates),
        "manifest",
    )
    if manifest.get("scene_id") != scene_id:
        raise InteractionSitePlanError("manifest_scene_id_mismatch")
    xml_bodies = _xml_body_inventory(Path(source_mjcf).resolve())
    region_ids, region_bounds = _manifest_regions(manifest)
    assets = _asset_index(manifest, xml_bodies)
    robot = _mapping(manifest.get("robot"), "manifest_robot")
    robot_body = _identifier(robot.get("body"), "manifest_robot_body")
    if robot_body not in xml_bodies:
        raise InteractionSitePlanError(f"robot_body_unbound:{robot_body}")

    conversations: list[ConversationStation] = []
    handovers: list[HandoverStation] = []
    slots: list[SeatSlot] = []
    workstations: list[WorkstationBinding] = []
    all_ids: set[str] = set()

    for index, raw in enumerate(conversation_payloads):
        item = _mapping(raw, f"conversation_station_{index}")
        _exact(item, {"id", "region", "allowed_actor_pairs", "roles", "distance_m", "yaw_tolerance_rad"}, f"conversation_station_{index}")
        station_id = _identifier(item["id"], "conversation_station_id")
        pairs = _string_array(
            item["allowed_actor_pairs"],
            f"conversation_actor_pairs_invalid:{station_id}",
            allowed=frozenset({"npc_npc", "robot_npc"}),
        )
        roles = _parse_roles(item["roles"], {"speaker", "listener"}, station_id)
        minimum, maximum = _distance_contract(item["distance_m"], f"{station_id}_distance")
        tolerance = _number(item["yaw_tolerance_rad"], f"{station_id}_yaw_tolerance")
        if not 0 < tolerance <= math.pi:
            raise InteractionSitePlanError(f"yaw_tolerance_invalid:{station_id}")
        _validate_pair(roles, "speaker", "listener", minimum, maximum, tolerance, station_id)
        region = _identifier(item["region"], "region")
        if region not in region_ids:
            raise InteractionSitePlanError(f"station_region_unbound:{station_id}:{region}")
        _validate_region_positions(
            region_bounds,
            region=region,
            station_id=station_id,
            positions={role: pose.position for role, pose in roles.items()},
        )
        conversations.append(ConversationStation(station_id, region, pairs, roles, minimum, maximum, tolerance))
        all_ids.add(station_id)

    for index, raw in enumerate(handover_payloads):
        item = _mapping(raw, f"handover_station_{index}")
        _exact(item, {"id", "region", "modes", "roles", "transfer_position", "object_ids", "distance_m", "yaw_tolerance_rad"}, f"handover_station_{index}")
        station_id = _identifier(item["id"], "handover_station_id")
        modes = _string_array(
            item["modes"],
            f"handover_modes_invalid:{station_id}",
            allowed=frozenset({"npc_to_npc", "robot_to_npc", "npc_to_robot"}),
        )
        roles = _parse_roles(item["roles"], {"giver", "receiver", "robot"}, station_id)
        minimum, maximum = _distance_contract(item["distance_m"], f"{station_id}_distance")
        tolerance = _number(item["yaw_tolerance_rad"], f"{station_id}_yaw_tolerance")
        mode_roles = {
            "npc_to_npc": ("giver", "receiver"),
            "robot_to_npc": ("robot", "receiver"),
            "npc_to_robot": ("giver", "robot"),
        }
        for mode in modes:
            first, second = mode_roles[mode]
            _validate_pair(
                roles, first, second, minimum, maximum, tolerance, f"{station_id}:{mode}"
            )
        region = _identifier(item["region"], "region")
        if region not in region_ids:
            raise InteractionSitePlanError(f"station_region_unbound:{station_id}:{region}")
        transfer_position = _position(
            item["transfer_position"], f"{station_id}_transfer_position"
        )
        _validate_region_positions(
            region_bounds,
            region=region,
            station_id=station_id,
            positions={
                **{role: pose.position for role, pose in roles.items()},
                "transfer": transfer_position,
            },
        )
        object_ids = _string_array(
            item["object_ids"], f"handover_objects_invalid:{station_id}"
        )
        for object_id in object_ids:
            asset = assets.get(_identifier(object_id, "handover_object"))
            if asset is None or asset.get("dynamic") is not True or asset.get("graspable") is not True:
                raise InteractionSitePlanError(f"handover_object_unbound_or_not_graspable:{object_id}")
        handovers.append(HandoverStation(station_id, region, modes, roles, transfer_position, object_ids, minimum, maximum, tolerance))
        all_ids.add(station_id)

    seat_categories = {"meeting_chairs", "task_chairs", "lounge_seating", "chairs", "sofas", "benches", "stools", "ottomans", "seating_furniture"}
    for index, raw in enumerate(seat_slot_payloads):
        item = _mapping(raw, f"seat_slot_{index}")
        _exact(item, {"id", "owner_entity", "seat_type", "slot_index", "ingress", "sit", "clearance_radius_m"}, f"seat_slot_{index}")
        slot_id = _identifier(item["id"], "seat_slot_id")
        owner = _identifier(item["owner_entity"], "seat_owner")
        asset = assets.get(owner)
        if asset is None or asset.get("category") not in seat_categories:
            raise InteractionSitePlanError(f"seat_owner_unbound_or_not_seating:{owner}")
        slot_index = item["slot_index"]
        if isinstance(slot_index, bool) or not isinstance(slot_index, int) or slot_index < 1:
            raise InteractionSitePlanError(f"seat_slot_index_invalid:{slot_id}")
        clearance = _number(item["clearance_radius_m"], f"{slot_id}_clearance")
        if clearance <= 0:
            raise InteractionSitePlanError(f"seat_clearance_invalid:{slot_id}")
        ingress = StationPose.from_dict(item["ingress"], f"{slot_id}_ingress")
        sit = StationPose.from_dict(item["sit"], f"{slot_id}_sit")
        ingress_distance = _distance(ingress, sit)
        if not clearance < ingress_distance <= MAX_SEAT_INGRESS_DISTANCE_M:
            raise InteractionSitePlanError(f"seat_ingress_distance_invalid:{slot_id}")
        slots.append(SeatSlot(slot_id, owner, _identifier(item["seat_type"], "seat_type"), slot_index, ingress, sit, clearance))
        all_ids.add(slot_id)

    slot_ids = {slot.slot_id for slot in slots}
    for index, raw in enumerate(workstation_payloads):
        item = _mapping(raw, f"workstation_{index}")
        _exact(item, {"id", "workstation_entity", "computer_entity", "seat_slot", "work_site"}, f"workstation_{index}")
        binding_id = _identifier(item["id"], "workstation_id")
        workstation_entity = _identifier(item["workstation_entity"], "workstation_entity")
        computer_entity = _identifier(item["computer_entity"], "computer_entity")
        workstation_asset = assets.get(workstation_entity)
        computer_asset = assets.get(computer_entity)
        if workstation_asset is None or workstation_asset.get("category") not in {"workstation_pods", "desks", "meeting_tables"}:
            raise InteractionSitePlanError(f"workstation_entity_unbound:{workstation_entity}")
        if computer_asset is None or computer_asset.get("category") not in {"displays", "computers"}:
            raise InteractionSitePlanError(f"computer_entity_not_computer:{computer_entity}")
        _validate_workstation_support(
            assets=assets,
            binding_id=binding_id,
            workstation_asset=workstation_asset,
            computer_asset=computer_asset,
        )
        seat_slot = _identifier(item["seat_slot"], "workstation_seat_slot")
        if seat_slot not in slot_ids:
            raise InteractionSitePlanError(f"workstation_seat_slot_unbound:{seat_slot}")
        workstations.append(WorkstationBinding(binding_id, workstation_entity, computer_entity, seat_slot, StationPose.from_dict(item["work_site"], f"{binding_id}_work_site")))
        all_ids.add(binding_id)

    exemptions: list[Mapping[str, str]] = []
    for index, raw in enumerate(seat_exemption_payloads):
        item = _mapping(raw, f"seat_exemption_{index}")
        _exact(item, {"owner_entity", "reason", "evidence"}, f"seat_exemption_{index}")
        owner = _identifier(item["owner_entity"], "seat_exemption_owner")
        if owner not in assets:
            raise InteractionSitePlanError(f"seat_exemption_owner_unbound:{owner}")
        exemptions.append({key: _identifier(item[key], f"seat_exemption_{key}") for key in item})

    expected_count = len(conversations) + len(handovers) + len(slots) + len(workstations)
    if len(all_ids) != expected_count:
        raise InteractionSitePlanError("duplicate_resource_id")
    return SceneInteractionPlan(plan_path, scene_id, tuple(conversations), tuple(handovers), tuple(slots), tuple(workstations), tuple(exemptions))
