"""Strict, portable source configuration for scene-wide NPC semantics.

This deliberately uses JSON for the first public version: the repository
already ships JSON scene manifests and it lets the loader reject duplicate
keys without adding a YAML dependency.  Generated artifacts are projections,
never configuration input.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SceneConfigError(ValueError):
    """A scene_npc_config/v1 document violates its strict contract."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SceneConfigError(f"duplicate_key:{key}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError as error:
        raise SceneConfigError(f"invalid_json:{path}:{error.msg}") from error
    if not isinstance(payload, dict):
        raise SceneConfigError(f"root_must_be_object:{path}")
    return payload


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SceneConfigError(f"{name}_must_be_object")
    return value


def _finite_point(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) not in (2, 3):
        raise SceneConfigError(f"{name}_must_be_2_or_3_coordinates")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise SceneConfigError(f"{name}_must_be_finite")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise SceneConfigError(f"{name}_must_be_finite")
    return (result[0], result[1], result[2] if len(result) == 3 else 0.025)


@dataclass(frozen=True)
class PopulationMember:
    npc: str
    spawn: str
    initial_region: str


@dataclass(frozen=True)
class SceneInteractionPlanRef:
    path: Path
    required_capabilities: frozenset[str]


@dataclass(frozen=True)
class RobotNavigationConfig:
    footprint_radius: float
    clearance: float
    resolution: float


@dataclass(frozen=True)
class SceneNpcConfig:
    path: Path
    scene_id: str
    kind: str
    source_mjcf: Path
    source_manifest: Path
    semantic_policy: Path
    npc_catalog: Path
    navigation: dict[str, Any]
    registration: dict[str, Any]
    population: tuple[PopulationMember, ...]
    spawn_policy: dict[str, Any]
    route_coverage: dict[str, Any]
    entity_overrides: dict[str, dict[str, Any]]
    region_overrides: dict[str, dict[str, Any]]
    custom_targets: dict[str, dict[str, Any]]
    routes: tuple[dict[str, Any], ...]
    traffic: dict[str, Any]
    interactions: dict[str, dict[str, Any]]
    runtime: dict[str, Any]
    schema: str
    interaction_plan: SceneInteractionPlanRef | None
    robot_navigation: RobotNavigationConfig | None


def load_scene_npc_config(path: str | Path) -> SceneNpcConfig:
    """Load a strict v1 compatibility config or authoritative-plan v2 config."""
    source = Path(path).resolve()
    payload = load_strict_json(source)
    schema = payload.get("schema")
    is_v2 = schema == "scene_npc_config/v2"
    raw_plan: dict[str, Any] | None = None
    raw_robot_navigation: dict[str, Any] | None = None
    if is_v2:
        v2_required_fields = {
            "schema",
            "scene",
            "semantic_registration",
            "population",
            "route_coverage",
            "routes",
            "traffic",
            "runtime",
            "interaction_plan",
            "robot_navigation",
        }
        v2_allowed_fields = v2_required_fields | {"spawn_policy"}
        if set(payload) - v2_allowed_fields or v2_required_fields - set(payload):
            raise SceneConfigError("scene_config_v2_fields_are_not_exact")
        raw_plan = _require_mapping(payload["interaction_plan"], "interaction_plan")
        if set(raw_plan) != {"path", "required_capabilities"}:
            raise SceneConfigError("interaction_plan_fields_are_not_exact")
        capabilities = raw_plan["required_capabilities"]
        allowed_capabilities = {"conversation", "handover", "sit", "work"}
        if (
            not isinstance(capabilities, list)
            or not capabilities
            or not all(isinstance(item, str) and item for item in capabilities)
            or len(set(capabilities)) != len(capabilities)
            or set(capabilities) - allowed_capabilities
        ):
            raise SceneConfigError("interaction_plan_required_capabilities_invalid")
        raw_robot_navigation = _require_mapping(
            payload["robot_navigation"], "robot_navigation"
        )
        if set(raw_robot_navigation) != {"footprint_radius", "clearance", "resolution"}:
            raise SceneConfigError("robot_navigation_fields_are_not_exact")
        payload = dict(payload)
        payload.pop("interaction_plan")
        payload.pop("robot_navigation")
        payload["schema"] = "scene_npc_config/v1"
        payload["interactions"] = {
            "conversation": {
                "status": "unsupported",
                "reason": "v2_candidate_requires_physical_acceptance_receipt",
            },
            "robot_handover": {
                "status": "unsupported",
                "reason": "v2_candidate_requires_physical_acceptance_receipt",
            },
        }
    allowed = {
        "schema",
        "scene",
        "semantic_registration",
        "population",
        "route_coverage",
        "routes",
        "traffic",
        "interactions",
        "runtime",
        "spawn_policy",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise SceneConfigError("unknown_root_fields:" + ",".join(sorted(unknown)))
    if payload.get("schema") != "scene_npc_config/v1":
        raise SceneConfigError("scene_config_schema_must_be_scene_npc_config_v1")
    scene = _require_mapping(payload.get("scene"), "scene")
    if set(scene) != {
        "id",
        "kind",
        "source_mjcf",
        "source_manifest",
        "semantic_policy",
        "npc_catalog",
        "navigation",
    }:
        raise SceneConfigError("scene_fields_are_not_exact")
    scene_id, kind = scene["id"], scene["kind"]
    if not isinstance(scene_id, str) or not scene_id or kind not in {"office", "home"}:
        raise SceneConfigError("scene_id_or_kind_invalid")
    navigation = _require_mapping(scene["navigation"], "navigation")
    if set(navigation) - {
        "surface",
        "planner",
        "agent_radius",
        "clearance",
        "resolution",
    }:
        raise SceneConfigError("navigation_unknown_fields")
    for key in ("surface", "planner"):
        if not isinstance(navigation.get(key), str) or not navigation[key]:
            raise SceneConfigError(f"navigation_{key}_invalid")
    for key in ("agent_radius", "clearance"):
        value = navigation.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise SceneConfigError(f"navigation_{key}_invalid")
    resolution = navigation.get("resolution", 0.08)
    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, (int, float))
        or not math.isfinite(resolution)
        or not 0.01 <= resolution <= 0.2
    ):
        raise SceneConfigError("navigation_resolution_invalid")
    navigation = dict(navigation, resolution=float(resolution))
    registration = _require_mapping(payload.get("semantic_registration"), "semantic_registration")
    if set(registration) - {
        "strict_coverage",
        "discover",
        "region_overrides",
        "entity_overrides",
        "custom_targets",
    }:
        raise SceneConfigError("semantic_registration_unknown_fields")
    if registration.get("strict_coverage") is not True:
        raise SceneConfigError("strict_coverage_must_be_true")
    discover = _require_mapping(registration.get("discover"), "discover")
    if set(discover) != {"manifest_assets", "manifest_zones", "xml_semantic_sites"} or not all(
        discover.values()
    ):
        raise SceneConfigError("discover_must_enable_all_sources")
    entity_overrides = _require_mapping(
        registration.get("entity_overrides", {}), "entity_overrides"
    )
    region_overrides = _require_mapping(
        registration.get("region_overrides", {}), "region_overrides"
    )
    custom_targets = _require_mapping(registration.get("custom_targets", {}), "custom_targets")
    allowed_override_fields = {
        "semantic_class",
        "affordances",
        "navigation_requirement",
        "point_bundle",
        "explicit_point",
        "action_navigation_site",
        "exemption",
    }
    allowed_point_bundles = {
        "region_samples",
        "perimeter_approach",
        "observation",
        "seat",
        "graspable_on_support",
        "door_portal",
        "conversation_pair",
        "handover_pair",
        "robot_service",
    }
    for entity_id, override in entity_overrides.items():
        definition = _require_mapping(override, f"entity_override:{entity_id}")
        unknown_override_fields = set(definition) - allowed_override_fields
        if unknown_override_fields:
            raise SceneConfigError(
                f"entity_override_unknown_fields:{entity_id}:"
                + ",".join(sorted(unknown_override_fields))
            )
        requirement = definition.get("navigation_requirement")
        if requirement not in {"region", "approach", "observe", "none"}:
            raise SceneConfigError(f"entity_override_requirement_invalid:{entity_id}")
        semantic_class = definition.get("semantic_class")
        if not isinstance(semantic_class, str) or not semantic_class:
            raise SceneConfigError(f"entity_override_semantic_class_invalid:{entity_id}")
        affordances = definition.get("affordances")
        if (
            not isinstance(affordances, list)
            or not all(isinstance(value, str) and value for value in affordances)
            or len(set(affordances)) != len(affordances)
        ):
            raise SceneConfigError(f"entity_override_affordances_invalid:{entity_id}")
        point_bundle = definition.get("point_bundle")
        if requirement != "none" and point_bundle not in allowed_point_bundles:
            raise SceneConfigError(f"entity_override_point_bundle_invalid:{entity_id}")
        if "explicit_point" in definition:
            _finite_point(definition["explicit_point"], f"entity_override_point:{entity_id}")
        if requirement == "none":
            if point_bundle is not None or "explicit_point" in definition:
                raise SceneConfigError(f"entity_override_exemption_has_navigation:{entity_id}")
            exemption = _require_mapping(
                definition.get("exemption"), f"entity_override_exemption:{entity_id}"
            )
            if (
                set(exemption) != {"reason"}
                or not isinstance(exemption["reason"], str)
                or not exemption["reason"]
            ):
                raise SceneConfigError(f"entity_override_exemption_invalid:{entity_id}")
        elif "exemption" in definition:
            raise SceneConfigError(f"entity_override_unexpected_exemption:{entity_id}")
    for point_id, raw_point in custom_targets.items():
        if not isinstance(point_id, str) or not point_id.startswith("point."):
            raise SceneConfigError(f"custom_target_id_invalid:{point_id}")
        point = _require_mapping(raw_point, f"custom_target:{point_id}")
        if set(point) != {"owner", "position", "yaw", "usages"}:
            raise SceneConfigError(f"custom_target_fields_invalid:{point_id}")
        if not isinstance(point["owner"], str) or not point["owner"]:
            raise SceneConfigError(f"custom_target_owner_invalid:{point_id}")
        _finite_point(point["position"], f"custom_target_position:{point_id}")
        yaw = point["yaw"]
        if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
            raise SceneConfigError(f"custom_target_yaw_invalid:{point_id}")
        usages = point["usages"]
        if (
            not isinstance(usages, list)
            or not usages
            or not all(isinstance(value, str) and value for value in usages)
            or len(set(usages)) != len(usages)
        ):
            raise SceneConfigError(f"custom_target_usages_invalid:{point_id}")
    for region_id, override in region_overrides.items():
        if not isinstance(region_id, str) or not region_id.startswith(("room.", "zone.")):
            raise SceneConfigError("region_override_id_invalid")
        definition = _require_mapping(override, f"region_override:{region_id}")
        if (
            set(definition) != {"semantic_class", "geometry", "required"}
            or definition["required"] is not True
        ):
            raise SceneConfigError(f"region_override_invalid:{region_id}")
        if not isinstance(definition["semantic_class"], str) or not definition[
            "semantic_class"
        ].startswith("region."):
            raise SceneConfigError(f"region_override_semantic_class_invalid:{region_id}")
        geometry = _require_mapping(definition["geometry"], f"region_geometry:{region_id}")
        if geometry.get("mode") != "seed_and_component" or set(geometry) != {"mode", "seed"}:
            raise SceneConfigError(f"region_geometry_invalid:{region_id}")
        _finite_point(geometry["seed"], f"region_geometry_seed:{region_id}")
    raw_spawn_policy_raw = payload.get("spawn_policy")
    if raw_spawn_policy_raw is None:
        spawn_policy = None
    else:
        raw_spawn_policy = _require_mapping(raw_spawn_policy_raw, "spawn_policy")
        if set(raw_spawn_policy) != {"capacity", "minimum_separation_m", "allocation", "anchors"}:
            raise SceneConfigError("spawn_policy_fields_invalid")
        capacity = raw_spawn_policy["capacity"]
        minimum_separation = raw_spawn_policy["minimum_separation_m"]
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise SceneConfigError("spawn_policy_capacity_invalid")
        if isinstance(minimum_separation, bool) or not isinstance(minimum_separation, (int, float)) or not math.isfinite(float(minimum_separation)) or float(minimum_separation) <= 0:
            raise SceneConfigError("spawn_policy_minimum_separation_invalid")
        anchors = raw_spawn_policy["anchors"]
        if not isinstance(anchors, list) or len(anchors) != capacity or len(set(anchors)) != len(anchors) or not all(isinstance(item, str) and item.startswith("point.") for item in anchors):
            raise SceneConfigError("spawn_policy_anchors_invalid")
        spawn_policy = {"capacity": capacity, "minimum_separation_m": float(minimum_separation), "allocation": raw_spawn_policy["allocation"], "anchors": list(anchors)}
        for anchor in spawn_policy["anchors"]:
            point = custom_targets.get(anchor)
            if point is None or "spawn" not in point["usages"] or "spawn_anchor" not in point["usages"]:
                raise SceneConfigError(f"spawn_policy_anchor_unbound:{anchor}")
    raw_population = _require_mapping(payload.get("population"), "population")
    if set(raw_population) != {"members"} or not isinstance(raw_population["members"], list):
        raise SceneConfigError("population_members_required")
    members: list[PopulationMember] = []
    for index, item in enumerate(raw_population["members"]):
        entry = _require_mapping(item, f"population_member:{index}")
        if set(entry) != {"npc", "spawn", "initial_region"}:
            raise SceneConfigError(f"population_member_fields_invalid:{index}")
        if not all(isinstance(entry[key], str) and entry[key] for key in entry):
            raise SceneConfigError(f"population_member_values_invalid:{index}")
        members.append(PopulationMember(entry["npc"], entry["spawn"], entry["initial_region"]))
    if not members or len({member.npc for member in members}) != len(members):
        raise SceneConfigError("population_members_must_be_unique_and_nonempty")
    if len({member.spawn for member in members}) != len(members):
        raise SceneConfigError("population_spawn_points_must_be_unique")
    coverage = _require_mapping(payload.get("route_coverage"), "route_coverage")
    expected_coverage = {
        "origins": "all_population_spawns",
        "targets": "all_required_navigation_points",
        "connectivity": "strongly_connected",
        "preflight": "required",
    }
    if coverage != expected_coverage:
        raise SceneConfigError("route_coverage_contract_invalid")
    raw_routes = payload.get("routes", [])
    if not isinstance(raw_routes, list):
        raise SceneConfigError("routes_must_be_array")
    routes: list[dict[str, Any]] = []
    route_ids: set[str] = set()
    for index, raw_route in enumerate(raw_routes):
        route = _require_mapping(raw_route, f"route:{index}")
        if set(route) != {"id", "from", "to", "mode", "actions"}:
            raise SceneConfigError(f"route_fields_invalid:{index}")
        if not all(isinstance(route[key], str) and route[key] for key in ("id", "from", "to")):
            raise SceneConfigError(f"route_endpoints_invalid:{index}")
        if route["id"] in route_ids or route["from"] == route["to"]:
            raise SceneConfigError(f"route_identity_invalid:{route['id']}")
        if route["mode"] not in {"fixed_contract", "audited_dynamic", "optional"}:
            raise SceneConfigError(f"route_mode_invalid:{route['id']}")
        actions = route["actions"]
        if (
            not isinstance(actions, list)
            or not actions
            or not all(isinstance(action, str) and action for action in actions)
        ):
            raise SceneConfigError(f"route_actions_invalid:{route['id']}")
        route_ids.add(route["id"])
        routes.append(dict(route))
    traffic = _require_mapping(payload.get("traffic"), "traffic")
    if set(traffic) != {
        "policy",
        "dynamic_obstacles",
        "on_block",
        "max_wait_s",
        "reservation_horizon_s",
        "no_direct_fallback",
    }:
        raise SceneConfigError("traffic_fields_are_not_exact")
    if traffic["policy"] != "sequential_route_reservation":
        raise SceneConfigError("traffic_policy_invalid")
    if not isinstance(traffic["dynamic_obstacles"], bool):
        raise SceneConfigError("traffic_dynamic_obstacles_invalid")
    if traffic["on_block"] not in {"wait_then_replan", "retry_then_fail"}:
        raise SceneConfigError("traffic_on_block_invalid")
    for key in ("max_wait_s", "reservation_horizon_s"):
        value = traffic[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SceneConfigError(f"traffic_{key}_invalid")
        if not math.isfinite(value) or value < 0:
            raise SceneConfigError(f"traffic_{key}_invalid")
    if traffic["reservation_horizon_s"] <= 0:
        raise SceneConfigError("traffic_reservation_horizon_invalid")
    if traffic["no_direct_fallback"] is not True:
        raise SceneConfigError("traffic_no_direct_fallback_must_be_true")
    interactions = _require_mapping(payload.get("interactions", {}), "interactions")
    if set(interactions) != {"conversation", "robot_handover"}:
        raise SceneConfigError("interaction_capabilities_must_be_explicit")
    for interaction_id, raw_interaction in interactions.items():
        interaction = _require_mapping(raw_interaction, f"interaction:{interaction_id}")
        status = interaction.get("status")
        if status == "unsupported":
            if set(interaction) != {"status", "reason"}:
                raise SceneConfigError(f"interaction_unsupported_fields:{interaction_id}")
            if not isinstance(interaction["reason"], str) or not interaction["reason"]:
                raise SceneConfigError(f"interaction_unsupported_reason_invalid:{interaction_id}")
            continue
        if status != "supported":
            raise SceneConfigError(f"interaction_status_invalid:{interaction_id}")
        if interaction_id == "conversation":
            if set(interaction) != {"status", "sites"}:
                raise SceneConfigError("conversation_supported_fields_invalid")
            sites = interaction["sites"]
            if (
                not isinstance(sites, list)
                or len(sites) < 2
                or len(set(sites)) != len(sites)
                or not all(isinstance(site, str) and site for site in sites)
            ):
                raise SceneConfigError("conversation_supported_sites_invalid")
        else:
            if set(interaction) != {"status", "sites", "object", "robot_approach"}:
                raise SceneConfigError("robot_handover_supported_fields_invalid")
            sites = interaction["sites"]
            if (
                not isinstance(sites, list)
                or len(sites) < 2
                or len(set(sites)) != len(sites)
                or not all(isinstance(site, str) and site for site in sites)
            ):
                raise SceneConfigError("robot_handover_supported_sites_invalid")
            for key in ("object", "robot_approach"):
                if not isinstance(interaction[key], str) or not interaction[key]:
                    raise SceneConfigError(f"robot_handover_{key}_invalid")
        if "reason" in interaction:
            raise SceneConfigError(f"interaction_unknown_fields:{interaction_id}")
    runtime = _require_mapping(payload.get("runtime", {}), "runtime")
    if set(runtime) != {"max_replans", "route_failure"}:
        raise SceneConfigError("runtime_fields_invalid")
    if (
        not isinstance(runtime["max_replans"], int)
        or isinstance(runtime["max_replans"], bool)
        or runtime["max_replans"] < 0
    ):
        raise SceneConfigError("runtime_max_replans_invalid")
    if runtime["route_failure"] != "fail":
        raise SceneConfigError("runtime_route_failure_invalid")

    def resolve(relative: Any, key: str) -> Path:
        if not isinstance(relative, str) or not relative:
            raise SceneConfigError(f"scene_{key}_invalid")
        return (source.parent / relative).resolve()

    interaction_plan = None
    robot_navigation = None
    if is_v2:
        assert raw_plan is not None and raw_robot_navigation is not None
        plan_path = raw_plan["path"]
        if not isinstance(plan_path, str) or not plan_path:
            raise SceneConfigError("interaction_plan_path_invalid")
        interaction_plan = SceneInteractionPlanRef(
            path=(source.parent / plan_path).resolve(),
            required_capabilities=frozenset(raw_plan["required_capabilities"]),
        )
        values: dict[str, float] = {}
        for key in ("footprint_radius", "clearance", "resolution"):
            value = raw_robot_navigation[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise SceneConfigError(f"robot_navigation_{key}_invalid")
            values[key] = float(value)
        robot_navigation = RobotNavigationConfig(**values)

    return SceneNpcConfig(
        path=source,
        scene_id=scene_id,
        kind=kind,
        source_mjcf=resolve(scene["source_mjcf"], "source_mjcf"),
        source_manifest=resolve(scene["source_manifest"], "source_manifest"),
        semantic_policy=resolve(scene["semantic_policy"], "semantic_policy"),
        npc_catalog=resolve(scene["npc_catalog"], "npc_catalog"),
        navigation=navigation,
        registration=registration,
        population=tuple(members),
        spawn_policy=spawn_policy,
        route_coverage=coverage,
        entity_overrides=entity_overrides,
        region_overrides=region_overrides,
        custom_targets=custom_targets,
        routes=tuple(routes),
        traffic=dict(traffic),
        interactions={key: dict(value) for key, value in interactions.items()},
        runtime=runtime,
        schema="scene_npc_config/v2" if is_v2 else "scene_npc_config/v1",
        interaction_plan=interaction_plan,
        robot_navigation=robot_navigation,
    )
