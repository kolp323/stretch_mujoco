"""Private candidate compiler for ``scene_npc_config/v2``."""

from __future__ import annotations

from copy import deepcopy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

from stretch_mujoco.humanoid.navigation import NavigationPathError, OfficeNavigationMesh
from stretch_mujoco.semantics import SemanticWorld
from stretch_mujoco.semantics.scene_discovery import compile_semantics

from .interaction_projection import project_interaction_plan
from .interaction_station import SceneInteractionPlan, load_interaction_site_plan
from .scene_compiler import _append_sites, _sha256, _source_sites, _write_json
from .scene_config import SceneNpcConfig
from .schema import NpcPopulation
from .trajectory_profile import NpcTrajectoryProfile


VALIDATOR_VERSION = "scene_compiler_phase1b2c/v1"


def _validate_required_capabilities(
    config: SceneNpcConfig, plan: SceneInteractionPlan
) -> None:
    available = {
        "conversation": bool(plan.conversation_stations),
        "handover": bool(plan.handover_stations),
        "sit": bool(plan.seat_slots),
        "work": bool(plan.workstations),
    }
    assert config.interaction_plan is not None
    for capability in sorted(config.interaction_plan.required_capabilities):
        if not available[capability]:
            raise ValueError(f"required_capability_missing:{capability}")


def _targeted_merge(target: dict[str, Any], patch: Mapping[str, Any]) -> None:
    """Merge projection-owned fields without discarding discovered metadata."""
    for key, value in patch.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _targeted_merge(current, value)
        elif isinstance(current, list) and isinstance(value, list):
            current.extend(item for item in value if item not in current)
        else:
            target[key] = deepcopy(value)


def _entity_aliases(base: Mapping[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for entity_id, definition in base["entities"].items():
        source = definition.get("source", {})
        binding = source.get("xml_binding", {})
        candidates = {
            entity_id,
            source.get("id"),
            binding.get("name"),
        }
        if isinstance(binding.get("name"), str):
            candidates.add(f"object.{binding['name']}")
        if isinstance(source.get("id"), str):
            candidates.add(f"object.{source['id']}")
        for alias in candidates:
            if not isinstance(alias, str) or not alias:
                continue
            previous = aliases.get(alias)
            if previous is not None and previous != entity_id:
                raise ValueError(f"semantic_entity_alias_conflict:{alias}")
            aliases[alias] = entity_id
    return aliases


def _resolve_projection_semantics(base: Mapping[str, Any], projection):
    aliases = _entity_aliases(base)
    new_entity_ids = set(projection.semantic_entities)
    for entity_id in sorted(new_entity_ids):
        if entity_id in aliases:
            raise ValueError(f"interaction_entity_id_conflict:{entity_id}")

    def resolve(reference: str, *, required: bool = True) -> str:
        if reference in new_entity_ids:
            return reference
        resolved = aliases.get(reference)
        if resolved is not None:
            return resolved
        if required:
            raise ValueError(f"interaction_semantic_reference_unbound:{reference}")
        return reference

    updates: dict[str, dict[str, Any]] = {}
    for authored_id, patch in projection.semantic_entity_updates.items():
        canonical_id = resolve(authored_id)
        if canonical_id in updates:
            raise ValueError(f"interaction_entity_update_conflict:{canonical_id}")
        updates[canonical_id] = deepcopy(patch)

    points: dict[str, dict[str, Any]] = {}
    for point_id, raw in projection.semantic_points.items():
        point = deepcopy(raw)
        point["owner"] = resolve(point["owner"])
        point["target"] = resolve(point["target"], required=False)
        points[point_id] = point

    relations = []
    for raw in projection.semantic_relations:
        relation = dict(raw)
        relation["subject"] = resolve(relation["subject"])
        relation["object"] = resolve(relation["object"])
        relations.append(relation)
    return projection.semantic_entities, updates, points, tuple(relations)


def _merge_semantic_v2(
    base: dict[str, Any],
    *,
    new_entities: Mapping[str, Mapping[str, Any]],
    entity_updates: Mapping[str, Mapping[str, Any]],
    new_points: Mapping[str, Mapping[str, Any]],
    new_relations: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    merged = deepcopy(base)
    entities = merged.setdefault("entities", {})
    points = merged.setdefault("points", {})
    relations = merged.setdefault("relations", [])
    if (
        not isinstance(entities, dict)
        or not isinstance(points, dict)
        or not isinstance(relations, list)
    ):
        raise ValueError("base_semantic_v2_shape_invalid")

    conflicts = sorted(set(entities) & set(new_entities))
    if conflicts:
        raise ValueError(f"interaction_entity_id_conflict:{conflicts[0]}")
    for entity_id, definition in new_entities.items():
        entities[entity_id] = deepcopy(definition)
    for entity_id, patch in entity_updates.items():
        if entity_id not in entities:
            raise ValueError(f"interaction_entity_update_unbound:{entity_id}")
        _targeted_merge(entities[entity_id], patch)

    conflicts = sorted(set(points) & set(new_points))
    if conflicts:
        raise ValueError(f"interaction_point_id_conflict:{conflicts[0]}")
    for point_id, definition in new_points.items():
        points[point_id] = deepcopy(definition)

    seen: dict[tuple[str, str], str] = {}
    for relation in [*relations, *new_relations]:
        if not isinstance(relation, dict) or set(relation) != {
            "subject",
            "relation",
            "object",
        }:
            raise ValueError("semantic_relation_shape_invalid")
        key = (relation["subject"], relation["relation"])
        previous = seen.get(key)
        if previous is not None:
            error = (
                "duplicate_semantic_relation"
                if previous == relation["object"]
                else "conflicting_semantic_relation"
            )
            raise ValueError(f"{error}:{relation['subject']}:{relation['relation']}")
        seen[key] = relation["object"]
    relations.extend(deepcopy(new_relations))
    return merged


def _population_v3(
    config: SceneNpcConfig,
    *,
    scene_name: str,
    trajectory_name: str,
    base_sites: list[dict[str, Any]],
    interaction_stations: Mapping[str, Any],
    output_parent: Path,
) -> dict[str, Any]:
    catalog_path = config.npc_catalog
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    point_by_id = {point["id"]: point for point in base_sites}
    selected: dict[str, Any] = {}
    for member in config.population:
        definition = catalog.get("npcs", {}).get(member.npc)
        if definition is None:
            raise ValueError(f"npc_catalog_member_missing:{member.npc}")
        if member.spawn not in point_by_id:
            raise ValueError(f"population_spawn_unbound:{member.npc}:{member.spawn}")
        spawn = point_by_id[member.spawn]
        selected[member.npc] = dict(
            definition,
            spawn={
                "location": member.initial_region,
                "site": spawn["site"],
                "yaw": float(spawn["yaw"]),
            },
        )

    population = {
        key: deepcopy(value)
        for key, value in catalog.items()
        if key not in {"npcs", "interaction_templates", "trajectory_profile"}
    }
    population["schema_version"] = 3
    for resource_key in ("asset_manifest", "appearance_catalog"):
        resource = catalog.get(resource_key)
        if isinstance(resource, str):
            population[resource_key] = os.path.relpath(
                (catalog_path.parent / resource).resolve(), output_parent
            )
    assert config.interaction_plan is not None
    population.update(
        {
            "scene": scene_name,
            "trajectory_profile": trajectory_name,
            "traffic_policy": dict(config.traffic),
            "interaction_capabilities": {
                "required": sorted(config.interaction_plan.required_capabilities),
                "production_evidence": False,
            },
            "interaction_stations": deepcopy(interaction_stations),
            "npcs": selected,
        }
    )
    if config.spawn_policy is not None:
        population["spawn_policy"] = dict(config.spawn_policy)
    return population


def _trajectory_profile(
    config: SceneNpcConfig,
    *,
    scene_name: str,
    scene_sha256: str,
    semantic_v2: Mapping[str, Any],
    base_sites: list[dict[str, Any]],
) -> dict[str, Any]:
    navigation_sites = [
        point for point in base_sites if point.get("kind", "navigation") == "navigation"
    ]
    anchors = {
        point["id"]: {"site": point["site"], "role": "semantic_navigation"}
        for point in navigation_sites
    }

    def resolve_endpoint(endpoint: str) -> str:
        if endpoint in anchors:
            return endpoint
        entity = semantic_v2["entities"].get(endpoint)
        if entity is None:
            raise ValueError(f"route_endpoint_unbound:{endpoint}")
        candidates = entity["points"]["navigation"]
        if not candidates:
            raise ValueError(f"route_endpoint_has_no_navigation_point:{endpoint}")
        return sorted(candidates)[0]

    business_routes = [
        {
            "route_id": f"business_{route['id']}",
            "from": resolve_endpoint(route["from"]),
            "to": resolve_endpoint(route["to"]),
            "actions": list(route["actions"]),
            "mode": route["mode"],
        }
        for route in config.routes
    ]
    if any(route["from"] == route["to"] for route in business_routes):
        raise ValueError("business_route_resolves_to_same_navigation_point")
    coverage_routes = [
        {
            "route_id": f"coverage_{origin_index:02d}_{target_index:04d}",
            "from": member.spawn,
            "to": point["id"],
            "actions": ["move_to"],
        }
        for origin_index, member in enumerate(config.population)
        for target_index, point in enumerate(navigation_sites)
        if point["id"] != member.spawn
    ]
    routes = business_routes + coverage_routes
    if not routes:
        raise ValueError("trajectory_profile_requires_distinct_navigation_points")
    manifest = json.loads(config.source_manifest.read_text(encoding="utf-8"))
    robot = manifest.get("robot", {})
    robot_body = robot.get("body") if isinstance(robot, dict) else None
    return {
        "schema_version": 1,
        "profile_id": config.scene_id + "_candidate_coverage",
        "scene": scene_name,
        "scene_sha256": scene_sha256,
        "navigation": {
            "surface": config.navigation["surface"],
            "agent_radius": config.navigation["agent_radius"],
            "clearance": config.navigation["clearance"],
            "resolution": config.navigation["resolution"],
            "exclude_body_roots": (
                [robot_body] if isinstance(robot_body, str) and robot_body else []
            ),
        },
        "anchors": anchors,
        "routes": routes,
    }


def _interaction_navigation_targets(plan: SceneInteractionPlan, projection):
    sites = {site["id"]: site for site in projection.derived_sites}
    npc_targets: list[dict[str, Any]] = []
    robot_targets: list[dict[str, Any]] = []
    npc_index: dict[tuple[str, str], dict[str, Any]] = {}
    robot_index: dict[tuple[str, str], dict[str, Any]] = {}

    def add_target(
        collection: list[dict[str, Any]],
        index: dict[tuple[str, str], dict[str, Any]],
        point_id: str,
        usage: str,
        consumer: str,
    ) -> None:
        site = sites[point_id]["site"]
        key = (point_id, site)
        existing = index.get(key)
        if existing is not None:
            if usage not in existing["usages"]:
                existing["usages"].append(usage)
            if consumer not in existing["consumers"]:
                existing["consumers"].append(consumer)
            return
        definition = {
            "target": point_id,
            "point": point_id,
            "site": site,
            "role": usage,
            "usages": [usage],
            "consumers": [consumer],
        }
        index[key] = definition
        collection.append(definition)

    for station in plan.conversation_stations:
        for role in ("speaker", "listener"):
            point_id = f"point.{station.station_id}.{role}"
            add_target(
                npc_targets,
                npc_index,
                point_id,
                f"conversation_{role}",
                station.station_id,
            )
            if "robot_npc" in station.allowed_actor_pairs:
                add_target(
                    robot_targets,
                    robot_index,
                    point_id,
                    f"conversation_{role}",
                    station.station_id,
                )

    for station in plan.handover_stations:
        npc_roles: set[str] = set()
        if "npc_to_npc" in station.modes:
            npc_roles.update(("giver", "receiver"))
        if "robot_to_npc" in station.modes:
            npc_roles.add("receiver")
        if "npc_to_robot" in station.modes:
            npc_roles.add("giver")
        for role in ("giver", "receiver"):
            if role in npc_roles:
                point_id = f"point.{station.station_id}.{role}"
                add_target(
                    npc_targets,
                    npc_index,
                    point_id,
                    f"handover_{role}",
                    station.station_id,
                )
        if set(station.modes) & {"robot_to_npc", "npc_to_robot"}:
            point_id = f"point.{station.station_id}.robot"
            add_target(
                robot_targets,
                robot_index,
                point_id,
                "handover_robot",
                station.station_id,
            )

    seat_by_id = {slot.slot_id: slot for slot in plan.seat_slots}
    for slot in plan.seat_slots:
        point_id = f"point.{slot.slot_id}.ingress"
        add_target(
            npc_targets,
            npc_index,
            point_id,
            "seat_ingress",
            slot.slot_id,
        )
    for workstation in plan.workstations:
        slot = seat_by_id[workstation.seat_slot]
        point_id = f"point.{slot.slot_id}.ingress"
        add_target(
            npc_targets,
            npc_index,
            point_id,
            "workstation_seat_ingress",
            workstation.binding_id,
        )
    return npc_targets, robot_targets


def _round_xy(value: np.ndarray) -> list[float]:
    return [round(float(value[0]), 9), round(float(value[1]), 9)]


def _path_length(path: tuple[np.ndarray, ...]) -> float:
    return round(
        sum(
            float(np.linalg.norm(path[index] - path[index - 1]))
            for index in range(1, len(path))
        ),
        9,
    )


def _mesh_evidence(
    mesh: OfficeNavigationMesh,
    *,
    surface: str,
    radius: float,
    clearance: float,
) -> dict[str, Any]:
    return {
        "surface": surface,
        "footprint": {
            "radius": radius,
            "clearance": clearance,
            "effective_radius": radius + clearance,
            "resolution": mesh.resolution,
        },
        "primary_component": {
            "id": int(mesh.primary_component_id),
            "cell_count": int(mesh.component_sizes[mesh.primary_component_id]),
            "component_count": len(mesh.component_sizes),
            "bounds": [
                round(mesh.x_min, 9),
                round(mesh.x_max, 9),
                round(mesh.y_min, 9),
                round(mesh.y_max, 9),
            ],
        },
    }


def _build_mesh(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    surface: str,
    resolution: float,
    effective_radius: float,
    robot_body: str,
    actor: str,
) -> OfficeNavigationMesh:
    try:
        return OfficeNavigationMesh.from_model(
            model,
            data,
            resolution=resolution,
            agent_radius=effective_radius,
            floor_geom_name=surface,
            exclude_body_roots=(robot_body,),
        )
    except (NavigationPathError, ValueError) as error:
        raise ValueError(f"{actor}_navigation_mesh_unavailable") from error


def _site_xy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site: str,
    *,
    actor: str,
    target: str,
) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if site_id < 0:
        raise ValueError(f"{actor}_navigation_site_unbound:{target}:{site}")
    return np.asarray(data.site_xpos[site_id, :2], dtype=float)


def _preflight_navigation(
    config: SceneNpcConfig,
    plan: SceneInteractionPlan,
    projection,
    base_sites: list[dict[str, Any]],
    scene_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert config.robot_navigation is not None
    manifest = json.loads(config.source_manifest.read_text(encoding="utf-8"))
    robot = manifest.get("robot", {})
    robot_body = robot.get("body") if isinstance(robot, dict) else None
    if not isinstance(robot_body, str) or not robot_body:
        raise ValueError("robot_navigation_body_unbound")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    surface = str(config.navigation["surface"])
    npc_mesh = _build_mesh(
        model,
        data,
        surface=surface,
        resolution=float(config.navigation["resolution"]),
        effective_radius=float(config.navigation["agent_radius"])
        + float(config.navigation["clearance"]),
        robot_body=robot_body,
        actor="npc",
    )
    npc_targets, robot_targets = _interaction_navigation_targets(plan, projection)
    base_site_by_id = {site["id"]: site["site"] for site in base_sites}

    npc_origins = []
    npc_routes = []
    resolved_npc_targets = [
        dict(
            definition,
            xy=_round_xy(
                _site_xy(
                    model,
                    data,
                    definition["site"],
                    actor="npc",
                    target=definition["target"],
                )
            ),
        )
        for definition in npc_targets
    ]
    for definition in resolved_npc_targets:
        target_xy = np.asarray(definition["xy"], dtype=float)
        if not npc_mesh.is_world_free(
            target_xy, component_id=npc_mesh.primary_component_id
        ):
            raise ValueError(f"npc_navigation_target_unavailable:{definition['target']}")
    for member in config.population:
        origin_site = base_site_by_id[member.spawn]
        origin_xy = _site_xy(
            model,
            data,
            origin_site,
            actor="npc",
            target=member.spawn,
        )
        if not npc_mesh.is_world_free(
            origin_xy, component_id=npc_mesh.primary_component_id
        ):
            raise ValueError(f"npc_navigation_origin_unavailable:{member.npc}:{member.spawn}")
        npc_origins.append(
            {
                "npc": member.npc,
                "origin": member.spawn,
                "site": origin_site,
                "xy": _round_xy(origin_xy),
            }
        )
        for target in resolved_npc_targets:
            try:
                path = npc_mesh.plan(origin_xy, np.asarray(target["xy"], dtype=float))
            except NavigationPathError as error:
                raise ValueError(
                    f"npc_navigation_route_unavailable:{member.spawn}:{target['target']}"
                ) from error
            npc_routes.append(
                {
                    "origin": member.spawn,
                    "target": target["target"],
                    "site": target["site"],
                    "path_length_m": _path_length(path),
                }
            )

    robot_mesh = _build_mesh(
        model,
        data,
        surface=surface,
        resolution=float(config.robot_navigation.resolution),
        effective_radius=float(config.robot_navigation.footprint_radius)
        + float(config.robot_navigation.clearance),
        robot_body=robot_body,
        actor="robot",
    )
    robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body)
    if robot_body_id < 0:
        raise ValueError(f"robot_navigation_body_unbound:{robot_body}")
    robot_origin = np.asarray(data.xpos[robot_body_id, :2], dtype=float)
    if not robot_mesh.is_world_free(
        robot_origin, component_id=robot_mesh.primary_component_id
    ):
        raise ValueError(f"robot_navigation_origin_unavailable:{robot_body}")
    resolved_robot_targets = [
        dict(
            definition,
            xy=_round_xy(
                _site_xy(
                    model,
                    data,
                    definition["site"],
                    actor="robot",
                    target=definition["target"],
                )
            ),
        )
        for definition in robot_targets
    ]
    robot_routes = []
    for target in resolved_robot_targets:
        target_xy = np.asarray(target["xy"], dtype=float)
        if not robot_mesh.is_world_free(
            target_xy, component_id=robot_mesh.primary_component_id
        ):
            raise ValueError(f"robot_navigation_target_unavailable:{target['target']}")
        try:
            path = robot_mesh.plan(robot_origin, target_xy)
        except NavigationPathError as error:
            raise ValueError(
                f"robot_navigation_route_unavailable:{robot_body}:{target['target']}"
            ) from error
        robot_routes.append(
            {
                "origin": robot_body,
                "target": target["target"],
                "site": target["site"],
                "path_length_m": _path_length(path),
            }
        )

    npc_evidence = {
        "status": "passed",
        "planner": str(config.navigation["planner"]),
        **_mesh_evidence(
            npc_mesh,
            surface=surface,
            radius=float(config.navigation["agent_radius"]),
            clearance=float(config.navigation["clearance"]),
        ),
        "origins": npc_origins,
        "targets": resolved_npc_targets,
        "routes": npc_routes,
    }
    robot_evidence = {
        "status": "passed",
        "planner": str(config.navigation["planner"]),
        **_mesh_evidence(
            robot_mesh,
            surface=surface,
            radius=float(config.robot_navigation.footprint_radius),
            clearance=float(config.robot_navigation.clearance),
        ),
        "origin": {"body": robot_body, "xy": _round_xy(robot_origin)},
        "targets": resolved_robot_targets,
        "routes": robot_routes,
    }
    return npc_evidence, robot_evidence


def compile_scene_npc_config_v2(
    config: SceneNpcConfig,
    output: str | Path,
    *,
    check_navigation: bool = True,
) -> dict[str, Path]:
    """Atomically emit a non-production v2 candidate without physical preflight."""
    if config.interaction_plan is None or config.robot_navigation is None:
        raise ValueError("scene_config_v2_requires_plan_and_robot_navigation")
    plan = load_interaction_site_plan(
        config.interaction_plan.path,
        source_manifest=config.source_manifest,
        source_mjcf=config.source_mjcf,
        expected_scene_id=config.scene_id,
    )
    _validate_required_capabilities(config, plan)
    projection = project_interaction_plan(plan)
    derived_names = {site["site"] for site in projection.derived_sites}
    source_conflicts = sorted(derived_names & _source_sites(config.source_mjcf))
    if source_conflicts:
        raise ValueError(f"interaction_source_site_conflict:{source_conflicts[0]}")

    base_v2, base_v1, coverage, base_sites = compile_semantics(
        config, check_navigation=check_navigation
    )
    base_site_conflicts = sorted(
        derived_names & {site["site"] for site in base_sites}
    )
    if base_site_conflicts:
        raise ValueError(f"interaction_base_site_conflict:{base_site_conflicts[0]}")
    base_entity_count = len(base_v2["entities"])
    base_point_count = len(base_v2["points"])
    new_entities, entity_updates, new_points, new_relations = (
        _resolve_projection_semantics(base_v2, projection)
    )
    semantic_v2 = _merge_semantic_v2(
        base_v2,
        new_entities=new_entities,
        entity_updates=entity_updates,
        new_points=new_points,
        new_relations=new_relations,
    )
    base_v1["projection_scope"] = "base_compatibility_only"
    base_v1["interaction_station_support"] = False

    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scene-npc-v2-", dir=output_path.parent) as raw:
        staging = Path(raw)
        scene_path = staging / f"{config.scene_id}_npc.xml"
        semantic_v2_path = staging / f"{config.scene_id}.semantic.v2.json"
        semantic_v1_path = staging / f"{config.scene_id}.semantic.json"
        coverage_path = staging / f"{config.scene_id}.semantic_coverage.json"
        population_path = staging / f"{config.scene_id}.population.json"
        trajectory_path = staging / f"{config.scene_id}.trajectory_profile.json"
        receipt_path = staging / f"{config.scene_id}.receipt.json"

        all_sites = base_sites + list(projection.derived_sites)
        _append_sites(
            config.source_mjcf,
            scene_path,
            all_sites,
            runtime_parent=output_path.parent,
        )
        if check_navigation:
            preflight_scene_path = staging / f"{config.scene_id}.preflight.xml"
            _append_sites(
                config.source_mjcf,
                preflight_scene_path,
                all_sites,
                runtime_parent=staging,
            )
            npc_navigation, robot_navigation = _preflight_navigation(
                config, plan, projection, base_sites, preflight_scene_path
            )
        else:
            disabled = {
                "status": "not_run",
                "reason": "check_navigation_false",
                "candidate_only": True,
            }
            npc_navigation = dict(disabled)
            robot_navigation = dict(disabled)
        coverage["interaction_navigation_preflight"] = {
            "requested": check_navigation,
            "candidate_only": not check_navigation,
            "npc": {
                "status": npc_navigation["status"],
                "target_count": len(npc_navigation.get("targets", [])),
                "route_count": len(npc_navigation.get("routes", [])),
            },
            "robot": {
                "status": robot_navigation["status"],
                "target_count": len(robot_navigation.get("targets", [])),
                "route_count": len(robot_navigation.get("routes", [])),
            },
        }
        _write_json(semantic_v2_path, semantic_v2)
        _write_json(semantic_v1_path, base_v1)
        _write_json(coverage_path, coverage)
        trajectory = _trajectory_profile(
            config,
            scene_name=scene_path.name,
            scene_sha256=_sha256(scene_path),
            semantic_v2=semantic_v2,
            base_sites=base_sites,
        )
        _write_json(trajectory_path, trajectory)
        population = _population_v3(
            config,
            scene_name=scene_path.name,
            trajectory_name=trajectory_path.name,
            base_sites=base_sites,
            interaction_stations=projection.interaction_stations,
            output_parent=output_path.parent,
        )
        all_site_names = {site["site"] for site in all_sites}
        NpcPopulation.from_dict(
            population,
            source_path=output_path.with_name(population_path.name),
            sites=all_site_names,
        )
        _write_json(population_path, population)

        SemanticWorld.from_json(semantic_v2_path)
        NpcTrajectoryProfile.from_json(trajectory_path)

        validators = deepcopy(projection.candidate_receipt["validators"])
        validators["source_bindings"] = {"status": "passed"}
        validators["npc_navigation"] = npc_navigation
        validators["robot_navigation"] = robot_navigation
        outputs_for_hash = {
            "scene": scene_path,
            "semantic_v2": semantic_v2_path,
            "semantic_v1": semantic_v1_path,
            "coverage": coverage_path,
            "population": population_path,
            "trajectory_profile": trajectory_path,
        }
        receipt = {
            "schema": "scene_interaction_compile_receipt/v1",
            "generator": "stretch_mujoco.npc.scene_compiler_v2",
            "validator_version": VALIDATOR_VERSION,
            "scene_id": config.scene_id,
            "validated_for_runtime": False,
            "projection": deepcopy(projection.candidate_receipt),
            "counts": {
                **projection.candidate_receipt["counts"],
                "base_semantic_entities": base_entity_count,
                "base_semantic_points": base_point_count,
                "semantic_entities": len(semantic_v2["entities"]),
                "semantic_points": len(semantic_v2["points"]),
                "semantic_relations": len(semantic_v2["relations"]),
            },
            "validators": validators,
            "sha256": {
                "inputs": {
                    "scene_config": _sha256(config.path),
                    "interaction_plan": _sha256(plan.path),
                    "source_mjcf": _sha256(config.source_mjcf),
                    "source_manifest": _sha256(config.source_manifest),
                    "semantic_policy": _sha256(config.semantic_policy),
                    "npc_catalog": _sha256(config.npc_catalog),
                },
                "outputs": {
                    name: _sha256(path) for name, path in outputs_for_hash.items()
                },
            },
            "semantic_v1": {
                "scope": "base_compatibility_only",
                "interaction_station_support": False,
            },
        }
        _write_json(receipt_path, receipt)

        staged = {**outputs_for_hash, "receipt": receipt_path}
        final = {name: output_path.with_name(path.name) for name, path in staged.items()}
        for name, path in staged.items():
            os.replace(path, final[name])
    return final
