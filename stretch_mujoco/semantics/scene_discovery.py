"""Full semantic discovery and strict coverage for NPC scene compilation."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from stretch_mujoco.humanoid.navigation import NavigationPathError, OfficeNavigationMesh
from stretch_mujoco.npc.scene_config import (
    SceneConfigError,
    SceneNpcConfig,
    _finite_point,
    load_strict_json,
)


class SemanticCoverageError(ValueError):
    """Discovery, registration, or reachability did not close strictly."""


@dataclass(frozen=True)
class DiscoveredEntity:
    semantic_id: str
    source_id: str
    kind: str
    body: str | None
    site: str | None
    label: str
    category: str | None
    bounds: tuple[float, float, float, float] | None
    support: str | None = None
    action_position: tuple[float, float, float] | None = None
    action_yaw: float = 0.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy(path: Path) -> dict[str, Any]:
    payload = load_strict_json(path)
    if set(payload) != {
        "schema",
        "kind",
        "fallback",
        "rules",
        "semantic_name_taxonomy",
    }:
        raise SemanticCoverageError(f"policy_fields_are_not_exact:{path}")
    if payload.get("schema") != "semantic_policy/v1" or not isinstance(payload.get("rules"), list):
        raise SemanticCoverageError(f"policy_invalid:{path}")
    if payload.get("kind") not in {"office", "home", "fixture"}:
        raise SemanticCoverageError(f"policy_kind_invalid:{path}")
    if payload.get("fallback") != "unresolved":
        raise SemanticCoverageError(f"policy_fallback_must_be_unresolved:{path}")
    taxonomy = payload.get("semantic_name_taxonomy")
    if not isinstance(taxonomy, dict):
        raise SemanticCoverageError(f"policy_taxonomy_invalid:{path}")

    allowed_definition_fields = {
        "semantic_class",
        "affordances",
        "navigation_requirement",
        "point_bundle",
        "action_navigation_site",
        "exemption",
    }
    allowed_bundles = {
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

    def validate_definition(definition: Any, label: str) -> None:
        if not isinstance(definition, dict):
            raise SemanticCoverageError(f"policy_definition_invalid:{label}")
        unknown = set(definition) - allowed_definition_fields
        if unknown:
            raise SemanticCoverageError(
                f"policy_definition_unknown_fields:{label}:" + ",".join(sorted(unknown))
            )
        semantic_class = definition.get("semantic_class")
        requirement = definition.get("navigation_requirement")
        affordances = definition.get("affordances")
        if not isinstance(semantic_class, str) or not semantic_class:
            raise SemanticCoverageError(f"policy_semantic_class_invalid:{label}")
        if requirement not in {"region", "approach", "observe", "none"}:
            raise SemanticCoverageError(f"policy_requirement_invalid:{label}")
        if (
            not isinstance(affordances, list)
            or not all(isinstance(value, str) and value for value in affordances)
            or len(set(affordances)) != len(affordances)
        ):
            raise SemanticCoverageError(f"policy_affordances_invalid:{label}")
        bundle = definition.get("point_bundle")
        if requirement == "none":
            exemption = definition.get("exemption")
            if (
                bundle is not None
                or not isinstance(exemption, dict)
                or set(exemption) != {"reason"}
                or not isinstance(exemption["reason"], str)
                or not exemption["reason"]
            ):
                raise SemanticCoverageError(f"policy_exemption_invalid:{label}")
        elif bundle not in allowed_bundles:
            raise SemanticCoverageError(f"policy_point_bundle_invalid:{label}")

    rule_ids: set[str] = set()
    for index, rule in enumerate(payload["rules"]):
        if not isinstance(rule, dict) or set(rule) - (allowed_definition_fields | {"id", "match"}):
            raise SemanticCoverageError(f"policy_rule_fields_invalid:{index}")
        rule_id = rule.get("id")
        match = rule.get("match")
        if not isinstance(rule_id, str) or not rule_id or rule_id in rule_ids:
            raise SemanticCoverageError(f"policy_rule_id_invalid:{index}")
        if (
            not isinstance(match, dict)
            or not match
            or set(match)
            - {
                "kind",
                "category",
                "semantic_name",
                "semantic_name_contains",
            }
        ):
            raise SemanticCoverageError(f"policy_match_invalid:{rule_id}")
        if not all(isinstance(value, str) and value for value in match.values()):
            raise SemanticCoverageError(f"policy_match_value_invalid:{rule_id}")
        validate_definition(
            {key: value for key, value in rule.items() if key not in {"id", "match"}},
            f"rule:{rule_id}",
        )
        rule_ids.add(rule_id)
    for semantic_name, definition in taxonomy.items():
        if not isinstance(semantic_name, str) or not semantic_name:
            raise SemanticCoverageError("policy_taxonomy_name_invalid")
        validate_definition(definition, f"taxonomy:{semantic_name}")
    return payload


def _xml_inventory(xml_path: Path) -> tuple[set[str], set[str]]:
    root = ET.parse(xml_path).getroot()
    return (
        {node.get("name") for node in root.findall(".//body") if node.get("name")}
        | {"geom:" + node.get("name") for node in root.findall(".//geom") if node.get("name")},
        {node.get("name") for node in root.findall(".//site") if node.get("name")},
    )


def _manifest_body_binding(xml_path: Path, asset: dict[str, Any], bodies: set[str]) -> str | None:
    declared = asset.get("body") or asset.get("object_id")
    if declared is not None:
        return str(declared) if str(declared) in bodies else None
    instance = asset.get("instance")
    if not isinstance(instance, int):
        return None
    suffix = f"_{instance:03d}"
    asset_name = str(asset.get("asset_id", "")).casefold()
    for prefix in (
        ("procedural_waste bin", "geom:procedural_bin_"),
        ("procedural_area rug", "geom:procedural_rug_"),
    ):
        if asset_name == prefix[0]:
            candidate = prefix[1] + f"{instance:03d}"
            if candidate in bodies:
                return candidate
    if asset_name == "procedural_snack counter" and "snack_counter" in bodies:
        return "snack_counter"
    if asset_name == "procedural_snack storage cabinet":
        candidate = f"snack_cabinet_{instance:03d}"
        if candidate in bodies:
            return candidate
    candidates = sorted(
        name for name in bodies if not name.startswith("geom:") and name.endswith(suffix)
    )
    if len(candidates) == 1:
        return candidates[0]
    candidates = sorted(
        name for name in bodies if name.startswith("geom:") and name.endswith(suffix)
    )
    if len(candidates) == 1:
        return candidates[0]
    # Generated office assets commonly use asset_000_<asset-prefix> bodies.
    prefix = str(asset.get("asset_id", "")).replace(" ", "_").lower()[:8]
    candidates = sorted(
        name
        for name in bodies
        if name.startswith(f"asset_{instance:03d}_") and (not prefix or prefix in name.lower())
    )
    return candidates[0] if len(candidates) == 1 else None


def discover(config: SceneNpcConfig) -> tuple[DiscoveredEntity, ...]:
    """Discover manifest objects/zones and XML semantic sites with bidirectional binding."""
    if not config.source_mjcf.is_file() or not config.source_manifest.is_file():
        raise SemanticCoverageError("source_scene_or_manifest_missing")
    manifest = load_strict_json(config.source_manifest)
    bodies, sites = _xml_inventory(config.source_mjcf)
    entities: list[DiscoveredEntity] = []
    for zone in manifest.get("zones", []):
        if not isinstance(zone, dict) or not isinstance(zone.get("type"), str):
            raise SemanticCoverageError("manifest_zone_invalid")
        bounds = zone.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise SemanticCoverageError(f"manifest_zone_bounds_invalid:{zone['type']}")
        entities.append(
            DiscoveredEntity(
                f"zone.{zone['type']}",
                zone["type"],
                "region",
                None,
                f"zone_{zone['type']}_center" if f"zone_{zone['type']}_center" in sites else None,
                zone["type"],
                "zone",
                (
                    float(bounds[0]),
                    float(bounds[2]),
                    float(bounds[1]),
                    float(bounds[3]),
                ),
            )
        )
    for asset in manifest.get("assets", []):
        if not isinstance(asset, dict):
            raise SemanticCoverageError("manifest_asset_invalid")
        source_id = asset.get("object_id") or asset.get("body")
        if (
            source_id is None
            and asset.get("asset_id") is not None
            and asset.get("instance") is not None
        ):
            source_id = f"{asset['asset_id']}.{asset['instance']}"
        if not isinstance(source_id, str) or not source_id:
            raise SemanticCoverageError("manifest_asset_has_no_stable_binding")
        declared_body = asset.get("body") or asset.get("object_id")
        body = _manifest_body_binding(config.source_mjcf, asset, bodies)
        if body is None:
            raise SemanticCoverageError(
                f"manifest_xml_body_unbound:{source_id}:{declared_body or asset.get('instance')}"
            )
        grasp_site = asset.get("grasp_site")
        if (
            grasp_site is None
            and asset.get("category") == "furniture/snack_counter"
            and "snack_pickup_site" in sites
        ):
            grasp_site = "snack_pickup_site"
        if grasp_site is not None and grasp_site not in sites:
            raise SemanticCoverageError(f"manifest_xml_site_unbound:{source_id}:{grasp_site}")
        raw_bounds = asset.get("bounds_mujoco")
        bounds = None
        action_position = None
        if isinstance(raw_bounds, list) and len(raw_bounds) == 2:
            try:
                bounds = (
                    float(raw_bounds[0][0]),
                    float(raw_bounds[0][1]),
                    float(raw_bounds[1][0]),
                    float(raw_bounds[1][1]),
                )
                action_position = (
                    (float(raw_bounds[0][0]) + float(raw_bounds[1][0])) / 2.0,
                    (float(raw_bounds[0][1]) + float(raw_bounds[1][1])) / 2.0,
                    float(raw_bounds[1][2]),
                )
            except (TypeError, ValueError, IndexError) as error:
                raise SemanticCoverageError(f"manifest_bounds_invalid:{source_id}") from error
        position = asset.get("position")
        if bounds is None and isinstance(position, list) and len(position) >= 2:
            x, y = float(position[0]), float(position[1])
            radius = float(asset.get("collision_radius", 0.2))
            bounds = (x - radius, y - radius, x + radius, y + radius)
            action_position = (
                x,
                y,
                float(position[2]) if len(position) >= 3 else 0.45,
            )
        label = str(asset.get("semantic_name") or asset.get("name") or source_id)
        support = asset.get("support")
        if support is not None and (not isinstance(support, str) or not support):
            raise SemanticCoverageError(f"manifest_support_invalid:{source_id}")
        entities.append(
            DiscoveredEntity(
                f"object.{source_id}",
                source_id,
                "object",
                body,
                grasp_site,
                label,
                asset.get("category"),
                bounds,
                support,
                action_position,
                float(asset.get("yaw", 0.0)),
            )
        )
    manifest_site_names = {entity.site for entity in entities if entity.site}
    zone_ids = {entity.source_id for entity in entities if entity.kind == "region"}
    for site in sorted(sites):
        if site.startswith("zone_") and site.endswith("_center"):
            zone_name = site.removeprefix("zone_").removesuffix("_center")
            if zone_name in zone_ids:
                # This is binding evidence for the manifest region, not another entity.
                continue
        if site.startswith("zone_") or site.endswith("_grasp_site"):
            if site not in manifest_site_names:
                entities.append(
                    DiscoveredEntity(
                        f"site.{site}", site, "site", None, site, site, "xml_site", None
                    )
                )
    for region_id, override in config.region_overrides.items():
        if any(entity.semantic_id == region_id for entity in entities):
            raise SemanticCoverageError(f"region_override_conflicts_with_discovery:{region_id}")
        seed = _finite_point(override["geometry"]["seed"], f"region_seed:{region_id}")
        entities.append(
            DiscoveredEntity(
                region_id,
                region_id,
                "region",
                None,
                None,
                region_id,
                "override",
                (seed[0], seed[1], seed[0], seed[1]),
            )
        )
    duplicate_ids = {
        entity.semantic_id
        for entity in entities
        if sum(item.semantic_id == entity.semantic_id for item in entities) > 1
    }
    if duplicate_ids:
        raise SemanticCoverageError("semantic_id_not_unique:" + ",".join(sorted(duplicate_ids)))
    return tuple(sorted(entities, key=lambda item: item.semantic_id))


def _classify(
    entity: DiscoveredEntity, policy: dict[str, Any], overrides: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    override = overrides.get(entity.semantic_id)
    if override is not None:
        return dict(override)
    exact_names = policy.get("semantic_name_taxonomy", {})
    if entity.category == "default" and entity.label in exact_names:
        return dict(exact_names[entity.label])
    for rule in policy["rules"]:
        match = rule.get("match", {})
        if match.get("kind") and match["kind"] != entity.kind:
            continue
        if match.get("category") and match["category"] != entity.category:
            continue
        if (
            match.get("semantic_name")
            and match["semantic_name"].casefold() != entity.label.casefold()
        ):
            continue
        if (
            match.get("semantic_name_contains")
            and match["semantic_name_contains"].casefold() not in entity.label.casefold()
        ):
            continue
        return {key: value for key, value in rule.items() if key not in {"id", "match"}}
    return {"navigation_requirement": "unresolved"}


def _candidate_positions(
    entity: DiscoveredEntity, bundle: str, *, margin: float
) -> tuple[tuple[float, float, float], ...]:
    if entity.bounds is None:
        raise SemanticCoverageError(f"point_candidate_missing_bounds:{entity.semantic_id}")
    x0, y0, x1, y1 = entity.bounds
    if bundle == "region_samples":
        center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        candidates = [(center_x, center_y, 0.025)]
        step = max(0.4, 2.0 * margin)
        x = x0 + margin
        while x <= x1 - margin + 1e-9:
            y = y0 + margin
            while y <= y1 - margin + 1e-9:
                candidates.append((x, y, 0.025))
                y += step
            x += step
        return tuple(
            sorted(
                set(candidates),
                key=lambda point: (math.dist(point[:2], (center_x, center_y)), point),
            )
        )
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    candidates: list[tuple[float, float, float]] = []
    # Search several deterministic rings.  A single ring is insufficient for
    # wall-backed furniture and supported objects whose own bounds lie on a
    # table or counter footprint.
    for extra in (0.0, 0.25, 0.5, 0.8):
        offset = margin + extra
        candidates.extend(
            (
                (x1 + offset, center_y, 0.025),
                (x1 + offset, y1 + offset, 0.025),
                (center_x, y1 + offset, 0.025),
                (x0 - offset, y1 + offset, 0.025),
                (x0 - offset, center_y, 0.025),
                (x0 - offset, y0 - offset, 0.025),
                (center_x, y0 - offset, 0.025),
                (x1 + offset, y0 - offset, 0.025),
            )
        )
    return tuple(candidates)


def _observation_has_line_of_sight(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    entity: DiscoveredEntity,
    position: tuple[float, float, float],
) -> bool:
    if entity.bounds is None:
        return False
    x0, y0, x1, y1 = entity.bounds
    origin = np.asarray((position[0], position[1], 1.4), dtype=float)
    target = np.asarray(((x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.8), dtype=float)
    vector = target - origin
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-6:
        return True
    geom_id = np.asarray([-1], dtype=np.int32)
    # Scene resources consistently place collision proxies in group 3 and
    # render-only meshes in group 2.  Ray-casting against visual stage meshes
    # produces false wall hits in HSSD homes.
    geom_group = np.asarray([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    hit_distance = float(
        mujoco.mj_ray(
            model,
            data,
            origin,
            vector / distance,
            geom_group,
            1,
            -1,
            geom_id,
        )
    )
    if geom_id[0] >= 0 and entity.body:
        if entity.body.startswith("geom:"):
            hit_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0]))
            if hit_name == entity.body.removeprefix("geom:"):
                return True
        else:
            target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
            hit_body_id = int(model.geom_bodyid[int(geom_id[0])])
            while hit_body_id > 0:
                if hit_body_id == target_body_id:
                    return True
                hit_body_id = int(model.body_parentid[hit_body_id])
    target_radius = max(0.2, math.hypot(x1 - x0, y1 - y0) / 2.0)
    return hit_distance < 0.0 or hit_distance >= distance - target_radius - 0.05


def _inferred_support(
    entity: DiscoveredEntity,
    inventory: tuple[DiscoveredEntity, ...],
    policy: dict[str, Any],
    overrides: dict[str, dict[str, Any]],
) -> DiscoveredEntity | None:
    """Find the smallest furniture footprint containing an unsupported object."""
    if entity.bounds is None:
        return None
    center_x = (entity.bounds[0] + entity.bounds[2]) / 2.0
    center_y = (entity.bounds[1] + entity.bounds[3]) / 2.0
    candidates: list[tuple[float, str, DiscoveredEntity]] = []
    for candidate in inventory:
        if candidate is entity or candidate.kind != "object" or candidate.bounds is None:
            continue
        definition = _classify(candidate, policy, overrides)
        semantic_class = definition.get("semantic_class")
        if not isinstance(semantic_class, str) or not semantic_class.startswith("furniture."):
            continue
        x0, y0, x1, y1 = candidate.bounds
        if x0 - 0.1 <= center_x <= x1 + 0.1 and y0 - 0.1 <= center_y <= y1 + 0.1:
            candidates.append((max(0.0, (x1 - x0) * (y1 - y0)), candidate.semantic_id, candidate))
    return min(candidates)[2] if candidates else None


def _dynamic_actor_roots(config: SceneNpcConfig) -> tuple[str, ...]:
    manifest = load_strict_json(config.source_manifest)
    robot = manifest.get("robot", {})
    if not isinstance(robot, dict):
        raise SemanticCoverageError("manifest_robot_invalid")
    body = robot.get("body")
    if body is None:
        return ()
    if not isinstance(body, str) or not body:
        raise SemanticCoverageError("manifest_robot_body_invalid")
    return (body,)


def compile_semantics(
    config: SceneNpcConfig, *, check_navigation: bool = True
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Return v2 registry, v1 safe projection, coverage report and generated sites."""
    policy = _policy(config.semantic_policy)
    inventory = discover(config)
    entries: dict[str, Any] = {}
    generated_sites: list[dict[str, Any]] = []
    unresolved: list[str] = []
    unbound: list[str] = []
    exemptions: list[dict[str, str]] = []
    model: mujoco.MjModel | None = None
    data: mujoco.MjData | None = None
    navigation: OfficeNavigationMesh | None = None
    if check_navigation:
        model = mujoco.MjModel.from_xml_path(str(config.source_mjcf))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        navigation = OfficeNavigationMesh.from_model(
            model,
            data,
            resolution=float(config.navigation["resolution"]),
            agent_radius=float(config.navigation["agent_radius"])
            + float(config.navigation["clearance"]),
            floor_geom_name=str(config.navigation["surface"]),
            exclude_body_roots=_dynamic_actor_roots(config),
        )
    for entity in inventory:
        definition = _classify(entity, policy, config.entity_overrides)
        if entity.kind == "region":
            definition = dict(definition, semantic_class=f"region.{entity.source_id}")
        requirement = definition.get("navigation_requirement", "unresolved")
        if requirement not in {"region", "approach", "observe", "none"}:
            unresolved.append(entity.semantic_id)
            continue
        if requirement == "none":
            exemption = definition.get("exemption")
            if (
                not isinstance(exemption, dict)
                or not isinstance(exemption.get("reason"), str)
                or not exemption["reason"]
            ):
                unbound.append(f"exemption_reason_missing:{entity.semantic_id}")
            else:
                exemptions.append({"entity": entity.semantic_id, "reason": exemption["reason"]})
        point_ids: dict[str, list[str]] = {"navigation": [], "action": []}
        if requirement != "none":
            bundle = str(
                definition.get(
                    "point_bundle",
                    "region_samples" if requirement == "region" else "perimeter_approach",
                )
            )
            explicit = definition.get("explicit_point")
            candidate_entity = entity
            if entity.support:
                support_matches = [
                    item
                    for item in inventory
                    if item.source_id == entity.support or item.body == entity.support
                ]
                if len(support_matches) != 1:
                    unbound.append(f"support_unbound:{entity.semantic_id}:{entity.support}")
                    continue
                candidate_entity = support_matches[0]
            elif bundle in {"graspable_on_support", "observation", "seat"}:
                inferred_support = _inferred_support(
                    entity, inventory, policy, config.entity_overrides
                )
                if inferred_support is not None:
                    candidate_entity = inferred_support
                elif bundle == "graspable_on_support":
                    unbound.append(f"support_missing:{entity.semantic_id}")
                    continue
            candidates = (
                (_finite_point(explicit, f"explicit_point:{entity.semantic_id}"),)
                if explicit is not None
                else _candidate_positions(
                    candidate_entity,
                    bundle,
                    margin=float(config.navigation["agent_radius"])
                    + float(config.navigation["clearance"])
                    + 0.15,
                )
            )
            position = candidates[0]
            if navigation is not None:
                valid_candidates: list[tuple[float, float, float]] = []
                for candidate in candidates:
                    projected = candidate
                    if not navigation.is_world_free(
                        np.asarray(candidate[:2]),
                        component_id=navigation.primary_component_id,
                    ):
                        if explicit is not None:
                            continue
                        target_bounds = candidate_entity.bounds
                        assert target_bounds is not None
                        target_center = np.asarray(
                            (
                                (target_bounds[0] + target_bounds[2]) / 2.0,
                                (target_bounds[1] + target_bounds[3]) / 2.0,
                            )
                        )
                        try:
                            xy = navigation.nearest_free_world(
                                np.asarray(candidate[:2]),
                                toward=target_center,
                                max_distance=1.25,
                                component_id=navigation.primary_component_id,
                            )
                        except NavigationPathError:
                            continue
                        projected = (float(xy[0]), float(xy[1]), candidate[2])
                    if requirement == "observe" and not (
                        model is not None
                        and data is not None
                        and _observation_has_line_of_sight(model, data, entity, projected)
                    ):
                        continue
                    if projected not in valid_candidates:
                        valid_candidates.append(projected)
                if (
                    not valid_candidates
                    and requirement == "observe"
                    and entity.bounds is not None
                    and model is not None
                    and data is not None
                ):
                    target_center = np.asarray(
                        (
                            (entity.bounds[0] + entity.bounds[2]) / 2.0,
                            (entity.bounds[1] + entity.bounds[3]) / 2.0,
                        )
                    )
                    free_cells = np.argwhere(
                        navigation.component_labels == navigation.primary_component_id
                    )
                    fallback_candidates = sorted(
                        (
                            (
                                float(
                                    np.linalg.norm(
                                        navigation.cell_to_world(tuple(cell)) - target_center
                                    )
                                ),
                                tuple(int(value) for value in cell),
                            )
                            for cell in free_cells
                        ),
                        key=lambda item: (item[0], item[1]),
                    )
                    for distance_to_target, cell in fallback_candidates:
                        if distance_to_target < 0.45:
                            continue
                        if distance_to_target > 5.0:
                            break
                        xy = navigation.cell_to_world(cell)
                        projected = (float(xy[0]), float(xy[1]), 0.025)
                        if _observation_has_line_of_sight(model, data, entity, projected):
                            valid_candidates.append(projected)
                            break
                if not valid_candidates:
                    raise SemanticCoverageError(f"point_candidate_unavailable:{entity.semantic_id}")
                position = valid_candidates[0]
            point_id = f"point.{entity.semantic_id}.{'observation' if requirement == 'observe' else 'approach'}.01"
            site = "npc__" + config.scene_id + "__" + point_id.replace(".", "__")
            yaw = (
                math.atan2(
                    entity.bounds[0] + entity.bounds[2] - 2 * position[0],
                    2 * position[1] - entity.bounds[1] - entity.bounds[3],
                )
                if entity.bounds
                else 0.0
            )
            point_ids["navigation"].append(point_id)
            generated_sites.append(
                {
                    "id": point_id,
                    "site": site,
                    "position": position,
                    "yaw": yaw,
                    "owner": entity.semantic_id,
                    "required": True,
                    "kind": "navigation",
                    "usages": [
                        "navigation",
                        *(["spawn"] if requirement == "region" else []),
                    ],
                }
            )
            if entity.site is not None or bundle in {
                "graspable_on_support",
                "seat",
                "door_portal",
            }:
                action_site = entity.site
                if action_site is None and bundle == "seat":
                    action_id = f"point.{entity.semantic_id}.action.sit"
                    action_site = "npc__" + config.scene_id + "__" + action_id.replace(".", "__")
                    action_position = entity.action_position or position
                    generated_sites.append(
                        {
                            "id": action_id,
                            "site": action_site,
                            "position": action_position,
                            "yaw": entity.action_yaw,
                            "owner": entity.semantic_id,
                            "required": False,
                            "kind": "action",
                            "navigation_site": point_id,
                            "emit_site": True,
                        }
                    )
                if action_site is None:
                    unbound.append(f"action_site_missing:{entity.semantic_id}")
                else:
                    configured_navigation = definition.get("action_navigation_site", point_id)
                    if configured_navigation != point_id:
                        unbound.append(
                            f"action_navigation_unbound:{entity.semantic_id}:{configured_navigation}"
                        )
                    action_id = (
                        f"point.{entity.semantic_id}.action.sit"
                        if bundle == "seat"
                        else f"point.{entity.semantic_id}.action"
                    )
                    if not any(item["id"] == action_id for item in generated_sites):
                        generated_sites.append(
                            {
                                "id": action_id,
                                "site": action_site,
                                "owner": entity.semantic_id,
                                "required": False,
                                "kind": "action",
                                "navigation_site": point_id,
                                "emit_site": False,
                            }
                        )
                    point_ids["action"].append(action_id)
        binding_name = entity.body or entity.site
        binding_kind = (
            "geom"
            if binding_name and binding_name.startswith("geom:")
            else ("body" if entity.body else "site")
        )
        if binding_name and binding_name.startswith("geom:"):
            binding_name = binding_name.removeprefix("geom:")
        entries[entity.semantic_id] = {
            "semantic_class": definition.get("semantic_class"),
            "source": {
                "id": entity.source_id,
                "xml_binding": {"kind": binding_kind, "name": binding_name},
            },
            "labels": {"canonical": entity.label, "source_category": entity.category},
            "navigation_requirement": requirement,
            "affordances": definition.get("affordances", []),
            "points": point_ids,
        }
    if unresolved or unbound:
        raise SemanticCoverageError(
            "semantic_coverage_incomplete:" + ",".join(sorted(unresolved + unbound))
        )
    for point_id, definition in config.custom_targets.items():
        owner = definition["owner"]
        if owner not in entries:
            raise SemanticCoverageError(f"custom_target_owner_unbound:{point_id}:{owner}")
        if any(item["id"] == point_id for item in generated_sites):
            raise SemanticCoverageError(f"custom_target_id_conflict:{point_id}")
        position = _finite_point(definition["position"], f"custom_target:{point_id}")
        generated_sites.append(
            {
                "id": point_id,
                "site": "npc__" + config.scene_id + "__" + point_id.replace(".", "__"),
                "position": position,
                "yaw": float(definition["yaw"]),
                "owner": owner,
                "required": True,
                "kind": "navigation",
                "usages": list(definition["usages"]),
            }
        )
        entries[owner]["points"]["navigation"].append(point_id)
    navigation_sites = [
        item for item in generated_sites if item.get("kind", "navigation") == "navigation"
    ]
    point_by_id = {item["id"]: item for item in navigation_sites}
    for member in config.population:
        if member.spawn not in point_by_id:
            raise SemanticCoverageError(f"population_spawn_unbound:{member.npc}:{member.spawn}")
        if "spawn" not in point_by_id[member.spawn].get("usages", []):
            raise SemanticCoverageError(
                f"population_spawn_usage_missing:{member.npc}:{member.spawn}"
            )
    minimum_spawn_spacing = 2.0 * (
        float(config.navigation["agent_radius"]) + float(config.navigation["clearance"])
    )
    for index, member in enumerate(config.population):
        for other in config.population[index + 1 :]:
            first = np.asarray(point_by_id[member.spawn]["position"][:2])
            second = np.asarray(point_by_id[other.spawn]["position"][:2])
            if float(np.linalg.norm(first - second)) < minimum_spawn_spacing:
                raise SemanticCoverageError(
                    f"population_spawn_spacing_invalid:{member.npc}:{other.npc}"
                )
    reachability: list[dict[str, Any]] = []
    unreachable: list[str] = []
    if check_navigation:
        assert navigation is not None
        for point in navigation_sites:
            if not navigation.is_world_free(
                np.asarray(point["position"][:2]),
                component_id=navigation.primary_component_id,
            ):
                unreachable.append(f"point_not_free:{point['id']}")
        for member in config.population:
            origin = point_by_id[member.spawn]
            for target in navigation_sites:
                try:
                    path = navigation.plan(
                        np.asarray(origin["position"][:2]), np.asarray(target["position"][:2])
                    )
                    path_length = sum(
                        float(np.linalg.norm(path[index] - path[index - 1]))
                        for index in range(1, len(path))
                    )
                    reachability.append(
                        {
                            "origin": member.spawn,
                            "target": target["id"],
                            "path_length_m": round(path_length, 9),
                        }
                    )
                except NavigationPathError:
                    unreachable.append(f"unreachable:{member.spawn}:{target['id']}")
    if unreachable:
        raise SemanticCoverageError(
            "required_navigation_unreachable:" + ",".join(sorted(unreachable))
        )
    for member in config.population:
        if member.initial_region not in entries:
            raise SemanticCoverageError(
                f"population_initial_region_unbound:{member.npc}:{member.initial_region}"
            )
    v2 = {
        "schema_version": 2,
        "scene": config.source_mjcf.name,
        "scene_id": config.scene_id,
        "entities": entries,
        "points": {
            item["id"]: {
                key: value for key, value in item.items() if key not in {"id", "emit_site"}
            }
            for item in generated_sites
        },
    }
    safe_types = {
        "furniture.seat": "Chair",
        "furniture.bed": "Bed",
        "furniture.storage": "StorageCabinet",
        "furniture.workstation": "Workstation",
        "furniture.counter": "Counter",
        "region.work": "Workstation",
        "region.meeting": "MeetingTable",
        "region.lounge": "Counter",
        "region.snack": "Counter",
        "region.home": "Counter",
    }
    v1_objects, v1_points = {}, {}
    for entity_id, entry in entries.items():
        object_type = safe_types.get(entry["semantic_class"])
        if object_type and entry["source"]["xml_binding"]["name"]:
            v1_objects[entity_id] = {
                "type": object_type,
                "binding": entry["source"]["xml_binding"],
                "attributes": {"semantic_class": entry["semantic_class"]},
            }
            for point_id in entry["points"]["navigation"]:
                point = point_by_id[point_id]
                v1_points[point_id] = {
                    "role": "human_stand_site",
                    "owner": entity_id,
                    "site": point["site"],
                    "attributes": {"binding": "semantic_navigation", "target": entity_id},
                }
    v1 = {
        "schema_version": 1,
        "scene": config.source_mjcf.name,
        "objects": v1_objects,
        "relations": [],
        "interaction_points": v1_points,
    }
    matrix_bytes = json.dumps(reachability, sort_keys=True).encode("utf-8")
    coverage = {
        "schema_version": 1,
        "scene_id": config.scene_id,
        "counts": {
            "discovered": len(inventory),
            "classified": len(entries),
            "registered": len(entries) - len(exemptions),
            "explicitly_exempt": len(exemptions),
            "unresolved": 0,
            "unbound": 0,
        },
        "required_navigation_points": len(navigation_sites),
        "reachable_navigation_points": len(navigation_sites) if check_navigation else 0,
        "navigation_preflight": "passed" if check_navigation else "not_run",
        "strongly_connected": bool(check_navigation),
        "unreachable": [],
        "exemptions": exemptions,
        "reachability": {
            "origins": len(config.population),
            "targets": len(navigation_sites),
            "pair_count": len(reachability),
            "all_reachable": bool(check_navigation),
        },
        "reachability_matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "planner": {
            "name": config.navigation["planner"],
            "agent_radius": config.navigation["agent_radius"],
            "clearance": config.navigation["clearance"],
            "resolution": config.navigation["resolution"],
            "primary_component_id": (
                navigation.primary_component_id if navigation is not None else None
            ),
            "primary_component_cells": (
                navigation.component_sizes[navigation.primary_component_id]
                if navigation is not None
                else None
            ),
        },
    }
    return v2, v1, coverage, generated_sites
