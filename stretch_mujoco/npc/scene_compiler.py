"""Unified compiler for strict scene semantic registrations.

The compiler intentionally emits a full v2 registry plus a conservative v1
projection.  It does not invent legacy ``ObjectType`` values for new objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from stretch_mujoco.semantics.scene_discovery import compile_semantics

from .scene_config import load_scene_npc_config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sites(source: Path) -> set[str]:
    """Collect named sites from the source MJCF and its local includes."""
    root = ET.parse(source).getroot()
    sites = {node.get("name") for node in root.findall(".//site") if node.get("name")}
    for include in root.findall("include"):
        raw_file = include.get("file")
        if raw_file:
            included = Path(raw_file)
            included = included if included.is_absolute() else source.parent / included
            if included.is_file():
                sites.update(_source_sites(included.resolve()))
    return sites


def _source_site_yaw(source: Path, site_name: str) -> float:
    """Read the authored facing direction of a registered source site."""
    root = ET.parse(source).getroot()
    site = next(
        (node for node in root.findall(".//site") if node.get("name") == site_name),
        None,
    )
    if site is None:
        raise ValueError(f"interaction_site_unbound:{site_name}")
    euler = site.get("euler")
    if euler:
        values = [float(item) for item in euler.split()]
        if len(values) == 3:
            return values[2]
    return 0.0


def _validate_interaction_sources(config) -> None:
    """Reject capability claims that are not backed by source resources."""
    source_sites = _source_sites(config.source_mjcf)
    for capability, declaration in config.interactions.items():
        if declaration["status"] == "unsupported":
            continue
        for site in declaration["sites"]:
            if site not in source_sites:
                raise ValueError(f"interaction_site_unbound:{capability}:{site}")
        if capability == "robot_handover":
            manifest = json.loads(config.source_manifest.read_text(encoding="utf-8"))
            asset_ids = {
                str(asset.get("object_id") or asset.get("body"))
                for asset in manifest.get("assets", [])
                if isinstance(asset, dict) and (asset.get("object_id") or asset.get("body"))
            }
            if declaration["object"] not in asset_ids:
                raise ValueError(f"handover_object_unbound:{declaration['object']}")
            if declaration["robot_approach"] not in source_sites:
                raise ValueError(
                    f"handover_robot_approach_unbound:{declaration['robot_approach']}"
                )


def _append_sites(
    source: Path,
    destination: Path,
    sites: list[dict[str, Any]],
    *,
    runtime_parent: Path,
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"scene_worldbody_missing:{source}")
    asset_root: Path | None = None
    for include in root.findall("include"):
        raw_file = include.get("file")
        if not raw_file:
            raise ValueError(f"scene_include_file_missing:{source}")
        include_source = (
            Path(raw_file) if Path(raw_file).is_absolute() else (source.parent / raw_file).resolve()
        )
        if not include_source.is_file():
            raise ValueError(f"scene_include_missing:{include_source}")
        include.set("file", os.path.relpath(include_source, runtime_parent))
        included_root = ET.parse(include_source).getroot()
        included_compiler = included_root.find("compiler")
        if included_compiler is not None and included_compiler.get("assetdir"):
            raw_asset_root = Path(included_compiler.get("assetdir", ""))
            asset_root = (
                raw_asset_root
                if raw_asset_root.is_absolute()
                else (include_source.parent / raw_asset_root).resolve()
            )
    compiler = root.find("compiler")
    if compiler is not None:
        raw_asset_root = compiler.get("assetdir")
        if raw_asset_root:
            configured_root = Path(raw_asset_root)
            asset_root = (
                configured_root
                if configured_root.is_absolute()
                else (source.parent / configured_root).resolve()
            )
        if asset_root is not None:
            compiler.set("assetdir", os.path.relpath(asset_root, runtime_parent))
    existing = {site.get("name") for site in worldbody.findall(".//site")}
    for point in sites:
        if point.get("emit_site") is False:
            continue
        if point["site"] in existing:
            raise ValueError(f"generated_semantic_site_conflict:{point['site']}")
        x, y, z = point["position"]
        ET.SubElement(
            worldbody,
            "site",
            name=point["site"],
            pos=f"{x:.9g} {y:.9g} {z:.9g}",
            euler=f"0 0 {point['yaw']:.9g}",
            size="0.015",
            rgba="0 0 0 0",
        )
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode", xml_declaration=False)


def compile_scene_npc_config(
    config_path: str | Path, output: str | Path, *, check_navigation: bool = True
) -> dict[str, Path]:
    """Compile atomically; no output is replaced until all strict checks pass."""
    config = load_scene_npc_config(config_path)
    if config.schema == "scene_npc_config/v2":
        from .scene_compiler_v2 import compile_scene_npc_config_v2

        return compile_scene_npc_config_v2(
            config, output, check_navigation=check_navigation
        )
    _validate_interaction_sources(config)
    v2, v1, coverage, sites = compile_semantics(config, check_navigation=check_navigation)
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scene-npc-", dir=output_path.parent) as temporary:
        staging = Path(temporary)
        scene_path = staging / f"{config.scene_id}_npc.xml"
        v2_path = staging / f"{config.scene_id}.semantic.v2.json"
        v1_path = staging / f"{config.scene_id}.semantic.json"
        coverage_path = staging / f"{config.scene_id}.semantic_coverage.json"
        receipt_path = staging / f"{config.scene_id}.receipt.json"
        population_path = staging / f"{config.scene_id}.population.json"
        trajectory_path = staging / f"{config.scene_id}.trajectory_profile.json"
        _append_sites(
            config.source_mjcf,
            scene_path,
            sites,
            runtime_parent=output_path.parent,
        )
        # The base semantic compiler intentionally focuses on scene furniture
        # and navigation.  A supported showcase capability additionally needs
        # its physical delivery object and robot body in the runtime graph.
        # Project those source-manifest-backed bindings here; they are not
        # inferred from a video or from a logical transcript.
        handover = config.interactions["robot_handover"]
        if handover["status"] == "supported":
            manifest = json.loads(config.source_manifest.read_text(encoding="utf-8"))
            manifest_asset = next(
                asset
                for asset in manifest.get("assets", [])
                if asset.get("object_id") == handover["object"]
            )
            object_id = handover["object"]
            destination = "zone.snack" if config.kind == "office" else "zone.home"
            v1.setdefault("objects", {}).update(
                {
                    object_id: {
                        "type": "Snack",
                        "binding": {"kind": "body", "name": handover["object"]},
                        "attributes": {
                            "graspable": bool(manifest_asset.get("graspable", True)),
                            "dynamic": bool(manifest_asset.get("dynamic", True)),
                            "available": True,
                            "location": destination,
                            "source_manifest_object": handover["object"],
                        },
                    },
                    "stretch_3": {
                        "type": "StretchRobot",
                        "binding": {"kind": "body", "name": "base_link"},
                        "attributes": {"simulator_robot": True},
                    },
                }
            )
            v1.setdefault("relations", []).append(
                {"subject": object_id, "relation": "ON", "object": destination}
            )
            v1.setdefault("interaction_points", {})["showcase_robot_request"] = {
                "owner": "stretch_3",
                "role": "robot_request_site",
                "site": handover["robot_approach"],
                "attributes": {"binding": "robot_request", "target": "stretch_3"},
            }
            conversation_sites = config.interactions["conversation"]["sites"]
            handover_sites = handover["sites"]
            conversation_yaws = [_source_site_yaw(config.source_mjcf, site) for site in conversation_sites]
            handover_yaws = [_source_site_yaw(config.source_mjcf, site) for site in handover_sites]
            point_owner = "zone.meeting" if config.kind == "office" else "zone.home"
            v1["interaction_points"].update(
                {
                    "showcase_conversation_speaker": {
                        "owner": point_owner,
                        "role": "conversation_site",
                        "site": conversation_sites[0],
                        "attributes": {"yaw": conversation_yaws[0]},
                    },
                    "showcase_conversation_listener": {
                        "owner": point_owner,
                        "role": "conversation_site",
                        "site": conversation_sites[1],
                        "attributes": {"yaw": conversation_yaws[1]},
                    },
                    "showcase_handover_giver": {
                        "owner": "stretch_3",
                        "role": "handover_site",
                        "site": handover_sites[0],
                        "attributes": {"yaw": handover_yaws[0]},
                    },
                    "showcase_handover_receiver": {
                        "owner": "stretch_3",
                        "role": "handover_site",
                        "site": handover_sites[1],
                        "attributes": {"yaw": handover_yaws[1]},
                    },
                }
            )
        _write_json(v2_path, v2)
        _write_json(v1_path, v1)
        _write_json(coverage_path, coverage)
        catalog_path = config.npc_catalog
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        points = {point["id"]: point for point in sites}
        selected = {
            member.npc: dict(
                catalog["npcs"][member.npc],
                spawn={
                    "location": member.initial_region,
                    "site": points[member.spawn]["site"],
                    "yaw": float(points[member.spawn]["yaw"]),
                },
            )
            for member in config.population
            if member.npc in catalog["npcs"]
        }
        if len(selected) != len(config.population):
            raise ValueError("npc_catalog_member_missing")
        population = {
            key: value
            for key, value in catalog.items()
            if key not in {"npcs", "trajectory_profile", "interaction_templates"}
        }
        if config.spawn_policy is not None:
            spawn_policy = dict(config.spawn_policy)
        final_parent = output_path.parent
        for resource_key in ("asset_manifest", "appearance_catalog"):
            resource = catalog.get(resource_key)
            if isinstance(resource, str):
                population[resource_key] = os.path.relpath(
                    (catalog_path.parent / resource).resolve(), final_parent
                )
        population.update(
            {
                "scene": scene_path.name,
                "trajectory_profile": trajectory_path.name,
                "traffic_policy": dict(config.traffic),
                "interaction_capabilities": {
                    key: dict(value) for key, value in config.interactions.items()
                },
                "npcs": selected,
            }
        )
        if config.spawn_policy is not None:
            population["spawn_policy"] = spawn_policy
        # Interaction templates are a generated projection of the source
        # capability contract.  Keeping them here makes the runtime consume
        # the same named stations that the compiler validated, rather than a
        # legacy Python constant.
        if all(
            config.interactions[key]["status"] == "supported"
            for key in ("conversation", "robot_handover")
        ):
            conversation_sites = config.interactions["conversation"]["sites"]
            handover_sites = config.interactions["robot_handover"]["sites"]
            conversation_yaws = [_source_site_yaw(config.source_mjcf, site) for site in conversation_sites]
            handover_yaws = [_source_site_yaw(config.source_mjcf, site) for site in handover_sites]
            population["interaction_templates"] = {
                "conversation": {
                    "speaker": {"site": conversation_sites[0], "yaw": conversation_yaws[0]},
                    "listener": {
                        "site": conversation_sites[1],
                        "yaw": conversation_yaws[1],
                    },
                },
                "handover": {
                    "giver": {"site": handover_sites[0], "yaw": handover_yaws[0]},
                    "receiver": {
                        "site": handover_sites[1],
                        "yaw": handover_yaws[1],
                    },
                },
            }
        _write_json(population_path, population)
        source_manifest = json.loads(config.source_manifest.read_text(encoding="utf-8"))
        robot = source_manifest.get("robot", {})
        robot_body = robot.get("body") if isinstance(robot, dict) else None
        navigation_sites = [
            point for point in sites if point.get("kind", "navigation") == "navigation"
        ]
        anchors = {
            point["id"]: {
                "site": point["site"],
                "role": "semantic_navigation",
            }
            for point in navigation_sites
        }

        def resolve_route_endpoint(endpoint: str) -> str:
            if endpoint in anchors:
                return endpoint
            entity = v2["entities"].get(endpoint)
            if entity is None:
                raise ValueError(f"route_endpoint_unbound:{endpoint}")
            candidates = entity["points"]["navigation"]
            if not candidates:
                raise ValueError(f"route_endpoint_has_no_navigation_point:{endpoint}")
            return sorted(candidates)[0]

        business_routes = [
            {
                "route_id": f"business_{route['id']}",
                "from": resolve_route_endpoint(route["from"]),
                "to": resolve_route_endpoint(route["to"]),
                "actions": list(route["actions"]),
                "mode": route["mode"],
            }
            for route in config.routes
        ]
        if any(route["from"] == route["to"] for route in business_routes):
            raise ValueError("business_route_resolves_to_same_navigation_point")
        routes = business_routes + [
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
        if not routes:
            raise ValueError("trajectory_profile_requires_distinct_navigation_points")
        _write_json(
            trajectory_path,
            {
                "schema_version": 1,
                "profile_id": config.scene_id + "_coverage",
                "scene": scene_path.name,
                "scene_sha256": _sha256(scene_path),
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
            },
        )
        receipt = {
            "schema_version": 1,
            "generator": "stretch_mujoco.npc.scene_compiler",
            "scene_id": config.scene_id,
            "sha256": {
                "source_mjcf": _sha256(config.source_mjcf),
                "source_manifest": _sha256(config.source_manifest),
                "semantic_policy": _sha256(config.semantic_policy),
                "scene_config": _sha256(config.path),
                "npc_catalog": _sha256(catalog_path),
                "generated_scene": _sha256(scene_path),
                "semantic_v2": _sha256(v2_path),
                "semantic_v1": _sha256(v1_path),
                "coverage": _sha256(coverage_path),
                "population": _sha256(population_path),
                "trajectory_profile": _sha256(trajectory_path),
            },
        }
        _write_json(receipt_path, receipt)
        outputs = {
            "scene": scene_path,
            "semantic_v2": v2_path,
            "semantic_v1": v1_path,
            "coverage": coverage_path,
            "population": population_path,
            "trajectory_profile": trajectory_path,
            "receipt": receipt_path,
        }
        final = {name: output_path.with_name(path.name) for name, path in outputs.items()}
        for name, path in outputs.items():
            os.replace(path, final[name])
    return final


def config_only_inventory(config_path: str | Path) -> dict[str, Any]:
    """Run strict discovery/classification but never compile MuJoCo or write XML."""
    config = load_scene_npc_config(config_path)
    try:
        _validate_interaction_sources(config)
        _, _, coverage, _ = compile_semantics(config, check_navigation=False)
        return {"scene_id": config.scene_id, "status": "closed", "coverage": coverage}
    except ValueError as error:
        return {"scene_id": config.scene_id, "status": "unresolved", "error": str(error)}
