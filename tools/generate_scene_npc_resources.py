"""Deterministically materialize taxonomy and all active scene configs."""

from __future__ import annotations
import json
from pathlib import Path

import mujoco
import numpy as np

from stretch_mujoco.humanoid.navigation import OfficeNavigationMesh
try:
    from tools.propose_interaction_sites import propose_plan
except ModuleNotFoundError:
    from propose_interaction_sites import propose_plan

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "stretch_mujoco" / "models"
AGENT_RADIUS = 0.16
OFFICE_CLEARANCE = 0.06
HOME_CLEARANCE = 0.0
OFFICE_RESOLUTION = 0.08
HOME_RESOLUTION = 0.06
SPAWN_ANCHOR_COUNT = 10
SPAWN_ANCHOR_SEPARATION = 0.70


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def spawn_anchor_targets(
    mjcf_path: Path,
    manifest: dict,
    kind: str,
    existing_targets: dict[str, dict],
) -> dict[str, dict]:
    """Choose ten deterministic, collision-free spawn anchors per scene."""
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = manifest.get("robot", {})
    robot_body = robot.get("body") if isinstance(robot, dict) else None
    navigation = OfficeNavigationMesh.from_model(
        model,
        data,
        resolution=OFFICE_RESOLUTION if kind == "office" else HOME_RESOLUTION,
        agent_radius=AGENT_RADIUS + (OFFICE_CLEARANCE if kind == "office" else HOME_CLEARANCE),
        floor_geom_name="office_floor" if kind == "office" else "hssd_floor_collision",
        exclude_body_roots=(robot_body,) if isinstance(robot_body, str) and robot_body else (),
    )
    free_cells = np.argwhere(navigation.component_labels == navigation.primary_component_id)
    candidates = [navigation.cell_to_world(tuple(cell)) for cell in free_cells]
    existing = [np.asarray(v["position"][:2], dtype=float) for v in existing_targets.values()]
    selected: list[np.ndarray] = []
    # Farthest-point sampling gives useful coverage across the room instead of
    # placing all ten anchors in one large open patch.
    for _ in range(SPAWN_ANCHOR_COUNT):
        pool = []
        for point in candidates:
            xy = np.asarray(point[:2], dtype=float)
            prior = existing + selected
            if prior and min(float(np.linalg.norm(xy - item)) for item in prior) < SPAWN_ANCHOR_SEPARATION:
                continue
            score = min((float(np.linalg.norm(xy - item)) for item in prior), default=1e9)
            pool.append((score, float(xy[0]), float(xy[1]), xy))
        if not pool:
            raise ValueError(f"{kind} scene cannot provide {SPAWN_ANCHOR_COUNT} spawn anchors")
        selected.append(max(pool, key=lambda item: (item[0], item[1], item[2]))[3])
    owner = "zone.home" if kind == "home" else "zone.work"
    return {
        f"point.spawn.anchor.{index:02d}": {
            "owner": owner,
            "position": [round(float(point[0]), 9), round(float(point[1]), 9), 0.025],
            "yaw": 0.0,
            "usages": ["navigation", "spawn", "spawn_anchor"],
        }
        for index, point in enumerate(selected, start=1)
    }


def default_semantic_definition(name: str) -> dict:
    normalized = name.casefold()
    if any(word in normalized for word in ("table", "counter", "island", "desk")):
        return {
            "semantic_class": "furniture.counter",
            "affordances": ["approach"],
            "navigation_requirement": "approach",
            "point_bundle": "perimeter_approach",
        }
    if any(
        word in normalized
        for word in ("cabinet", "wardrobe", "dresser", "drawer", "shelf", "bookcase")
    ):
        return {
            "semantic_class": "furniture.storage",
            "affordances": ["approach"],
            "navigation_requirement": "approach",
            "point_bundle": "perimeter_approach",
        }
    if any(
        word in normalized
        for word in ("chair", "sofa", "couch", "stool", "bench", "bed", "ottoman")
    ):
        semantic_class = "furniture.bed" if "bed" in normalized else "furniture.seat"
        return {
            "semantic_class": semantic_class,
            "affordances": ["approach", "sit"],
            "navigation_requirement": "approach",
            "point_bundle": "seat",
        }
    if any(word in normalized for word in ("toilet", "bath", "shower", "sink", "basin")):
        return {
            "semantic_class": "furniture.fixture",
            "affordances": ["approach"],
            "navigation_requirement": "approach",
            "point_bundle": "perimeter_approach",
        }
    if "door" in normalized:
        return {
            "semantic_class": "architecture.door",
            "affordances": ["observe"],
            "navigation_requirement": "observe",
            "point_bundle": "observation",
        }
    return {
        "semantic_class": "object.observable",
        "affordances": ["observe"],
        "navigation_requirement": "observe",
        "point_bundle": "observation",
    }


def policy(kind: str, defaults: set[str]) -> dict:
    rules = []
    categories = {
        "office": {
            "interactive_objects": ("object.graspable", "approach", "graspable_on_support"),
            "meeting_chairs": ("furniture.seat", "approach", "seat"),
            "props/plants": ("object.observable", "observe", "observation"),
            "props/trash_bins": ("object.observable", "observe", "observation"),
            "decoration/rugs": ("object.observable", "observe", "observation"),
            "legacy_sofas": ("furniture.seat", "approach", "seat"),
            "workstation_pods": ("furniture.workstation", "approach", "perimeter_approach"),
            "meeting_tables": ("furniture.counter", "approach", "perimeter_approach"),
            "displays": ("object.observable", "observe", "observation"),
            "whiteboards": ("object.observable", "observe", "observation"),
            "furniture/snack_counter": ("furniture.counter", "approach", "perimeter_approach"),
            "furniture/storage": ("furniture.storage", "approach", "perimeter_approach"),
        },
        "home": {
            "storage_furniture": ("furniture.storage", "approach", "perimeter_approach"),
            "interactive_objects": ("object.graspable", "approach", "graspable_on_support"),
            "lighting": ("object.observable", "observe", "observation"),
            "seating_furniture": ("furniture.seat", "approach", "seat"),
            "decor": ("object.observable", "observe", "observation"),
            "support_furniture": ("furniture.support", "approach", "perimeter_approach"),
            "plant": ("object.observable", "observe", "observation"),
            "floor_covering": ("object.observable", "observe", "observation"),
            "electronics": ("object.observable", "observe", "observation"),
            "desks": ("furniture.workstation", "approach", "perimeter_approach"),
            "computers": ("object.observable", "observe", "observation"),
            "chairs": ("furniture.seat", "approach", "seat"),
        },
    }[kind]
    rules.append(
        {
            "id": "regions",
            "match": {"kind": "region"},
            "semantic_class": f"region.{kind}",
            "affordances": ["navigate"],
            "navigation_requirement": "region",
            "point_bundle": "region_samples",
        }
    )
    for category, (semantic_class, requirement, bundle) in categories.items():
        rules.append(
            {
                "id": "category_" + category.replace("/", "_"),
                "match": {"category": category},
                "semantic_class": semantic_class,
                "affordances": [requirement],
                "navigation_requirement": requirement,
                "point_bundle": bundle,
            }
        )
    taxonomy = {name: default_semantic_definition(name) for name in sorted(defaults)}
    return {
        "schema": "semantic_policy/v1",
        "kind": kind,
        "fallback": "unresolved",
        "rules": rules,
        "semantic_name_taxonomy": taxonomy,
    }


def project_home_targets(
    manifest_path: Path, mjcf_path: Path, plan: dict
) -> tuple[dict, dict, dict, dict]:
    """Move legacy demo slots onto distinct cells in the main walkable component."""
    manifest = json.loads(manifest_path.read_text())
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = manifest.get("robot", {})
    robot_body = robot.get("body") if isinstance(robot, dict) else None
    navigation = OfficeNavigationMesh.from_model(
        model,
        data,
        resolution=HOME_RESOLUTION,
        agent_radius=AGENT_RADIUS + HOME_CLEARANCE,
        floor_geom_name="hssd_floor_collision",
        exclude_body_roots=(robot_body,) if isinstance(robot_body, str) and robot_body else (),
    )
    free_cells = np.argwhere(navigation.component_labels == navigation.primary_component_id)
    selected: list[np.ndarray] = []
    projected_sites: dict[str, dict] = {}
    migrations: dict[str, dict] = {}
    for site_id, definition in sorted(plan["sites"].items()):
        desired = np.asarray(definition["position"], dtype=float)[:2]
        candidates = sorted(
            (
                (
                    float(np.linalg.norm(navigation.cell_to_world(tuple(cell)) - desired)),
                    tuple(int(value) for value in cell),
                )
                for cell in free_cells
            ),
            key=lambda item: (item[0], item[1]),
        )
        chosen = next(
            navigation.cell_to_world(cell)
            for _, cell in candidates
            if all(
                float(np.linalg.norm(navigation.cell_to_world(cell) - prior)) >= 0.55
                for prior in selected
            )
        )
        selected.append(chosen)
        position = [round(float(chosen[0]), 9), round(float(chosen[1]), 9)]
        projected_sites[site_id] = dict(definition, position=position)
        migrations[site_id] = {
            "legacy_position": [float(value) for value in desired],
            "registered_position": position,
            "displacement_m": round(float(np.linalg.norm(chosen - desired)), 9),
            "component_id": int(navigation.primary_component_id),
        }
    room_names = {
        "bathroom": "bathroom",
        "bedroom": "bedroom",
        "kitchen": "kitchen",
        "living room": "living_room",
        "office": "office",
    }
    region_overrides: dict[str, dict] = {}
    room_provenance: dict[str, dict] = {}
    for source_room, room_id in room_names.items():
        anchors = [
            asset
            for asset in manifest["assets"]
            if str(asset.get("found_in", "")).strip() == source_room
            and isinstance(asset.get("position"), list)
            and len(asset["position"]) >= 2
        ]
        if not anchors:
            continue
        positions = [np.asarray(asset["position"][:2], dtype=float) for asset in anchors]
        anchor_index = min(
            range(len(anchors)),
            key=lambda index: (
                sum(float(np.linalg.norm(positions[index] - other)) for other in positions),
                str(anchors[index].get("object_id") or anchors[index].get("asset_id")),
            ),
        )
        desired = positions[anchor_index]
        _, room_cell = min(
            (
                (
                    float(np.linalg.norm(navigation.cell_to_world(tuple(cell)) - desired)),
                    tuple(int(value) for value in cell),
                )
                for cell in free_cells
            ),
            key=lambda item: (item[0], item[1]),
        )
        seed = navigation.cell_to_world(room_cell)
        semantic_id = f"room.{room_id}"
        region_overrides[semantic_id] = {
            "semantic_class": f"region.room.{room_id}",
            "geometry": {
                "mode": "seed_and_component",
                "seed": [round(float(seed[0]), 9), round(float(seed[1]), 9), 0.025],
            },
            "required": True,
        }
        room_provenance[semantic_id] = {
            "source_hint": source_room,
            "anchor_object": str(
                anchors[anchor_index].get("object_id") or anchors[anchor_index].get("asset_id")
            ),
            "anchor_position": [float(value) for value in desired],
            "registered_seed": region_overrides[semantic_id]["geometry"]["seed"],
            "candidate_object_count": len(anchors),
            "component_id": int(navigation.primary_component_id),
        }
    return projected_sites, migrations, region_overrides, room_provenance


def main() -> None:
    configs = MODELS / "scene_npc_configs"
    plans = json.loads((MODELS / "assets/home_scenes/npc_slot_overrides.json").read_text())[
        "scenes"
    ]
    home_entity_overrides = json.loads(
        (MODELS / "semantic_policies" / "home_scene_overrides.json").read_text()
    )["scenes"]
    office_population_plans = json.loads(
        (MODELS / "scene_npc_configs" / "office_population_plans.json").read_text()
    )["scenes"]
    home_slot_migrations = {}
    home_room_overlays = {}
    for kind, directory in [
        ("office", MODELS / "assets/office_scenes"),
        ("home", MODELS / "assets/home_scenes"),
    ]:
        defaults = set()
        for manifest in sorted(directory.glob(f"{kind}_*.json")):
            if manifest.name.endswith("_npc.json"):
                continue
            stem = manifest.stem
            mjcf_path = directory / f"{stem}.xml"
            interaction_plan = propose_plan(mjcf_path, manifest, kind)
            plan_path = MODELS / "scene_interaction_plans" / kind / f"{stem}.json"
            write(plan_path, interaction_plan)
            data = json.loads(manifest.read_text())
            defaults.update(
                str(a.get("semantic_name") or a.get("name") or a.get("object_id") or "")
                for a in data["assets"]
                if a.get("category") == "default"
            )
            if kind == "home":
                plan = plans[stem]
                projected_sites, migrations, region_overrides, room_provenance = (
                    project_home_targets(
                        manifest,
                        mjcf_path,
                        plan,
                    )
                )
                home_slot_migrations[stem] = migrations
                home_room_overlays[stem] = room_provenance
                custom_targets = {
                    f"point.home.{site_id}": {
                        "owner": "zone.home",
                        "position": definition["position"],
                        "yaw": float(definition.get("yaw", 0.0)),
                        "usages": ["navigation", *definition.get("tags", [])],
                    }
                    for site_id, definition in projected_sites.items()
                }
                custom_targets.update(
                    spawn_anchor_targets(mjcf_path, data, kind, custom_targets)
                )
                roster = [
                    {
                        "npc": entry["npc_id"],
                        "spawn": f"point.home.{entry['site']}",
                        "initial_region": entry.get("location", "zone.home"),
                    }
                    for entry in plan["roster"]
                ]
                routes = [
                    {
                        "id": "living_to_kitchen",
                        "from": "room.living_room",
                        "to": "room.kitchen",
                        "mode": "audited_dynamic",
                        "actions": ["move_to", "household_activity"],
                    },
                    {
                        "id": "bedroom_to_living",
                        "from": "room.bedroom",
                        "to": "room.living_room",
                        "mode": "fixed_contract",
                        "actions": ["move_to", "relax"],
                    },
                ]
            else:
                custom_targets = spawn_anchor_targets(mjcf_path, data, kind, {})
                region_overrides = {}
                roster = office_population_plans[stem]
                routes = [
                    {
                        "id": "work_to_meeting",
                        "from": "zone.work",
                        "to": "zone.meeting",
                        "mode": "fixed_contract",
                        "actions": ["move_to", "attend_meeting"],
                    },
                    {
                        "id": "meeting_to_snack",
                        "from": "zone.meeting",
                        "to": "zone.snack",
                        "mode": "audited_dynamic",
                        "actions": ["move_to", "take_break"],
                    },
                ]
            payload = {
                "schema": "scene_npc_config/v2",
                "scene": {
                    "id": stem,
                    "kind": kind,
                    "source_mjcf": f"../../assets/{kind}_scenes/{stem}.xml",
                    "source_manifest": f"../../assets/{kind}_scenes/{stem}.json",
                    "semantic_policy": f"../../semantic_policies/{kind}.json",
                    "npc_catalog": "../../office_population.production.example.json",
                    "navigation": {
                        "surface": "office_floor" if kind == "office" else "hssd_floor_collision",
                        "planner": "collision_geometry_v1",
                        "agent_radius": AGENT_RADIUS,
                        "clearance": (OFFICE_CLEARANCE if kind == "office" else HOME_CLEARANCE),
                        "resolution": (OFFICE_RESOLUTION if kind == "office" else HOME_RESOLUTION),
                    },
                },
                "semantic_registration": {
                    "strict_coverage": True,
                    "discover": {
                        "manifest_assets": True,
                        "manifest_zones": True,
                        "xml_semantic_sites": True,
                    },
                    "region_overrides": region_overrides,
                    "entity_overrides": (
                        home_entity_overrides.get(stem, {}) if kind == "home" else {}
                    ),
                    "custom_targets": custom_targets,
                },
                "population": {"members": roster},
                "spawn_policy": {
                    "capacity": SPAWN_ANCHOR_COUNT,
                    "minimum_separation_m": SPAWN_ANCHOR_SEPARATION,
                    "allocation": "exclusive",
                    "anchors": sorted(
                        point_id for point_id, definition in custom_targets.items()
                        if "spawn_anchor" in definition.get("usages", [])
                    ),
                },
                "route_coverage": {
                    "origins": "all_population_spawns",
                    "targets": "all_required_navigation_points",
                    "connectivity": "strongly_connected",
                    "preflight": "required",
                },
                "routes": routes,
                "traffic": {
                    "policy": "sequential_route_reservation",
                    "dynamic_obstacles": True,
                    "on_block": "wait_then_replan",
                    "max_wait_s": 12.0,
                    "reservation_horizon_s": 30.0,
                    "no_direct_fallback": True,
                },
                "interaction_plan": {
                    "path": f"../../scene_interaction_plans/{kind}/{stem}.json",
                    "required_capabilities": ["conversation", "handover", "sit", "work"],
                },
                "robot_navigation": {
                    "footprint_radius": 0.32,
                    "clearance": 0.08,
                    "resolution": (OFFICE_RESOLUTION if kind == "office" else HOME_RESOLUTION),
                },
                "runtime": {"max_replans": 3, "route_failure": "fail"},
            }
            write(configs / kind / f"{stem}.json", payload)
        write(MODELS / "semantic_policies" / f"{kind}.json", policy(kind, defaults))
    write(
        configs / "catalog.json",
        {
            "schema_version": 1,
            "configs": [
                str(path.relative_to(configs))
                for path in sorted(configs.glob("office/*.json"))
                + sorted(configs.glob("home/*.json"))
            ],
        },
    )
    write(
        MODELS / "generated_scene_npc" / "provenance" / "home_slot_migration.json",
        {
            "schema_version": 1,
            "method": "nearest_distinct_cell_in_primary_component",
            "agent_radius": AGENT_RADIUS,
            "clearance": HOME_CLEARANCE,
            "resolution": HOME_RESOLUTION,
            "scenes": home_slot_migrations,
        },
    )
    write(
        MODELS / "generated_scene_npc" / "provenance" / "home_room_overlays.json",
        {
            "schema_version": 1,
            "method": "single_room_hint_medoid_projected_to_primary_component",
            "agent_radius": AGENT_RADIUS,
            "clearance": HOME_CLEARANCE,
            "resolution": HOME_RESOLUTION,
            "scenes": home_room_overlays,
        },
    )


if __name__ == "__main__":
    main()
