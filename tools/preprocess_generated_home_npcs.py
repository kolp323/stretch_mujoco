"""Prepare the ten generated HSSD home scenes for population-driven NPCs.

The source home XMLs are intentionally left untouched.  This preprocessor
adds an explicit named-site contract to a sibling ``*_npc.xml``, plus a
navigation-floor alias and an overview camera. Population and semantic
sidecars are written below ``models/generated_home_npc``.

The HSSD mesh cache is not required to author these files, but is required by
MuJoCo when compiling/rendering the resulting scenes.  A source-adjacent
``npc_slot_overrides.json`` is the authoritative named-site and roster config.
It is required because the manifest alone cannot represent the indoor floor
topology of these scenes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from stretch_mujoco.npc.scene_site_config import SceneSitePlan, load_scene_site_plans


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = PROJECT_ROOT / "stretch_mujoco" / "models"
DEFAULT_SOURCE = MODELS / "assets" / "home_scenes"
DEFAULT_OUTPUT = MODELS / "generated_home_npc"
POPULATION_TEMPLATE = (
    MODELS / "generated_office_npc" / "populations" / "office_01_linear_bench.population.json"
)
SLOT_OVERRIDE_FILENAME = "npc_slot_overrides.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_stem(path: Path) -> str:
    if (
        not path.name.startswith("home_")
        or path.suffix != ".xml"
        or path.name.endswith("_robot.xml")
    ):
        raise ValueError(f"Not a home scene XML: {path}")
    return path.stem


def _object_boxes(manifest: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    boxes = []
    for item in manifest.get("assets", []):
        bounds = item.get("bounds_mujoco")
        if not isinstance(bounds, list) or len(bounds) != 2:
            continue
        try:
            boxes.append(
                (float(bounds[0][0]), float(bounds[0][1]), float(bounds[1][0]), float(bounds[1][1]))
            )
        except (IndexError, TypeError, ValueError):
            continue
    return boxes


def _validate_site_plan(plan: SceneSitePlan, manifest: dict[str, Any]) -> None:
    """Check source-authored sites before emitting XML or derived sidecars."""
    bounds = manifest.get("bounds_mujoco")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("Home manifest requires bounds_mujoco")
    x0, y0 = float(bounds[0][0]), float(bounds[0][1])
    x1, y1 = float(bounds[1][0]), float(bounds[1][1])
    for site in plan.sites.values():
        x, y, _ = site.position
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            raise ValueError(f"{plan.scene_id}: site '{site.name}' lies outside manifest bounds")
    spawn_sites = [plan.sites[entry.site].position for entry in plan.roster]
    for index, left in enumerate(spawn_sites):
        for right in spawn_sites[index + 1 :]:
            if math.hypot(left[0] - right[0], left[1] - right[1]) < 0.55:
                raise ValueError(f"{plan.scene_id}: spawn sites are closer than 0.55 m")


def _bed_sites(
    manifest: dict[str, Any], *, sit_clearance: float = 0.30, approach_clearance: float = 0.75
) -> tuple[dict[str, Any], ...]:
    """Derive a bed-edge sit cue and an exterior navigation ingress cue."""
    bounds = manifest.get("bounds_mujoco")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("Home manifest requires bounds_mujoco")
    x0, y0 = float(bounds[0][0]), float(bounds[0][1])
    x1, y1 = float(bounds[1][0]), float(bounds[1][1])
    boxes = _object_boxes(manifest)
    beds: list[dict[str, Any]] = []
    for asset in manifest.get("assets", []):
        semantic_name = str(asset.get("semantic_name") or "")
        if (
            not re.search(r"\bbed\b", semantic_name, flags=re.IGNORECASE)
            or "pet bed" in semantic_name.lower()
        ):
            continue
        raw_bounds = asset.get("bounds_mujoco")
        if not isinstance(raw_bounds, list) or len(raw_bounds) != 2:
            continue
        try:
            bx0, by0, bz0 = (float(value) for value in raw_bounds[0])
            bx1, by1, bz1 = (float(value) for value in raw_bounds[1])
        except (TypeError, ValueError):
            continue
        center = ((bx0 + bx1) / 2.0, (by0 + by1) / 2.0)
        half_x, half_y = abs(bx1 - bx0) / 2.0, abs(by1 - by0) / 2.0
        directions = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
        candidates = []
        for dx, dy in directions:
            half_extent = half_x if dx else half_y
            sit = (
                center[0] + dx * (half_extent + sit_clearance),
                center[1] + dy * (half_extent + sit_clearance),
            )
            approach = (
                center[0] + dx * (half_extent + approach_clearance),
                center[1] + dy * (half_extent + approach_clearance),
            )
            # The NPC faces outward, leaving its back toward the mattress.
            yaw = math.atan2(dx, -dy)
            candidates.append((sit, approach, yaw))
        valid = [
            candidate
            for candidate in candidates
            if x0 + sit_clearance <= candidate[0][0] <= x1 - sit_clearance
            and y0 + sit_clearance <= candidate[0][1] <= y1 - sit_clearance
            and x0 + approach_clearance <= candidate[1][0] <= x1 - approach_clearance
            and y0 + approach_clearance <= candidate[1][1] <= y1 - approach_clearance
        ]
        if not valid:
            continue

        def candidate_clearance(
            candidate: tuple[tuple[float, float], tuple[float, float], float],
        ) -> float:
            x, y = candidate[1]
            distances = [
                math.hypot(max(lo - x, 0.0, x - hi), max(low - y, 0.0, y - high))
                for lo, low, hi, high in boxes
                if (lo, low, hi, high) != (bx0, by0, bx1, by1)
            ]
            return min(distances, default=10.0)

        sit, approach, yaw = max(valid, key=candidate_clearance)
        beds.append(
            {
                "object_id": str(asset.get("object_id") or ""),
                "semantic_name": semantic_name,
                "sit": sit,
                "approach": approach,
                "yaw": yaw,
                "bounds_z": (bz0, bz1),
            }
        )
    return tuple(beds)


def _append_npc_contract(
    source: Path,
    destination: Path,
    plan: SceneSitePlan,
    beds: tuple[dict[str, Any], ...],
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{source}: missing worldbody")
    existing_sites = {node.get("name") for node in worldbody.findall(".//site")}
    bed_site_names = tuple(
        name
        for index in range(1, len(beds) + 1)
        for name in (
            f"npc_home_bed_{index:02d}_sit_site",
            f"npc_home_bed_{index:02d}_approach_site",
        )
    )
    if any(site in existing_sites for site in (*plan.sites, *bed_site_names)):
        raise ValueError(f"{source}: NPC contract sites already exist; use a fresh destination")
    for site in plan.sites.values():
        x, y, z = site.position
        ET.SubElement(
            worldbody,
            "site",
            name=site.name,
            pos=f"{x:.9g} {y:.9g} {z:.9g}",
            euler=f"0 0 {site.yaw:.9g}",
            size="0.015",
            rgba="0 0 0 0",
        )
    for index, bed in enumerate(beds, start=1):
        sit_x, sit_y = bed["sit"]
        approach_x, approach_y = bed["approach"]
        ET.SubElement(
            worldbody,
            "site",
            name=f"npc_home_bed_{index:02d}_sit_site",
            pos=f"{sit_x:.9g} {sit_y:.9g} 0.025",
            euler="0 0 0",
            size="0.02",
            rgba="0 0 0 0",
        )
        ET.SubElement(
            worldbody,
            "site",
            name=f"npc_home_bed_{index:02d}_approach_site",
            pos=f"{approach_x:.9g} {approach_y:.9g} 0.025",
            euler="0 0 0",
            size="0.02",
            rgba="0 0 0 0",
        )

    floor = worldbody.find("./geom[@name='hssd_floor_collision']")
    if floor is None:
        raise ValueError(f"{source}: missing hssd_floor_collision")
    nav_floor = copy.deepcopy(floor)
    nav_floor.set("name", "office_floor")
    nav_floor.set("contype", "0")
    nav_floor.set("conaffinity", "0")
    nav_floor.set("rgba", "0 0 0 0")
    worldbody.append(nav_floor)

    camera = worldbody.find("./camera[@name='habitat_overview']")
    if camera is not None and worldbody.find("./camera[@name='office_overview']") is None:
        overview = copy.deepcopy(camera)
        overview.set("name", "office_overview")
        worldbody.append(overview)
    root.insert(0, ET.Comment(" Generated by preprocess_generated_home_npcs.py; do not edit. "))
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode", xml_declaration=False)


def _population(template: dict[str, Any], scene: Path, plan: SceneSitePlan) -> dict[str, Any]:
    payload = copy.deepcopy(template)
    payload["schema_version"] = 2
    payload["scene"] = Path("../../assets/home_scenes").joinpath(scene.name).as_posix()
    payload.pop("trajectory_profile", None)
    payload.pop("interaction_templates", None)
    selected: dict[str, Any] = {}
    for spawn in plan.roster:
        if spawn.npc_id not in payload["npcs"]:
            raise ValueError(f"{plan.scene_id}: roster NPC '{spawn.npc_id}' is absent from template")
        definition = copy.deepcopy(payload["npcs"][spawn.npc_id])
        definition["spawn"] = {
            "location": spawn.location,
            "site": spawn.site,
            "yaw": spawn.yaw,
        }
        selected[spawn.npc_id] = definition
    payload["npcs"] = selected
    return payload


def _semantic(
    scene: Path,
    plan: SceneSitePlan,
    beds: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    activity_site = plan.demo.activity_site if plan.demo is not None else plan.roster[0].site
    payload = {
        "schema_version": 1,
        "scene": scene.name,
        "objects": {
            "home": {
                "type": "Counter",
                "binding": {"kind": "site", "name": activity_site},
                "attributes": {"region": "home", "capacity": len(plan.roster)},
            }
        },
        "relations": [],
        "interaction_points": {
            f"home_spawn_{spawn.npc_id}": {
                "role": "human_stand_site",
                "owner": "home",
                "site": spawn.site,
                "attributes": {
                    "binding": "location",
                    "target": "home",
                    "slot_id": spawn.site,
                    "npc_id": spawn.npc_id,
                    "usage": "spawn",
                    "yaw": spawn.yaw,
                },
            }
            for spawn in plan.roster
        },
    }
    if plan.demo is not None:
        payload["interaction_points"]["home_activity"] = {
            "role": "human_stand_site",
            "owner": "home",
            "site": plan.demo.activity_site,
            "attributes": {
                "binding": "location",
                "target": "home",
                "usage": "activity",
                "yaw": plan.sites[plan.demo.activity_site].yaw,
            },
        }
    for index, bed in enumerate(beds, start=1):
        bed_id = f"bed_{index:02d}"
        sit_site = f"npc_home_bed_{index:02d}_sit_site"
        approach_site = f"npc_home_bed_{index:02d}_approach_site"
        payload["objects"][bed_id] = {
            "type": "Bed",
            "binding": {"kind": "body", "name": bed["object_id"]},
            "attributes": {
                "region": "home",
                "capacity": 1,
                "furniture_kind": "bed",
                "semantic_name": bed["semantic_name"],
            },
        }
        payload["interaction_points"][f"{bed_id}_sit"] = {
            "role": "chair_sit_site",
            "owner": bed_id,
            "site": sit_site,
            "attributes": {
                "binding": "location",
                "target": bed_id,
                "seat_type": "bed",
                "yaw": bed["yaw"],
            },
        }
        payload["interaction_points"][f"{bed_id}_approach"] = {
            "role": "human_stand_site",
            "owner": bed_id,
            "site": approach_site,
            "attributes": {
                "binding": "seat_navigation",
                "target": bed_id,
            },
        }
    return payload


def preprocess(source: Path, output: Path, template_path: Path) -> list[Path]:
    output = output.resolve()
    template = json.loads(template_path.read_text(encoding="utf-8"))
    plan_path = source / SLOT_OVERRIDE_FILENAME
    if not plan_path.is_file():
        raise FileNotFoundError(f"Home scene site config is required: {plan_path}")
    site_plans = load_scene_site_plans(plan_path)
    scenes_output = []
    for source_scene in sorted(source.glob("home_*.xml")):
        if source_scene.name.endswith("_robot.xml") or source_scene.name.endswith("_npc.xml"):
            continue
        stem = _source_stem(source_scene)
        manifest_path = source_scene.with_suffix(".json")
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            plan = site_plans[stem]
        except KeyError as error:
            raise ValueError(f"Home scene '{stem}' has no named site plan in {plan_path}") from error
        _validate_site_plan(plan, manifest)
        beds = _bed_sites(manifest)
        generated_scene = source_scene.with_name(f"{stem}_npc.xml")
        _append_npc_contract(source_scene, generated_scene, plan, beds)
        population_dir = output / "populations"
        semantic_dir = output / "semantics"
        receipt_dir = output / "receipts"
        population_path = population_dir / f"{stem}.population.json"
        semantic_path = semantic_dir / f"{stem}.semantic.json"
        population_path.parent.mkdir(parents=True, exist_ok=True)
        semantic_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_dir.mkdir(parents=True, exist_ok=True)
        population_path.write_text(
            json.dumps(_population(template, generated_scene, plan), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        semantic_path.write_text(
            json.dumps(_semantic(generated_scene, plan, beds), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt = {
            "schema_version": 1,
            "generator": "tools/preprocess_generated_home_npcs.py",
            "scene": generated_scene.name,
            "source_scene": source_scene.name,
            "npc_ids": [entry.npc_id for entry in plan.roster],
            "sites": {
                name: {"position": list(site.position), "yaw": site.yaw, "tags": sorted(site.tags)}
                for name, site in plan.sites.items()
            },
            "roster": [
                {"npc_id": entry.npc_id, "site": entry.site, "location": entry.location, "yaw": entry.yaw}
                for entry in plan.roster
            ],
            "beds": beds,
            "sha256": {
                "source_scene": _sha256(source_scene),
                "generated_scene": _sha256(generated_scene),
                "population": _sha256(population_path),
                "semantic": _sha256(semantic_path),
            },
        }
        (receipt_dir / f"{stem}.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        scenes_output.append(generated_scene)
    if len(scenes_output) != 10:
        raise ValueError(f"Expected 10 home scenes, found {len(scenes_output)}")
    catalog = {
        "schema_version": 1,
        "scene_count": len(scenes_output),
        "source_directory": str(source),
        "scenes": [
            {
                "scene": scene.name,
                "population": f"populations/{scene.stem.removesuffix('_npc')}.population.json",
                "semantic_world": f"semantics/{scene.stem.removesuffix('_npc')}.semantic.json",
                "receipt": f"receipts/{scene.stem.removesuffix('_npc')}.json",
            }
            for scene in scenes_output
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "catalog.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return scenes_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template-population", type=Path, default=POPULATION_TEMPLATE)
    args = parser.parse_args()
    generated = preprocess(
        args.source.resolve(), args.output.resolve(), args.template_population.resolve()
    )
    print(f"Prepared {len(generated)} home NPC scenes under {args.output.resolve()}")


if __name__ == "__main__":
    main()
