"""Generate ten simplified home scenes from ten distinct HSSD scene IDs.

Unlike ``generate_office_scenes.py``, this script does not reuse one hand-made
layout with furniture permutations.  Each output scene is a conversion of a
different HSSD scene instance (the uncluttered split), with a Stretch robot
included and a manifest that records the source HSSD ID.

The converted HSSD assets are cached below the output directory for reproducible
reruns. HSSD itself is not redistributed by this repository; pass
``--hssd-root`` when it is installed somewhere else.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import click
import cv2
import mujoco
import numpy as np

from stretch_mujoco.habitat_scene_gallery import (
    DEFAULT_HSSD_ROOT,
    prepare_habitat_scene,
)
try:
    from office_interactive_assets import (
        corrected_interactive_collision_bounds,
        interactive_euler,
        load_interactive_assets,
    )
except ModuleNotFoundError:
    from tools.office_interactive_assets import (
        corrected_interactive_collision_bounds,
        interactive_euler,
        load_interactive_assets,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "stretch_mujoco" / "models"
STRETCH_XML = MODELS_ROOT / "stretch.xml"
DEFAULT_OUTPUT = MODELS_ROOT / "assets" / "home_scenes"
ASSETS_ROOT = MODELS_ROOT / "assets"

# These distinct HSSD scenes were selected by screening the uncluttered split
# for compact stage footprints, low object counts, and relatively simple
# top-down structure.  They are intended to provide short, tractable home
# navigation tests instead of large, many-room interiors.
DEFAULT_SCENE_IDS = (
    "102344115",
    "103997919_171031233",
    "107734119_175999932",
    "103997970_171031287",
    "102344094",
    "104348082_171512994",
    "104348463_171513588",
    "102344193",
    "107734110_175999914",
    "104862513_172226580",
)
HOME_INTERACTIVE_IDS = (
    "017_calculator", "043_book", "116_keyboard", "101_milk-tea", "snack_soda_can",
    "001_bottle", "025_chips-tub", "035_apple", "038_milk-box", "071_can",
    "green_apple", "075_bread", "069_vagetable"
)
# Office supplies belong on a desk, not a kitchen counter or coffee table.
OFFICE_ASSET_IDS = frozenset({"017_calculator", "043_book", "116_keyboard"})
# Food items that read naturally on a kitchen surface; vegetable is the
# clearest case (an explicit example from the task), fruit/bread follow.
KITCHEN_PREFERRED_ASSET_IDS = ("069_vagetable", "035_apple", "green_apple", "075_bread")


def _numbers(values: tuple[float, ...] | np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _scene_name(index: int, hssd_id: str) -> str:
    return f"home_{index:02d}_{hssd_id}"


def _asset_file_path(path: str | Path) -> str:
    """Return a portable path relative to MuJoCo's shared asset directory."""
    resolved = Path(path).resolve()
    return Path(os.path.relpath(resolved, ASSETS_ROOT.resolve())).as_posix()


def _make_asset_paths_portable(scene_xml: Path) -> None:
    """Rewrite converted HSSD asset references before committing the scene."""
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    for node in root.findall(".//*[@file]"):
        file_path = node.get("file")
        if file_path and Path(file_path).is_absolute():
            node.set("file", _asset_file_path(file_path))
    ET.indent(root, space="  ")
    tree.write(scene_xml, encoding="unicode", xml_declaration=False)


def _write_robot_include(
    output_dir: Path, scene_name: str, bounds: np.ndarray, yaw: float, start_xy=None
) -> tuple[Path, tuple[float, float, float]]:
    """Write a robot-only include and choose a deterministic open-area start.

    HSSD scene geoms are visual-only, so this start is intentionally a simple
    geometric choice near the scene center.  It is recorded in the manifest and
    can be changed later without regenerating the converted HSSD assets.
    """
    tree = ET.parse(STRETCH_XML)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("stretch.xml has no compiler element")
    compiler.set(
        "assetdir",
        os.path.relpath(ASSETS_ROOT.resolve(), output_dir.resolve()),
    )
    base = root.find("./worldbody/body[@name='base_link']")
    if base is None:
        raise RuntimeError("stretch.xml has no base_link body")
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    # Stay away from the outer shell while retaining a stable, reproducible
    # position for all source scenes.
    xy = np.asarray(start_xy, dtype=float) if start_xy is not None else lo[:2] + 0.50 * (hi[:2] - lo[:2])
    start = (float(xy[0]), float(xy[1]), float(yaw))
    base.set("pos", _numbers((start[0], start[1], 0.0)))
    base.set("quat", _numbers((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))))
    path = output_dir / f"{scene_name}_robot.xml"
    ET.indent(root, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=False)
    return path, start


def _add_robot_include(scene_xml: Path, robot_xml: Path, scene_name: str) -> None:
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    root.set("model", scene_name)
    # The include is placed before the scene compiler, matching the existing
    # office generator and allowing Stretch's defaults/assets to be merged.
    root.insert(0, ET.Element("include", {"file": robot_xml.name}))
    ET.indent(root, space="  ")
    tree.write(scene_xml, encoding="unicode", xml_declaration=False)


def _wall_collision_boxes(scene_xml: Path) -> list[np.ndarray]:
    """World AABBs of the stage's wall/doorway colliders in a compiled scene.

    `object_records` only tracks furniture, so a clearance search built from
    it alone can happily settle on a spot squeezed into a doorway or hallway
    -- narrower than the home's outer bounds but wide open by furniture
    standards. These wall segments (see `_add_stage_wall_collisions`) are
    axis-aligned boxes directly on the world body, so their AABB is exact.
    """
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    boxes = []
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if not name.startswith("habitat_wall_collision_"):
            continue
        pos, size = model.geom_pos[geom_id], model.geom_size[geom_id]
        boxes.append(np.stack((pos - size, pos + size)))
    return boxes


def _choose_open_start(
    bounds: np.ndarray, object_records: list[dict], wall_boxes: list[np.ndarray] = ()
) -> tuple[float, float]:
    """Choose the grid point with maximum clearance from furniture and walls."""
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    boxes = [np.asarray(r["bounds_mujoco"], float) for r in object_records if r.get("bounds_mujoco")]
    boxes.extend(wall_boxes)
    best, best_score = None, -1.0
    for x in np.linspace(lo[0] + 1.0, hi[0] - 1.0, 15):
        for y in np.linspace(lo[1] + 1.0, hi[1] - 1.0, 15):
            distances = []
            for box in boxes:
                dx = max(box[0, 0] - x, 0.0, x - box[1, 0])
                dy = max(box[0, 1] - y, 0.0, y - box[1, 1])
                distances.append(float(np.hypot(dx, dy)))
            score = min(distances, default=10.0)
            if score > best_score:
                best, best_score = (float(x), float(y)), score
    # Match the office generator's validated minimum clearance while keeping
    # a generous one-metre border from the imported stage bounds.
    if best is None or best_score < 0.45:
        raise RuntimeError(f"{len(object_records)} objects leave no safe robot start (clearance={best_score:.2f}m)")
    return best


def _render_preview(xml_path: Path, output_path: Path) -> dict[str, int]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = "habitat_top" if mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "habitat_top"
    ) >= 0 else -1
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    renderer.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return {"textures": model.ntex, "materials": model.nmat, "meshes": model.nmesh, "geoms": model.ngeom}


def _support_zone(support: dict) -> str:
    """Classify a support surface by the room/context HSSD tagged it with."""
    text = (support.get("semantic_name", "") + " " + support.get("found_in", "")).lower()
    if "kitchen" in text:
        return "kitchen"
    if "office" in text:
        return "office"
    if "bedroom" in text or "kids" in text:
        return "bedroom"
    if "dining" in text:
        return "dining"
    return "living"


def _plan_support_assignments(supports: list[dict]) -> tuple[dict, dict[str, list[str]]]:
    """Pick one uncluttered surface as the main grasp table and spread a few
    semantically-appropriate items across the rest, instead of piling the
    whole interactive pool onto whichever table happens to rank first.
    """
    areas = {}
    zones = {}
    for support in supports:
        sb = np.asarray(support["bounds_mujoco"], dtype=float)
        areas[support["object_id"]] = float((sb[1, 0] - sb[0, 0]) * (sb[1, 1] - sb[0, 1]))
        zones[support["object_id"]] = _support_zone(support)

    # The main grasping station should be a table that isn't already spoken
    # for by a specific room role (office desk, bedroom nightstand); prefer
    # the largest such surface so there is room for most of the pool.
    open_supports = [s for s in supports if zones[s["object_id"]] not in ("office", "bedroom")]
    primary = max(open_supports or supports, key=lambda s: areas[s["object_id"]])
    secondary = sorted(
        (s for s in supports if s is not primary),
        key=lambda s: areas[s["object_id"]],
        reverse=True,
    )

    leftover = list(HOME_INTERACTIVE_IDS)

    def take(predicate, limit: int) -> list[str]:
        picked = [asset_id for asset_id in leftover if predicate(asset_id)][:limit]
        for asset_id in picked:
            leftover.remove(asset_id)
        return picked

    assignment: dict[str, list[str]] = {s["object_id"]: [] for s in supports}
    for support in secondary:
        zone = zones[support["object_id"]]
        if zone == "office":
            picks = take(lambda a: a in OFFICE_ASSET_IDS, 3)
        elif zone == "kitchen":
            picks = take(lambda a: a in KITCHEN_PREFERRED_ASSET_IDS, 2)
            if not picks:
                picks = take(lambda a: a not in OFFICE_ASSET_IDS, 1)
        else:
            # A believable side/dining table: a couple of snack items, no
            # office supplies.
            picks = take(lambda a: a not in OFFICE_ASSET_IDS, 2)
        assignment[support["object_id"]] = picks
    # Whatever the themed secondary tables didn't need -- the bulk of the
    # snack/fruit pool -- becomes the main grasp table's spread.
    assignment[primary["object_id"]].extend(leftover)
    return primary, assignment


def _place_on_support(
    asset,
    support: dict,
    rng: np.random.Generator,
    placed_xy: list[tuple[np.ndarray, np.ndarray]],
    obstacles: list[np.ndarray],
):
    """Find a free, edge-friendly spot for `asset` on `support`'s top surface.

    Furniture is frequently placed at an arbitrary yaw. Sampling candidate
    points directly in `support["bounds_mujoco"]` (its world-axis-aligned
    bounding box) can place an item well outside the table's actual rotated
    footprint -- looking "on the table" by AABB but resting over empty air,
    so it falls or slides off once physics starts. Instead sample in the
    support's own local (unrotated) frame, using `local_bounds`, and rotate
    the chosen point into world coordinates with its recorded `rotation`.

    `obstacles` are other scene objects' world AABBs (e.g. a dining chair
    tucked under the table): its own oversized bounding-box collider can
    still poke up through the tabletop at the table's edge, so a spawn point
    there interpenetrates a static body and gets launched by the resulting
    contact impulse once physics starts.
    """
    cb = corrected_interactive_collision_bounds(asset)
    margin = 0.06
    local_bounds = np.asarray(support["local_bounds"], dtype=float)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(support["rotation"], dtype=float))
    rotation = rotation.reshape(3, 3)
    position = np.asarray(support["position"], dtype=float)
    # The object itself is never rotated to match the table (see the
    # `interactive_euler`-only body below), so its axis-aligned footprint is
    # approximated as a bounding circle -- guaranteeing clearance from the
    # rotated edge regardless of the table's yaw.
    footprint_radius = 0.5 * float(np.hypot(*(cb[1, :2] - cb[0, :2])))
    inset = margin + footprint_radius
    lo = local_bounds[0, :2] + inset
    hi = local_bounds[1, :2] - inset
    if np.any(hi < lo):
        return None
    sb = np.asarray(support["bounds_mujoco"], dtype=float)
    z = float(sb[1, 2] - cb[0, 2] + 0.002)
    z_lo, z_hi = z + cb[0, 2], z + cb[1, 2]
    # Sample points along the exposed edges (preferred for Stretch grasping)
    # plus one interior point, then reject AABB overlaps with objects already
    # placed on this (or any) table, and with any other scene object whose
    # bounding box intrudes into this item's actual height band.
    local = []
    for u in np.linspace(0.18, 0.82, 4):
        local.extend(((lo[0] + (hi[0] - lo[0]) * u, lo[1]),
                      (lo[0] + (hi[0] - lo[0]) * u, hi[1]),
                      (lo[0], lo[1] + (hi[1] - lo[1]) * u),
                      (hi[0], lo[1] + (hi[1] - lo[1]) * u)))
    local.append(tuple(rng.uniform(lo, hi)))
    rng.shuffle(local)
    for local_x, local_y in local:
        x0, y0 = position[:2] + rotation[:2, :2] @ np.asarray((local_x, local_y))
        box = (cb[0, :2] + (x0, y0), cb[1, :2] + (x0, y0))
        if any(box[0][0] < old[1][0] and box[1][0] > old[0][0]
               and box[0][1] < old[1][1] and box[1][1] > old[0][1]
               for old in placed_xy):
            continue
        if any(box[0][0] < ob[1, 0] and box[1][0] > ob[0, 0]
               and box[0][1] < ob[1, 1] and box[1][1] > ob[0, 1]
               and z_lo < ob[1, 2] and z_hi > ob[0, 2]
               for ob in obstacles):
            continue
        return cb, float(x0), float(y0), z, box
    return None


def _add_interactive_objects(xml_path: Path, prepared) -> list[dict]:
    """Place the office interactive pool on semantic HSSD table surfaces.

    One uncluttered surface (preferring a living/dining/kitchen table over an
    office desk or nightstand) becomes the main grasp station and gets most of
    the snack/fruit pool; the remaining tables get a small, semantically
    matched subset (office supplies on a desk, food on a kitchen counter).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    assets_node = root.find("asset")
    world = root.find("worldbody")
    if assets_node is None or world is None:
        raise RuntimeError(f"{xml_path}: missing asset/worldbody")
    pool = {asset.asset_id: asset for asset in load_interactive_assets()}
    selected = [pool[asset_id] for asset_id in HOME_INTERACTIVE_IDS if asset_id in pool]
    existing = {node.get("name") for node in assets_node}
    for asset in selected:
        for definition in asset.definitions:
            if definition.attributes.get("name") not in existing:
                attributes = dict(definition.attributes)
                if "file" in attributes:
                    attributes["file"] = _asset_file_path(attributes["file"])
                ET.SubElement(assets_node, definition.tag, **attributes)
                existing.add(definition.attributes.get("name"))
    supports = [r for r in prepared.object_records if r.get("support_surface") and r.get("bounds_mujoco")]
    if not supports:
        raise RuntimeError(f"{prepared.scene_id}: no semantic table/counter surface found")
    rng = np.random.default_rng(7103 + sum(ord(c) for c in prepared.scene_id))
    records = []
    placed_xy: list[tuple[np.ndarray, np.ndarray]] = []

    _primary, assignment = _plan_support_assignments(supports)
    support_for_asset_id = {
        asset_id: support_id for support_id, asset_ids in assignment.items() for asset_id in asset_ids
    }
    support_by_id = {s["object_id"]: s for s in supports}
    # A dining chair tucked under a table, an ottoman beside a coffee table,
    # a mixer already on a counter, etc. can still poke into the tabletop's
    # height band even though none of them is a dedicated support surface.
    # Two exclusions keep this from rejecting perfectly good spots:
    #  - "storage_furniture"/"support_furniture"/"floor_covering" are
    #    routinely a merged multi-segment mesh (a whole kitchen run, a
    #    wardrobe wall, a rug) whose tight AABB spans far more empty space
    #    than the object actually occupies.
    #  - multi-seat sofas/sectionals (empirically >1.5 m^2 footprint here,
    #    vs. under 1.4 m^2 for every single chair/stool/ottoman/bench across
    #    the generated scenes) are frequently L-shaped, so their rectangular
    #    AABB also overstates their solid footprint -- shrink those toward
    #    their own center instead of trusting or discarding the full box, so
    #    a table corner that is genuinely buried in the sofa's bulk (a
    #    "cuddler"/chaise wraparound) is still caught.
    OBSTACLE_EXCLUDED_CATEGORIES = {"storage_furniture", "support_furniture", "floor_covering"}
    OBSTACLE_MAX_FOOTPRINT_M2 = 1.5
    OBSTACLE_SHRINK_FACTOR = 0.6
    obstacle_pool = []
    for r in prepared.object_records:
        bounds = r.get("bounds_mujoco")
        if not bounds or r.get("category") in OBSTACLE_EXCLUDED_CATEGORIES:
            continue
        box = np.asarray(bounds, dtype=float)
        footprint = float((box[1, 0] - box[0, 0]) * (box[1, 1] - box[0, 1]))
        if footprint > OBSTACLE_MAX_FOOTPRINT_M2:
            center, half = box.mean(axis=0), (box[1] - box[0]) / 2.0
            box = np.stack((center - half * OBSTACLE_SHRINK_FACTOR, center + half * OBSTACLE_SHRINK_FACTOR))
        obstacle_pool.append((r["object_id"], box))

    for index, asset in enumerate(selected):
        object_id = f"{asset.asset_id}_home_{index:02d}"
        support = support_by_id.get(support_for_asset_id.get(asset.asset_id))
        if support is None:
            continue
        obstacles = [b for oid, b in obstacle_pool if oid != support["object_id"]]
        chosen = _place_on_support(asset, support, rng, placed_xy, obstacles)
        if chosen is None:
            # Small homes may not have enough exposed tabletop area for the
            # complete pool. Keep the scene valid and retain the earlier
            # objects (including the fruit/snack essentials).
            click.echo(f"Warning: {prepared.scene_id}: skipping {asset.asset_id}; no free tabletop area")
            continue
        cb, x, y, z, world_box = chosen
        body = ET.SubElement(world, "body", name=object_id, pos=_numbers((x, y, z)))
        ET.SubElement(body, "freejoint", name=f"{object_id}_freejoint")
        dx, dy, dz = cb[1] - cb[0]
        inertia = asset.mass_kg / 12.0 * np.asarray((dy * dy + dz * dz, dx * dx + dz * dz, dx * dx + dy * dy))
        ET.SubElement(body, "inertial", pos=_numbers((cb[:, 0].mean(), cb[:, 1].mean(), cb[:, 2].mean())), mass=str(asset.mass_kg), diaginertia=_numbers(np.maximum(inertia, 1e-6)))
        orientation = ET.SubElement(body, "body", name=f"{object_id}_orientation", euler=_numbers(interactive_euler(asset.asset_id)))
        visual = {"name": f"{object_id}_visual", "type": "mesh", "mesh": asset.visual_mesh, "mass": "0", "contype": "0", "conaffinity": "0", "group": "2"}
        if asset.material:
            visual["material"] = asset.material
        ET.SubElement(orientation, "geom", **visual)
        ET.SubElement(body, "geom", name=f"{object_id}_collision", type="box", pos=_numbers((cb[:, 0].mean(), cb[:, 1].mean(), cb[:, 2].mean())), size=_numbers((cb[1]-cb[0])/2), mass="0", contype="1", conaffinity="1", friction=asset.friction, rgba="0 0 0 0")
        site = ET.SubElement(orientation, "site", name=f"{object_id}_grasp_site", pos=_numbers((0, 0, max(0.01, asset.height * 0.5))), size="0.018", rgba="0 0 0 0")
        placed_xy.append(world_box)
        records.append({"object_id": object_id, "asset_id": asset.asset_id, "name": asset.name, "category": "interactive_objects", "position": [float(x), float(y), z], "bounds_mujoco": [[float(v) for v in world_box[0]] + [float(cb[0, 2] + z)], [float(v) for v in world_box[1]] + [float(cb[1, 2] + z)]], "mass_kg": asset.mass_kg, "dynamic": True, "graspable": True, "support": support["object_id"], "grasp_site": site.get("name")})
    ET.indent(root, space="  ")
    tree.write(xml_path, encoding="unicode", xml_declaration=False)
    return records


def _write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        """# Generated simplified home scenes

Ten scenes generated from ten different HSSD scene IDs. The script uses the
HSSD `scenes-uncluttered` split, converts each stage/object to MuJoCo, and adds
a Hello Robot Stretch include. Converted meshes and textures are stored in
`_hssd_cache/`.

```bash
python examples/generated_home_scene.py --scene 1
```

Regenerate with a different dataset location using:

```bash
python tools/generate_home_scenes.py --hssd-root /path/to/hssd-hab
```
""",
        encoding="utf-8",
    )


def generate(
    output: Path,
    hssd_root: Path,
    scene_ids: tuple[str, ...],
    *,
    skip_previews: bool,
    rebuild: bool,
    ktx_command: Path | None,
) -> None:
    if len(scene_ids) != 10:
        raise click.ClickException(f"Expected exactly 10 HSSD scene IDs, got {len(scene_ids)}")
    if len(set(scene_ids)) != len(scene_ids):
        raise click.ClickException("HSSD scene IDs must be unique")
    hssd_root = hssd_root.resolve()
    missing = [
        scene_id
        for scene_id in scene_ids
        if not (hssd_root / "scenes-uncluttered" / f"{scene_id}.scene_instance.json").is_file()
    ]
    if missing:
        raise click.ClickException(
            "Missing HSSD scenes-uncluttered files for: " + ", ".join(missing)
        )
    output.mkdir(parents=True, exist_ok=True)
    for old_file in output.glob("home_*.*"):
        old_file.unlink()
    cache_root = output / "_hssd_cache"
    if rebuild and cache_root.exists():
        shutil.rmtree(cache_root)

    catalog = []
    for index, hssd_id in enumerate(scene_ids, start=1):
        scene_name = _scene_name(index, hssd_id)
        click.echo(f"Generating {scene_name} from HSSD {hssd_id}")
        prepared = prepare_habitat_scene(
            scene_id=hssd_id,
            hssd_root=hssd_root,
            cache_root=cache_root,
            # This is the simplification knob: remove clutter while retaining
            # the architectural stage and primary furniture.
            uncluttered=True,
            stage_alpha=1.0,
            include_stage=True,
            include_collision=True,
            include_grasp_sites=False,
            rebuild=rebuild,
            ktx_command=ktx_command,
        )
        scene_xml = output / f"{scene_name}.xml"
        shutil.copy2(prepared.xml_path, scene_xml)
        _make_asset_paths_portable(scene_xml)
        interactive_records = _add_interactive_objects(scene_xml, prepared)
        yaw = (index - 1) % 4 * (math.pi / 2)
        wall_boxes = _wall_collision_boxes(scene_xml)
        start_xy = _choose_open_start(prepared.bounds, prepared.object_records + interactive_records, wall_boxes)
        robot_xml, robot_start = _write_robot_include(output, scene_name, prepared.bounds, yaw, start_xy)
        _add_robot_include(scene_xml, robot_xml, scene_name)

        manifest = {
            "scene_id": scene_name,
            "title": f"Simplified Home {index:02d}",
            "source_dataset": "HSSD",
            "hssd_scene_id": hssd_id,
            "hssd_scene_split": "scenes-uncluttered",
            "layout": "imported_hssd_home",
            "simplification": {"uncluttered": True, "stage_alpha": 1.0},
            "object_count": prepared.object_count,
            "unique_template_count": prepared.unique_template_count,
            "category_counts": prepared.category_counts,
            "bounds_mujoco": prepared.bounds.tolist(),
            "zones": [
                {
                    "type": "home",
                    "bounds": [
                        float(prepared.bounds[0, 0]),
                        float(prepared.bounds[1, 0]),
                        float(prepared.bounds[0, 1]),
                        float(prepared.bounds[1, 1]),
                    ],
                }
            ],
            "assets": prepared.object_records + interactive_records,
            "robot": {
                "model": "Hello Robot Stretch",
                "initial_pose": list(robot_start),
                "body": "base_link",
            },
            "mjcf": scene_xml.name,
            "robot_include": robot_xml.name,
            "preview": f"{scene_name}.png",
        }
        manifest_path = output / f"{scene_name}.json"
        if not skip_previews:
            manifest["model"] = _render_preview(scene_xml, output / manifest["preview"])
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        catalog.append(
            {
                "scene_id": scene_name,
                "hssd_scene_id": hssd_id,
                "title": manifest["title"],
                "mjcf": scene_xml.name,
                "manifest": manifest_path.name,
                "preview": manifest["preview"],
                "object_count": prepared.object_count,
                "robot_initial_pose": list(robot_start),
            }
        )

    (output / "catalog.json").write_text(
        json.dumps(
            {
                "scene_count": 10,
                "source_dataset": "HSSD",
                "scene_split": "scenes-uncluttered",
                "scenes": catalog,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_readme(output)
    click.echo(f"Generated {len(catalog)} scenes in {output}")


@click.command()
@click.option("--output", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT, show_default=True)
@click.option("--hssd-root", type=click.Path(path_type=Path), default=DEFAULT_HSSD_ROOT, show_default=True)
@click.option(
    "--scene-id",
    "scene_ids",
    multiple=True,
    help="HSSD scene ID; repeat exactly ten times. Defaults to the curated home list.",
)
@click.option("--skip-previews", is_flag=True, help="Skip EGL preview rendering.")
@click.option("--rebuild", is_flag=True, help="Discard converted HSSD cache before generating.")
@click.option("--ktx-command", type=click.Path(path_type=Path), help="Path to Khronos ktx for BasisU textures.")
def main(
    output: Path,
    hssd_root: Path,
    scene_ids: tuple[str, ...],
    skip_previews: bool,
    rebuild: bool,
    ktx_command: Path | None,
) -> None:
    """Generate ten simplified homes, each from a distinct HSSD scene ID."""
    generate(
        output,
        hssd_root,
        tuple(scene_ids) or DEFAULT_SCENE_IDS,
        skip_previews=skip_previews,
        rebuild=rebuild,
        ktx_command=ktx_command,
    )


if __name__ == "__main__":
    main()
