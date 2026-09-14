"""Convert an HSSD Habitat scene instance into a static MuJoCo visualization."""

from __future__ import annotations

import csv
import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh

from stretch_mujoco.gltf_material_converter import (
    HABITAT_TO_MUJOCO,
    ConvertedGltfAsset,
    convert_glb_with_materials,
)
from stretch_mujoco.paths import cache_root, configured_path, require_external_directory


DEFAULT_HSSD_ROOT = configured_path("STRETCH_MUJOCO_HSSD_ROOT")
DEFAULT_SCENE_ID = "108294417_176709879"
DEFAULT_CACHE_ROOT = cache_root() / "habitat_scenes"

CATEGORY_COLORS = {
    "seating_furniture": "0.23 0.52 0.70 1",
    "support_furniture": "0.62 0.43 0.25 1",
    "storage_furniture": "0.48 0.40 0.31 1",
    "electronics": "0.18 0.23 0.28 1",
    "lighting": "0.90 0.68 0.20 1",
    "plant": "0.20 0.55 0.28 1",
    "decor": "0.65 0.34 0.52 1",
    "floor_covering": "0.49 0.35 0.58 1",
    "drinkware": "0.28 0.63 0.67 1",
    "default": "0.68 0.69 0.66 1",
}


@dataclass(frozen=True)
class ConvertedTemplate:
    template_name: str
    source_glb: Path
    asset: ConvertedGltfAsset
    category: str
    found_in: str = ""
    semantic_name: str = ""


@dataclass(frozen=True)
class PreparedHabitatScene:
    scene_id: str
    xml_path: Path
    object_count: int
    unique_template_count: int
    bounds: np.ndarray
    category_counts: dict[str, int]
    object_records: list[dict[str, Any]]


def _scene_path(root: Path, scene_id: str, uncluttered: bool) -> Path:
    directory = "scenes-uncluttered" if uncluttered else "scenes"
    return root / directory / f"{scene_id}.scene_instance.json"


def _find_object_config(root: Path, template_name: str) -> Path:
    base_id = _template_base_id(template_name)
    candidates = (
        root / "objects" / template_name[0] / f"{template_name}.object_config.json",
        root / "objects" / "decomposed" / base_id / f"{template_name}.object_config.json",
        root / "objects" / "openings" / f"{template_name}.object_config.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Object config not found for template {template_name}")


def _template_base_id(template_name: str) -> str:
    return re.sub(r"_part_\d+$", "", template_name)


def _semantic_categories(root: Path) -> dict[str, str]:
    path = root / "semantics" / "objects.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row["id"]: row.get("super_category") or row.get("main_category") or "default"
            for row in csv.DictReader(stream)
        }


def _material_category(raw_category: str) -> str:
    category = raw_category.lower()
    if category in CATEGORY_COLORS:
        return category
    if "light" in category:
        return "lighting"
    if "plant" in category:
        return "plant"
    if category in {"picture", "decoration", "decor", "clock"}:
        return "decor"
    if category in {"cup", "mug", "bottle", "drinkware"}:
        return "drinkware"
    return "default"


def _habitat_quat_to_mujoco(quaternion: list[float]) -> np.ndarray:
    w, x, y, z = quaternion
    rotation = np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    target_rotation = HABITAT_TO_MUJOCO @ rotation @ HABITAT_TO_MUJOCO.T
    target_quaternion = np.empty(4)
    mujoco.mju_mat2Quat(target_quaternion, target_rotation.reshape(-1))
    return target_quaternion


def _numbers(values: Any) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)


def _register_material_parts(
    assets: ET.Element,
    prefix: str,
    converted: ConvertedGltfAsset,
    *,
    alpha_multiplier: float = 1.0,
) -> list[str]:
    material_names = []
    for index, part in enumerate(converted.parts):
        part_prefix = f"{prefix}_part_{index:03d}"
        texture_name = None
        if part.texture_path is not None:
            texture_name = f"{part_prefix}_texture"
            ET.SubElement(
                assets,
                "texture",
                name=texture_name,
                type="2d",
                file=str(part.texture_path.resolve()),
            )
        rgba = (*part.rgba[:3], part.rgba[3] * alpha_multiplier)
        attributes = {
            "name": f"{part_prefix}_material",
            "rgba": _numbers(rgba),
            "specular": f"{part.specular:.6g}",
            "shininess": f"{part.shininess:.6g}",
            "reflectance": f"{part.reflectance:.6g}",
        }
        if texture_name is not None:
            attributes["texture"] = texture_name
        ET.SubElement(assets, "material", **attributes)
        material_names.append(attributes["name"])
    return material_names


def _transform_bounds(
    bounds: np.ndarray, position: np.ndarray, rotation: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    corners = np.array(
        [
            (x, y, z)
            for x in (bounds[0, 0], bounds[1, 0])
            for y in (bounds[0, 1], bounds[1, 1])
            for z in (bounds[0, 2], bounds[1, 2])
        ]
    )
    w, x, y, z = rotation
    matrix = np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    transformed = (corners * scale) @ matrix.T + position
    return np.stack((transformed.min(axis=0), transformed.max(axis=0)))


def _add_stage_wall_collisions(world: ET.Element, stage_asset: ConvertedGltfAsset) -> int:
    """Approximate structural HSSD wall components with non-degenerate boxes.

    MuJoCo convexifies collision meshes. Several HSSD stage parts (ceilings,
    floors and wall caps) are exactly planar, which makes QHull fail. Connected
    component AABBs avoid that failure while preserving the vertical barriers
    needed by the MuJoCo navigation grid.
    """
    count = 0
    for part in stage_asset.parts:
        loaded = trimesh.load(part.obj_path, force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
            continue
        for component in loaded.split(only_watertight=False):
            lo, hi = np.asarray(component.bounds, dtype=float)
            extent = hi - lo
            horizontal = np.sort(extent[:2])
            # HSSD walls are tall, thin vertical components. Ignore floors,
            # ceilings, trim, and large non-wall aggregates.
            if extent[2] < 1.0 or horizontal[0] > 0.35 or horizontal[1] < 0.25:
                continue
            center = (lo + hi) / 2.0
            half = np.maximum(extent / 2.0, (0.025, 0.025, 0.025))
            ET.SubElement(
                world,
                "geom",
                name=f"habitat_wall_collision_{count:04d}",
                type="box",
                pos=_numbers(center),
                size=_numbers(half),
                rgba="0 0 0 0",
                contype="1",
                conaffinity="1",
                group="3",
                friction="1 0.01 0.001",
            )
            count += 1
    return count


def _mesh_world_vertices(obj_path: Path, scale: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """World-space vertices MuJoCo renders for `obj_path` at pos=0, quat=identity."""
    xml = (
        f'<mujoco><asset><mesh name="m" file="{obj_path.resolve()}" scale="{_numbers(scale)}"/></asset>'
        '<worldbody><geom name="g" type="mesh" mesh="m" mass="0" '
        'contype="0" conaffinity="0"/></worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model.mesh_vert @ data.geom_xmat[0].reshape(3, 3).T + data.geom_xpos[0]


def _mesh_pose_correction(
    obj_path: Path, scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> tuple[np.ndarray, np.ndarray] | None:
    """geom pos/quat that reproduce this OBJ's own authored world coordinates.

    MuJoCo derives each mesh's own center/orientation frame from its
    volumetric inertia to place a mesh geom correctly, even for a massless,
    non-colliding visual geom -- that frame is an asset-level property, not a
    per-geom one. HSSD stage parts and furniture/object parts alike are
    routinely paper-thin (wall caps, trim, door frames, rugs, glass panes) or
    merge several disconnected islands under one material (lamp posts,
    mirrors); for those the inertia estimate is ill-posed and MuJoCo silently
    compiles the wrong frame, which visibly displaces the geom (walls
    detached from their doors, a lamp rendered a meter from its base, an
    object floating above or sunk into the surface it sits on) -- reproduced
    even compiling the single mesh in isolation, so it's a MuJoCo mesh-
    compile quirk, not something introduced by scene assembly.

    Rather than depend on MuJoCo's estimate, measure the actual rigid
    transform it applied to this specific mesh (by compiling it alone and
    comparing to its own OBJ file) and return the pose that cancels it out,
    so the geom renders at its true authored position regardless of how
    degenerate the source mesh is. MuJoCo also splits vertices at hard edges
    to compute flat-shaded normals, appending duplicates rather than
    reordering, so the file's own N vertices are reliably the compiled
    array's first N entries in the same order -- verified empirically to
    match to floating-point precision -- which lets this use an exact,
    correspondence-based Kabsch fit instead of a symmetry-prone heuristic.
    Returns None if the fit's residual isn't tiny, leaving the part
    unmodified rather than risking making it worse.
    """
    raw_vertices = np.asarray(
        trimesh.load(obj_path, force="mesh", process=False).vertices, dtype=float
    ) * np.asarray(scale, dtype=float)
    compiled_vertices = _mesh_world_vertices(obj_path, scale)
    if len(compiled_vertices) < len(raw_vertices):
        return None
    compiled_vertices = compiled_vertices[: len(raw_vertices)]
    raw_mean = raw_vertices.mean(axis=0)
    compiled_mean = compiled_vertices.mean(axis=0)
    covariance = (compiled_vertices - compiled_mean).T @ (raw_vertices - raw_mean)
    u, _, vt = np.linalg.svd(covariance)
    handedness = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag((1.0, 1.0, handedness)) @ u.T
    translation = raw_mean - rotation @ compiled_mean
    residual = float(np.abs(compiled_vertices @ rotation.T + translation - raw_vertices).max())
    if residual > 0.005:
        return None
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
    return translation, quat


def prepare_habitat_scene(
    *,
    scene_id: str = DEFAULT_SCENE_ID,
    hssd_root: Path | None = DEFAULT_HSSD_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    uncluttered: bool = False,
    stage_alpha: float = 0.32,
    include_stage: bool = True,
    include_collision: bool = False,
    include_grasp_sites: bool = False,
    rebuild: bool = False,
    ktx_command: Path | None = None,
) -> PreparedHabitatScene:
    hssd_root = require_external_directory(
        hssd_root,
        environment_variable="STRETCH_MUJOCO_HSSD_ROOT",
        description="HSSD dataset root",
    )
    cache_dir = cache_root / scene_id
    if rebuild and cache_dir.exists():
        shutil.rmtree(cache_dir)

    scene_path = _scene_path(hssd_root, scene_id, uncluttered)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    instances = scene.get("object_instances", [])
    categories = _semantic_categories(hssd_root)
    with (hssd_root / "semantics" / "objects.csv").open(newline="", encoding="utf-8") as stream:
        semantic_rows = list(csv.DictReader(stream))

    stage_config_path = hssd_root / f"{scene['stage_instance']['template_name']}.stage_config.json"
    stage_config = json.loads(stage_config_path.read_text(encoding="utf-8"))
    stage_source = stage_config_path.parent / stage_config["render_asset"]
    stage_asset = (
        convert_glb_with_materials(
            stage_source, cache_dir / "converted" / "stage", ktx_command=ktx_command
        )
        if include_stage
        else None
    )

    templates: dict[str, ConvertedTemplate] = {}
    for template_name in dict.fromkeys(instance["template_name"] for instance in instances):
        config_path = _find_object_config(hssd_root, template_name)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source = config_path.parent / config["render_asset"]
        converted = convert_glb_with_materials(
            source,
            cache_dir / "converted" / "objects" / _safe_name(template_name),
            ktx_command=ktx_command,
        )
        base_id = _template_base_id(template_name)
        raw_category = categories.get(base_id, "default")
        semantic_row = next((row for row in semantic_rows if row.get("id") == base_id), {})
        templates[template_name] = ConvertedTemplate(
            template_name, source, converted, _material_category(raw_category),
            semantic_row.get("foundIn", ""), semantic_row.get("name", "")
        )

    root = ET.Element("mujoco", model=f"Habitat scene {scene_id}")
    ET.SubElement(root, "compiler", angle="radian", balanceinertia="true")
    ET.SubElement(root, "option", gravity="0 0 -9.81" if include_collision else "0 0 0")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        ambient="0.38 0.38 0.38",
        diffuse="0.68 0.68 0.68",
        specular="0.06 0.06 0.06",
    )
    ET.SubElement(visual, "rgba", haze="0.78 0.83 0.86 1")
    assets = ET.SubElement(root, "asset")
    stage_materials = []
    if stage_asset is not None:
        stage_materials = _register_material_parts(
            assets, "habitat_stage", stage_asset, alpha_multiplier=stage_alpha
        )
        for part_index, part in enumerate(stage_asset.parts):
            ET.SubElement(
                assets,
                "mesh",
                name=f"habitat_stage_mesh_{part_index:03d}",
                file=str(part.obj_path.resolve()),
            )

    template_materials = {
        template_name: _register_material_parts(
            assets, f"object_{template_index:03d}", template.asset
        )
        for template_index, (template_name, template) in enumerate(templates.items())
    }

    mesh_names: dict[tuple[str, tuple[float, ...], int], str] = {}
    # Object instance meshes suffer the identical MuJoCo mesh-compile quirk
    # documented on `_mesh_pose_correction`: multi-island or thin-shell props
    # (lamp posts, rugs, mirror glass, ...) get silently displaced from their
    # authored position. Correct each unique (template, scale, part) mesh once
    # -- the correction is a property of the compiled mesh asset, so every
    # instance sharing it needs the same local pos/quat fix-up.
    mesh_corrections: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    object_records: list[dict[str, Any]] = []
    grasp_site_count = 0
    for index, instance in enumerate(instances):
        template_name = instance["template_name"]
        source_scale = np.asarray(instance.get("non_uniform_scale", (1, 1, 1)), dtype=float)
        target_scale = source_scale[[0, 2, 1]]
        scale_key = tuple(round(float(value), 7) for value in target_scale)
        for part_index, part in enumerate(templates[template_name].asset.parts):
            key = (template_name, scale_key, part_index)
            if key in mesh_names:
                continue
            mesh_name = f"object_mesh_{len(mesh_names):03d}"
            mesh_names[key] = mesh_name
            ET.SubElement(
                assets,
                "mesh",
                name=mesh_name,
                file=str(part.obj_path.resolve()),
                scale=_numbers(target_scale),
            )
            mesh_corrections[mesh_name] = _mesh_pose_correction(
                part.obj_path, tuple(float(v) for v in target_scale)
            )

    world = ET.SubElement(root, "worldbody")
    if stage_asset is not None:
        for part_index, (material_name, part) in enumerate(zip(stage_materials, stage_asset.parts)):
            geom_attributes = {
                "name": f"habitat_stage_visual_{part_index:03d}",
                "type": "mesh",
                "mesh": f"habitat_stage_mesh_{part_index:03d}",
                "material": material_name,
                "contype": "0",
                "conaffinity": "0",
                "group": "2",
                # NOT shellinertia: on a mesh geom it makes MuJoCo apply the
                # mesh's own (frequently degenerate/wrong, see
                # _mesh_pose_correction) inertial-frame compensation a
                # second time on top of this geom's pos/quat -- confirmed by
                # a minimal repro -- which is moot anyway for a mass="0" geom.
                "mass": "0",
            }
            correction = _mesh_pose_correction(part.obj_path)
            if correction is not None:
                pos, quat = correction
                geom_attributes["pos"] = _numbers(pos)
                geom_attributes["quat"] = _numbers(quat)
            ET.SubElement(world, "geom", **geom_attributes)
        if include_collision:
            _add_stage_wall_collisions(world, stage_asset)

    all_bounds = [stage_asset.bounds] if stage_asset is not None else []
    category_counts: dict[str, int] = {}
    for index, instance in enumerate(instances):
        template_name = instance["template_name"]
        template = templates[template_name]
        position = HABITAT_TO_MUJOCO @ np.asarray(instance.get("translation", (0, 0, 0)))
        rotation = _habitat_quat_to_mujoco(instance.get("rotation", (1, 0, 0, 0)))
        source_scale = np.asarray(instance.get("non_uniform_scale", (1, 1, 1)), dtype=float)
        target_scale = source_scale[[0, 2, 1]]
        scale_key = tuple(round(float(value), 7) for value in target_scale)
        category_counts[template.category] = category_counts.get(template.category, 0) + 1
        object_id = f"hssd_object_{index:03d}"
        body = ET.SubElement(
            world,
            "body",
            name=object_id,
            pos=_numbers(position),
            quat=_numbers(rotation),
        )
        ET.SubElement(
            body,
            "inertial",
            pos="0 0 0",
            mass="0.001",
            diaginertia="0.001 0.001 0.001",
        )
        for part_index, material_name in enumerate(template_materials[template_name]):
            mesh_name = mesh_names[(template_name, scale_key, part_index)]
            geom_attributes = {
                "name": f"object_visual_{index:03d}_{part_index:03d}",
                "type": "mesh",
                "mesh": mesh_name,
                "material": material_name,
                "contype": "0",
                "conaffinity": "0",
                "group": "2",
                # NOT shellinertia: see the identical note on the stage visual
                # geoms above; this is likewise moot for a mass="0" geom.
                "mass": "0",
            }
            correction = mesh_corrections.get(mesh_name)
            if correction is not None:
                pos, quat = correction
                geom_attributes["pos"] = _numbers(pos)
                geom_attributes["quat"] = _numbers(quat)
            ET.SubElement(body, "geom", **geom_attributes)
        local_bounds = np.asarray(template.asset.bounds, dtype=float)
        local_center = local_bounds.mean(axis=0) * target_scale
        local_half = np.maximum((local_bounds[1] - local_bounds[0]) / 2.0 * target_scale, 0.015)
        if include_collision:
            ET.SubElement(
                body,
                "geom",
                name=f"{object_id}_collision",
                type="box",
                pos=_numbers(local_center),
                size=_numbers(local_half),
                rgba="0 0 0 0",
                contype="1",
                conaffinity="1",
                group="3",
                mass="0.001",
            )
        # Keep the generated grasp benchmark tractable: expose a deterministic
        # subset of primary furniture/props instead of hundreds of decorative
        # wall, window, and lighting instances.
        graspable = include_grasp_sites and grasp_site_count < 20 and template.category not in {
            "default", "decor", "lighting", "floor_covering"
        }
        if graspable:
            ET.SubElement(
                body,
                "site",
                name=f"{object_id}_grasp_site",
                pos=_numbers(local_center),
                size="0.025",
                rgba="0.9 0.2 0.1 0.35",
            )
            grasp_site_count += 1
        semantic_text = (template.semantic_name + " " + template.found_in).lower()
        # Kitchen islands/counters are HSSD-tagged "storage_furniture" (their
        # defining feature is the cabinets below), but their flat top is just
        # as usable a grasp surface as a "support_furniture" table/desk.
        support_category_ok = template.category == "support_furniture" or (
            template.category == "storage_furniture"
            and any(token in semantic_text for token in ("island", "counter"))
        )
        # A "2 piece nesting" set, "3 piece" stacking tables, etc. are two or
        # more separate tables of different heights merged into one HSSD
        # instance/template. Its single AABB records only one flat top, but
        # the actual surface height varies across its footprint -- an item
        # anchored to that one height can end up hovering over the shorter
        # piece or clipped into the taller one. Skip these rather than place
        # anything on them.
        is_multi_piece_set = any(
            token in semantic_text for token in ("nesting", "2 piece", "3 piece", "piece set")
        )
        object_records.append(
            {
                "object_id": object_id,
                "asset_id": _template_base_id(template_name),
                "template_name": template_name,
                "name": template_name,
                "category": template.category,
                "semantic_name": template.semantic_name,
                "found_in": template.found_in,
                "support_surface": support_category_ok
                and not is_multi_piece_set
                and any(token in semantic_text
                        for token in ("table", "desk", "counter", "console", "island")),
                "position": position.tolist(),
                # Furniture is frequently placed at a yaw other than 0/90/180/
                # 270 degrees. Consumers that need the flat top surface (e.g.
                # placing graspable objects on it) must sample candidate
                # points in this body-local, axis-aligned frame and rotate
                # them into world coordinates -- the world AABB in
                # "bounds_mujoco" below is a looser box that can extend well
                # past the table's actual (rotated) footprint.
                "rotation": rotation.tolist(),
                "local_bounds": (local_bounds * target_scale).tolist(),
                "grasp_site": f"{object_id}_grasp_site" if graspable else None,
            }
        )
        all_bounds.append(
            _transform_bounds(template.asset.bounds, position, rotation, target_scale)
        )
        object_records[-1]["bounds_mujoco"] = all_bounds[-1].tolist()

    bounds = np.stack(
        (
            np.min([item[0] for item in all_bounds], axis=0),
            np.max([item[1] for item in all_bounds], axis=0),
        )
    )
    # HSSD stage meshes are visual-only in the gallery converter. Add a
    # lightweight floor collider for navigation and physical Stretch tests.
    floor_center = (bounds[0] + bounds[1]) / 2.0
    floor_half = np.maximum((bounds[1] - bounds[0]) / 2.0, 1.0)
    ET.SubElement(
        world,
        "geom",
        name="hssd_floor_collision",
        type="box",
        pos=_numbers((floor_center[0], floor_center[1], -0.06)),
        size=_numbers((floor_half[0], floor_half[1], 0.06)),
        friction="1 0.01 0.001",
        rgba="0 0 0 0",
        contype="1",
        conaffinity="1",
        group="3",
    )
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    radius = max(float(extent[0]), float(extent[1]))
    target = ET.SubElement(
        world, "body", name="scene_center", pos=_numbers((center[0], center[1], 1.0))
    )
    ET.SubElement(target, "site", size="0.01", rgba="0 0 0 0")
    ET.SubElement(
        world,
        "site",
        name="zone_home_center",
        pos=_numbers((center[0], center[1], 0.025)),
        size="0.015",
        rgba="0 0 0 0",
    )
    ET.SubElement(
        world,
        "camera",
        name="habitat_overview",
        mode="targetbody",
        target="scene_center",
        pos=_numbers((center[0], center[1] - radius * 1.05, max(10.0, radius * 0.75))),
        fovy="48",
    )
    # habitat_top is meant to be a true top-down floor-plan view. A perspective
    # camera makes the ~2.8 m tall opaque walls (and ceiling) visibly lean
    # toward the image center relative to the floor and furniture near z=0,
    # which reads as "misaligned" walls even though the geometry is correct.
    # An orthographic projection removes that parallax entirely.
    # MuJoCo's orthographic fovy is the *full* vertical extent of the view
    # (in length units), not a half-extent, so match the wider of the two
    # scene dimensions accounting for the render aspect ratio.
    preview_aspect = 4.0 / 3.0  # matches the 640x480 preview renderer
    top_full_extent = max(float(extent[1]), float(extent[0]) / preview_aspect)
    ET.SubElement(
        world,
        "camera",
        name="habitat_top",
        mode="targetbody",
        target="scene_center",
        pos=_numbers((center[0], center[1], max(18.0, radius * 1.25))),
        orthographic="true",
        fovy=f"{max(top_full_extent * 1.15, 3.0):.6g}",
    )
    ET.SubElement(root, "statistic", center=_numbers(center), extent=f"{radius:.6g}")

    xml_path = cache_dir / "scene.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="unicode", xml_declaration=False)
    return PreparedHabitatScene(
        scene_id=scene_id,
        xml_path=xml_path,
        object_count=len(instances),
        unique_template_count=len(templates),
        bounds=bounds,
        category_counts=category_counts,
        object_records=object_records,
    )


def load_habitat_scene_model(scene: PreparedHabitatScene) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(scene.xml_path))
