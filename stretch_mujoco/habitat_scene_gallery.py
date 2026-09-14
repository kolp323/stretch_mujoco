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


@dataclass(frozen=True)
class PreparedHabitatScene:
    scene_id: str
    xml_path: Path
    object_count: int
    unique_template_count: int
    bounds: np.ndarray
    category_counts: dict[str, int]


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


def prepare_habitat_scene(
    *,
    scene_id: str = DEFAULT_SCENE_ID,
    hssd_root: Path | None = DEFAULT_HSSD_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    uncluttered: bool = False,
    stage_alpha: float = 0.32,
    include_stage: bool = True,
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
        raw_category = categories.get(_template_base_id(template_name), "default")
        templates[template_name] = ConvertedTemplate(
            template_name, source, converted, _material_category(raw_category)
        )

    root = ET.Element("mujoco", model=f"Habitat scene {scene_id}")
    ET.SubElement(root, "compiler", angle="radian", balanceinertia="true")
    ET.SubElement(root, "option", gravity="0 0 0")
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

    world = ET.SubElement(root, "worldbody")
    if stage_asset is not None:
        for part_index, material_name in enumerate(stage_materials):
            ET.SubElement(
                world,
                "geom",
                name=f"habitat_stage_visual_{part_index:03d}",
                type="mesh",
                mesh=f"habitat_stage_mesh_{part_index:03d}",
                material=material_name,
                contype="0",
                conaffinity="0",
                group="2",
                shellinertia="true",
                mass="0",
            )

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
        body = ET.SubElement(
            world,
            "body",
            name=f"object_{index:03d}_{_safe_name(template_name)[:48]}",
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
            ET.SubElement(
                body,
                "geom",
                name=f"object_visual_{index:03d}_{part_index:03d}",
                type="mesh",
                mesh=mesh_names[(template_name, scale_key, part_index)],
                material=material_name,
                contype="0",
                conaffinity="0",
                group="2",
                shellinertia="true",
                mass="0",
            )
        all_bounds.append(
            _transform_bounds(template.asset.bounds, position, rotation, target_scale)
        )

    bounds = np.stack(
        (
            np.min([item[0] for item in all_bounds], axis=0),
            np.max([item[1] for item in all_bounds], axis=0),
        )
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
        "camera",
        name="habitat_overview",
        mode="targetbody",
        target="scene_center",
        pos=_numbers((center[0], center[1] - radius * 1.05, max(10.0, radius * 0.75))),
        fovy="48",
    )
    ET.SubElement(
        world,
        "camera",
        name="habitat_top",
        mode="targetbody",
        target="scene_center",
        pos=_numbers((center[0], center[1], max(18.0, radius * 1.25))),
        fovy="45",
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
    )


def load_habitat_scene_model(scene: PreparedHabitatScene) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(scene.xml_path))
