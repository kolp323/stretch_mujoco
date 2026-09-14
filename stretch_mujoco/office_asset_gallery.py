"""Generate and launch a textured MuJoCo gallery for shortlisted HSSD office assets."""

from __future__ import annotations

import json
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from stretch_mujoco.gltf_material_converter import (
    ConvertedGltfAsset,
    convert_glb_with_materials,
)
from stretch_mujoco.paths import cache_root


ASSET_ROOT = Path(__file__).resolve().parent / "models" / "assets" / "office_assets"
DEFAULT_CATALOG = ASSET_ROOT / "catalog" / "hssd_candidates.json"
DEFAULT_PREVIEW_MANIFEST = ASSET_ROOT / "previews" / "manifest.json"
DEFAULT_CACHE = cache_root() / "office_asset_gallery"

CATEGORY_COLORS = {
    "furniture/chairs": "0.26 0.52 0.68 1",
    "furniture/desks": "0.58 0.42 0.26 1",
    "furniture/sofas": "0.42 0.58 0.48 1",
    "furniture/storage": "0.50 0.43 0.36 1",
    "electronics": "0.28 0.32 0.37 1",
    "lighting": "0.84 0.68 0.28 1",
    "props/plants": "0.24 0.55 0.30 1",
    "props/trash_bins": "0.42 0.47 0.50 1",
    "props/books": "0.66 0.27 0.24 1",
    "props/drinkware": "0.30 0.55 0.62 1",
    "decoration/rugs": "0.55 0.32 0.45 1",
    "decoration/curtains": "0.50 0.46 0.68 1",
    "decoration/wall_art": "0.75 0.50 0.22 1",
}


@dataclass(frozen=True)
class GalleryAsset:
    asset_id: str
    display_name: str
    category: str
    source_glb: Path
    converted_dir: Path


@dataclass(frozen=True)
class PreparedGallery:
    xml_path: Path
    assets: tuple[GalleryAsset, ...]
    display_scales: tuple[float, ...]


def load_shortlist(
    catalog_path: Path = DEFAULT_CATALOG,
    manifest_path: Path = DEFAULT_PREVIEW_MANIFEST,
    per_category: int = 3,
) -> tuple[GalleryAsset, ...]:
    if per_category < 1:
        raise ValueError("per_category must be positive")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {candidate["asset_id"]: candidate for candidate in catalog["candidates"]}
    counts: dict[str, int] = {}
    assets = []
    for preview in manifest:
        category = preview["category"]
        if preview.get("status") != "rendered" or counts.get(category, 0) >= per_category:
            continue
        candidate = by_id[preview["asset_id"]]
        assets.append(
            GalleryAsset(
                asset_id=candidate["asset_id"],
                display_name=candidate["display_name"],
                category=category,
                source_glb=Path(candidate["source"]["render_asset"]),
                converted_dir=ASSET_ROOT / category / candidate["asset_id"],
            )
        )
        counts[category] = counts.get(category, 0) + 1
    return tuple(assets)


def _display_scale(bounds: np.ndarray, real_scale: bool) -> float:
    dimensions = bounds[1] - bounds[0]
    return 1.0 if real_scale else min(1.35 / max(float(dimensions.max()), 1e-6), 4.0)


def _numbers(values: tuple[float, ...] | np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _write_asset_metadata(asset: GalleryAsset, converted: ConvertedGltfAsset) -> None:
    metadata = {
        "asset_id": asset.asset_id,
        "display_name": asset.display_name,
        "category": asset.category,
        "source": {
            "dataset": "HSSD",
            "render_asset": str(asset.source_glb),
            "license": "CC-BY-NC-4.0",
        },
        "coordinate_system": "MuJoCo Z-up",
        "bounds_m": converted.bounds.tolist(),
        "material_part_count": len(converted.parts),
    }
    (asset.converted_dir / "asset.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def convert_gallery_assets(
    assets: tuple[GalleryAsset, ...],
    *,
    rebuild: bool = False,
    ktx_command: Path | None = None,
) -> list[tuple[GalleryAsset, ConvertedGltfAsset]]:
    converted_assets = []
    for asset in assets:
        if rebuild and asset.converted_dir.exists():
            shutil.rmtree(asset.converted_dir)
        converted = convert_glb_with_materials(
            asset.source_glb, asset.converted_dir, ktx_command=ktx_command
        )
        _write_asset_metadata(asset, converted)
        converted_assets.append((asset, converted))
    return converted_assets


def write_office_asset_registry(
    converted: list[tuple[GalleryAsset, ConvertedGltfAsset]],
    output: Path = ASSET_ROOT / "mjcf" / "office_assets.xml",
) -> None:
    """Write a relocatable registry whose assets are relative to the MJCF file."""
    root = ET.Element("mujoco", model="textured office assets")
    assets = ET.SubElement(root, "asset")
    for asset, converted_asset in converted:
        asset_prefix = f"office_{asset.asset_id}"
        for part_index, part in enumerate(converted_asset.parts):
            prefix = f"{asset_prefix}_part_{part_index:03d}"
            texture_name = None
            if part.texture_path is not None:
                texture_name = f"{prefix}_texture"
                ET.SubElement(
                    assets,
                    "texture",
                    name=texture_name,
                    type="2d",
                    file=os.path.relpath(part.texture_path.resolve(), output.parent.resolve()),
                )
            material_attributes = {
                "name": f"{prefix}_material",
                "rgba": _numbers(part.rgba),
                "specular": f"{part.specular:.6g}",
                "shininess": f"{part.shininess:.6g}",
                "reflectance": f"{part.reflectance:.6g}",
            }
            if texture_name is not None:
                material_attributes["texture"] = texture_name
            ET.SubElement(assets, "material", **material_attributes)
            ET.SubElement(
                assets,
                "mesh",
                name=f"{prefix}_mesh",
                file=os.path.relpath(part.obj_path.resolve(), output.parent.resolve()),
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output, encoding="unicode", xml_declaration=False)


def _add_assets(
    root: ET.Element,
    converted: list[tuple[GalleryAsset, ConvertedGltfAsset]],
    real_scale: bool,
) -> dict[int, list[tuple[str, str]]]:
    assets = ET.SubElement(root, "asset")
    ET.SubElement(
        assets,
        "material",
        name="gallery_floor",
        rgba="0.12 0.14 0.16 1",
        reflectance="0.05",
    )
    ET.SubElement(
        assets,
        "material",
        name="gallery_pedestal",
        rgba="0.68 0.70 0.69 1",
    )
    part_assets: dict[int, list[tuple[str, str]]] = {}
    for asset_index, (_, converted_asset) in enumerate(converted):
        scale = _display_scale(converted_asset.bounds, real_scale)
        part_assets[asset_index] = []
        for part_index, part in enumerate(converted_asset.parts):
            prefix = f"gallery_{asset_index:02d}_{part_index:03d}"
            texture_name = None
            if part.texture_path is not None:
                texture_name = f"{prefix}_texture"
                ET.SubElement(
                    assets,
                    "texture",
                    name=texture_name,
                    type="2d",
                    file=str(part.texture_path.resolve()),
                )
            material_attributes = {
                "name": f"{prefix}_material",
                "rgba": _numbers(part.rgba),
                "specular": f"{part.specular:.6g}",
                "shininess": f"{part.shininess:.6g}",
                "reflectance": f"{part.reflectance:.6g}",
            }
            if texture_name is not None:
                material_attributes["texture"] = texture_name
            ET.SubElement(assets, "material", **material_attributes)
            mesh_name = f"{prefix}_mesh"
            ET.SubElement(
                assets,
                "mesh",
                name=mesh_name,
                file=str(part.obj_path.resolve()),
                scale=f"{scale} {scale} {scale}",
            )
            part_assets[asset_index].append((mesh_name, material_attributes["name"]))
    return part_assets


def _add_world(
    root: ET.Element,
    converted: list[tuple[GalleryAsset, ConvertedGltfAsset]],
    part_assets: dict[int, list[tuple[str, str]]],
    real_scale: bool,
) -> list[float]:
    world = ET.SubElement(root, "worldbody")
    categories = list(dict.fromkeys(asset.category for asset, _ in converted))
    category_index = {category: index for index, category in enumerate(categories)}
    category_totals = {
        category: sum(asset.category == category for asset, _ in converted)
        for category in categories
    }
    column_count = 4
    bay_width = 5.4
    bay_depth = 4.2
    row_count = (len(categories) + column_count - 1) // column_count
    gallery_width = column_count * bay_width
    gallery_depth = row_count * bay_depth
    gallery_center_y = (row_count - 1) * bay_depth / 2

    ET.SubElement(
        world,
        "geom",
        name="gallery_floor",
        type="plane",
        size=f"{gallery_width / 2 + 1} {gallery_depth / 2 + 1} 0.1",
        material="gallery_floor",
        friction="1 0.01 0.001",
    )
    target = ET.SubElement(world, "body", name="gallery_center", pos=f"0 {gallery_center_y} 0")
    ET.SubElement(target, "site", size="0.01", rgba="0 0 0 0")
    ET.SubElement(
        world,
        "camera",
        name="gallery_overview",
        mode="targetbody",
        target="gallery_center",
        pos=f"0 {-gallery_depth * 1.15} {max(14.0, gallery_depth * 1.05)}",
        fovy="48",
    )
    ET.SubElement(
        world,
        "light",
        name="gallery_key",
        pos=f"{-gallery_width / 3} {gallery_center_y - 2} 12",
        dir="0 0 -1",
        diffuse="0.85 0.84 0.80",
        castshadow="false",
    )
    ET.SubElement(
        world,
        "light",
        name="gallery_fill",
        pos=f"{gallery_width / 3} {gallery_center_y + 4} 10",
        dir="0 0 -1",
        diffuse="0.62 0.67 0.72",
        castshadow="false",
    )

    category_asset_counts: dict[str, int] = {}
    display_scales = []
    for asset_index, (asset, converted_asset) in enumerate(converted):
        bay = category_index[asset.category]
        bay_column = bay % column_count
        bay_row = bay // column_count
        local_index = category_asset_counts.get(asset.category, 0)
        category_asset_counts[asset.category] = local_index + 1
        bay_x = (bay_column - (column_count - 1) / 2) * bay_width
        bay_y = bay_row * bay_depth
        x = bay_x + (local_index - (category_totals[asset.category] - 1) / 2) * 1.55
        y = bay_y
        scale = _display_scale(converted_asset.bounds, real_scale)
        display_scales.append(scale)

        if local_index == 0:
            ET.SubElement(
                world,
                "geom",
                name=f"zone_{bay:02d}_{asset.category.replace('/', '_')}",
                type="box",
                pos=f"{bay_x} {bay_y} 0.025",
                size="2.45 1.55 0.025",
                rgba=CATEGORY_COLORS[asset.category].replace(" 1", " 0.22"),
                contype="0",
                conaffinity="0",
            )
        ET.SubElement(
            world,
            "geom",
            name=f"pedestal_{asset_index + 1:02d}",
            type="box",
            pos=f"{x} {y} 0.14",
            size="0.68 0.68 0.14",
            material="gallery_pedestal",
        )
        body = ET.SubElement(
            world,
            "body",
            name=f"item_{asset_index + 1:02d}_{asset.asset_id[:8]}",
            pos=f"{x} {y} 0.28",
        )
        ET.SubElement(
            body,
            "inertial",
            pos="0 0 0",
            mass="0.001",
            diaginertia="0.001 0.001 0.001",
        )
        center = converted_asset.bounds.mean(axis=0)
        visual_offset = np.array((-center[0], -center[1], -converted_asset.bounds[0, 2])) * scale
        visual_body = ET.SubElement(body, "body", pos=_numbers(visual_offset))
        ET.SubElement(
            visual_body,
            "inertial",
            pos="0 0 0",
            mass="0.001",
            diaginertia="0.001 0.001 0.001",
        )
        for part_index, (mesh_name, material_name) in enumerate(part_assets[asset_index]):
            ET.SubElement(
                visual_body,
                "geom",
                name=f"visual_{asset_index + 1:02d}_{part_index:03d}",
                type="mesh",
                mesh=mesh_name,
                material=material_name,
                mass="0",
                shellinertia="true",
                contype="0",
                conaffinity="0",
                group="2",
            )
    return display_scales


def prepare_gallery(
    *,
    cache_dir: Path = DEFAULT_CACHE,
    per_category: int = 3,
    real_scale: bool = False,
    rebuild: bool = False,
    ktx_command: Path | None = None,
) -> PreparedGallery:
    assets = load_shortlist(per_category=per_category)
    if rebuild and cache_dir.exists():
        shutil.rmtree(cache_dir)
    converted = convert_gallery_assets(assets, rebuild=rebuild, ktx_command=ktx_command)
    write_office_asset_registry(converted)

    root = ET.Element("mujoco", model="Textured HSSD office asset gallery")
    ET.SubElement(root, "compiler", angle="radian", balanceinertia="true")
    ET.SubElement(root, "statistic", center="0 6 1", extent="18")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        ambient="0.32 0.32 0.32",
        diffuse="0.65 0.65 0.65",
        specular="0.08 0.08 0.08",
    )
    part_assets = _add_assets(root, converted, real_scale)
    display_scales = _add_world(root, converted, part_assets, real_scale)
    xml_path = cache_dir / ("gallery_real_scale.xml" if real_scale else "gallery.xml")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="unicode", xml_declaration=False)
    return PreparedGallery(xml_path, assets, tuple(display_scales))


def load_gallery_model(gallery: PreparedGallery) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(gallery.xml_path))
