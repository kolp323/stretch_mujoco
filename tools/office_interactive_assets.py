"""Shared import, orientation, and geometry helpers for office task objects."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


MODELS_ROOT = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"
SNACK_ASSETS = MODELS_ROOT / "assets" / "office_snacks"

# Fixed body-local XYZ corrections in radians.  This is intentionally the one
# table used by both scene generation and interactive-object previews.
INTERACTIVE_ROTATION_CORRECTIONS = {
    "017_calculator": (math.pi / 2, 0.0, 0.0),
    "043_book": (math.pi / 2, 0.0, 0.0),
    "116_keyboard": (math.pi / 2, 0.0, 0.0),
    "101_milk-tea": (math.pi, 0.0, 0.0),
    "snack_soda_can": (0.0, 0.0, 0.0),
    "001_bottle": (math.pi, 0.0, 0.0),
    "025_chips-tub": (0.0, 0.0, 0.0),
    "035_apple": (0.0, 0.0, 0.0),
    "038_milk-box": (0.0, 0.0, 0.0),
    "071_can": (0.0, 0.0, 0.0),
    "green_apple": (math.pi / 2, 0.0, 0.0),
    "075_bread": (math.pi / 2, 0.0, 0.0),
}


@dataclass(frozen=True)
class ImportedDefinition:
    tag: str
    attributes: dict[str, str]


@dataclass(frozen=True)
class AssetPart:
    mesh: str
    material: str | None


@dataclass(frozen=True)
class InteractiveAsset:
    asset_id: str
    name: str
    visual_mesh: str
    collision_mesh: str
    material: str | None
    definitions: tuple[ImportedDefinition, ...]
    bounds: np.ndarray
    mass_kg: float
    mesh_min_z: float = 0.0
    friction: str = "0.7 0.01 0.001"

    @property
    def height(self) -> float:
        return float(self.bounds[1, 2] - self.bounds[0, 2])


def numbers(values: object) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)  # type: ignore[union-attr]


def interactive_euler(asset_id: str) -> tuple[float, float, float]:
    return INTERACTIVE_ROTATION_CORRECTIONS.get(asset_id, (0.0, 0.0, 0.0))


def _values(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.asarray([float(value) for value in text.split()] if text else default, dtype=float)


def _compile_part_boxes(
    definitions: tuple[ImportedDefinition, ...],
    components: list[tuple[np.ndarray, np.ndarray, tuple[AssetPart, ...]]],
    yaw: float = 0.0,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Compile imported meshes and return their MuJoCo-rendered AABBs."""
    root = ET.Element("mujoco", model="bounds_probe")
    ET.SubElement(root, "compiler", angle="radian", balanceinertia="true")
    assets = ET.SubElement(root, "asset")
    for definition in definitions:
        ET.SubElement(assets, definition.tag, definition.attributes)
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name="probe", euler=numbers((0, 0, yaw)))
    ET.SubElement(body, "inertial", pos="0 0 0", mass="0.001", diaginertia="0.001 0.001 0.001")
    for component_index, (pos, quat, parts) in enumerate(components):
        component = ET.SubElement(body, "body", name=f"component_{component_index}", pos=numbers(pos), quat=numbers(quat))
        ET.SubElement(component, "inertial", pos="0 0 0", mass="0.001", diaginertia="0.001 0.001 0.001")
        for part_index, part in enumerate(parts):
            ET.SubElement(component, "geom", name=f"part_{component_index}_{part_index}", type="mesh", mesh=part.mesh, material=part.material, mass="0", contype="0", conaffinity="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    boxes = {}
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if not name.startswith("part_"):
            continue
        _, component_index, part_index = name.split("_")
        mesh_id = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start : start + count]
        world_vertices = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        boxes[(int(component_index), int(part_index))] = (world_vertices.min(axis=0), world_vertices.max(axis=0))
    return boxes


def _compile_bbox(
    definitions: tuple[ImportedDefinition, ...],
    components: list[tuple[np.ndarray, np.ndarray, tuple[AssetPart, ...]]],
    yaw: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = _compile_part_boxes(definitions, components, yaw)
    if not boxes:
        raise ValueError("no mesh geoms found while probing bounds")
    lows, highs = zip(*boxes.values())
    return np.minimum.reduce(lows), np.maximum.reduce(highs)


_BOUNDS_CACHE: dict[tuple[str, str, float], np.ndarray] = {}


def _corrected_bounds(asset: InteractiveAsset, mesh: str, yaw: float = 0.0) -> np.ndarray:
    key = (asset.asset_id, mesh, round(yaw, 8))
    if key not in _BOUNDS_CACHE:
        # _compile_bbox cannot express the body-local correction, so probe it
        # with the same nested bodies used in the generated scene.
        root = ET.Element("mujoco", model="interactive_bounds")
        ET.SubElement(root, "compiler", angle="radian", balanceinertia="true")
        xml_assets = ET.SubElement(root, "asset")
        for definition in asset.definitions:
            ET.SubElement(xml_assets, definition.tag, definition.attributes)
        world = ET.SubElement(root, "worldbody")
        body = ET.SubElement(world, "body", euler=numbers((0, 0, yaw)))
        ET.SubElement(body, "freejoint")
        orientation = ET.SubElement(body, "body", euler=numbers(interactive_euler(asset.asset_id)))
        ET.SubElement(orientation, "geom", name="mesh", type="mesh", mesh=mesh, mass="1")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        geom_id = model.geom("mesh").id
        mesh_id = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start : start + count]
        world_vertices = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        _BOUNDS_CACHE[key] = np.asarray((world_vertices.min(axis=0), world_vertices.max(axis=0)))
    return _BOUNDS_CACHE[key].copy()


def corrected_interactive_bounds(asset: InteractiveAsset, yaw: float = 0.0) -> np.ndarray:
    return _corrected_bounds(asset, asset.visual_mesh, yaw)


def corrected_interactive_collision_bounds(asset: InteractiveAsset, yaw: float = 0.0) -> np.ndarray:
    return _corrected_bounds(asset, asset.collision_mesh, yaw)


def _load_interactive_mjcf(path: Path, asset_id: str | None = None) -> InteractiveAsset:
    root = ET.parse(path).getroot()
    source_id = asset_id or path.stem
    prefix = f"interactive_{source_id.replace('-', '_')}"
    renamed = {
        tag: {node.get("name"): f"{prefix}_{tag}_{index:03d}" for index, node in enumerate(root.findall(f"./asset/{tag}"))}
        for tag in ("texture", "material", "mesh")
    }
    definitions = []
    for tag in ("texture", "material", "mesh"):
        for node in root.findall(f"./asset/{tag}"):
            attrs = dict(node.attrib)
            attrs["name"] = renamed[tag][node.get("name")]
            if "file" in attrs:
                attrs["file"] = str((path.parent / attrs["file"]).resolve())
            if tag == "material" and "texture" in attrs:
                attrs["texture"] = renamed["texture"][attrs["texture"]]
            definitions.append(ImportedDefinition(tag, attrs))
    body = root.find("./worldbody/body")
    if body is None:
        raise ValueError(f"{path}: no worldbody/body")
    visual = next((geom for geom in body.findall("geom") if geom.get("contype", "0") == "0"), None)
    collision = next((geom for geom in body.findall("geom") if geom.get("contype", "0") != "0"), None)
    if visual is None or collision is None:
        raise ValueError(f"{path}: expected visual and collision geoms")
    visual_mesh = renamed["mesh"][visual.get("mesh")]
    collision_mesh = renamed["mesh"][collision.get("mesh")]
    material = renamed["material"].get(visual.get("material"))
    minimum, maximum = _compile_bbox(tuple(definitions), [(_values(body.get("pos"), (0, 0, 0)), _values(body.get("quat"), (1, 0, 0, 0)), (AssetPart(visual_mesh, material),))])
    origin = np.asarray(((minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, minimum[2]))
    try:
        metadata_min_z = float(json.loads(path.with_name("metadata.json").read_text(encoding="utf-8")).get("min_z"))
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
        metadata_min_z = float(minimum[2])
    return InteractiveAsset(source_id, source_id, visual_mesh, collision_mesh, material, tuple(definitions), np.asarray((minimum - origin, maximum - origin)), float(visual.get("mass", "0.1")), metadata_min_z, collision.get("friction", "0.7 0.01 0.001"))


def _make_snack_asset(name: str, mesh_file: str, texture_file: str, scale: str, mass: float) -> InteractiveAsset:
    prefix = f"interactive_snack_{name}"
    mesh, texture, material = f"{prefix}_mesh", f"{prefix}_texture", f"{prefix}_material"
    definitions = (
        ImportedDefinition("mesh", {"name": mesh, "file": str((SNACK_ASSETS / mesh_file).resolve()), "scale": scale}),
        ImportedDefinition("texture", {"name": texture, "type": "2d", "file": str((SNACK_ASSETS / texture_file).resolve())}),
        ImportedDefinition("material", {"name": material, "texture": texture, "specular": "0.1", "shininess": "0.08"}),
    )
    minimum, maximum = _compile_bbox(definitions, [(np.zeros(3), np.asarray((1, 0, 0, 0)), (AssetPart(mesh, material),))])
    origin = np.asarray(((minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, minimum[2]))
    return InteractiveAsset(f"snack_{name}", name.replace("_", " ").title(), mesh, mesh, material, definitions, np.asarray((minimum - origin, maximum - origin)), mass, float(minimum[2]))


def load_interactive_assets() -> tuple[InteractiveAsset, ...]:
    assets = [_load_interactive_mjcf(path) for path in sorted((MODELS_ROOT / "assets" / "grasp_objects").glob("*/*.xml"))]
    assets.extend((
        _make_snack_asset("soda_can", "can.msh", "soda.png", "1.4 1.4 1.4", 0.157),
        _make_snack_asset("cereal_box", "cereal.msh", "cereal.png", "0.85 0.85 0.85", 0.25),
        _make_snack_asset("bread_snack", "bread.msh", "bread.png", "0.9 0.9 0.9", 0.20),
        _make_snack_asset("lemon", "lemon.msh", "lemon.png", "1.5 1 1", 0.08),
    ))
    return tuple(assets)
