"""Generate ten textured open-plan MuJoCo offices with a Stretch robot."""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import click
import cv2
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "stretch_mujoco" / "models"
OFFICE_ASSETS = MODELS_ROOT / "assets" / "office_assets"
SNACK_ASSETS = MODELS_ROOT / "assets" / "office_snacks"
STRETCH_XML = MODELS_ROOT / "stretch.xml"
DEFAULT_OUTPUT = MODELS_ROOT / "assets" / "office_scenes"

ZONE_COLORS = {
    "work": "0.22 0.34 0.41 1",
    "meeting": "0.34 0.29 0.40 1",
    "lounge": "0.29 0.40 0.35 1",
    "snack": "0.43 0.35 0.24 1",
    "circulation": "0.19 0.22 0.24 1",
}
DESK_SURFACE_HEIGHT_M = 0.74


@dataclass(frozen=True)
class Zone:
    zone_type: str
    bounds: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bounds[0] + self.bounds[1]) / 2, (self.bounds[2] + self.bounds[3]) / 2)

    @property
    def width(self) -> float:
        return self.bounds[1] - self.bounds[0]

    @property
    def depth(self) -> float:
        return self.bounds[3] - self.bounds[2]


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    title: str
    width: float
    depth: float
    zones: tuple[Zone, ...]
    work_style: str
    meeting_style: str
    robot_start: tuple[float, float, float]


@dataclass(frozen=True)
class ImportedDefinition:
    tag: str
    attributes: dict[str, str]


@dataclass(frozen=True)
class AssetPart:
    mesh: str
    material: str


@dataclass(frozen=True)
class AssetComponent:
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    parts: tuple[AssetPart, ...]


@dataclass(frozen=True)
class AssetInfo:
    asset_id: str
    name: str
    category: str
    bounds: np.ndarray
    components: tuple[AssetComponent, ...]
    definitions: tuple[ImportedDefinition, ...]

    @property
    def part_count(self) -> int:
        return sum(len(component.parts) for component in self.components)


def scene_specs() -> tuple[SceneSpec, ...]:
    return (
        SceneSpec(
            "office_01_linear_bench",
            "Linear Bench Office",
            20,
            11,
            (
                Zone("work", (-10.0, 0.0, -5.5, 5.5)),
                Zone("meeting", (0.0, 10.0, 1.0, 5.5)),
                Zone("lounge", (0.0, 6.0, -5.5, 1.0)),
                Zone("snack", (6.0, 10.0, -5.5, 1.0)),
            ),
            "rows_x",
            "boardroom",
            (-0.7, -3.8, 0.0),
        ),
        SceneSpec(
            "office_02_cross_axis",
            "Cross Axis Office",
            18,
            12,
            (
                Zone("work", (-9.0, 9.0, -6.0, 0.0)),
                Zone("meeting", (-9.0, -2.0, 0.0, 6.0)),
                Zone("lounge", (-2.0, 4.5, 0.0, 6.0)),
                Zone("snack", (4.5, 9.0, 0.0, 6.0)),
            ),
            "rows_x",
            "huddle",
            (-7.7, -0.7, math.pi),
        ),
        SceneSpec(
            "office_03_long_gallery",
            "Long Gallery Office",
            21,
            10,
            (
                Zone("work", (-10.5, -2.0, -5.0, 5.0)),
                Zone("meeting", (-2.0, 5.0, -5.0, 0.0)),
                Zone("lounge", (-2.0, 5.0, 0.0, 5.0)),
                Zone("snack", (5.0, 10.5, -5.0, 5.0)),
            ),
            "rows_y",
            "boardroom",
            (9.2, 3.5, math.pi / 2),
        ),
        SceneSpec(
            "office_04_central_meeting",
            "Central Meeting Office",
            21,
            10.4,
            (
                Zone("work", (-10.5, 10.5, 0.5, 5.2)),
                Zone("meeting", (-10.5, -2.0, -5.2, 0.5)),
                Zone("lounge", (-2.0, 4.5, -5.2, 0.5)),
                Zone("snack", (4.5, 10.5, -5.2, 0.5)),
            ),
            "rows_x",
            "round",
            (9.2, 1.3, 0.0),
        ),
        SceneSpec(
            "office_05_team_clusters",
            "Team Cluster Office",
            20,
            12,
            (
                Zone("work", (-10.0, 2.0, -6.0, 1.0)),
                Zone("meeting", (2.0, 10.0, -6.0, 1.0)),
                Zone("lounge", (-10.0, 4.0, 1.0, 6.0)),
                Zone("snack", (4.0, 10.0, 1.0, 6.0)),
            ),
            "rows_x",
            "boardroom",
            (-8.7, 3.7, -math.pi / 2),
        ),
        SceneSpec(
            "office_06_diagonal_flow",
            "Diagonal Flow Office",
            21,
            11,
            (
                Zone("work", (-10.5, 2.0, -5.5, 1.0)),
                Zone("meeting", (2.0, 10.5, -5.5, 1.0)),
                Zone("lounge", (-10.5, 4.0, 1.0, 5.5)),
                Zone("snack", (4.0, 10.5, 1.0, 5.5)),
            ),
            "rows_x",
            "huddle",
            (-9.2, 4.2, -math.pi / 3),
        ),
        SceneSpec(
            "office_07_u_bench",
            "U Bench Office",
            20,
            12,
            (
                Zone("work", (-10.0, -1.0, -6.0, 6.0)),
                Zone("meeting", (-1.0, 10.0, -6.0, -1.0)),
                Zone("lounge", (-1.0, 5.0, -1.0, 6.0)),
                Zone("snack", (5.0, 10.0, -1.0, 6.0)),
            ),
            "rows_y",
            "round",
            (-0.7, -4.7, math.pi),
        ),
        SceneSpec(
            "office_08_dual_island",
            "Dual Island Office",
            20,
            11,
            (
                Zone("work", (-10.0, 10.0, -5.5, -0.5)),
                Zone("meeting", (-10.0, -3.0, -0.5, 5.5)),
                Zone("lounge", (-3.0, 4.0, -0.5, 5.5)),
                Zone("snack", (4.0, 10.0, -0.5, 5.5)),
            ),
            "rows_x",
            "huddle",
            (-8.7, -2.2, math.pi / 2),
        ),
        SceneSpec(
            "office_09_staggered_rows",
            "Staggered Rows Office",
            20,
            11,
            (
                Zone("work", (-10.0, 3.0, -5.5, 1.0)),
                Zone("meeting", (3.0, 10.0, -5.5, 1.0)),
                Zone("lounge", (-10.0, 4.0, 1.0, 5.5)),
                Zone("snack", (4.0, 10.0, 1.0, 5.5)),
            ),
            "rows_x",
            "boardroom",
            (-8.7, 3.8, -math.pi / 2),
        ),
        SceneSpec(
            "office_10_social_core",
            "Social Core Office",
            20,
            12,
            (
                Zone("work", (-10.0, -1.0, -6.0, 6.0)),
                Zone("meeting", (-1.0, 10.0, 1.0, 6.0)),
                Zone("lounge", (-1.0, 5.0, -6.0, 1.0)),
                Zone("snack", (5.0, 10.0, -6.0, 1.0)),
            ),
            "rows_y",
            "round",
            (8.5, 1.3, math.pi),
        ),
    )


def _values(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.asarray([float(value) for value in text.split()] if text else default, dtype=float)


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def _mesh_vertices(path: Path, scale: np.ndarray) -> np.ndarray:
    vertices = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(vertices) * scale


def load_mjcf_asset(
    relative_path: str,
    category: str,
    name: str,
    *,
    composite: bool = False,
    preserve_component_quats: bool = False,
    z_scale: float = 1.0,
    component_z_offsets: dict[int, float] | None = None,
) -> AssetInfo:
    source = OFFICE_ASSETS / relative_path
    root = ET.parse(source).getroot()
    prefix = "new_" + source.stem
    texture_names = {
        node.get("name"): f"{prefix}_texture_{index:03d}"
        for index, node in enumerate(root.findall("./asset/texture"))
    }
    material_names = {
        node.get("name"): f"{prefix}_material_{index:03d}"
        for index, node in enumerate(root.findall("./asset/material"))
    }
    mesh_names = {
        node.get("name"): f"{prefix}_mesh_{index:03d}"
        for index, node in enumerate(root.findall("./asset/mesh"))
    }
    definitions: list[ImportedDefinition] = []
    mesh_vertices: dict[str, np.ndarray] = {}
    for node in root.findall("./asset/texture"):
        attributes = dict(node.attrib)
        attributes["name"] = texture_names[node.get("name")]
        if "file" in attributes:
            attributes["file"] = str((source.parent / attributes["file"]).resolve())
        definitions.append(ImportedDefinition("texture", attributes))
    for node in root.findall("./asset/material"):
        attributes = dict(node.attrib)
        attributes["name"] = material_names[node.get("name")]
        if "texture" in attributes:
            attributes["texture"] = texture_names[attributes["texture"]]
        definitions.append(ImportedDefinition("material", attributes))
    for node in root.findall("./asset/mesh"):
        attributes = dict(node.attrib)
        original_name = node.get("name")
        mesh_path = (source.parent / attributes["file"]).resolve()
        scale = _values(attributes.get("scale"), (1.0, 1.0, 1.0)) * np.asarray((1.0, 1.0, z_scale))
        mesh_vertices[original_name] = _mesh_vertices(mesh_path, scale)
        attributes["name"] = mesh_names[original_name]
        attributes["file"] = str(mesh_path)
        attributes["scale"] = numbers(scale)
        definitions.append(ImportedDefinition("mesh", attributes))

    bodies = root.findall("./worldbody/body")
    if not composite:
        bodies = bodies[:1]
    raw_components = []
    all_vertices = []
    for component_index, body in enumerate(bodies):
        pos = _values(body.get("pos"), (0.0, 0.0, 0.0)) * np.asarray((1.0, 1.0, z_scale))
        pos[2] += (component_z_offsets or {}).get(component_index, 0.0)
        source_quat = _values(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
        quat = source_quat if preserve_component_quats else np.asarray((1.0, 0.0, 0.0, 0.0))
        rotation = _quat_matrix(quat)
        parts = []
        for geom in body.findall("geom"):
            original_mesh = geom.get("mesh")
            transformed = mesh_vertices[original_mesh] @ rotation.T + pos
            all_vertices.append(transformed)
            parts.append(
                AssetPart(
                    mesh=mesh_names[original_mesh],
                    material=material_names[geom.get("material")],
                )
            )
        raw_components.append((pos, quat, tuple(parts)))
    combined = np.vstack(all_vertices)
    minimum = combined.min(axis=0)
    maximum = combined.max(axis=0)
    floor_origin = 0.0 if min(pos[2] for pos, _, _ in raw_components) < 0.1 else minimum[2]
    origin = np.asarray(
        ((minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, floor_origin)
    )
    components = tuple(
        AssetComponent(
            pos=tuple((pos - origin).tolist()),
            quat=tuple(quat.tolist()),
            parts=parts,
        )
        for pos, quat, parts in raw_components
    )
    bounds = np.asarray(
        (
            (minimum[0] - origin[0], minimum[1] - origin[1], 0.0),
            (maximum[0] - origin[0], maximum[1] - origin[1], maximum[2] - origin[2]),
        )
    )
    return AssetInfo(prefix, name, category, bounds, components, tuple(definitions))


def asset_file_path(path: str | Path) -> str:
    """Return a portable file path relative to the model's compiler asset root."""
    return os.path.relpath(Path(path).resolve(), (MODELS_ROOT / "assets").resolve())


def load_assets() -> dict[str, list[AssetInfo]]:
    specifications = (
        ("desks/adjustable_desk.xml", "office_desks", "Adjustable Office Desk", False, False),
        (
            "desks/cb_desk_2400.xml",
            "workstation_pods",
            "CB Desk 2400 Workstation Pod",
            True,
            True,
        ),
        ("desks/oak_square_table.xml", "office_desks", "Oak Square Table", False, False),
        ("desks/reception_desk.xml", "reception_desks", "Curved Reception Desk", False, False),
        ("chairs/office_chair.xml", "office_chairs", "Cornell Swivel Office Chair", False, False),
        ("chairs/home_office_chair.xml", "office_chairs", "Home Office Chair", False, False),
        ("chairs/dining_chair.xml", "meeting_chairs", "Dining Meeting Chair", False, False),
        ("chairs/stance_chair.xml", "meeting_chairs", "Stance Meeting Chair", False, False),
        ("displays/imac_24.xml", "displays", "iMac 24", False, False),
        ("displays/imac_27.xml", "displays", "iMac 27", False, False),
        (
            "meeting_table/teaming_table.xml",
            "meeting_tables",
            "Teaming Meeting Table",
            True,
            False,
            DESK_SURFACE_HEIGHT_M / 0.9778999366941336,
        ),
        ("whiteboard/dry_wipe_module.xml", "whiteboards", "Double Dry Wipe Board", True, True),
    )
    by_category: dict[str, list[AssetInfo]] = defaultdict(list)
    for spec in specifications:
        relative_path, category, name, composite, preserve_quats, *overrides = spec
        asset = load_mjcf_asset(
            relative_path,
            category,
            name,
            composite=composite,
            preserve_component_quats=preserve_quats,
            z_scale=overrides[0] if overrides else 1.0,
            component_z_offsets=overrides[1] if len(overrides) > 1 else None,
        )
        by_category[category].append(asset)
    registry_path = OFFICE_ASSETS / "mjcf" / "office_assets.xml"
    legacy_registry = ET.parse(registry_path).getroot()
    registry_definitions = []
    for node in legacy_registry.find("asset"):
        attributes = dict(node.attrib)
        if "file" in attributes:
            source_path = Path(attributes["file"])
            if not source_path.is_absolute():
                source_path = registry_path.parent / source_path
            attributes["file"] = str(source_path.resolve())
        registry_definitions.append(ImportedDefinition(node.tag, attributes))
    for metadata_path in sorted((OFFICE_ASSETS / "furniture" / "sofas").glob("*/asset.json")):
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        asset_id = data["asset_id"]
        bounds = np.asarray(data["bounds_m"], dtype=float)
        center = bounds.mean(axis=0)
        prefix = f"office_{asset_id}_part_"
        definitions = tuple(
            definition
            for definition in registry_definitions
            if definition.attributes.get("name", "").startswith(prefix)
        )
        components = (
            AssetComponent(
                pos=(-float(center[0]), -float(center[1]), -float(bounds[0, 2])),
                quat=(1.0, 0.0, 0.0, 0.0),
                parts=tuple(
                    AssetPart(
                        mesh=f"{prefix}{part_index:03d}_mesh",
                        material=f"{prefix}{part_index:03d}_material",
                    )
                    for part_index in range(int(data["material_part_count"]))
                ),
            ),
        )
        by_category["legacy_sofas"].append(
            AssetInfo(
                asset_id,
                data["display_name"],
                "legacy_sofas",
                bounds,
                components,
                definitions,
            )
        )
    return dict(by_category)


def numbers(values: Any) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def add_wall_segment(
    world: ET.Element,
    name: str,
    axis: str,
    coordinate: float,
    start: float,
    end: float,
    *,
    gap_center: float | None = None,
    gap_width: float = 1.8,
) -> None:
    intervals = [(start, end)]
    if gap_center is not None:
        intervals = [
            (start, gap_center - gap_width / 2),
            (gap_center + gap_width / 2, end),
        ]
    for index, (part_start, part_end) in enumerate(intervals):
        if part_end - part_start < 0.1:
            continue
        middle = (part_start + part_end) / 2
        half_length = (part_end - part_start) / 2
        pos = (coordinate, middle, 1.4) if axis == "x" else (middle, coordinate, 1.4)
        size = (0.07, half_length, 1.4) if axis == "x" else (half_length, 0.07, 1.4)
        ET.SubElement(
            world,
            "geom",
            name=f"{name}_{index}",
            type="box",
            pos=numbers(pos),
            size=numbers(size),
            material="scene_wall",
        )


def add_open_shell(world: ET.Element, spec: SceneSpec) -> None:
    half_width = spec.width / 2
    half_depth = spec.depth / 2
    ET.SubElement(
        world,
        "geom",
        name="office_floor",
        type="box",
        pos="0 0 -0.04",
        size=numbers((half_width, half_depth, 0.04)),
        material="scene_floor",
        friction="1 0.01 0.001",
    )
    add_wall_segment(
        world,
        "front_wall",
        "y",
        -half_depth,
        -half_width,
        half_width,
        gap_center=0,
    )
    add_wall_segment(world, "back_wall", "y", half_depth, -half_width, half_width)
    add_wall_segment(world, "left_wall", "x", -half_width, -half_depth, half_depth)
    add_wall_segment(world, "right_wall", "x", half_width, -half_depth, half_depth)
    for zone in spec.zones:
        ET.SubElement(
            world,
            "site",
            name=f"zone_{zone.zone_type}_center",
            pos=numbers((*zone.center, 0.025)),
            size="0.015",
            rgba="0 0 0 0",
        )
        ET.SubElement(
            world,
            "light",
            name=f"light_{zone.zone_type}",
            pos=numbers((*zone.center, 2.7)),
            dir="0 0 -1",
            diffuse="0.52 0.51 0.48",
            attenuation="0.15 0.05 0.02",
            castshadow="false",
        )


def validate_zone_floor(spec: SceneSpec) -> None:
    half_width = spec.width / 2
    half_depth = spec.depth / 2
    total_area = 0.0
    for index, zone in enumerate(spec.zones):
        xmin, xmax, ymin, ymax = zone.bounds
        if xmin < -half_width or xmax > half_width or ymin < -half_depth or ymax > half_depth:
            raise ValueError(f"{spec.scene_id}: {zone.zone_type} extends beyond the floor")
        total_area += zone.width * zone.depth
        for other in spec.zones[index + 1 :]:
            oxmin, oxmax, oymin, oymax = other.bounds
            overlap_x = min(xmax, oxmax) - max(xmin, oxmin)
            overlap_y = min(ymax, oymax) - max(ymin, oymin)
            if overlap_x > 1e-6 and overlap_y > 1e-6:
                raise ValueError(f"{spec.scene_id}: {zone.zone_type} overlaps {other.zone_type}")
    floor_area = spec.width * spec.depth
    if not math.isclose(total_area, floor_area, abs_tol=1e-6):
        raise ValueError(
            f"{spec.scene_id}: zones cover {total_area:.2f} of {floor_area:.2f} square metres"
        )


class Furnisher:
    def __init__(self, world: ET.Element, asset_pool: dict[str, list[AssetInfo]]) -> None:
        self.world = world
        self.asset_pool = asset_pool
        self.counters: dict[str, int] = defaultdict(int)
        self.instance_count = 0
        self.placements: list[dict[str, Any]] = []

    def pick(self, category: str) -> AssetInfo:
        values = self.asset_pool[category]
        index = self.counters[category] % len(values)
        self.counters[category] += 1
        return values[index]

    def select(self, category: str, index: int) -> AssetInfo:
        values = self.asset_pool[category]
        return values[index % len(values)]

    def place(
        self,
        category: str,
        x: float,
        y: float,
        *,
        z: float = 0.0,
        yaw: float = 0.0,
        collision: bool = False,
        asset: AssetInfo | None = None,
    ) -> AssetInfo:
        asset = asset or self.pick(category)
        index = self.instance_count
        self.instance_count += 1
        body = ET.SubElement(
            self.world,
            "body",
            name=f"asset_{index:03d}_{asset.asset_id[:8]}",
            pos=numbers((x, y, z)),
            euler=numbers((0, 0, yaw)),
        )
        ET.SubElement(
            body,
            "inertial",
            pos="0 0 0",
            mass="0.001",
            diaginertia="0.001 0.001 0.001",
        )
        part_index = 0
        for component_index, component in enumerate(asset.components):
            visual = ET.SubElement(
                body,
                "body",
                name=f"asset_component_{index:03d}_{component_index:02d}",
                pos=numbers(component.pos),
                quat=numbers(component.quat),
            )
            ET.SubElement(
                visual,
                "inertial",
                pos="0 0 0",
                mass="0.001",
                diaginertia="0.001 0.001 0.001",
            )
            for part in component.parts:
                ET.SubElement(
                    visual,
                    "geom",
                    name=f"asset_visual_{index:03d}_{part_index:03d}",
                    type="mesh",
                    mesh=part.mesh,
                    material=part.material,
                    mass="0",
                    shellinertia="true",
                    contype="0",
                    conaffinity="0",
                    group="2",
                )
                part_index += 1
        dimensions = asset.bounds[1] - asset.bounds[0]
        collision_radius = 0.0
        if collision:
            collision_radius = float(np.linalg.norm(dimensions[:2]) / 2)
            ET.SubElement(
                body,
                "geom",
                name=f"asset_collision_{index:03d}",
                type="box",
                pos=numbers((0, 0, max(0.02, dimensions[2] / 2))),
                size=numbers(
                    (
                        max(0.03, dimensions[0] / 2),
                        max(0.03, dimensions[1] / 2),
                        max(0.02, dimensions[2] / 2),
                    )
                ),
                rgba="0 0 0 0",
                friction="0.9 0.01 0.001",
            )
        self.placements.append(
            {
                "instance": index,
                "asset_id": asset.asset_id,
                "name": asset.name,
                "category": category,
                "position": [x, y, z],
                "yaw": yaw,
                "collision_radius": collision_radius,
            }
        )
        return asset

    def record_procedural(
        self,
        name: str,
        category: str,
        x: float,
        y: float,
        radius: float,
        *,
        yaw: float = 0.0,
    ) -> None:
        self.placements.append(
            {
                "instance": self.instance_count,
                "asset_id": f"procedural_{name}",
                "name": name,
                "category": category,
                "position": [x, y, 0.0],
                "yaw": yaw,
                "collision_radius": radius,
            }
        )
        self.instance_count += 1


def furnish_work_zone(furnisher: Furnisher, zone: Zone, style: str, variant: int) -> None:
    pod = furnisher.select("workstation_pods", variant)
    size = pod.bounds[1] - pod.bounds[0]
    cx, cy = zone.center
    yaw = math.pi / 2 if style == "rows_y" else 0.0
    long_size = float(size[1] if style == "rows_y" else size[0])
    available = zone.depth if style == "rows_y" else zone.width
    count = max(1, min(2, int((available - 0.8) // (long_size + 0.45))))
    offsets = (np.arange(count) - (count - 1) / 2) * (long_size + 0.35)
    for offset in offsets:
        x = cx if style == "rows_y" else cx + float(offset)
        y = cy + float(offset) if style == "rows_y" else cy
        furnisher.place("workstation_pods", x, y, yaw=yaw, collision=True, asset=pod)
    xmin, xmax, ymin, ymax = zone.bounds
    add_plant(furnisher, xmax - 0.5, ymax - 0.5)
    add_trash_bin(furnisher, xmin + 0.5, ymin + 0.5)


def furnish_meeting_zone(furnisher: Furnisher, zone: Zone, style: str, variant: int) -> None:
    cx, cy = zone.center
    table = furnisher.select("meeting_tables", variant)
    chair = furnisher.select("meeting_chairs", variant)
    whiteboard = furnisher.select("whiteboards", variant)
    display = furnisher.select("displays", variant)
    add_rug(furnisher, cx, cy, min(2.4, zone.width / 3), min(1.8, zone.depth / 3))
    if style == "boardroom":
        table_yaw = 0.0
        chair_count, rx, ry = 8, min(2.45, zone.width / 3), min(1.45, zone.depth / 3)
    elif style == "huddle":
        table_yaw = math.pi / 2
        chair_count, rx, ry = 6, min(1.55, zone.width / 3), min(2.15, zone.depth / 3)
    else:
        table_yaw = 0.0
        chair_count, rx, ry = 6, min(2.2, zone.width / 3), min(1.4, zone.depth / 3)
    furnisher.place("meeting_tables", cx, cy, yaw=table_yaw, collision=True, asset=table)
    table_height = float((table.bounds[1] - table.bounds[0])[2])
    furnisher.place(
        "displays",
        cx,
        cy,
        z=table_height - 0.015,
        yaw=table_yaw,
        asset=display,
    )
    for index in range(chair_count):
        angle = 2 * math.pi * index / chair_count
        furnisher.place(
            "meeting_chairs",
            cx + rx * math.cos(angle),
            cy + ry * math.sin(angle),
            yaw=angle - math.pi / 2,
            collision=True,
            asset=chair,
        )
    xmin, xmax, ymin, ymax = zone.bounds
    board_x, board_y, board_yaw = xmax - 0.35, cy, 0.0
    if zone.width > zone.depth * 1.8:
        board_x, board_y, board_yaw = cx, ymax - 0.35, math.pi / 2
    furnisher.place(
        "whiteboards", board_x, board_y, yaw=board_yaw, collision=True, asset=whiteboard
    )
    add_plant(furnisher, xmin + 0.45, ymax - 0.45)


def furnish_lounge_zone(furnisher: Furnisher, zone: Zone, variant: int) -> None:
    cx, cy = zone.center
    _, xmax, ymin, ymax = zone.bounds
    sofa = furnisher.select("legacy_sofas", variant)
    add_rug(furnisher, cx, cy, min(2.2, zone.width / 3), min(1.7, zone.depth / 3))
    furnisher.place("legacy_sofas", cx, ymax - 0.75, yaw=0.0, collision=True, asset=sofa)
    if zone.depth > 4.0:
        furnisher.place("legacy_sofas", cx, ymin + 0.75, yaw=math.pi, collision=True, asset=sofa)
    add_plant(furnisher, xmax - 0.45, ymax - 0.45)


def add_rug(furnisher: Furnisher, x: float, y: float, half_x: float, half_y: float) -> None:
    index = furnisher.instance_count
    ET.SubElement(
        furnisher.world,
        "geom",
        name=f"procedural_rug_{index:03d}",
        type="box",
        pos=numbers((x, y, 0.018)),
        size=numbers((half_x, half_y, 0.015)),
        material="lounge_rug",
        contype="0",
        conaffinity="0",
    )
    furnisher.record_procedural("Area Rug", "decoration/rugs", x, y, 0.0)


def add_plant(furnisher: Furnisher, x: float, y: float) -> None:
    index = furnisher.instance_count
    body = ET.SubElement(
        furnisher.world, "body", name=f"procedural_plant_{index:03d}", pos=numbers((x, y, 0))
    )
    ET.SubElement(
        body, "geom", type="cylinder", pos="0 0 0.2", size="0.18 0.2", material="plant_pot"
    )
    ET.SubElement(
        body,
        "geom",
        type="sphere",
        pos="0 0 0.7",
        size="0.38",
        material="plant_leaf",
        contype="0",
        conaffinity="0",
    )
    furnisher.record_procedural("Office Plant", "props/plants", x, y, 0.2)


def add_trash_bin(furnisher: Furnisher, x: float, y: float) -> None:
    index = furnisher.instance_count
    ET.SubElement(
        furnisher.world,
        "geom",
        name=f"procedural_bin_{index:03d}",
        type="cylinder",
        pos=numbers((x, y, 0.22)),
        size="0.18 0.22",
        material="snack_counter_metal",
    )
    furnisher.record_procedural("Waste Bin", "props/trash_bins", x, y, 0.18)


def add_snack_counter(furnisher: Furnisher, zone: Zone, variant: int) -> None:
    cx, cy = zone.center
    yaw = math.pi / 2 if variant % 3 == 1 else 0.0
    body = ET.SubElement(
        furnisher.world,
        "body",
        name="snack_counter",
        pos=numbers((cx, cy, 0)),
        euler=numbers((0, 0, yaw)),
    )
    ET.SubElement(
        body,
        "geom",
        name="snack_counter_top",
        type="box",
        pos="0 0 0.74",
        size="0.85 0.38 0.04",
        material="snack_counter_wood",
        friction="0.9 0.02 0.002",
    )
    for x in (-0.70, 0.70):
        ET.SubElement(
            body,
            "geom",
            type="box",
            pos=numbers((x, 0, 0.35)),
            size="0.04 0.34 0.35",
            material="snack_counter_metal",
        )
    snack_specs = (
        ("snack_can_mesh", "snack_soda", -0.52, 0.84),
        ("snack_cereal_mesh", "snack_cereal", -0.17, 0.84),
        ("snack_bread_mesh", "snack_bread", 0.18, 0.81),
        ("snack_lemon_mesh", "snack_lemon", 0.52, 0.81),
    )
    for mesh, material, x, z in snack_specs:
        ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=mesh,
            material=material,
            pos=numbers((x, -0.03, z)),
            mass="0",
            shellinertia="true",
            contype="0",
            conaffinity="0",
            group="2",
        )
    ET.SubElement(
        body, "site", name="snack_pickup_site", pos="0 -0.48 0.82", size="0.02", rgba="0 0 0 0"
    )
    furnisher.record_procedural("Snack Counter", "furniture/snack_counter", cx, cy, 0.95, yaw=yaw)


def furnish_snack_zone(furnisher: Furnisher, zone: Zone, variant: int) -> None:
    add_snack_counter(furnisher, zone, variant)
    xmin, xmax, ymin, ymax = zone.bounds
    cabinet_x, cabinet_y = xmin + 0.65, ymax - 0.45
    cabinet = ET.SubElement(
        furnisher.world,
        "body",
        name=f"snack_cabinet_{furnisher.instance_count:03d}",
        pos=numbers((cabinet_x, cabinet_y, 0)),
    )
    ET.SubElement(
        cabinet,
        "geom",
        type="box",
        pos="0 0 0.6",
        size="0.55 0.28 0.6",
        material="snack_cabinet",
    )
    furnisher.record_procedural(
        "Snack Storage Cabinet", "furniture/storage", cabinet_x, cabinet_y, 0.62
    )
    add_trash_bin(furnisher, xmax - 0.45, ymin + 0.45)
    if zone.width > 4.2:
        add_plant(furnisher, xmin + 0.45, ymin + 0.45)


def add_scene_assets(root: ET.Element) -> None:
    assets = ET.SubElement(root, "asset")
    ET.SubElement(
        assets,
        "texture",
        name="scene_floor_texture",
        type="2d",
        builtin="checker",
        rgb1="0.24 0.27 0.29",
        rgb2="0.19 0.21 0.23",
        width="512",
        height="512",
    )
    ET.SubElement(
        assets,
        "material",
        name="scene_floor",
        texture="scene_floor_texture",
        texrepeat="8 5",
        reflectance="0.04",
    )
    ET.SubElement(assets, "material", name="scene_wall", rgba="0.84 0.86 0.85 1", specular="0.03")
    ET.SubElement(assets, "material", name="snack_counter_wood", rgba="0.55 0.36 0.20 1")
    ET.SubElement(assets, "material", name="snack_counter_metal", rgba="0.14 0.16 0.17 1")
    ET.SubElement(assets, "material", name="snack_cabinet", rgba="0.70 0.68 0.61 1")
    ET.SubElement(assets, "material", name="lounge_rug", rgba="0.45 0.39 0.31 1")
    ET.SubElement(assets, "material", name="plant_pot", rgba="0.26 0.19 0.13 1")
    ET.SubElement(assets, "material", name="plant_leaf", rgba="0.12 0.38 0.18 1")
    for name, filename, scale in (
        ("snack_can_mesh", "can.msh", "1.4 1.4 1.4"),
        ("snack_cereal_mesh", "cereal.msh", "0.85 0.85 0.85"),
        ("snack_bread_mesh", "bread.msh", "0.9 0.9 0.9"),
        ("snack_lemon_mesh", "lemon.msh", "1.5 1 1"),
    ):
        ET.SubElement(
            assets,
            "mesh",
            name=name,
            file=asset_file_path(SNACK_ASSETS / filename),
            scale=scale,
        )
    for material, texture, filename in (
        ("snack_soda", "snack_soda_tex", "soda.png"),
        ("snack_cereal", "snack_cereal_tex", "cereal.png"),
        ("snack_bread", "snack_bread_tex", "bread.png"),
        ("snack_lemon", "snack_lemon_tex", "lemon.png"),
    ):
        ET.SubElement(
            assets,
            "texture",
            name=texture,
            type="2d",
            file=asset_file_path(SNACK_ASSETS / filename),
        )
        ET.SubElement(
            assets, "material", name=material, texture=texture, specular="0.1", shininess="0.08"
        )


def add_imported_assets(
    root: ET.Element,
    asset_pool: dict[str, list[AssetInfo]],
) -> None:
    assets = ET.SubElement(root, "asset")
    seen = set()
    for values in asset_pool.values():
        for asset in values:
            if asset.asset_id in seen:
                continue
            seen.add(asset.asset_id)
            for definition in asset.definitions:
                attributes = dict(definition.attributes)
                if "file" in attributes:
                    attributes["file"] = asset_file_path(attributes["file"])
                ET.SubElement(assets, definition.tag, **attributes)


def write_robot_include(spec: SceneSpec, output_dir: Path) -> Path:
    tree = ET.parse(STRETCH_XML)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("stretch.xml has no compiler element")
    compiler.set(
        "assetdir",
        os.path.relpath((MODELS_ROOT / "assets").resolve(), output_dir.resolve()),
    )
    base = root.find("./worldbody/body[@name='base_link']")
    if base is None:
        raise RuntimeError("stretch.xml has no base_link body")
    x, y, yaw = spec.robot_start
    base.set("pos", numbers((x, y, 0)))
    base.set("quat", numbers((math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))))
    path = output_dir / f"{spec.scene_id}_robot.xml"
    ET.indent(root, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=False)
    return path


def validate_layout(spec: SceneSpec, placements: list[dict[str, Any]]) -> float:
    x, y, _ = spec.robot_start
    wall_clearance = min(spec.width / 2 - abs(x), spec.depth / 2 - abs(y))
    if wall_clearance < 1.0:
        raise ValueError(f"{spec.scene_id}: robot start is too close to an outer wall")
    clearances = []
    for placement in placements:
        radius = placement["collision_radius"]
        if radius <= 0:
            continue
        px, py = placement["position"][:2]
        clearances.append(math.hypot(px - x, py - y) - radius - 0.35)
    minimum = min(clearances) if clearances else wall_clearance
    if minimum < 0.45:
        raise ValueError(
            f"{spec.scene_id}: robot start clearance is only {minimum:.2f} m; choose a wider point"
        )
    return minimum


def build_scene(
    spec: SceneSpec,
    scene_index: int,
    asset_pool: dict[str, list[AssetInfo]],
    output_dir: Path,
) -> tuple[Path, Path]:
    validate_zone_floor(spec)
    robot_include = write_robot_include(spec, output_dir)
    root = ET.Element("mujoco", model=spec.title)
    ET.SubElement(root, "include", file=robot_include.name)
    ET.SubElement(root, "compiler", angle="radian", balanceinertia="true")
    ET.SubElement(root, "statistic", center="0 0 1", extent="16")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        ambient="0.31 0.31 0.31",
        diffuse="0.66 0.66 0.66",
        specular="0.08 0.08 0.08",
    )
    add_scene_assets(root)
    add_imported_assets(root, asset_pool)
    world = ET.SubElement(root, "worldbody")
    add_open_shell(world, spec)
    furnisher = Furnisher(world, asset_pool)
    zones = {zone.zone_type: zone for zone in spec.zones}
    furnish_work_zone(furnisher, zones["work"], spec.work_style, scene_index)
    furnish_meeting_zone(furnisher, zones["meeting"], spec.meeting_style, scene_index)
    furnish_lounge_zone(furnisher, zones["lounge"], scene_index)
    furnish_snack_zone(furnisher, zones["snack"], scene_index)
    clearance = validate_layout(spec, furnisher.placements)

    center = ET.SubElement(world, "body", name="scene_center", pos="0 0 1")
    ET.SubElement(center, "site", size="0.01", rgba="0 0 0 0")
    radius = max(spec.width, spec.depth)
    ET.SubElement(
        world,
        "camera",
        name="overview",
        mode="targetbody",
        target="scene_center",
        pos=numbers((0, -radius * 1.12, radius * 0.95)),
        fovy="48",
    )
    ET.SubElement(
        world,
        "camera",
        name="top",
        mode="targetbody",
        target="scene_center",
        pos=numbers((0, 0, radius * 1.2)),
        fovy="45",
    )

    xml_path = output_dir / f"{spec.scene_id}.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="unicode", xml_declaration=False)
    manifest = {
        "scene_id": spec.scene_id,
        "title": spec.title,
        "layout": "single_open_area",
        "dimensions_m": [spec.width, spec.depth],
        "area_m2": spec.width * spec.depth,
        "zones": [{"type": zone.zone_type, "bounds": list(zone.bounds)} for zone in spec.zones],
        "workstation_layout": spec.work_style,
        "meeting_layout": spec.meeting_style,
        "robot": {
            "model": "Hello Robot Stretch",
            "initial_pose": list(spec.robot_start),
            "minimum_clearance_m": clearance,
            "body": "base_link",
        },
        "asset_instance_count": len(furnisher.placements),
        "assets": furnisher.placements,
        "mjcf": xml_path.name,
        "robot_include": robot_include.name,
        "preview": f"{spec.scene_id}.png",
    }
    manifest_path = output_dir / f"{spec.scene_id}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return xml_path, manifest_path


def render_preview(
    xml_path: Path, output: Path, expected_start: tuple[float, float, float]
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    actual_start = data.xpos[base_id].copy()
    if np.linalg.norm(actual_start[:2] - np.asarray(expected_start[:2])) > 1e-5:
        raise ValueError(f"Robot start mismatch in {xml_path}: {actual_start}")
    renderer = mujoco.Renderer(model, height=480, width=640)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    renderer.update_scene(data, camera="top", scene_option=scene_option)
    image = renderer.render()
    renderer.close()
    cv2.imwrite(str(output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return {
        "textures": model.ntex,
        "materials": model.nmat,
        "meshes": model.nmesh,
        "geoms": model.ngeom,
        "robot_start_xyz": actual_start.tolist(),
    }


def write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        """# Generated open-plan office scenes

Ten textured, single-area MuJoCo offices. Every layout contains a multi-workstation work zone,
a meeting zone, a lounge, a snack counter, and a Hello Robot Stretch placed at a scene-specific
clear starting point. There are no internal room walls; floor bands and furniture define zones.

Open a scene with:

```bash
.venv/bin/python examples/generated_office_scene.py --scene 1
```

The viewer starts in free-camera mode. Use left-drag to rotate, right-drag to pan, and the mouse
wheel to zoom. The scenes reference the centralized `office_assets` library and include a
scene-specific generated Stretch XML that sets the robot's initial freejoint pose.
""",
        encoding="utf-8",
    )


@click.command()
@click.option(
    "--output", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT, show_default=True
)
@click.option("--skip-previews", is_flag=True, help="Generate XML and JSON without EGL previews.")
def main(output: Path, skip_previews: bool) -> None:
    """Generate all ten open-plan office scenes."""
    output.mkdir(parents=True, exist_ok=True)
    asset_pool = load_assets()
    catalog = []
    preview_images = []
    for index, spec in enumerate(scene_specs(), start=1):
        click.echo(f"Generating {spec.scene_id}")
        xml_path, manifest_path = build_scene(spec, index, asset_pool, output)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not skip_previews:
            preview_path = output / f"{spec.scene_id}.png"
            manifest["model"] = render_preview(xml_path, preview_path, spec.robot_start)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            preview_images.append(cv2.imread(str(preview_path)))
        catalog.append(
            {
                "scene_id": spec.scene_id,
                "title": spec.title,
                "layout": "single_open_area",
                "area_m2": manifest["area_m2"],
                "workstation_layout": manifest["workstation_layout"],
                "asset_instance_count": manifest["asset_instance_count"],
                "robot_initial_pose": manifest["robot"]["initial_pose"],
                "robot_clearance_m": manifest["robot"]["minimum_clearance_m"],
                "mjcf": manifest["mjcf"],
                "manifest": manifest_path.name,
                "preview": manifest["preview"],
            }
        )
    (output / "catalog.json").write_text(
        json.dumps({"scene_count": 10, "scenes": catalog}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if preview_images:
        thumbnails = [cv2.resize(image, (320, 240)) for image in preview_images]
        contact_sheet = np.vstack(
            [np.hstack(thumbnails[index : index + 2]) for index in range(0, len(thumbnails), 2)]
        )
        cv2.imwrite(str(output / "contact_sheet.png"), contact_sheet)
    write_readme(output)
    click.echo(f"Generated {len(catalog)} scenes in {output}")


if __name__ == "__main__":
    main()
