"""Generate ten textured open-plan MuJoCo offices with a Stretch robot."""

from __future__ import annotations

import json
import math
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MUJOCO_GL", "egl")

import click
import cv2
import mujoco
import numpy as np

from office_interactive_assets import (
    INTERACTIVE_ROTATION_CORRECTIONS,
    InteractiveAsset,
    corrected_interactive_collision_bounds,
    interactive_euler,
    load_interactive_assets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "stretch_mujoco" / "models"
OFFICE_ASSETS = MODELS_ROOT / "assets" / "office_assets"
SNACK_ASSETS = MODELS_ROOT / "assets" / "office_snacks"
STRETCH_XML = MODELS_ROOT / "stretch.xml"
DEFAULT_OUTPUT = MODELS_ROOT / "assets" / "office_scenes"

# Interactive-object pools are deliberately role-specific.  This prevents a
# desk from receiving food packaging intended for the snack counter and makes
# the generated scenes easier to use for task-conditioned grasping.
WORKSTATION_INTERACTIVE_ASSET_IDS = (
    "017_calculator",
    "043_book",
    "116_keyboard",
    "101_milk-tea",
    "snack_soda_can",
)
SNACK_ZONE_INTERACTIVE_ASSET_IDS = (
    "001_bottle",
    "025_chips-tub",
    "035_apple",
    "038_milk-box",
    "071_can",
    "green_apple",
    "075_bread",
)
SNACK_DUPLICATE_PROBABILITY = 0.15

ZONE_COLORS = {
    "work": "0.22 0.34 0.41 1",
    "meeting": "0.34 0.29 0.40 1",
    "lounge": "0.29 0.40 0.35 1",
    "snack": "0.43 0.35 0.24 1",
    "circulation": "0.19 0.22 0.24 1",
}
DESK_SURFACE_HEIGHT_M = 0.74
MONITOR_DESK_EDGE_MARGIN_M = 0.03


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


def _compile_part_boxes(
    definitions: tuple["ImportedDefinition", ...],
    raw_components: list[tuple[np.ndarray, np.ndarray, tuple["AssetPart", ...]]],
    yaw: float = 0.0,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Compile *raw_components* standalone and return MuJoCo's true world-space
    (lo, hi) box per individual (component_index, part_index).
    """
    root = ET.Element("mujoco", model="probe")
    ET.SubElement(root, "compiler", angle="radian", balanceinertia="true")
    assets = ET.SubElement(root, "asset")
    for definition in definitions:
        ET.SubElement(assets, definition.tag, definition.attributes)
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name="probe", euler=numbers((0, 0, yaw)))
    ET.SubElement(body, "inertial", pos="0 0 0", mass="0.001", diaginertia="0.001 0.001 0.001")
    for component_index, (pos, quat, parts) in enumerate(raw_components):
        visual = ET.SubElement(
            body,
            "body",
            name=f"probe_component_{component_index:02d}",
            pos=numbers(pos),
            quat=numbers(quat),
        )
        ET.SubElement(
            visual, "inertial", pos="0 0 0", mass="0.001", diaginertia="0.001 0.001 0.001"
        )
        for part_index, part in enumerate(parts):
            ET.SubElement(
                visual,
                "geom",
                name=f"probe_geom_{component_index:02d}_{part_index:03d}",
                type="mesh",
                mesh=part.mesh,
                material=part.material,
                mass="0",
                shellinertia="true",
                contype="0",
                conaffinity="0",
            )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    boxes: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if not name.startswith("probe_geom_"):
            continue
        component_index, part_index = (int(token) for token in name.split("_")[2:4])
        meshid = model.geom_dataid[gid]
        vadr, vnum = model.mesh_vertadr[meshid], model.mesh_vertnum[meshid]
        verts = model.mesh_vert[vadr : vadr + vnum]
        xmat = data.geom_xmat[gid].reshape(3, 3)
        xpos = data.geom_xpos[gid]
        world = verts @ xmat.T + xpos
        boxes[(component_index, part_index)] = (world.min(axis=0), world.max(axis=0))
    return boxes


def _compile_bbox(
    definitions: tuple["ImportedDefinition", ...],
    raw_components: list[tuple[np.ndarray, np.ndarray, tuple["AssetPart", ...]]],
    yaw: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compile *raw_components* standalone and return MuJoCo's true aggregate
    world bbox, by collapsing _compile_part_boxes()'s per-part boxes into one.
    """
    boxes = _compile_part_boxes(definitions, raw_components, yaw=yaw)
    if not boxes:
        raise ValueError("no mesh geoms found while probing bbox")
    los, his = zip(*boxes.values())
    return np.minimum.reduce(los), np.maximum.reduce(his)


def settle_z_offset(
    support_boxes: list[tuple[np.ndarray, np.ndarray]],
    mover_lo: np.ndarray,
    mover_hi: np.ndarray,
    *,
    drop_clearance: float = 0.3,
    settle_time: float = 1.2,
) -> tuple[float, bool]:
    """Drop a box shaped like [mover_lo, mover_hi] straight down onto
    support_boxes under gravity and real contact/friction, and return the z
    offset that needs to be added to the mover's current position so it
    actually rests on whatever is beneath its footprint - plus whether
    contact force was actually present at rest (the thing this whole function
    exists to guarantee, instead of eyeballing a render).

    support_boxes are candidate world-space (lo, hi) AABBs - e.g. every
    sub-part of a desk, not just the one a human guesses is "the worktop".
    Physics resolves which one the mover's footprint actually lands on.
    """
    root = ET.Element("mujoco", model="settle")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", cone="elliptic")
    world = ET.SubElement(root, "worldbody")
    highest_support_z = max(float(hi[2]) for _, hi in support_boxes)
    friction = "1.0 0.01 0.0001"
    for index, (lo, hi) in enumerate(support_boxes):
        center = (lo + hi) / 2
        half_size = np.maximum((hi - lo) / 2, 0.004)
        ET.SubElement(
            world,
            "geom",
            name=f"support_{index:03d}",
            type="box",
            pos=numbers(center),
            size=numbers(half_size),
            friction=friction,
            contype="1",
            conaffinity="1",
        )
    mover_half_size = np.maximum((mover_hi - mover_lo) / 2, 0.004)
    mover_center_xy = (mover_lo[:2] + mover_hi[:2]) / 2
    drop_start_z = highest_support_z + drop_clearance + mover_half_size[2]
    body = ET.SubElement(
        world,
        "body",
        name="mover",
        pos=numbers((mover_center_xy[0], mover_center_xy[1], drop_start_z)),
    )
    ET.SubElement(body, "joint", name="drop", type="slide", axis="0 0 1", damping="0.4")
    ET.SubElement(
        body,
        "geom",
        name="mover_geom",
        type="box",
        size=numbers(mover_half_size),
        friction=friction,
        contype="1",
        conaffinity="1",
        mass="1.0",
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    joint_id = model.joint("drop").id
    dof_adr = model.jnt_dofadr[joint_id]
    for _ in range(int(settle_time / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if abs(data.qvel[dof_adr]) < 1e-4 and data.time > 0.3:
            break
    settled_body_z = drop_start_z + data.qpos[model.jnt_qposadr[joint_id]]
    settled_mover_bottom = settled_body_z - mover_half_size[2]
    mover_id = model.geom("mover_geom").id
    force = np.zeros(6)
    has_contact = False
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.geom1 == mover_id or contact.geom2 == mover_id:
            mujoco.mj_contactForce(model, data, i, force)
            if abs(force[0]) > 1e-6:
                has_contact = True
                break
    offset = settled_mover_bottom - float(mover_lo[2])
    return offset, has_contact


def load_mjcf_asset(
    relative_path: str,
    category: str,
    name: str,
    *,
    composite: bool = False,
    preserve_component_quats: bool = False,
    z_scale: float = 1.0,
    component_z_offsets: dict[int, float] | None = None,
    calibrate_component_z_offsets: "Callable[[tuple[ImportedDefinition, ...], list[tuple[np.ndarray, np.ndarray, tuple[AssetPart, ...]]]], dict[int, float]] | None" = None,
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
        attributes["name"] = mesh_names[original_name]
        attributes["file"] = str(mesh_path)
        attributes["scale"] = numbers(scale)
        definitions.append(ImportedDefinition("mesh", attributes))

    bodies = root.findall("./worldbody/body")
    if not composite:
        bodies = bodies[:1]
    unshifted_components = []
    for component_index, body in enumerate(bodies):
        pos = _values(body.get("pos"), (0.0, 0.0, 0.0)) * np.asarray((1.0, 1.0, z_scale))
        source_quat = _values(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
        quat = source_quat if preserve_component_quats else np.asarray((1.0, 0.0, 0.0, 0.0))
        parts = tuple(
            AssetPart(
                mesh=mesh_names[geom.get("mesh")],
                material=material_names[geom.get("material")],
            )
            for geom in body.findall("geom")
        )
        unshifted_components.append((pos, quat, parts))

    offsets = dict(component_z_offsets or {})
    if calibrate_component_z_offsets is not None:
        offsets.update(calibrate_component_z_offsets(tuple(definitions), unshifted_components))

    raw_components = []
    for component_index, (pos, quat, parts) in enumerate(unshifted_components):
        pos = pos.copy()
        pos[2] += offsets.get(component_index, 0.0)
        raw_components.append((pos, quat, parts))
    minimum, maximum = _compile_bbox(tuple(definitions), raw_components)
    origin = np.asarray(
        ((minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, minimum[2])
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


def measure_top_z(asset: AssetInfo, yaw: float = 0.0) -> float:
    """Return the true world-space top z of *asset* when placed at yaw, per
    MuJoCo's own compiled geometry (see ``_compile_bbox``). Use this instead
    of ``asset.bounds`` when something needs to sit flush on top of it."""
    raw_components = [
        (np.asarray(component.pos), np.asarray(component.quat), component.parts)
        for component in asset.components
    ]
    _, hi = _compile_bbox(asset.definitions, raw_components, yaw=yaw)
    return float(hi[2])


def _calibrate_cb_desk_monitor_offsets(
    definitions: tuple[ImportedDefinition, ...],
    unshifted_components: list[tuple[np.ndarray, np.ndarray, tuple[AssetPart, ...]]],
) -> dict[int, float]:
    """Align cb-desk monitors to the broad slab and correct mirrored laptops."""
    desk_indices = [i for i, (_, _, parts) in enumerate(unshifted_components) if len(parts) == 4]
    monitor_indices = [i for i, (_, _, parts) in enumerate(unshifted_components) if len(parts) == 7]
    if not desk_indices or not monitor_indices:
        return {}
    worktop_bounds = {}
    for desk_index in desk_indices:
        worktop_bounds[desk_index] = _compile_bbox(definitions, [unshifted_components[desk_index]])
    max_footprint = max((hi[0] - lo[0]) * (hi[1] - lo[1]) for lo, hi in worktop_bounds.values())
    worktop_indices = [
        index
        for index, (lo, hi) in worktop_bounds.items()
        if (hi[0] - lo[0]) * (hi[1] - lo[1]) > max_footprint / 2
    ]
    part_boxes = _compile_part_boxes(definitions, unshifted_components)
    offsets: dict[int, float] = {}
    for monitor_index in monitor_indices:
        monitor_pos = unshifted_components[monitor_index][0]
        nearest_desk = min(
            worktop_indices,
            key=lambda d: np.hypot(
                unshifted_components[d][0][0] - monitor_pos[0],
                unshifted_components[d][0][1] - monitor_pos[1],
            ),
        )
        worktop_lo, worktop_hi = worktop_bounds[nearest_desk]
        mover_lo, mover_hi = _compile_bbox(definitions, [unshifted_components[monitor_index]])
        overlap_x = max(0.0, min(mover_hi[0], worktop_hi[0]) - max(mover_lo[0], worktop_lo[0]))
        overlap_y = max(0.0, min(mover_hi[1], worktop_hi[1]) - max(mover_lo[1], worktop_lo[1]))
        mover_area = (mover_hi[0] - mover_lo[0]) * (mover_hi[1] - mover_lo[1])
        if overlap_x * overlap_y < 0.7 * mover_area:
            pos, quat, parts = unshifted_components[monitor_index]
            flipped_quat = np.zeros(4)
            mujoco.mju_mulQuat(flipped_quat, np.array((0.0, 0.0, 0.0, 1.0)), quat)
            unshifted_components[monitor_index] = (pos, flipped_quat, parts)
            mover_lo, mover_hi = _compile_bbox(definitions, [unshifted_components[monitor_index]])
        # Every monitor on this worktop sits near one of its short ends (2
        # users per long edge, side by side along x), each authored with its
        # own leftover gap to that end - from a few cm to, for the outer
        # seat above, an outright overhang. Re-anchor every one of them (not
        # just the ones that were overhanging) to the same fixed margin from
        # its own nearest end, so all 4 monitors on a desk read as evenly
        # placed instead of at whatever gap each happened to be authored at.
        monitor_center_x = (mover_lo[0] + mover_hi[0]) / 2
        worktop_center_x = (worktop_lo[0] + worktop_hi[0]) / 2
        if monitor_center_x < worktop_center_x:
            nudge_x = (worktop_lo[0] + MONITOR_DESK_EDGE_MARGIN_M) - mover_lo[0]
        else:
            nudge_x = (worktop_hi[0] - MONITOR_DESK_EDGE_MARGIN_M) - mover_hi[0]
        monitor_pos[0] += nudge_x
        mover_lo[0] += nudge_x
        mover_hi[0] += nudge_x
        # The worktop component contains raised perimeter rails.  Settling
        # against every sub-part consequently selects a rail top and lifts
        # the monitor by ~16 cm.  Use the broad desktop slab height instead,
        # and anchor the monitor's *visual* mesh (not its collision hull) to
        # that plane so the rendered asset visibly touches the desktop.
        slab_candidates = [
            float(hi[2])
            for (ci, _pi), (lo, hi) in part_boxes.items()
            if ci == nearest_desk
            and (hi[0] - lo[0]) * (hi[1] - lo[1]) > 1.0
            and hi[2] - lo[2] < 0.08
            and hi[2] < 0.65
        ]
        if not slab_candidates:
            candidates = [
                float(hi[2])
                for (ci, _pi), (lo, hi) in part_boxes.items()
                if ci == nearest_desk
                and (hi[0] - lo[0]) * (hi[1] - lo[1]) > 1.0
                and hi[2] - lo[2] < 0.08
            ]
            if not candidates:
                raise ValueError(f"cb_desk_2400 monitor component {monitor_index}: desktop slab not found")
            slab_candidates = candidates
        slab_z = max(slab_candidates)
        visual_lo, _ = _compile_bbox(
            definitions,
            [(unshifted_components[monitor_index][0],
              unshifted_components[monitor_index][1],
              unshifted_components[monitor_index][2])],
        )
        offsets[monitor_index] = float(slab_z - visual_lo[2])
    return offsets


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
            calibrate_component_z_offsets=(
                _calibrate_cb_desk_monitor_offsets
                if relative_path == "desks/cb_desk_2400.xml"
                else None
            ),
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
        static_bounds = np.asarray(data["bounds_m"], dtype=float)
        center_xy = static_bounds.mean(axis=0)[:2]
        prefix = f"office_{asset_id}_part_"
        definitions = tuple(
            definition
            for definition in registry_definitions
            if definition.attributes.get("name", "").startswith(prefix)
        )
        parts = tuple(
            AssetPart(
                mesh=f"{prefix}{part_index:03d}_mesh",
                material=f"{prefix}{part_index:03d}_material",
            )
            for part_index in range(int(data["material_part_count"]))
        )
        # asset.json's bounds_m comes from a separate extraction tool and can
        # disagree with what MuJoCo actually renders (see _compile_bbox), so
        # measure the true bbox instead of trusting it for floor anchoring.
        probe_pos = np.asarray((-center_xy[0], -center_xy[1], 0.0))
        probe_quat = np.asarray((1.0, 0.0, 0.0, 0.0))
        minimum, maximum = _compile_bbox(definitions, [(probe_pos, probe_quat, parts)])
        components = (
            AssetComponent(
                pos=(-float(center_xy[0]), -float(center_xy[1]), -float(minimum[2])),
                quat=(1.0, 0.0, 0.0, 0.0),
                parts=parts,
            ),
        )
        bounds = np.asarray(
            (
                (minimum[0], minimum[1], 0.0),
                (maximum[0], maximum[1], maximum[2] - minimum[2]),
            )
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


def _asset_part_boxes(asset: AssetInfo) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Return the compiled, asset-local bounds of every imported mesh part."""
    components = [
        (np.asarray(component.pos), np.asarray(component.quat), component.parts)
        for component in asset.components
    ]
    return _compile_part_boxes(asset.definitions, components)


def _component_boxes(
    part_boxes: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Merge part bounds into one box per imported asset component."""
    boxes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for (component_index, _), (lo, hi) in part_boxes.items():
        old = boxes.get(component_index)
        boxes[component_index] = (
            (lo.copy(), hi.copy())
            if old is None
            else (np.minimum(old[0], lo), np.maximum(old[1], hi))
        )
    return boxes


def _xy_overlaps(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    *,
    minimum: float = 0.015,
) -> bool:
    """Whether two axis-aligned boxes overlap enough to count as a collision."""
    return bool(np.all(np.minimum(first[1][:2], second[1][:2]) - np.maximum(first[0][:2], second[0][:2]) > minimum))


def _yaw_rotated_xy_bounds(bounds: np.ndarray, yaw: float) -> np.ndarray:
    """Return the XY AABB of an existing axis-aligned box after yaw rotation."""
    corners = np.asarray(
        [
            (bounds[x_index, 0], bounds[y_index, 1])
            for x_index in (0, 1)
            for y_index in (0, 1)
        ]
    )
    rotation = np.asarray(
        ((math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw)))
    )
    rotated = corners @ rotation.T
    result = bounds.copy()
    result[0, :2] = rotated.min(axis=0)
    result[1, :2] = rotated.max(axis=0)
    return result


def cb_desk_worktop_boxes(
    pod: AssetInfo,
    part_boxes: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return the real local boxes of the two cb_desk_2400 worktop halves."""
    if pod.asset_id != "new_cb_desk_2400":
        raise ValueError(f"Expected cb_desk_2400, got {pod.asset_id}")
    component_boxes = _component_boxes(part_boxes or _asset_part_boxes(pod))
    areas = {
        component_index: float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        for component_index, (lo, hi) in component_boxes.items()
    }
    largest_area = max(areas.values())
    return tuple(
        component_boxes[component_index]
        for component_index in sorted(areas)
        if areas[component_index] >= largest_area * 0.99
    )


def cb_desk_worktop_component_indices(
    pod: AssetInfo,
    part_boxes: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> set[int]:
    """Identify the broad four-part worktop components of cb_desk_2400."""
    if pod.asset_id != "new_cb_desk_2400":
        raise ValueError(f"Expected cb_desk_2400, got {pod.asset_id}")
    component_boxes = _component_boxes(part_boxes or _asset_part_boxes(pod))
    areas = {
        ci: float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        for ci, (lo, hi) in component_boxes.items()
        if len(pod.components[ci].parts) == 4
    }
    if not areas:
        return set()
    largest = max(areas.values())
    return {ci for ci, area in areas.items() if area > largest / 2}


def cb_desk_worktop_surface_z(
    pod: AssetInfo,
    part_boxes: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> float:
    """Return the broad desktop slab's top z, excluding raised rails."""
    boxes = part_boxes or _asset_part_boxes(pod)
    worktop_indices = cb_desk_worktop_component_indices(pod, boxes)
    surfaces: list[float] = []
    for (component_index, _part_index), (lo, hi) in boxes.items():
        if component_index not in worktop_indices:
            continue
        area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        if area < 1.0:
            continue
        thickness = float(hi[2] - lo[2])
        # The actual slab is a broad thin layer. Rails/legs are either much
        # thicker or extend to the aggregate rail height.
        if thickness < 0.08 and hi[2] > 0.45:
            surfaces.append(float(hi[2]))
    if not surfaces:
        raise ValueError("Could not identify CB-desk desktop surface layer")
    return float(max(surfaces))


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
            # Use the imported mesh parts themselves as collision geometry.
            # MuJoCo compiles mesh geoms to convex collision hulls, preserving
            # legs/chairs/openings instead of collapsing an entire asset into
            # one world-space AABB.  The same geoms are consumed by the
            # polygon occupancy backend (their compiled vertices are projected
            # to XY), so physics and navigation share one source of truth.
            collision_part_index = 0
            worktop_component_indices = (
                cb_desk_worktop_component_indices(asset)
                if asset.asset_id == "new_cb_desk_2400"
                else set()
            )
            for component_index, component in enumerate(asset.components):
                collision_body = ET.SubElement(
                    body,
                    "body",
                    name=f"collision_component_{index:03d}_{component_index:02d}",
                    pos=numbers(component.pos),
                    quat=numbers(component.quat),
                )
                # This body only groups static collision geoms; it still
                # needs a tiny inertial because MuJoCo requires every moving
                # body in the kinematic tree to have positive mass.
                ET.SubElement(
                    collision_body,
                    "inertial",
                    pos="0 0 0",
                    mass="0.001",
                    diaginertia="0.001 0.001 0.001",
                )
                for part_index, part in enumerate(component.parts):
                    # This disconnected mesh becomes a large convex hull and
                    # intersects props placed on the desktop.
                    if (
                        asset.asset_id == "new_cb_desk_2400"
                        and component_index in worktop_component_indices
                        and part_index == 1
                    ):
                        continue
                    ET.SubElement(
                        collision_body,
                        "geom",
                        name=(f"nav_collision_{index:03d}_{collision_part_index:03d}"),
                        type="mesh",
                        mesh=part.mesh,
                        mass="0",
                        shellinertia="true",
                        contype="1",
                        conaffinity="1",
                        group="3",
                        rgba="0 0 0 0",
                        friction="0.9 0.01 0.001",
                    )
                    collision_part_index += 1
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

    def place_interactive(
        self,
        asset: InteractiveAsset,
        x: float,
        y: float,
        *,
        z: float,
        yaw: float = 0.0,
        support: str | None = None,
    ) -> str:
        """Place a real freejoint object with a mesh visual and stable box collision."""
        index = self.instance_count
        self.instance_count += 1
        object_id = f"{asset.asset_id}_{index:03d}"
        body = ET.SubElement(
            self.world,
            "body",
            name=object_id,
            pos=numbers((x, y, z)),
            euler=numbers((0.0, 0.0, yaw)),
        )
        ET.SubElement(body, "freejoint", name=f"{object_id}_freejoint")
        collision_bounds = corrected_interactive_collision_bounds(asset)
        collision_size = np.maximum(collision_bounds[1] - collision_bounds[0], 0.004)
        collision_center = (collision_bounds[0] + collision_bounds[1]) / 2
        # Mesh-derived inertia and contact can be poorly conditioned for thin
        # or non-watertight imported meshes (notably keyboards and cans),
        # amplifying ordinary desk contacts into huge angular accelerations.
        # Use the collision mesh's corrected AABB for both inertia and contact.
        dx, dy, dz = collision_size
        inertia = asset.mass_kg / 12.0 * np.asarray(
            (dy * dy + dz * dz, dx * dx + dz * dz, dx * dx + dy * dy)
        )
        ET.SubElement(
            body,
            "inertial",
            pos=numbers(collision_center),
            mass=numbers((asset.mass_kg,)),
            diaginertia=numbers(np.maximum(inertia, 1e-6)),
        )
        orientation_euler = numbers(interactive_euler(asset.asset_id))
        visual_attrs = {
            "name": f"{object_id}_visual",
            "type": "mesh",
            "mesh": asset.visual_mesh,
            "mass": "0",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
            "euler": orientation_euler,
        }
        if asset.material:
            visual_attrs["material"] = asset.material
        ET.SubElement(body, "geom", **visual_attrs)
        ET.SubElement(
            body,
            "geom",
            name=f"{object_id}_collision",
            type="box",
            pos=numbers(collision_center),
            size=numbers(collision_size / 2),
            mass="0",
            contype="1",
            conaffinity="1",
            friction=asset.friction,
            rgba="0 0 0 0",
        )
        ET.SubElement(
            body,
            "site",
            name=f"{object_id}_grasp_site",
            pos=numbers((0, 0, max(0.01, asset.height * 0.5))),
            size="0.018",
            rgba="0 0 0 0",
        )
        self.placements.append(
            {
                "instance": index,
                "asset_id": asset.asset_id,
                "name": asset.name,
                "category": "interactive_objects",
                "object_id": object_id,
                "body": object_id,
                "position": [x, y, z],
                "yaw": yaw,
                "euler_correction": list(
                    INTERACTIVE_ROTATION_CORRECTIONS.get(
                        asset.asset_id, (0.0, 0.0, 0.0)
                    )
                ),
                "mass_kg": asset.mass_kg,
                "dynamic": True,
                "graspable": True,
                "support": support,
                "grasp_site": f"{object_id}_grasp_site",
                "collision_radius": float(np.linalg.norm((asset.bounds[1] - asset.bounds[0])[:2]) / 2),
            }
        )
        return object_id


def _place_random_workstation_objects(
    furnisher: Furnisher,
    pod: AssetInfo,
    pod_instance_index: int,
    x: float,
    y: float,
    yaw: float,
    selected_assets: tuple[InteractiveAsset, ...],
    rng: random.Random,
) -> None:
    """Place selected work objects on one pod's free desk edge.

    Candidates are tested against the pod's monitor component AABBs in pod-local
    coordinates, so the generated object footprint cannot overlap an existing
    monitor. The front edge (local y=-1.0) is preferred because it is visible
    and reachable by Stretch; the rear edge is a fallback for unusually wide
    objects.
    """
    if not selected_assets:
        return
    part_boxes = _asset_part_boxes(pod)
    monitor_boxes = [
        box for (component_index, _), box in part_boxes.items()
        if len(pod.components[component_index].parts) == 7
    ]
    # Generate candidates from the actual worktop extent. The outer strips are
    # tried first because the monitor row occupies the middle of the desk.
    candidates = []
    for worktop_lo, worktop_hi in cb_desk_worktop_boxes(pod, part_boxes):
        x_positions = np.linspace(worktop_lo[0] + 0.14, worktop_hi[0] - 0.14, 5)
        y_positions = (
            worktop_lo[1] + 0.13,
            worktop_hi[1] - 0.13,
            worktop_lo[1] + 0.42,
            worktop_hi[1] - 0.42,
        )
        candidates.extend(
            (float(local_x), float(local_y), worktop_lo, worktop_hi)
            for local_y in y_positions
            for local_x in x_positions
        )
    placed_boxes: list[tuple[np.ndarray, np.ndarray]] = []
    support = f"asset_{pod_instance_index:03d}_{pod.asset_id[:8]}"
    # Do not use worktop_hi[2] here: that is the top of the raised perimeter
    # rail.  All candidate strips share the broad desktop plane.
    worktop_surface_z = cb_desk_worktop_surface_z(pod, part_boxes)
    for asset in selected_assets:
        # These boxes are checked in the unrotated pod frame.  The pod yaw is
        # applied only when converting the accepted local position to world
        # coordinates below.
        # The collision mesh is the geometry MuJoCo actually resolves against
        # the desk.  Some imported grasp meshes have a slightly lower
        # collision hull than their render mesh; anchoring from the visual
        # bounds therefore starts the freejoint body interpenetrating the
        # desktop and can produce an explosive contact impulse on frame one.
        corrected_bounds = corrected_interactive_collision_bounds(asset)
        candidate_order = list(candidates)
        rng.shuffle(candidate_order)
        for local_x, local_y, worktop_lo, worktop_hi in candidate_order:
            object_box = (
                corrected_bounds[0, :2] + (local_x, local_y),
                corrected_bounds[1, :2] + (local_x, local_y),
            )
            lo, hi = object_box
            if lo[0] < worktop_lo[0] + 0.05 or hi[0] > worktop_hi[0] - 0.05:
                continue
            if lo[1] < worktop_lo[1] + 0.05 or hi[1] > worktop_hi[1] - 0.05:
                continue
            if any(_xy_overlaps(object_box, box) for box in monitor_boxes):
                continue
            if any(_xy_overlaps(object_box, old) for old in placed_boxes):
                continue
            # Place the corrected mesh on the compiled CB-desk worktop.  Do
            # not use pod.bounds[1, 2] here: that aggregate bbox includes the
            # monitors and would float objects roughly 25 cm above the desk.
            # Collision meshes can extend below the visual mesh by a few mm;
            # use the rendered mesh's true lowest point for visual contact.
            # Keep the collision hull (rather than merely the visible mesh)
            # just above the desktop.  The tiny clearance avoids numerical
            # penetration while remaining visually flush; the settling
            # dynamics can then establish stable contact without launching
            # the object.
            collision_clearance = 0.002
            local_z = float(worktop_surface_z - corrected_bounds[0, 2] + collision_clearance)
            world_xy = np.asarray((x, y)) + np.asarray(
                (math.cos(yaw) * local_x - math.sin(yaw) * local_y,
                 math.sin(yaw) * local_x + math.cos(yaw) * local_y)
            )
            furnisher.place_interactive(
                asset,
                float(world_xy[0]),
                float(world_xy[1]),
                z=local_z,
                yaw=yaw,
                support=support,
            )
            placed_boxes.append(object_box)
            break
        else:
            raise ValueError(
                f"Could not place workstation object {asset.asset_id} on pod {pod_instance_index}"
            )


def furnish_work_zone(
    furnisher: Furnisher,
    zone: Zone,
    style: str,
    variant: int,
    interactive_assets: tuple[InteractiveAsset, ...] = (),
) -> None:
    pod = furnisher.select("workstation_pods", variant)
    if pod.asset_id != "new_cb_desk_2400":
        raise ValueError(
            f"Interactive workstation objects must be placed on cb_desk_2400, got {pod.asset_id}"
        )
    size = pod.bounds[1] - pod.bounds[0]
    cx, cy = zone.center
    yaw = math.pi / 2 if style == "rows_y" else 0.0
    # A "rows_y" placement rotates the pod 90 degrees, so its footprint along
    # the offset axis (world y) is governed by the pod's pre-rotation *x*
    # extent, not its y extent - using size[1] here understated a
    # ~5.6m-wide pod's spacing need as its ~2.4m depth, letting two pods
    # overlap into what looked like one 8-monitor desk.
    long_size = float(size[0])
    available = zone.depth if style == "rows_y" else zone.width
    count = max(1, min(2, int((available - 0.8) // (long_size + 0.45))))
    offsets = (np.arange(count) - (count - 1) / 2) * (long_size + 0.35)
    pod_placements: list[tuple[int, float, float]] = []
    for pod_index, offset in enumerate(offsets):
        x = cx if style == "rows_y" else cx + float(offset)
        y = cy + float(offset) if style == "rows_y" else cy
        pod_instance_index = furnisher.instance_count
        furnisher.place("workstation_pods", x, y, yaw=yaw, collision=True, asset=pod)
        pod_placements.append((pod_instance_index, x, y))

    asset_by_id = {asset.asset_id: asset for asset in interactive_assets}
    available = [asset_by_id[asset_id] for asset_id in WORKSTATION_INTERACTIVE_ASSET_IDS if asset_id in asset_by_id]
    if len(available) != len(WORKSTATION_INTERACTIVE_ASSET_IDS):
        missing = sorted(set(WORKSTATION_INTERACTIVE_ASSET_IDS) - set(asset_by_id))
        raise ValueError(f"Missing workstation interactive assets: {', '.join(missing)}")
    rng = random.Random(1701 + variant * 31)
    selected = tuple(rng.sample(available, rng.randint(2, 3)))
    assigned: list[list[InteractiveAsset]] = [[] for _ in pod_placements]
    for asset in selected:
        assigned[rng.randrange(len(assigned))].append(asset)
    for (pod_instance_index, x, y), assets_for_pod in zip(pod_placements, assigned):
        _place_random_workstation_objects(
            furnisher, pod, pod_instance_index, x, y, yaw, tuple(assets_for_pod), rng
        )
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
    table_top_z = measure_top_z(table, table_yaw)
    furnisher.place(
        "displays",
        cx,
        cy,
        z=table_top_z - 0.015,
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


def add_snack_counter(
    furnisher: Furnisher,
    zone: Zone,
    variant: int,
    interactive_assets: tuple[InteractiveAsset, ...] = (),
) -> None:
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
        size="1.1 0.48 0.04",
        material="snack_counter_wood",
        friction="0.9 0.02 0.002",
    )
    for x in (-0.94, 0.94):
        ET.SubElement(
            body,
            "geom",
            type="box",
            pos=numbers((x, 0, 0.35)),
            size="0.04 0.44 0.35",
            material="snack_counter_metal",
        )
    asset_by_id = {asset.asset_id: asset for asset in interactive_assets}
    missing = sorted(set(SNACK_ZONE_INTERACTIVE_ASSET_IDS) - set(asset_by_id))
    if missing:
        raise ValueError(f"Missing snack-zone interactive assets: {', '.join(missing)}")
    rng = random.Random(4103 + variant * 47)
    # Every selected type appears once; a small fraction receive a second
    # instance for a natural-looking but bounded snack display.
    selected = rng.sample(
        [asset_by_id[asset_id] for asset_id in SNACK_ZONE_INTERACTIVE_ASSET_IDS], 7
    )
    snack_assets = [
        asset
        for asset in selected
        for _ in range(1 + int(rng.random() < SNACK_DUPLICATE_PROBABILITY))
    ]
    # Sample continuously in the counter's front band (negative local Y,
    # where snack_pickup_site and the robot approach are located).  This keeps
    # props reachable instead of allowing them to drift toward the far edge.
    # Place larger footprints first so rejection sampling reliably finds room
    # for every selected object.
    counter_half_x, counter_half_y = 1.1, 0.48
    edge_margin = 0.045
    object_gap = 0.025
    reachable_y = (-0.34, -0.08)
    candidates = []
    for asset in snack_assets:
        relative_yaw = rng.uniform(-math.pi / 7, math.pi / 7)
        # place_interactive uses the corrected collision AABB as its stable
        # box collider. Rotate that exact box here so placement and runtime
        # collision checks operate on identical geometry.
        bounds = _yaw_rotated_xy_bounds(
            corrected_interactive_collision_bounds(asset), relative_yaw
        )
        footprint_area = float(np.prod(bounds[1, :2] - bounds[0, :2]))
        candidates.append((footprint_area, rng.random(), asset, relative_yaw, bounds))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    placed_boxes: list[tuple[np.ndarray, np.ndarray]] = []
    placements = []
    for _area, _tie_breaker, asset, relative_yaw, corrected_bounds in candidates:
        x_min = -counter_half_x + edge_margin - float(corrected_bounds[0, 0])
        x_max = counter_half_x - edge_margin - float(corrected_bounds[1, 0])
        y_min = max(
            -counter_half_y + edge_margin - float(corrected_bounds[0, 1]),
            reachable_y[0],
        )
        y_max = min(
            counter_half_y - edge_margin - float(corrected_bounds[1, 1]),
            reachable_y[1],
        )
        if x_min > x_max or y_min > y_max:
            raise ValueError(f"Snack object {asset.asset_id} does not fit in reachable counter band")

        for _attempt in range(1000):
            local_x = rng.uniform(x_min, x_max)
            local_y = rng.uniform(y_min, y_max)
            object_box = (
                corrected_bounds[0, :2] + (local_x, local_y),
                corrected_bounds[1, :2] + (local_x, local_y),
            )
            if any(_xy_overlaps(object_box, old, minimum=-object_gap) for old in placed_boxes):
                continue
            placed_boxes.append(object_box)
            placements.append((asset, relative_yaw, corrected_bounds, local_x, local_y))
            break
        else:
            raise ValueError(
                f"Could not place snack object {asset.asset_id} without overlap "
                f"inside the reachable counter band"
            )

    for asset, relative_yaw, corrected_bounds, local_x, local_y in placements:
        world_xy = np.asarray((cx, cy)) + np.asarray(
            (math.cos(yaw) * local_x - math.sin(yaw) * local_y,
             math.sin(yaw) * local_x + math.cos(yaw) * local_y)
        )
        furnisher.place_interactive(
            asset,
            float(world_xy[0]),
            float(world_xy[1]),
            z=0.78 - corrected_bounds[0, 2],
            yaw=yaw + relative_yaw,
            support="snack_counter",
        )
    ET.SubElement(
        body, "site", name="snack_pickup_site", pos="0 -0.58 0.82", size="0.02", rgba="0 0 0 0"
    )
    furnisher.record_procedural("Snack Counter", "furniture/snack_counter", cx, cy, 1.2, yaw=yaw)


def furnish_snack_zone(
    furnisher: Furnisher,
    zone: Zone,
    variant: int,
    interactive_assets: tuple[InteractiveAsset, ...] = (),
) -> None:
    add_snack_counter(furnisher, zone, variant, interactive_assets)
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


def add_interactive_assets(root: ET.Element, assets_pool: tuple[InteractiveAsset, ...]) -> None:
    assets = root.find("asset")
    if assets is None:
        raise RuntimeError("scene has no asset section")
    seen: set[str] = set()
    for asset in assets_pool:
        for definition in asset.definitions:
            name = definition.attributes.get("name", "")
            if name in seen:
                continue
            seen.add(name)
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
    interactive_assets: tuple[InteractiveAsset, ...],
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
    add_interactive_assets(root, interactive_assets)
    world = ET.SubElement(root, "worldbody")
    add_open_shell(world, spec)
    furnisher = Furnisher(world, asset_pool)
    zones = {zone.zone_type: zone for zone in spec.zones}
    furnish_work_zone(
        furnisher, zones["work"], spec.work_style, scene_index, interactive_assets
    )
    furnish_meeting_zone(furnisher, zones["meeting"], spec.meeting_style, scene_index)
    furnish_lounge_zone(furnisher, zones["lounge"], scene_index)
    furnish_snack_zone(furnisher, zones["snack"], scene_index, interactive_assets)
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
    interactive_assets = load_interactive_assets()
    catalog = []
    preview_images = []
    for index, spec in enumerate(scene_specs(), start=1):
        click.echo(f"Generating {spec.scene_id}")
        xml_path, manifest_path = build_scene(spec, index, asset_pool, interactive_assets, output)
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
