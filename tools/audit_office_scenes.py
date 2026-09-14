"""Headless geometry audit for the generated office scenes.

Compiles each office_*.xml with MuJoCo, computes the true world-space bounding
box of every placed furniture/prop instance from its visual mesh vertices, and
flags instances whose lowest point does not rest (within `--tolerance`) on the
floor or on the top surface of whatever else occupies the same footprint below
it. Catches the class of bug where an object floats above its supporting
surface (e.g. a monitor hovering over a desk) or sinks into the floor.

Usage:
  .venv/bin/python tools/audit_office_scenes.py
  .venv/bin/python tools/audit_office_scenes.py --scene office_04_central_meeting
  .venv/bin/python tools/audit_office_scenes.py --tolerance 0.02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENES_DIR = PROJECT_ROOT / "stretch_mujoco" / "models" / "assets" / "office_scenes"
# Some valid grasp objects (notably the notebook) are very thin along one
# axis.  A 2 cm overlap requirement rejects their entire footprint even when
# they are fully on a desk, so use a smaller geometric contact threshold and
# rely on the z-gap test for the actual support decision.
MIN_FOOTPRINT_OVERLAP_M = 0.005


def _top_level_asset_body(model: mujoco.MjModel, body_id: int) -> int:
    """Climb to the body directly under worldbody (the placed instance root)."""
    cur = body_id
    top = body_id
    while cur > 0:
        top = cur
        cur = int(model.body_parentid[cur])
    return top


_BOX_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float
)


def instance_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World-space [min, max] xyz per placed instance.

    Covers visual (group 2) mesh geoms plus opaque box-primitive geoms
    (procedural furniture like the snack counter mixes both).  Support is
    deliberately resolved separately from physical collision geometry below:
    an aggregate visual AABB of a composite desk includes its legs, monitors
    and worktop, so it cannot identify the actual tabletop height.
    """
    geom_world: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("asset_collision"):
            continue
        xmat = data.geom_xmat[gid].reshape(3, 3)
        xpos = data.geom_xpos[gid]
        meshid = model.geom_dataid[gid]
        if meshid >= 0:
            if model.geom_group[gid] != 2:
                continue
            vadr, vnum = model.mesh_vertadr[meshid], model.mesh_vertnum[meshid]
            world = model.mesh_vert[vadr : vadr + vnum] @ xmat.T + xpos
        elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            world = (_BOX_CORNERS * model.geom_size[gid]) @ xmat.T + xpos
        else:
            continue
        geom_world[gid] = (world.min(axis=0), world.max(axis=0))

    inst_bounds: dict[str, list[np.ndarray]] = {}
    for gid, (lo, hi) in geom_world.items():
        body_id = int(model.geom_bodyid[gid])
        if body_id == 0:
            continue  # room shell (floor/walls), not a placed instance
        top = _top_level_asset_body(model, body_id)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, top)
        if name is None:
            continue
        if name not in inst_bounds:
            inst_bounds[name] = [lo.copy(), hi.copy()]
        else:
            inst_bounds[name][0] = np.minimum(inst_bounds[name][0], lo)
            inst_bounds[name][1] = np.maximum(inst_bounds[name][1], hi)
    return {name: (b[0], b[1]) for name, b in inst_bounds.items()}


def collision_support_boxes(
    model: mujoco.MjModel, data: mujoco.MjData
) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Return world-space boxes of individual physical collision geoms.

    This intentionally preserves each collision geom instead of merging the
    whole furniture instance.  For example, a calculator on a CB desk is
    supported by the worktop collision mesh at z=0.711 m, not by the desk's
    aggregate visual AABB (whose monitors extend to z≈0.98 m).
    """
    supports: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for gid in range(model.ngeom):
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        xmat = data.geom_xmat[gid].reshape(3, 3)
        xpos = data.geom_xpos[gid]
        meshid = int(model.geom_dataid[gid])
        if meshid >= 0:
            vadr, vnum = model.mesh_vertadr[meshid], model.mesh_vertnum[meshid]
            world = model.mesh_vert[vadr : vadr + vnum] @ xmat.T + xpos
        elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            world = (_BOX_CORNERS * model.geom_size[gid]) @ xmat.T + xpos
        else:
            # The audit's generated office supports are mesh/box geoms. Other
            # primitive types are irrelevant here and cannot safely be
            # represented by the box-corner helper.
            continue
        body_id = int(model.geom_bodyid[gid])
        top = _top_level_asset_body(model, body_id) if body_id else 0
        top_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, top) or "world"
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        supports.append((top_name, geom_name, world.min(axis=0), world.max(axis=0)))
    return supports


def audit_scene(xml_path: Path, tolerance: float) -> list[tuple[str, str, float, float, float]]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    instances = [(name, lo, hi) for name, (lo, hi) in instance_bounds(model, data).items()]
    supports = collision_support_boxes(model, data)
    flagged = []
    for name, lo, hi in instances:
        best_support_z = 0.0  # floor
        support_name = "floor"
        for other_name, geom_name, olo, ohi in supports:
            # A placed object's own collision mesh is not a support. Static
            # composite furniture may share one top-level body with its
            # monitor meshes, which is also correctly excluded here.
            if other_name == name:
                continue
            overlap_x = min(hi[0], ohi[0]) - max(lo[0], olo[0])
            overlap_y = min(hi[1], ohi[1]) - max(lo[1], olo[1])
            if (
                overlap_x > MIN_FOOTPRINT_OVERLAP_M
                and overlap_y > MIN_FOOTPRINT_OVERLAP_M
                and ohi[2] <= lo[2] + 0.08
            ):
                if ohi[2] > best_support_z:
                    best_support_z = float(ohi[2])
                    support_name = f"{other_name}/{geom_name}"
        gap = float(lo[2] - best_support_z)
        if abs(gap) > tolerance:
            flagged.append((name, support_name, gap, float(lo[2]), float(hi[2])))
    return flagged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", help="Only audit the scene with this id (e.g. office_01_linear_bench)")
    parser.add_argument("--tolerance", type=float, default=0.03, help="Allowed gap/penetration in metres")
    args = parser.parse_args()

    if args.scene:
        xml_paths = [SCENES_DIR / f"{args.scene}.xml"]
    else:
        xml_paths = sorted(p for p in SCENES_DIR.glob("office_*.xml") if not p.stem.endswith("_robot"))

    total_flagged = 0
    for xml_path in xml_paths:
        flagged = audit_scene(xml_path, args.tolerance)
        total_flagged += len(flagged)
        print(f"=== {xml_path.name} ===")
        if not flagged:
            print("  ok")
            continue
        for name, support, gap, lo_z, hi_z in flagged:
            tag = "FLOAT" if gap > 0 else "PENETRATE"
            print(f"  {tag:9s} gap={gap:+.3f}m  {name:28s} on {support:20s} z[{lo_z:.3f},{hi_z:.3f}]")

    print(f"\ntotal flagged: {total_flagged}")
    return 1 if total_flagged else 0


if __name__ == "__main__":
    sys.exit(main())
