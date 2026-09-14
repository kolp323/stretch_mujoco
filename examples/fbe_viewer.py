#!/usr/bin/env python3
"""FBE (Frontier-Based Exploration) in the MuJoCo viewer.

The robot starts with zero knowledge of the environment and autonomously
explores by scanning, detecting frontiers, and navigating to them with A*.

Controls
--------
  G          — start / pause exploration
  R          — reset to a new random start
  T          — top-down camera
  O          — overview camera
  Esc        — quit

Visualisation
-------------
  white quads  = known free space
  dark quads   = known obstacles
  cyan quads   = frontier cells
  red sphere   = current frontier target
  green line   = planned A* path
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import click
import mujoco
import mujoco.viewer
import numpy as np

from stretch_mujoco.habitat_scene_gallery import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_SCENE_ID,
    load_habitat_scene_model,
    prepare_habitat_scene,
)
from stretch_mujoco.navigations import NavigationController
from stretch_mujoco.navigations.FBE import (
    FBEPlanner,
    FBEState,
    detect_frontier_cells,
)
from examples.navigation_viewer import (
    _prepare_scene_xml,
    _inject_waypoint_geoms,
    _remove_waypoint_geoms,
    _camera_presets,
    _apply_preset,
    PathFollower,
)


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------
def _inject_local_map_quads(
    xml_str: str,
    local_data: np.ndarray,
    grid_bounds: tuple[float, float, float, float],
    resolution: float,
) -> str:
    """Add small quads showing the local occupancy map."""
    x_min, x_max, y_min, y_max = grid_bounds
    root = ET.fromstring(xml_str)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return xml_str

    for geom in list(worldbody.findall("geom")):
        if (geom.get("name") or "").startswith("fbe_"):
            worldbody.remove(geom)

    step = max(1, local_data.shape[0] // 70)
    z = 0.015
    s = resolution * step / 2

    # Subsampling stride (at least 1)
    stride = max(1, step * step // 2)

    rows, cols = np.where(local_data >= 0)  # known cells
    for i in range(0, len(rows), stride):
        r, c = rows[i], cols[i]
        val = local_data[r, c]
        wx = x_min + (c + 0.5) * resolution
        wy = y_min + (r + 0.5) * resolution
        rgba = "1 1 1 0.35" if val == 0 else "0.15 0.15 0.15 0.5"
        ET.SubElement(
            worldbody,
            "geom",
            name=f"fbe_map_{r}_{c}",
            type="box",
            pos=f"{wx:.4f} {wy:.4f} {z}",
            size=f"{s:.4f} {s:.4f} 0.002",
            rgba=rgba,
            contype="0",
            conaffinity="0",
        )
    return ET.tostring(root, encoding="unicode")


def _inject_frontier_quads(
    xml_str: str,
    frontier_mask: np.ndarray,
    grid_bounds: tuple[float, float, float, float],
    resolution: float,
) -> str:
    """Highlight frontier cells with cyan quads."""
    x_min, x_max, y_min, y_max = grid_bounds
    root = ET.fromstring(xml_str)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return xml_str

    for geom in list(worldbody.findall("geom")):
        if (geom.get("name") or "").startswith("fbe_front_"):
            worldbody.remove(geom)

    step = max(1, frontier_mask.shape[0] // 70)
    z = 0.022
    s = resolution * step / 2 * 1.1
    stride = max(1, step * step // 2)
    rows, cols = np.where(frontier_mask)
    for i in range(0, len(rows), stride):
        r, c = rows[i], cols[i]
        wx = x_min + (c + 0.5) * resolution
        wy = y_min + (r + 0.5) * resolution
        ET.SubElement(
            worldbody,
            "geom",
            name=f"fbe_front_{r}_{c}",
            type="box",
            pos=f"{wx:.4f} {wy:.4f} {z}",
            size=f"{s:.4f} {s:.4f} 0.003",
            rgba="0 1 1 0.7",
            contype="0",
            conaffinity="0",
        )
    return ET.tostring(root, encoding="unicode")


def _add_user_geom(user_scene, geom_type, size, pos, rgba) -> bool:
    """Append one transient viewer geom, returning False when capacity is full."""
    if user_scene.ngeom >= len(user_scene.geoms):
        return False
    mujoco.mjv_initGeom(
        user_scene.geoms[user_scene.ngeom],
        type=geom_type,
        size=np.asarray(size, dtype=float),
        pos=np.asarray(pos, dtype=float),
        mat=np.eye(3).reshape(-1),
        rgba=np.asarray(rgba, dtype=float),
    )
    user_scene.ngeom += 1
    return True


def _update_user_scene(user_scene, fbe, goal_xy, resolution) -> None:
    """Refresh map, frontier, path, and target overlays without rebuilding MuJoCo."""
    user_scene.ngeom = 0
    capacity = len(user_scene.geoms)
    if capacity == 0:
        return

    known_cells = np.argwhere(fbe.local_map.data >= 0)
    known_budget = max(1, min(500, capacity // 2))
    known_stride = max(1, int(np.ceil(len(known_cells) / known_budget)))
    half_size = resolution * 0.48
    for row, col in known_cells[::known_stride]:
        value = fbe.local_map.data[row, col]
        x, y = fbe.local_map.cell_to_world((int(row), int(col)))
        rgba = (1.0, 1.0, 1.0, 0.25) if value == 0 else (0.1, 0.1, 0.1, 0.55)
        if not _add_user_geom(
            user_scene,
            mujoco.mjtGeom.mjGEOM_BOX,
            (half_size, half_size, 0.002),
            (x, y, 0.015),
            rgba,
        ):
            return

    frontier_cells = np.argwhere(detect_frontier_cells(fbe.local_map))
    frontier_budget = max(1, min(250, capacity - user_scene.ngeom - 32))
    frontier_stride = max(1, int(np.ceil(len(frontier_cells) / frontier_budget)))
    for row, col in frontier_cells[::frontier_stride]:
        x, y = fbe.local_map.cell_to_world((int(row), int(col)))
        if not _add_user_geom(
            user_scene,
            mujoco.mjtGeom.mjGEOM_BOX,
            (half_size, half_size, 0.003),
            (x, y, 0.022),
            (0.0, 1.0, 1.0, 0.7),
        ):
            return

    for waypoint in fbe.diag.current_path:
        if not _add_user_geom(
            user_scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            (0.07, 0.0, 0.0),
            (waypoint[0], waypoint[1], 0.08),
            (0.0, 1.0, 0.0, 0.9),
        ):
            return

    if fbe._target_cluster is not None:
        target = fbe.local_map.cell_to_world(fbe._target_cluster.centroid_cell)
        _add_user_geom(
            user_scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            (0.16, 0.0, 0.0),
            (target[0], target[1], 0.14),
            (1.0, 0.0, 0.0, 0.9),
        )

    if goal_xy is not None:
        _add_user_geom(
            user_scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            (0.14, 0.0, 0.0),
            (goal_xy[0], goal_xy[1], 0.12),
            (0.0, 1.0, 0.0, 0.85),
        )


# ---------------------------------------------------------------------------
@click.command()
@click.option("--scene-id", default=DEFAULT_SCENE_ID, show_default=True)
@click.option(
    "--cache-root",
    type=click.Path(path_type=Path),
    default=DEFAULT_CACHE_ROOT,
    show_default=True,
)
@click.option("--resolution", default=0.25, show_default=True, help="Grid resolution (m/cell).")
@click.option("--agent-radius", default=0.3, show_default=True)
@click.option("--max-range", default=5.0, show_default=True, help="Laser max range (m).")
def main(scene_id, cache_root, resolution, agent_radius, max_range):
    """FBE exploration in the MuJoCo viewer."""
    # --- Scene & god grid ---
    scene = prepare_habitat_scene(scene_id=scene_id, cache_root=cache_root)
    base_xml = Path(scene.xml_path).read_text(encoding="utf-8")
    base_xml = _remove_waypoint_geoms(base_xml)
    bounds = scene.bounds

    mx, Mx = float(bounds[0, 0] + 0.8), float(bounds[1, 0] - 0.8)
    my, My = float(bounds[0, 1] + 0.8), float(bounds[1, 1] - 0.8)
    presets = _camera_presets(bounds)
    gbounds = (mx, Mx, my, My)

    model_g = load_habitat_scene_model(scene)
    data_g = mujoco.MjData(model_g)
    mujoco.mj_step(model_g, data_g)
    nav = NavigationController(
        model_g,
        data_g,
        bounds=(mx, Mx, my, My),
        resolution=resolution,
        agent_radius=agent_radius,
        minimum_obstacle_height=0.1,
        maximum_obstacle_height=3.0,
        require_collision=False,
        exclude_prefixes=("habitat_stage_",),
    )
    god_grid = nav.grid

    # --- Find start/goal pair (same logic as A* demo) ---
    rng = np.random.default_rng(42)
    free_pts = [
        (float(x), float(y))
        for x in np.linspace(mx + 0.5, Mx - 0.5, 10)
        for y in np.linspace(my + 0.5, My - 0.5, 10)
        if god_grid.is_world_free(np.array([x, y]))
    ]
    rng.shuffle(free_pts)

    start_pt, goal_pt = None, None
    for s in free_pts[:20]:
        for g in free_pts[1:]:
            if g == s:
                continue
            d = float(np.linalg.norm(np.array(s) - np.array(g)))
            if 3 < d < 15:
                try:
                    # Verify A* can find a path on god grid
                    nav.plan(start=s, goal=g)
                    start_pt, goal_pt = s, g
                    break
                except Exception:
                    continue
        if start_pt is not None:
            break

    if start_pt is None:
        start_pt = free_pts[0]
        goal_pt = free_pts[-1]

    # --- FBE ---
    fbe = FBEPlanner(
        god_grid,
        robot_xy=start_pt,
        robot_yaw=0.0,
        num_rays=180,
        max_range_m=max_range,
        fov_degrees=270,
        min_cluster_size=3,
        explore_threshold=0.85,
    )

    # --- Initial model with robot ---
    def _build_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
        xml = base_xml
        # Robot
        rx, ry = float(fbe._robot_xy[0]), float(fbe._robot_xy[1])
        xml = _prepare_scene_xml(xml, with_robot=True, robot_start=(rx, ry))
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        # Place robot
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if body_id >= 0:
            for j in range(m.njnt):
                if m.jnt_bodyid[j] == body_id:
                    qadr = m.jnt_qposadr[j]
                    m.qpos0[qadr : qadr + 3] = [rx, ry, 0.0]
                    m.qpos0[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
                    d.qpos[qadr : qadr + 3] = [rx, ry, 0.0]
                    break
        return m, d

    model, data = _build_scene()
    follower = PathFollower([np.array(start_pt)], speed=1.0)

    # --- State ---
    state: dict[str, Any] = {
        "pending_preset": "top",
        "run": False,
        "overlay_timer": 0,
        "path_signature": None,
    }

    def key_callback(keycode: int) -> None:
        nonlocal follower, fbe, start_pt, goal_pt
        if keycode == ord("G"):
            if fbe.is_finished():
                state["run"] = False
                print("[FBE] Exploration is finished; press R to reset before running again.")
                return
            state["run"] = not state["run"]
            print(
                f"[FBE] {'▶ Running' if state['run'] else '⏸ Paused'}  "
                f"explored={fbe.local_map.explored_ratio():.0%}"
            )
        elif keycode == ord("R"):
            # Pick new start/goal pair
            rng2 = np.random.default_rng()
            rng2.shuffle(free_pts)
            start_pt, goal_pt = None, None
            for s2 in free_pts[:20]:
                for g2 in free_pts[1:]:
                    if g2 == s2:
                        continue
                    d2 = float(np.linalg.norm(np.array(s2) - np.array(g2)))
                    if 3 < d2 < 15:
                        try:
                            nav.plan(start=s2, goal=g2)
                            start_pt, goal_pt = s2, g2
                            break
                        except Exception:
                            continue
                if start_pt is not None:
                    break
            if start_pt is None:
                start_pt, goal_pt = free_pts[0], free_pts[-1]

            fbe = FBEPlanner(
                god_grid,
                robot_xy=start_pt,
                robot_yaw=0.0,
                num_rays=180,
                max_range_m=max_range,
                fov_degrees=270,
                min_cluster_size=3,
                explore_threshold=0.85,
            )
            follower = PathFollower([np.array(start_pt)], speed=1.0)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
            if body_id >= 0:
                for joint_id in range(model.njnt):
                    if model.jnt_bodyid[joint_id] == body_id:
                        qadr = model.jnt_qposadr[joint_id]
                        data.qpos[qadr : qadr + 3] = [start_pt[0], start_pt[1], 0.0]
                        data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
                        data.qvel[model.jnt_dofadr[joint_id] : model.jnt_dofadr[joint_id] + 6] = 0
                        mujoco.mj_forward(model, data)
                        break
            state["run"] = False
            state["overlay_timer"] = 0
            state["path_signature"] = None
            print(
                f"[FBE] Reset — start=({start_pt[0]:.1f},{start_pt[1]:.1f})  "
                f"goal=({goal_pt[0]:.1f},{goal_pt[1]:.1f})"
            )
        elif keycode == ord("T"):
            state["pending_preset"] = "top"
        elif keycode == ord("O"):
            state["pending_preset"] = "overview"

    print(f"\n{'='*60}")
    print(f"  FBE Exploration — Habitat Scene {scene_id}")
    print(f"{'='*60}")
    print(
        f"Grid: {god_grid.occupancy.shape}  "
        f"|  God-free: {100*(~god_grid.occupancy).sum()/god_grid.occupancy.size:.0f}%"
    )
    print(
        f"Start: ({start_pt[0]:.1f}, {start_pt[1]:.1f})  "
        f"→  Goal: ({goal_pt[0]:.1f}, {goal_pt[1]:.1f})"
    )
    print(f"Laser: {fbe.num_rays} rays × {max_range}m × {fbe.fov_degrees}°")
    print(f"Controls: G=start/pause  R=reset  T=top  O=overview  Esc=quit\n")

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        _apply_preset(viewer, presets["top"])

        while viewer.is_running():
            preset_name = state.pop("pending_preset", None)
            if preset_name is not None:
                _apply_preset(viewer, presets[preset_name])

            if state["run"] and not fbe.is_finished():
                # Get robot pose from MuJoCo
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
                if body_id >= 0:
                    xpos = data.xpos[body_id]
                    xmat = data.xmat[body_id].reshape(3, 3)
                    rx, ry = float(xpos[0]), float(xpos[1])
                    ryaw = float(np.arctan2(xmat[1, 0], xmat[0, 0]))

                    # FBE step
                    fbe.step((rx, ry), ryaw)

                    # Update path follower
                    if fbe.has_path() and fbe.diag.current_path:
                        signature = tuple(
                            tuple(np.round(waypoint, 4)) for waypoint in fbe.diag.current_path
                        )
                        if signature != state["path_signature"]:
                            follower.reset(fbe.diag.current_path)
                            state["path_signature"] = signature
                    elif fbe.state == FBEState.FINISHED:
                        state["run"] = False
                        print(
                            f"[FBE] ✅ Exploration finished!  "
                            f"Explored: {fbe.local_map.explored_ratio():.1%}  "
                            f"Reason: {fbe.diag.finish_reason}"
                        )
                    else:
                        state["path_signature"] = None

                    # Move robot along path
                    if not follower.reached_goal:
                        follower.update(model, data)

                state["overlay_timer"] += 1
                if state["overlay_timer"] >= 10:
                    state["overlay_timer"] = 0
                    with viewer.lock():
                        _update_user_scene(viewer.user_scn, fbe, goal_pt, resolution)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
