#!/usr/bin/env python3
"""Real-time navigation visualisation in the MuJoCo viewer.

Loads a Habitat scene, plans a path, and opens the interactive viewer with
coloured sphere markers at the waypoints.  Optionally includes the Stretch
robot placed at the path start.

Controls
--------
  1 / 2      — switch between A* and FMM
  R          — re-plan with random start/goal
  T          — top-down camera
  O          — overview camera
  Mouse drag — rotate / pan
  Scroll     — zoom
  Esc        — quit

Usage
-----
  python examples/navigation_viewer.py
  python examples/navigation_viewer.py --with-robot
"""

from __future__ import annotations

import re
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
from stretch_mujoco.navigations import (
    Algorithm,
    NavigationController,
    NavigationPathError,
)
from stretch_mujoco.utils import get_absolute_path_stretch_xml, models_path


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def _prepare_scene_xml(
    base_xml: str,
    *,
    with_robot: bool = False,
    robot_start: tuple[float, float] | None = None,
) -> str:
    """Return a modified scene XML string.

    *with_robot* inserts gravity, a collision floor, and the Stretch robot.
    *robot_start* is the (x, y) base position for the robot.
    """
    xml = base_xml

    # Fix gravity (Habitat scenes ship with gravity="0 0 0")
    xml = re.sub(
        r'gravity="[^"]*"',
        'gravity="0 0 -9.81"',
        xml,
        count=1,
    )

    if with_robot:
        # Generate stretch XML with absolute mesh paths
        stretch_xml_path = get_absolute_path_stretch_xml()
        include_line = f'<include file="{stretch_xml_path}"/>'

        # Insert after the opening <mujoco ...> tag
        xml = re.sub(
            r'(<mujoco\s[^>]*>)',
            f"\\1\n    {include_line}",
            xml,
            count=1,
        )

        # Add a collision floor so the robot has something to drive on.
        # Must be at z=0 — the robot is designed to stand on a floor at z=0.
        floor_xml = (
            '<geom name="nav_collision_floor" type="plane"'
            ' pos="0 0 0" size="0 0 0.05"'
            ' friction="2.0 0.01 0.0001"'
            ' rgba="0.3 0.3 0.3 0.2"/>'
        )
        xml = re.sub(
            r"(<worldbody>)",
            f"\\1\n    {floor_xml}",
            xml,
            count=1,
        )

    return xml


def _inject_waypoint_geoms(xml_str: str, waypoints: list[np.ndarray]) -> str:
    """Insert small sphere geoms for each waypoint into the MuJoCo XML."""
    root = ET.fromstring(xml_str)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return xml_str

    for i, wp in enumerate(waypoints):
        if i == 0:
            rgba = "0.9 0.2 0.2 1.0"
            radius = "0.08"
        elif i == len(waypoints) - 1:
            rgba = "0.2 0.9 0.2 1.0"
            radius = "0.08"
        else:
            t = i / max(len(waypoints) - 1, 1)
            r, g, b = t, 0.4, 1.0 - t
            rgba = f"{r:.3f} {g:.3f} {b:.3f} 0.85"
            radius = "0.04"
        ET.SubElement(
            worldbody,
            "geom",
            name=f"nav_wp_{i:03d}",
            type="sphere",
            pos=f"{wp[0]:.4f} {wp[1]:.4f} 0.06",
            size=radius,
            rgba=rgba,
            contype="0",
            conaffinity="0",
        )
    return ET.tostring(root, encoding="unicode")


def _remove_waypoint_geoms(xml_str: str) -> str:
    """Strip previously injected navigation waypoint geoms."""
    root = ET.fromstring(xml_str)
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for geom in list(worldbody.findall("geom")):
            if (geom.get("name") or "").startswith("nav_wp_"):
                worldbody.remove(geom)
    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------------------
# Path-following controller (teleport — sets qpos directly)
# ---------------------------------------------------------------------------


class PathFollower:
    """Teleport the robot along a planned path by directly setting qpos.

    The Habitat scene is visual-only (no collision friction), so we bypass
    the physics and move the base_link free-joint position/orientation
    directly.  This gives smooth, reliable visualisation of the robot
    following the planned path.
    """

    def __init__(
        self,
        waypoints: list[np.ndarray],
        *,
        speed: float = 1.2,            # metres / second
        waypoint_tolerance: float = 0.15,
    ) -> None:
        self.waypoints = [np.asarray(wp, dtype=float)[:2] for wp in waypoints]
        self._wp_idx = 0
        self.speed = speed
        self.waypoint_tolerance = waypoint_tolerance
        self._reached = False
        # Interpolation state
        self._prev_wp: np.ndarray | None = None
        self._progress: float = 0.0     # 0..1 between prev and current target

    @property
    def reached_goal(self) -> bool:
        return self._reached

    @property
    def current_target(self) -> np.ndarray:
        return self.waypoints[min(self._wp_idx, len(self.waypoints) - 1)]

    def reset(self, waypoints: list[np.ndarray]) -> None:
        self.waypoints = [np.asarray(wp, dtype=float)[:2] for wp in waypoints]
        self._wp_idx = 0
        self._reached = False
        self._prev_wp = None
        self._progress = 0.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Advance the robot position for one viewer frame."""
        if self._reached:
            return

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if body_id < 0:
            self._reached = True
            return

        # Find the free joint for base_link
        joint_id = -1
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == body_id:
                joint_id = j
                break
        if joint_id < 0:
            self._reached = True
            return

        qadr = model.jnt_qposadr[joint_id]
        dof_adr = model.jnt_dofadr[joint_id]

        # Current position
        cur = np.array([data.qpos[qadr], data.qpos[qadr + 1]], dtype=float)
        target = self.current_target

        # Segment length for interpolation speed
        if self._prev_wp is None:
            self._prev_wp = cur.copy()

        seg_len = float(np.linalg.norm(target - self._prev_wp))
        if seg_len < 1e-6:
            self._advance_waypoint()
            return

        # Advance progress (use frame-based step for smoother motion at
        # any viewer frame rate; assume ~60 fps)
        frame_step = self.speed / 60.0
        self._progress += frame_step / max(seg_len, 0.01)

        if self._progress >= 1.0:
            # Snap to target and advance
            data.qpos[qadr] = target[0]
            data.qpos[qadr + 1] = target[1]
            self._orient_toward_next(model, data, qadr)
            self._prev_wp = target.copy()
            self._progress = 0.0
            self._advance_waypoint()
        else:
            # Interpolate
            new_pos = self._prev_wp + self._progress * (target - self._prev_wp)
            data.qpos[qadr] = new_pos[0]
            data.qpos[qadr + 1] = new_pos[1]
            angle = np.arctan2(target[1] - new_pos[1], target[0] - new_pos[0])
            self._set_yaw(data, qadr, angle)

        # Set Z to 0 and zero velocity
        data.qpos[qadr + 2] = 0.0
        data.qvel[dof_adr : dof_adr + 6] = 0.0

    def _advance_waypoint(self) -> None:
        self._wp_idx += 1
        if self._wp_idx >= len(self.waypoints):
            self._reached = True

    def _orient_toward_next(
        self, model: mujoco.MjModel, data: mujoco.MjData, qadr: int
    ) -> None:
        """Face the robot toward the next waypoint, or keep final heading."""
        if self._wp_idx + 1 < len(self.waypoints):
            nxt = self.waypoints[self._wp_idx + 1]
            cur = self.waypoints[self._wp_idx]
        elif self._wp_idx > 0:
            nxt = self.waypoints[self._wp_idx]
            cur = self.waypoints[self._wp_idx - 1]
        else:
            return
        angle = np.arctan2(nxt[1] - cur[1], nxt[0] - cur[0])
        self._set_yaw(data, qadr, angle)

    @staticmethod
    def _set_yaw(data: mujoco.MjData, qadr: int, yaw: float) -> None:
        """Set the quaternion in qpos to represent a yaw rotation."""
        half = yaw / 2.0
        data.qpos[qadr + 3] = np.cos(half)
        data.qpos[qadr + 4] = 0.0
        data.qpos[qadr + 5] = 0.0
        data.qpos[qadr + 6] = np.sin(half)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def _camera_presets(bounds: np.ndarray) -> dict[str, dict[str, Any]]:
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    distance = max(float(extent[0]), float(extent[1])) * 1.2
    lookat = np.array((center[0], center[1], max(0.8, center[2])), dtype=float)
    return {
        "overview": {"lookat": lookat, "distance": distance, "azimuth": 135, "elevation": -32},
        "top": {"lookat": lookat, "distance": distance * 1.05, "azimuth": 90, "elevation": -89},
    }


def _apply_preset(viewer, preset: dict[str, Any]) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = preset["lookat"]
    viewer.cam.distance = preset["distance"]
    viewer.cam.azimuth = preset["azimuth"]
    viewer.cam.elevation = preset["elevation"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--scene-id", default=DEFAULT_SCENE_ID, show_default=True)
@click.option("--cache-root", type=click.Path(path_type=Path),
              default=DEFAULT_CACHE_ROOT, show_default=True)
@click.option("--resolution", default=0.2, show_default=True)
@click.option("--agent-radius", default=0.3, show_default=True)
@click.option("--algorithm", "algo", type=click.Choice(["astar", "fmm"]),
              default="astar", show_default=True)
@click.option("--with-robot", is_flag=True,
              help="Include the Stretch robot in the scene.")
def main(scene_id, cache_root, resolution, agent_radius, algo, with_robot):
    """Interactive navigation viewer for Habitat scenes."""
    # --- Prepare scene ---
    scene = prepare_habitat_scene(scene_id=scene_id, cache_root=cache_root)
    base_xml = Path(scene.xml_path).read_text(encoding="utf-8")
    base_xml = _remove_waypoint_geoms(base_xml)
    bounds = scene.bounds
    presets = _camera_presets(bounds)

    mx, Mx = float(bounds[0, 0] + 0.8), float(bounds[1, 0] - 0.8)
    my, My = float(bounds[0, 1] + 0.8), float(bounds[1, 1] - 0.8)

    # --- Build navigation controller ---
    def _make_nav(algorithm: Algorithm) -> NavigationController:
        model = load_habitat_scene_model(scene)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        return NavigationController(
            model, data,
            algorithm=algorithm,
            bounds=(mx, Mx, my, My),
            resolution=resolution,
            agent_radius=agent_radius,
            minimum_obstacle_height=0.1,
            maximum_obstacle_height=3.0,
            require_collision=False,
            exclude_prefixes=("habitat_stage_",),
        )

    algorithm = Algorithm(algo)
    nav = _make_nav(algorithm)

    # --- Find initial free pair ---
    rng = np.random.default_rng(42)

    def _random_free_pair(nav: NavigationController):
        free = []
        for x in np.linspace(mx + 0.5, Mx - 0.5, 10):
            for y in np.linspace(my + 0.5, My - 0.5, 10):
                if nav.is_free((x, y)):
                    free.append((float(x), float(y)))
        rng.shuffle(free)
        for s in free[:20]:
            for g in free[1:]:
                if g == s:
                    continue
                d = float(np.linalg.norm(np.array(s) - np.array(g)))
                if 3 < d < 15:
                    try:
                        nav.plan(start=s, goal=g)
                        return s, g
                    except NavigationPathError:
                        continue
        return free[0], free[-1]

    start_pt, goal_pt = _random_free_pair(nav)
    waypoints = nav.plan(start=start_pt, goal=goal_pt)

    # --- Build XML with robot + markers ---
    xml_str = _prepare_scene_xml(
        base_xml,
        with_robot=with_robot,
        robot_start=start_pt,
    )
    xml_str = _inject_waypoint_geoms(xml_str, waypoints)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    # Place robot at start if requested
    if with_robot:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if body_id >= 0:
            joint_id = -1
            for j in range(model.njnt):
                if model.jnt_bodyid[j] == body_id:
                    joint_id = j
                    break
            if joint_id >= 0:
                qadr = model.jnt_qposadr[joint_id]
                # Set initial position: x,y on floor at z=0
                model.qpos0[qadr : qadr + 3] = [start_pt[0], start_pt[1], 0.0]
                model.qpos0[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
                data = mujoco.MjData(model)
                data.qpos[qadr : qadr + 3] = [start_pt[0], start_pt[1], 0.0]

    print(f"Scene: {scene_id}  |  Algorithm: {algorithm.value.upper()}"
          f"{'  |  Robot: yes' if with_robot else ''}")
    print(f"Grid: {nav.grid.occupancy.shape}  "
          f"|  Free: {(~nav.grid.occupancy).sum()}/{nav.grid.occupancy.size}")
    pts = np.array(waypoints)
    dist = sum(float(np.linalg.norm(pts[i+1]-pts[i])) for i in range(len(pts)-1))
    print(f"Path: {len(waypoints)} waypoints, {dist:.2f} m")
    print(f"Start: ({start_pt[0]:.2f}, {start_pt[1]:.2f})  "
          f"Goal: ({goal_pt[0]:.2f}, {goal_pt[1]:.2f})")
    ctrl_str = "  G=go/stop" if with_robot else ""
    print(f"\nControls: 1=A*  2=FMM  R=replan  T=top  O=overview{ctrl_str}  Esc=quit\n")

    # --- Path follower ---
    follower = PathFollower(waypoints) if with_robot else None

    # --- Viewer state ---
    state: dict[str, Any] = {
        "pending_preset": "top",
        "rebuild": False,
        "algorithm": algorithm,
        "run_robot": with_robot,  # auto-start
    }

    def _rebuild_model() -> None:
        nonlocal model, data, nav, waypoints, start_pt, goal_pt
        s, g = _random_free_pair(nav)
        try:
            new_wp = nav.plan(start=s, goal=g)
        except NavigationPathError:
            return
        start_pt, goal_pt = s, g
        waypoints = new_wp
        xml = _prepare_scene_xml(base_xml, with_robot=with_robot, robot_start=s)
        xml = _inject_waypoint_geoms(xml, waypoints)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        if with_robot:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
            if body_id >= 0:
                for j in range(model.njnt):
                    if model.jnt_bodyid[j] == body_id:
                        qadr = model.jnt_qposadr[j]
                        model.qpos0[qadr : qadr + 3] = [s[0], s[1], 0.0]
                        model.qpos0[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
                        data.qpos[qadr : qadr + 3] = [s[0], s[1], 0.0]
                        break
        pts = np.array(waypoints)
        d = sum(float(np.linalg.norm(pts[i+1]-pts[i])) for i in range(len(pts)-1))
        print(f"[{nav.algorithm.value.upper()}] {len(waypoints)} wp, {d:.2f} m  "
              f"({s[0]:.2f},{s[1]:.2f}) → ({g[0]:.2f},{g[1]:.2f})")

    def key_callback(keycode: int) -> None:
        nonlocal follower
        if keycode == ord("1"):
            state["algorithm"] = Algorithm.ASTAR
            state["rebuild"] = True
        elif keycode == ord("2"):
            state["algorithm"] = Algorithm.FMM
            state["rebuild"] = True
        elif keycode == ord("R"):
            state["rebuild"] = True
        elif keycode == ord("G"):
            if with_robot and follower is not None:
                state["run_robot"] = not state["run_robot"]
                if state["run_robot"]:
                    follower.reset(waypoints)
                    print("[Robot] Moving!")
                else:
                    print("[Robot] Stopped.")
        elif keycode == ord("T"):
            state["pending_preset"] = "top"
        elif keycode == ord("O"):
            state["pending_preset"] = "overview"

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        while viewer.is_running():
            preset_name = state.pop("pending_preset", None)
            if preset_name is not None:
                _apply_preset(viewer, presets[preset_name])

            if state.pop("rebuild", False):
                algo = state["algorithm"]
                if algo != nav.algorithm:
                    nav = _make_nav(algo)
                _rebuild_model()
                # Reset follower with new waypoints
                if with_robot:
                    follower = PathFollower(waypoints)
                    state["run_robot"] = True  # auto-start
                viewer.sync()
                _apply_preset(viewer, presets["top"])

            # Path-following
            if with_robot and follower is not None and state["run_robot"]:
                follower.update(model, data)
                if follower.reached_goal:
                    state["run_robot"] = False
                    print("[Robot] Goal reached!")

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
