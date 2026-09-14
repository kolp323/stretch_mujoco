#!/usr/bin/env python3
"""Run GraspGen-guided, physics-based grasps from successful office nav poses.

Each attempt observes with D435i, optionally refines with D405, closes the
gripper, then lifts in place. Success requires bilateral physical contact and
sufficient object lift; objects are never kinematically attached.

Example::

  MUJOCO_GL=egl python tools/test_scene/run_office_grasp_sim.py \
      --scene 1 --output-dir output/office_grasp_sim
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.grasp_task import GRASP_FRICTION
from stretch_mujoco.graspgen import (
    RGBDGraspClient,
    StretchGraspIK,
    d435i_rotated_camera_intrinsics,
    rotated_d435i_optical_pose,
)
from stretch_mujoco.graspgen.calibration import world_to_base_pose

try:
    from test_office_navigation import _is_descendant_of, _scene_paths
except ImportError:
    from tools.test_scene.test_office_navigation import _is_descendant_of, _scene_paths


NAV_LIFT_CLEARANCE = 1.1
SAFE_PIVOT_DISTANCE_M = 0.65
WRIST_REFINE_STANDOFF_M = 0.18
MOTION_SPEED = 1.0
GRIPPER_OPEN = 0.56
GRIPPER_CLOSE = -0.15
MIN_LIFT_M = 0.15
CAMERA_HZ = 10.0
CONTINUOUS_DEBUG_HZ = 5.0
# Keep the side-on grasp wrist flat; live alignment corrects resulting drift.
FLAT_WRIST_PITCH = 0.0
FLAT_WRIST_ROLL = 0.0
# Fixed Cartesian pre-close corrections.
POST_ALIGN_PUSH_M = 0.05
WRIST_YAW_LEVER_ARM_M = 0.26
PREGRASP_LATERAL_SHIFT_M = 0.015
PUSH_IN_DOWNWARD_SHIFT_M = 0.03
# Reject self-occluded/off-object GraspGen candidates.
MAX_GRASP_POINT_OBJECT_DISTANCE_M = 0.2


def _default_prompt(asset: dict, object_id: str) -> str:
    """Turn generated ids such as ``001_bottle_022`` into a useful prompt."""
    name = str(asset.get("asset_id") or asset.get("name") or object_id)
    if name.endswith("_" + object_id.rsplit("_", 1)[-1]):
        name = name.rsplit("_", 1)[0]
    if name[:4].isdigit() and name[3] == "_":
        name = name[4:]
    return name.replace("-", " ").replace("_", " ").strip()


def _project_target(sim: StretchMujocoSimulator, target_world: np.ndarray,
                    camera_k: np.ndarray | None = None) -> tuple[float, float, float]:
    """Return target pixel (u, v) and camera-frame depth after image rotation."""
    pose = rotated_d435i_optical_pose(sim.get_link_pose("camera_color_optical_frame"))
    point = (np.linalg.inv(pose) @ np.r_[np.asarray(target_world, dtype=float), 1.0])[:3]
    camera_k = d435i_rotated_camera_intrinsics() if camera_k is None else camera_k
    fx, fy, cx, cy = camera_k[0, 0], camera_k[1, 1], camera_k[0, 2], camera_k[1, 2]
    if point[2] <= 1e-5:
        return float("nan"), float("nan"), float(point[2])
    return float(fx * point[0] / point[2] + cx), float(fy * point[1] / point[2] + cy), float(point[2])


def _debug_image(rgb: np.ndarray, depth: np.ndarray, *, u: float, v: float,
                 stage: str, detail: str = "") -> np.ndarray:
    rgb = np.asarray(rgb).copy()
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.05)
    depth_vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        scaled = np.clip((depth - 0.05) / 2.95 * 255.0, 0, 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
        depth_vis[~valid] = 0
    canvas = np.concatenate([rgb, depth_vis], axis=1)
    if np.isfinite(u) and np.isfinite(v):
        cv2.drawMarker(canvas, (round(u), round(v)), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(canvas, f"{stage}  {detail[:100]}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _compose_debug_frame(sim: StretchMujocoSimulator, target: np.ndarray, stage: str,
                          camera_k: np.ndarray | None, detail: str) -> tuple[np.ndarray, dict[str, float]]:
    """Build a head RGB-D plus wrist RGB debug panel."""
    u, v, z = _project_target(sim, target, camera_k)
    cameras = sim.pull_camera_data()
    rgb = cameras.get_camera_data(StretchCameras.cam_d435i_rgb, auto_rotate=True,
                                  auto_correct_rgb=False)
    depth = cameras.get_camera_data(StretchCameras.cam_d435i_depth, auto_rotate=True)
    frame = _debug_image(rgb, depth, u=u, v=v, stage=stage,
                         detail=f"target=({u:.0f},{v:.0f}) z={z:.2f} {detail}")
    wrist_rgb = np.asarray(
        cameras.get_camera_data(StretchCameras.cam_d405_rgb, auto_rotate=True, auto_correct_rgb=False)
    )
    wrist_width = max(1, round(wrist_rgb.shape[1] * frame.shape[0] / wrist_rgb.shape[0]))
    wrist_panel = cv2.resize(wrist_rgb, (wrist_width, frame.shape[0]))
    frame = np.concatenate([frame, wrist_panel], axis=1)
    return frame, {"u": u, "v": v, "camera_z": z}


class _DebugRecorder:
    def __init__(self, directory: Path, enabled: bool) -> None:
        self.directory, self.enabled, self.frames = directory, enabled, []
        self.target = np.zeros(3)
        self.stage = "start"
        self._continuous_frames: list[np.ndarray] = []
        self._continuous_thread: threading.Thread | None = None
        self._continuous_stop = threading.Event()
        self._continuous_hz = CONTINUOUS_DEBUG_HZ
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)

    def capture(self, sim: StretchMujocoSimulator, target: np.ndarray, stage: str,
                camera_k: np.ndarray | None = None, detail: str = "") -> dict[str, float]:
        self.target, self.stage = np.asarray(target, dtype=float), stage
        if not self.enabled:
            u, v, z = _project_target(sim, target, camera_k)
            return {"u": u, "v": v, "camera_z": z}
        frame, projection = _compose_debug_frame(sim, target, stage, camera_k, detail)
        self.frames.append(frame)
        cv2.imwrite(str(self.directory / f"{len(self.frames):02d}_{stage.lower()}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return projection

    def start_continuous(self, sim: StretchMujocoSimulator, hz: float = CONTINUOUS_DEBUG_HZ) -> None:
        """Record the attempt continuously in addition to stage captures."""
        if not self.enabled or self._continuous_thread is not None:
            return
        self._continuous_hz = hz
        self._continuous_stop.clear()
        start_time = time.monotonic()

        def _loop() -> None:
            period = 1.0 / hz
            while not self._continuous_stop.is_set():
                loop_started = time.monotonic()
                try:
                    if sim.is_running():
                        frame, _ = _compose_debug_frame(
                            sim, self.target, self.stage, None,
                            f"t={time.monotonic() - start_time:.1f}s",
                        )
                        self._continuous_frames.append(frame)
                except Exception:
                    pass
                elapsed = time.monotonic() - loop_started
                self._continuous_stop.wait(max(0.0, period - elapsed))

        self._continuous_thread = threading.Thread(target=_loop, daemon=True)
        self._continuous_thread.start()

    def stop_continuous(self) -> None:
        if self._continuous_thread is None:
            return
        self._continuous_stop.set()
        self._continuous_thread.join(timeout=5.0)
        self._continuous_thread = None

    def close(self, name: str) -> None:
        self.stop_continuous()
        if self.enabled and self.frames:
            imageio.mimsave(self.directory / f"{name}.gif", self.frames, duration=0.7)
        if self.enabled and self._continuous_frames:
            imageio.mimsave(
                self.directory / f"{name}_full.gif",
                self._continuous_frames,
                duration=1.0 / self._continuous_hz,
            )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _find_asset(manifest: dict, object_id: str) -> dict:
    for asset in manifest.get("assets", []):
        if asset.get("object_id") == object_id:
            return asset
    raise ValueError(f"object_id {object_id!r} not found in manifest assets")


def _boost_friction(model: mujoco.MjModel, body_names: list[str], friction: float) -> int:
    """Raise subtree sliding friction without weakening higher-friction tips."""
    changed = 0
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        for geom_id in range(model.ngeom):
            if _is_descendant_of(model, int(model.geom_bodyid[geom_id]), body_id):
                model.geom_friction[geom_id, 0] = max(model.geom_friction[geom_id, 0], friction)
                changed += 1
    return changed


def _prepare_model(xml_path: Path, object_id: str, grasp_site: str) -> dict:
    """Load and validate the scene, then apply grasp friction."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
    if body_id < 0:
        raise ValueError(f"object body not found in XML: {object_id}")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, grasp_site)
    if site_id < 0:
        raise ValueError(f"grasp site not found in XML: {grasp_site}")

    jnt_adr = int(model.body_jntadr[body_id])
    jnt_num = int(model.body_jntnum[body_id])
    has_freejoint = jnt_num == 1 and jnt_adr >= 0 and int(model.jnt_type[jnt_adr]) == int(
        mujoco.mjtJoint.mjJNT_FREE
    )
    gravity_z = float(model.opt.gravity[2])
    subtree_mass = float(model.body_subtreemass[body_id])
    friction_geoms_patched = _boost_friction(
        model, [object_id, "rubber_tip_left", "rubber_tip_right"], GRASP_FRICTION
    )

    return {
        "model": model,
        "has_freejoint": has_freejoint,
        "gravity_z": gravity_z,
        "subtree_mass_kg": subtree_mass,
        "friction_geoms_patched": friction_geoms_patched,
    }


def _wait(
    sim: StretchMujocoSimulator, actuator: Actuators, timeout: float = 20.0, tolerance: float = 0.02
) -> bool:
    return sim.wait_until_at_setpoint(actuator, timeout=timeout, position_tolerance=tolerance)


def _set_flat_wrist(sim: StretchMujocoSimulator, yaw: float) -> None:
    """Set wrist yaw with the pitch and roll used for side-on grasps."""
    sim.move_to(Actuators.wrist_yaw, float(yaw))
    sim.move_to(Actuators.wrist_pitch, FLAT_WRIST_PITCH)
    sim.move_to(Actuators.wrist_roll, FLAT_WRIST_ROLL)
    for actuator in (Actuators.wrist_yaw, Actuators.wrist_pitch, Actuators.wrist_roll):
        _wait(sim, actuator, timeout=20.0)


def _rotate_to_yaw(
    sim: StretchMujocoSimulator, target_yaw: float, timeout: float = 12.0, tolerance: float = 0.03
) -> bool:
    """Rotate the base in place to face ``target_yaw``, mirroring the final
    in-place alignment controller in ``run_office_navigation_sim.py``."""
    started = time.monotonic()
    previous_w = 0.0
    while time.monotonic() - started < timeout:
        _, _, yaw = sim.get_base_pose()
        yaw_error = float(np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw)))
        if abs(yaw_error) <= tolerance and abs(previous_w) <= 0.05:
            sim.set_base_velocity(0.0, 0.0)
            return True
        desired_w = float(np.clip(2.2 * yaw_error, -0.8, 0.8))
        previous_w += float(np.clip(desired_w - previous_w, -0.04, 0.04))
        sim.set_base_velocity(0.0, previous_w)
        time.sleep(1.0 / 30.0)
    sim.set_base_velocity(0.0, 0.0)
    return False


def _aim_head_at(sim: StretchMujocoSimulator, target_world: np.ndarray) -> None:
    """Center a world point in the D435i image using finite differences."""
    target_world = np.asarray(target_world, dtype=float)
    for _ in range(4):
        u, v, z = _project_target(sim, target_world)
        status = sim.pull_status()
        if np.isfinite(u) and z > 0:
            pan0 = float(status.head_pan.pos)
            step = 0.06
            sim.move_to(Actuators.head_pan, float(np.clip(pan0 + step, -4.04, 1.73)))
            _wait(sim, Actuators.head_pan, timeout=4.0)
            u1, _, _ = _project_target(sim, target_world)
            derivative = (u1 - u) / step if np.isfinite(u1) else 0.0
            sim.move_to(Actuators.head_pan, float(np.clip(
                pan0 - (u - 120.0) / derivative if abs(derivative) > 1e-3 else pan0,
                -4.04, 1.73)))
            _wait(sim, Actuators.head_pan, timeout=6.0)

        u, v, z = _project_target(sim, target_world)
        status = sim.pull_status()
        if np.isfinite(v) and z > 0:
            tilt0 = float(status.head_tilt.pos)
            step = 0.06
            sim.move_to(Actuators.head_tilt, float(np.clip(tilt0 + step, -1.53, 0.79)))
            _wait(sim, Actuators.head_tilt, timeout=4.0)
            _, v1, _ = _project_target(sim, target_world)
            derivative = (v1 - v) / step if np.isfinite(v1) else 0.0
            sim.move_to(Actuators.head_tilt, float(np.clip(
                tilt0 - (v - 212.0) / derivative if abs(derivative) > 1e-3 else tilt0,
                -1.53, 0.79)))
            _wait(sim, Actuators.head_tilt, timeout=6.0)
        if np.isfinite(u) and np.isfinite(v) and abs(u - 120) < 18 and abs(v - 212) < 18:
            break


def _move_base_forward(
    sim: StretchMujocoSimulator, dx: float, speed: float = 0.08, timeout: float = 8.0
) -> None:
    """Drive ``dx`` metres along the current heading."""
    if abs(dx) < 0.005:
        return
    start_x, start_y, _ = sim.get_base_pose()
    sim.set_base_velocity(float(np.copysign(speed, dx)), 0.0)
    deadline = time.monotonic() + timeout
    traveled = 0.0
    while traveled < abs(dx) - 0.005 and time.monotonic() < deadline:
        time.sleep(0.02)
        x, y, _ = sim.get_base_pose()
        traveled = float(np.hypot(x - start_x, y - start_y))
    sim.set_base_velocity(0.0, 0.0)
    time.sleep(0.2)


def _align_ee(
    sim: StretchMujocoSimulator,
    target_world: np.ndarray,
    *,
    adjust_arm: bool,
    iterations: int = 3,
) -> None:
    """Closed-loop lift/wrist alignment, optionally including arm depth."""
    for _ in range(iterations):
        error_world = np.asarray(target_world, dtype=float) - sim.get_ee_pose()[:3, 3]
        base_yaw = float(sim.get_base_pose()[2])
        cosine, sine = np.cos(-base_yaw), np.sin(-base_yaw)
        error_x = cosine * error_world[0] - sine * error_world[1]
        error_y = sine * error_world[0] + cosine * error_world[1]
        error_z = error_world[2]
        status = sim.pull_status()
        sim.move_to(Actuators.lift, float(np.clip(status.lift.pos + error_z, 0.0, 1.1)))
        if adjust_arm:
            sim.move_to(Actuators.arm, float(np.clip(status.arm.pos - error_y, 0.0, 0.52)))
        sim.move_to(
            Actuators.wrist_yaw,
            float(
                np.clip(
                    status.wrist_yaw.pos
                    + np.clip(error_x / WRIST_YAW_LEVER_ARM_M, -0.04, 0.04),
                    -1.39,
                    4.42,
                )
            ),
        )
        _wait(sim, Actuators.lift, timeout=4.0)
        if adjust_arm:
            _wait(sim, Actuators.arm, timeout=4.0)
        _wait(sim, Actuators.wrist_yaw, timeout=4.0)
        time.sleep(0.2)


def _infer_world_grasp_poses(
    client: RGBDGraspClient,
    rgb: np.ndarray,
    depth: np.ndarray,
    camera_k: np.ndarray,
    camera_pose_world: np.ndarray,
    prompt: str,
    approach_camera: np.ndarray,
    *,
    max_approach_angle_deg: float = 45.0,
    min_depth: float = 0.05,
    max_depth: float = 3.0,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Infer GraspGen poses and transform them from camera to world frame."""
    # Empty/invalid depth frames can make the GraspGen server construct a
    # zero-length tensor and fail while reshaping it.  Treat that as a
    # per-object no-grasp result so the batch continues with other objects.
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    valid_depth = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    if rgb.size == 0 or depth.size == 0 or not valid_depth.any():
        return None, None, {"status": "no_valid_depth", "num_grasps": 0}
    try:
        response = client.infer(
            rgb,
            depth,
            camera_k,
            prompt,
            candidate_top_k=200,
            desired_approach_direction=tuple(approach_camera),
            max_approach_angle_deg=max_approach_angle_deg,
            approach_allow_opposite=True,
            min_depth=min_depth,
            max_depth=max_depth,
        )
    except Exception as error:
        return None, None, {
            "status": "server_error",
            "num_grasps": 0,
            "error": str(error),
        }
    if response.get("status") != "ok" or not response.get("num_grasps"):
        return None, None, response
    camera_tcp_poses = np.asarray(response["grasp_tcp_poses"], dtype=float)
    world_tcp_poses = np.asarray([camera_pose_world @ pose for pose in camera_tcp_poses])
    confidences = np.asarray(response["confidences"], dtype=float)
    return world_tcp_poses, confidences, response


def _select_reachable_candidate(
    grasp_ik: StretchGraspIK,
    world_tcp_poses: np.ndarray,
    confidences: np.ndarray,
    base_pose: tuple[float, float, float],
    sanity_point: np.ndarray,
    *,
    max_sanity_distance: float = MAX_GRASP_POINT_OBJECT_DISTANCE_M,
    max_wrist_yaw: float = 0.7,
):
    """Return the best reachable, on-object pose and its symmetry variant."""
    base_from_world = world_to_base_pose(base_pose)
    base_tcp_poses = np.asarray([base_from_world @ pose for pose in world_tcp_poses])
    symmetry = np.diag([-1.0, -1.0, 1.0, 1.0])
    candidates = grasp_ik.rank_candidates(
        np.asarray([variant for pose in base_tcp_poses for variant in (pose, pose @ symmetry)]),
        np.repeat(confidences, 2),
    )
    candidates = [c for c in candidates if abs(c.joints[3]) <= max_wrist_yaw]
    if not candidates:
        return None, None
    on_target = [
        c for c in candidates
        if np.linalg.norm(world_tcp_poses[c.index // 2][:3, 3] - sanity_point)
        <= max_sanity_distance
    ]
    if not on_target:
        best_off_target = float(min(
            np.linalg.norm(world_tcp_poses[c.index // 2][:3, 3] - sanity_point)
            for c in candidates
        ))
        return None, best_off_target
    return on_target[0], None


def _solve_standoff_pose(
    grasp_ik: StretchGraspIK,
    grasp_pose_world: np.ndarray,
    base_pose: tuple[float, float, float],
    final_joints: np.ndarray,
    standoff_m: float,
) -> np.ndarray | None:
    """Solve the shorter-arm of the two possible approach-axis standoffs."""
    approach_axis = grasp_pose_world[:3, 2]
    norm = float(np.linalg.norm(approach_axis))
    if norm < 1e-6:
        return None
    approach_axis = approach_axis / norm
    base_from_world = world_to_base_pose(base_pose)
    best_joints: np.ndarray | None = None
    for sign in (1.0, -1.0):
        pose = grasp_pose_world.copy()
        pose[:3, 3] = grasp_pose_world[:3, 3] + sign * approach_axis * standoff_m
        candidate_joints, position_error, _ = grasp_ik.solve(
            base_from_world @ pose, initial=final_joints
        )
        if position_error > 0.02 or candidate_joints[2] >= final_joints[2] - 1e-3:
            continue
        if best_joints is None or candidate_joints[2] < best_joints[2]:
            best_joints = candidate_joints
    return best_joints


def _shift_ee_world(
    sim: StretchMujocoSimulator, grasp_ik: StretchGraspIK, delta_world: np.ndarray,
) -> bool:
    """Move the end effector by an IK-validated world-frame displacement."""
    delta_world = np.asarray(delta_world, dtype=float)
    if not np.any(delta_world):
        return False
    ee_pose = np.asarray(sim.get_ee_pose(), dtype=float)
    desired_ee = ee_pose.copy()
    desired_ee[:3, 3] += delta_world
    base_pose = sim.get_base_pose()
    target_base = world_to_base_pose(base_pose) @ desired_ee
    status = sim.pull_status()
    current_joints = np.array([
        0.0,
        float(status.lift.pos),
        float(status.arm.pos),
        float(status.wrist_yaw.pos),
        float(status.wrist_pitch.pos),
        float(status.wrist_roll.pos),
    ])
    joints, position_error, _ = grasp_ik.solve(target_base, initial=current_joints)
    if position_error > 0.025:
        return False

    if abs(float(joints[0])) > 0.005:
        _move_base_forward(sim, float(joints[0]))
    sim.move_to(Actuators.arm, float(np.clip(joints[2], 0.0, 0.52)))
    sim.move_to(Actuators.lift, float(np.clip(joints[1], 0.0, 1.1)))
    sim.move_to(Actuators.wrist_yaw, float(np.clip(joints[3], -1.39, 4.42)))
    _wait(sim, Actuators.arm, timeout=6.0, tolerance=0.01)
    _wait(sim, Actuators.lift, timeout=6.0, tolerance=0.01)
    _wait(sim, Actuators.wrist_yaw, timeout=6.0, tolerance=0.01)
    time.sleep(0.2)
    return True


def _push_in_along_approach(
    sim: StretchMujocoSimulator,
    grasp_ik: StretchGraspIK,
    push_m: float,
) -> float:
    """Push toward the object along base-local -Y, expressed in world frame."""
    if push_m <= 0:
        return 0.0
    base_yaw = float(sim.get_base_pose()[2])
    approach = np.array([np.sin(base_yaw), -np.cos(base_yaw), 0.0])
    return push_m if _shift_ee_world(sim, grasp_ik, approach * push_m) else 0.0


def _wrist_refine_grasp(
    sim: StretchMujocoSimulator,
    grasp_ik: StretchGraspIK,
    args: argparse.Namespace,
    scene_id: str,
    object_id: str,
    prompt: str,
    fallback_grasp_point: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Optionally refine a head-camera grasp with the D405 wrist camera."""
    diag: dict[str, Any] = {"wrist_refine_used": False}
    cameras = sim.pull_camera_data()
    rgb = cameras.get_camera_data(
        StretchCameras.cam_d405_rgb, auto_rotate=True, auto_correct_rgb=False
    )
    depth = cameras.get_camera_data(StretchCameras.cam_d405_depth, auto_rotate=True)
    camera_k = np.asarray(cameras.cam_d405_K, dtype=np.float32)
    if rgb.shape[:2] != depth.shape or camera_k.shape != (3, 3):
        diag["wrist_refine_skip_reason"] = "camera_data_inconsistent"
        return None, None, diag

    camera_pose_world = sim.get_link_pose("gripper_camera_color_optical_frame")
    base_pose = sim.get_base_pose()
    base_from_world = world_to_base_pose(base_pose)
    approach_base = np.array([0.0, -1.0, 0.0], dtype=float)
    approach_camera = (base_from_world[:3, :3] @ camera_pose_world[:3, :3]).T @ approach_base

    with RGBDGraspClient(args.graspgen_host, args.graspgen_port) as client:
        world_tcp_poses, confidences, response = _infer_world_grasp_poses(
            client, rgb, depth, camera_k, camera_pose_world, prompt, approach_camera,
            max_depth=0.8,
        )
    (args.output_dir / f"{scene_id}_{object_id}_wrist_refine_response.json").write_text(
        json.dumps(_json_compatible(response), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if world_tcp_poses is None:
        diag["wrist_refine_skip_reason"] = f"graspgen_{response.get('status', 'no_grasps')}"
        return None, None, diag

    object_now = np.asarray(
        sim.pull_grasp_metrics().get("object_position", fallback_grasp_point), dtype=float
    )
    candidate, best_off_target = _select_reachable_candidate(
        grasp_ik, world_tcp_poses, confidences, base_pose, object_now,
    )
    if candidate is None:
        diag["wrist_refine_skip_reason"] = (
            "ik_unreachable" if best_off_target is None else "candidate_off_target"
        )
        if best_off_target is not None:
            diag["wrist_refine_best_off_target_m"] = best_off_target
        return None, None, diag

    source_index = candidate.index // 2
    refined_point = world_tcp_poses[source_index][:3, 3]
    diag.update(
        wrist_refine_used=True,
        wrist_refine_position_error_m=candidate.position_error,
        wrist_refine_orientation_error_rad=candidate.orientation_error,
        wrist_refine_shift_m=float(np.linalg.norm(refined_point - fallback_grasp_point)),
    )
    return candidate.joints, refined_point, diag


def run_object(
    xml_path: Path,
    manifest: dict,
    scene_id: str,
    report_path: Path,
    args: argparse.Namespace,
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    goal_site = report["goal_site"]
    object_id = goal_site.removesuffix("_grasp_site")
    result: dict[str, Any] = {
        "object_id": object_id,
        "grasp_site": goal_site,
        "nav_report": str(report_path),
        "final_pose": report.get("final_pose"),
    }

    if not report.get("reached", False):
        result.update(success=False, skip_reason="navigation_not_reached")
        return result

    asset = _find_asset(manifest, object_id)

    prepared = _prepare_model(xml_path, object_id, goal_site)
    result.update(
        gravity_z=prepared["gravity_z"],
        friction_geoms_patched=prepared["friction_geoms_patched"],
        subtree_mass_kg=prepared["subtree_mass_kg"],
        manifest_mass_kg=asset.get("mass_kg"),
    )
    if not prepared["has_freejoint"]:
        result.update(success=False, skip_reason="not_dynamic")
        return result
    if prepared["gravity_z"] >= 0:
        result.update(success=False, skip_reason="gravity_disabled")
        return result

    x, y, yaw0 = report["final_pose"]
    sim = StretchMujocoSimulator(
        model=prepared["model"],
        cameras_to_use=[
            StretchCameras.cam_d435i_rgb,
            StretchCameras.cam_d435i_depth,
            StretchCameras.cam_d405_rgb,
            StretchCameras.cam_d405_depth,
        ],
        camera_hz=CAMERA_HZ,
        start_translation=[float(x), float(y), 0.0],
        start_rotation_quat=[float(np.cos(yaw0 / 2.0)), 0.0, 0.0, float(np.sin(yaw0 / 2.0))],
    )
    debug = _DebugRecorder(
        (args.output_dir / "debug") / object_id,
        args.debug,
    )
    try:
        sim.start(headless=True, use_passive_viewer=False)
        try:
            sim.get_base_pose()
        except RuntimeError as error:
            raise RuntimeError(
                "MuJoCo physics process stopped during startup; inspect the chained "
                f"exception for the MJCF/runtime cause: {error}"
            ) from error
        sim.set_robot_motion_speed(MOTION_SPEED)
        debug.start_continuous(sim)

        # Observe before pivoting so the retracted wrist does not occlude the object.
        sim.move_to(Actuators.arm, 0.0)
        sim.move_to(Actuators.lift, NAV_LIFT_CLEARANCE)
        _wait(sim, Actuators.arm, timeout=20.0)
        _wait(sim, Actuators.lift, timeout=20.0)
        sim.request_grasp_metrics(object_id)
        time.sleep(0.3)
        initial_metrics = sim.pull_grasp_metrics()
        initial_object_position = np.asarray(initial_metrics["object_position"], dtype=float)
        debug.capture(sim, initial_object_position, "01_nav_pose", detail="arm retracted, lift high")

        obj_x, obj_y = report["requested_goal"]
        base_x0, base_y0, _ = sim.get_base_pose()
        bearing = float(np.arctan2(obj_y - base_y0, obj_x - base_x0))
        grasp_yaw = float(np.arctan2(np.sin(bearing + np.pi / 2), np.cos(bearing + np.pi / 2)))
        result.update(bearing_to_object=bearing, grasp_stance_yaw=grasp_yaw)
        grasp_ik = StretchGraspIK(sim.urdf_model)

        _aim_head_at(sim, initial_object_position)
        debug.capture(sim, initial_object_position, "02_camera_aim")

        sim.move_to(Actuators.gripper, GRIPPER_OPEN)
        open_started = time.monotonic()
        gripper_opening = float(sim.pull_status().gripper.pos)
        while gripper_opening < 0.50 and time.monotonic() - open_started < 30.0:
            time.sleep(0.1)
            gripper_opening = float(sim.pull_status().gripper.pos)
        result["pregrasp_gripper_opening"] = gripper_opening

        cameras = sim.pull_camera_data()
        rgb = cameras.get_camera_data(
            StretchCameras.cam_d435i_rgb, auto_rotate=True, auto_correct_rgb=False
        )
        depth = cameras.get_camera_data(StretchCameras.cam_d435i_depth, auto_rotate=True)
        camera_k = np.asarray(cameras.cam_d435i_K, dtype=np.float32)
        if rgb.shape[:2] != depth.shape or camera_k.shape != (3, 3):
            raise RuntimeError("D435i RGB, depth, and intrinsics are inconsistent")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        observation_path = args.output_dir / f"{scene_id}_{object_id}_observation.npz"
        np.savez_compressed(observation_path, rgb=rgb, depth=depth, camera_K=camera_k)
        result["observation_npz"] = str(observation_path)

        # The image is taken pre-pivot; use the future grasp stance for approach bias.
        raw_camera_pose = sim.get_link_pose("camera_color_optical_frame")
        world_camera_pose = rotated_d435i_optical_pose(raw_camera_pose)
        future_base_from_world = world_to_base_pose((base_x0, base_y0, grasp_yaw))
        approach_base = np.array([0.0, -1.0, 0.0], dtype=float)
        approach_camera = (future_base_from_world[:3, :3] @ world_camera_pose[:3, :3]).T @ approach_base
        prompt = _default_prompt(asset, object_id)
        projection = debug.capture(sim, initial_object_position, "03_observation",
                                   camera_k, detail=f"prompt={prompt}")
        result.update(graspgen_prompt=prompt, camera_target_projection=projection)
        with RGBDGraspClient(args.graspgen_host, args.graspgen_port) as client:
            health = client.health()
            if not health.get("ready"):
                raise RuntimeError(f"GraspGen service is not ready: {health}")
            world_tcp_poses, confidences, response = _infer_world_grasp_poses(
                client, rgb, depth, camera_k, world_camera_pose, prompt, approach_camera,
            )
        (args.output_dir / f"{scene_id}_{object_id}_graspgen_response.json").write_text(
            json.dumps(_json_compatible(response), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if world_tcp_poses is None:
            result.update(success=False, skip_reason=f"graspgen_{response.get('status', 'no_grasps')}")
            return result
        best_pose = world_tcp_poses[int(np.argmax(confidences))]

        # Adjust distance before pivoting; afterward the base cannot move along reach.
        base_x0, base_y0, _ = sim.get_base_pose()
        distance_to_target = float(np.hypot(best_pose[0, 3] - base_x0, best_pose[1, 3] - base_y0))
        if distance_to_target < SAFE_PIVOT_DISTANCE_M:
            _move_base_forward(sim, distance_to_target - SAFE_PIVOT_DISTANCE_M)

        base_x0, base_y0, _ = sim.get_base_pose()
        _, reach_shortfall_m, _ = grasp_ik.solve(
            world_to_base_pose((base_x0, base_y0, grasp_yaw)) @ best_pose
        )
        result["reach_shortfall_m"] = reach_shortfall_m
        if reach_shortfall_m > 0.02:
            _move_base_forward(sim, reach_shortfall_m + 0.05)

        # Pivot so the arm's local -Y axis faces the object.
        rotated = _rotate_to_yaw(sim, grasp_yaw)
        result.update(rotated_to_stance=rotated)

        # Compensate object motion since the RGB-D observation.
        object_now = np.asarray(
            sim.pull_grasp_metrics().get("object_position", initial_object_position), dtype=float
        )
        object_drift = object_now - initial_object_position
        result["object_drift_m"] = float(np.linalg.norm(object_drift))
        if np.linalg.norm(object_drift) > MAX_GRASP_POINT_OBJECT_DISTANCE_M:
            result.update(success=False, skip_reason="object_moved_too_much")
            return result
        world_tcp_poses = world_tcp_poses.copy()
        world_tcp_poses[:, :3, 3] += object_drift

        candidate, best_off_target = _select_reachable_candidate(
            grasp_ik, world_tcp_poses, confidences, sim.get_base_pose(), object_now,
        )
        if candidate is None:
            if best_off_target is None:
                result.update(success=False, skip_reason="ik_unreachable")
            else:
                result.update(
                    success=False,
                    skip_reason="candidate_off_target",
                    best_candidate_object_distance_m=best_off_target,
                )
            return result

        joints = candidate.joints
        source_index = candidate.index // 2
        grasp_pose_world = world_tcp_poses[source_index]
        grasp_point = grasp_pose_world[:3, 3]
        result.update(
            grasp_point=grasp_point,
            ik_position_error_m=candidate.position_error,
            ik_orientation_error_rad=candidate.orientation_error,
            graspgen_confidence=candidate.confidence,
            graspgen_candidate=source_index,
            ik_joints=joints,
        )
        _aim_head_at(sim, grasp_point)
        debug.capture(sim, grasp_point, "04_grasp_candidate", camera_k,
                      detail=f"confidence={candidate.confidence:.3f}")

        if args.dry_run:
            result.update(success=None, skip_reason="dry_run")
            return result

        # Raise and retract before extending toward the pregrasp.
        _move_base_forward(sim, float(joints[0]))
        safe_lift = NAV_LIFT_CLEARANCE
        sim.move_to(Actuators.arm, 0.0)
        _wait(sim, Actuators.arm, timeout=30.0)
        sim.move_to(Actuators.lift, safe_lift)
        _wait(sim, Actuators.lift, timeout=30.0)

        # Use a Cartesian standoff for the wrist-camera refinement.
        if PREGRASP_LATERAL_SHIFT_M != 0:
            base_yaw = float(sim.get_base_pose()[2])
            delta_world = PREGRASP_LATERAL_SHIFT_M * np.array(
                [np.cos(base_yaw), np.sin(base_yaw), 0.0]
            )
            _shift_ee_world(sim, grasp_ik, delta_world)

        standoff_joints = _solve_standoff_pose(
            grasp_ik, grasp_pose_world, sim.get_base_pose(), joints, WRIST_REFINE_STANDOFF_M,
        )
        result["wrist_refine_standoff_solved"] = standoff_joints is not None
        approach_joints = standoff_joints if standoff_joints is not None else joints

        _set_flat_wrist(sim, approach_joints[3])

        sim.move_to(Actuators.arm, float(approach_joints[2]))
        _wait(sim, Actuators.arm, timeout=25.0, tolerance=0.01)
        sim.move_to(Actuators.lift, float(approach_joints[1]))
        _wait(sim, Actuators.lift, timeout=25.0, tolerance=0.01)

        refined_joints, refined_grasp_point, refine_diag = None, None, {"wrist_refine_used": False}
        if standoff_joints is not None:
            refined_joints, refined_grasp_point, refine_diag = _wrist_refine_grasp(
                sim, grasp_ik, args, scene_id, object_id, prompt, grasp_point,
            )
        else:
            refine_diag["wrist_refine_skip_reason"] = "standoff_unsolvable"
        result.update(refine_diag)
        if refined_joints is not None:
            joints, grasp_point = refined_joints, refined_grasp_point
        debug.capture(sim, grasp_point, "04b_wrist_refine", camera_k,
                      detail=f"used={refined_joints is not None}")

        _set_flat_wrist(sim, joints[3])

        _align_ee(sim, grasp_point, adjust_arm=False)
        sim.move_to(Actuators.arm, float(joints[2]))
        arm_reached = _wait(sim, Actuators.arm, timeout=25.0, tolerance=0.01)
        sim.move_to(Actuators.lift, float(joints[1]))
        lift_reached = _wait(sim, Actuators.lift, timeout=25.0, tolerance=0.01)
        time.sleep(0.3)

        status = sim.pull_status()
        result.update(
            pregrasp_arm_reached=arm_reached,
            pregrasp_lift_reached=lift_reached,
            pregrasp_arm_error_m=float(joints[2]) - float(status.arm.pos),
            pregrasp_lift_error_m=float(joints[1]) - float(status.lift.pos),
            pregrasp_wrist_yaw_error=float(joints[3]) - float(status.wrist_yaw.pos),
            pregrasp_wrist_pitch_error=FLAT_WRIST_PITCH - float(status.wrist_pitch.pos),
            pregrasp_wrist_roll_error=FLAT_WRIST_ROLL - float(status.wrist_roll.pos),
            pregrasp_center_distance_m=sim.pull_grasp_metrics().get("center_distance_m"),
        )
        result["pregrasp_target_distance_m"] = float(
            np.linalg.norm(sim.get_ee_pose()[:3, 3] - grasp_point)
        )
        debug.capture(sim, grasp_point, "05_pregrasp", camera_k,
                      detail=f"dist={result['pregrasp_target_distance_m']:.3f}m")

        _align_ee(sim, grasp_point, adjust_arm=True)
        time.sleep(0.2)
        debug.capture(sim, grasp_point, "06_aligned", camera_k)

        if POST_ALIGN_PUSH_M > 0:
            real_push_m = _push_in_along_approach(sim, grasp_ik, POST_ALIGN_PUSH_M)
            if PUSH_IN_DOWNWARD_SHIFT_M != 0:
                _shift_ee_world(sim, grasp_ik, np.array([0.0, 0.0, -PUSH_IN_DOWNWARD_SHIFT_M]))

            debug.capture(sim, grasp_point, "06b_push_in", camera_k,
                          detail=f"push={real_push_m:.3f}m")

        sim.move_to(Actuators.gripper, GRIPPER_CLOSE)
        close_started = time.monotonic()
        gripper_position = float(sim.pull_status().gripper.pos)
        grasp_metrics = sim.pull_grasp_metrics()
        while (
            not grasp_metrics.get("bilateral_contact", False)
            and time.monotonic() - close_started < 30.0
        ):
            time.sleep(0.1)
            gripper_position = float(sim.pull_status().gripper.pos)
            grasp_metrics = sim.pull_grasp_metrics()
            if gripper_position <= 0.0 and time.monotonic() - close_started >= 2.0:
                break
        result["grasp_metrics_at_close"] = grasp_metrics
        bilateral_contact = bool(grasp_metrics.get("bilateral_contact", False))
        left_contacts = int(grasp_metrics.get("left_finger_contacts", 0))
        right_contacts = int(grasp_metrics.get("right_finger_contacts", 0))
        debug.capture(sim, grasp_point, "07_close", camera_k,
                      detail=f"L={left_contacts} R={right_contacts}")
        if not (bilateral_contact and left_contacts > 0 and right_contacts > 0):
            result.update(success=False, skip_reason="no_bilateral_contact")
            return result

        # Keep physical contact and gravity enabled; do not attach the object.
        time.sleep(0.3)

        # Lift in place without changing arm or wrist pose.
        current_lift = float(sim.pull_status().lift.pos)
        sim.move_to(Actuators.lift, float(np.clip(current_lift + 0.20, 0.0, 1.1)))
        _wait(sim, Actuators.lift, timeout=30.0)
        time.sleep(0.5)  # hold the lifted pose while physics settles

        final_metrics = sim.pull_grasp_metrics()
        lift_delta = float(final_metrics["object_position"][2] - initial_object_position[2])
        final_bilateral = bool(final_metrics.get("bilateral_contact", False))
        result.update(
            initial_object_position=initial_object_position,
            final_grasp_metrics=final_metrics,
            lift_delta_m=lift_delta,
            bilateral_contact=final_bilateral,
            success=final_bilateral and lift_delta >= MIN_LIFT_M,
        )
        debug.capture(sim, np.asarray(final_metrics["object_position"], dtype=float), "08_lift", camera_k,
                      detail=f"lift={lift_delta:.3f}m")
        if not result["success"]:
            result["skip_reason"] = "insufficient_lift_or_lost_contact"
        return result
    finally:
        debug.close(f"{scene_id}_{object_id}_process")
        if sim.is_running():
            sim.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="1", help="1..10 or scene id")
    parser.add_argument("--scene-root", type=Path, default=None,
                        help="Generated scene directory; defaults to office_scenes")
    parser.add_argument("--nav-report-dir", type=Path, default=None,
                        help="Navigation report directory (defaults to output/<scene family>_nav_sim)")
    parser.add_argument("--object-id", default=None, help="comma-separated object ids to restrict to")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (defaults to output/<scene family>_grasp_sim)")
    parser.add_argument("--graspgen-host", default=os.getenv("GRASPGEN_HOST", "10.29.150.95"))
    parser.add_argument("--graspgen-port", type=int, default=5557)
    parser.add_argument("--debug", action="store_true",
                        help="Save per-stage RGB-D debug PNGs and a process GIF")
    parser.add_argument("--list", action="store_true", help="List discovered reports and exit")
    parser.add_argument("--dry-run", action="store_true", help="IK planning only, no motion")
    args = parser.parse_args()
    if args.scene_root is not None:
        os.environ["STRETCH_SCENE_ROOT"] = str(args.scene_root.resolve())
    # Keep grasp tests aligned with run_office_navigation_sim.py.  The
    # historical defaults were office-only, which made a home run silently
    # search the wrong directory for navigation reports.
    root_name = Path(os.environ.get("STRETCH_SCENE_ROOT", "office_scenes")).name
    family = "home" if "home" in root_name.lower() else "office"
    if args.nav_report_dir is None:
        args.nav_report_dir = Path("output") / f"{family}_nav_sim"
    if args.output_dir is None:
        args.output_dir = Path("output") / f"{family}_grasp_sim"

    xml_path, manifest_path = _scene_paths(args.scene)[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_id = manifest["scene_id"]

    report_paths = sorted(args.nav_report_dir.glob(f"{scene_id}_*_grasp_site.json"))
    if args.object_id:
        wanted = {value.strip() for value in args.object_id.split(",") if value.strip()}
        report_paths = [
            p for p in report_paths if p.name.removeprefix(f"{scene_id}_").removesuffix(
                "_grasp_site.json"
            ) in wanted
        ]

    if args.list:
        for path in report_paths:
            report = json.loads(path.read_text(encoding="utf-8"))
            print(f"{report['goal_site']}: reached={report.get('reached')} path={path}")
        return 0

    if not report_paths:
        print(f"No navigation reports found for {scene_id} in {args.nav_report_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for report_path in report_paths:
        result = run_object(xml_path, manifest, scene_id, report_path, args)
        results.append(result)
        object_json = args.output_dir / f"{scene_id}_{result['object_id']}_grasp.json"
        object_json.write_text(
            json.dumps(_json_compatible(result), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        status = "SKIP" if result.get("skip_reason") and result.get("success") is None else (
            "PASS" if result.get("success") else "FAIL"
        )
        print(
            f"{status} {result['object_id']}: success={result.get('success')} "
            f"reason={result.get('skip_reason')} lift_delta_m={result.get('lift_delta_m')} "
            f"report={object_json}"
        )

    summary_path = args.output_dir / f"{scene_id}_grasp_summary.json"
    summary = {
        "scene_id": scene_id,
        "results": [
            {
                "object_id": r["object_id"],
                "success": r.get("success"),
                "skip_reason": r.get("skip_reason"),
                "lift_delta_m": r.get("lift_delta_m"),
            }
            for r in results
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary: {summary_path}")

    attempted = [r for r in results if r.get("success") is not None]
    failed = [r for r in attempted if not r.get("success")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
