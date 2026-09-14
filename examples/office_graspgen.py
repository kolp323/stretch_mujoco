"""Detect and grasp an office snack with GraspGen, optionally recording MP4."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import click
import cv2
import numpy as np

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.graspgen import (
    RGBDGraspClient,
    StretchGraspIK,
    rotated_d435i_optical_pose,
    world_to_base_pose,
)
from stretch_mujoco.paths import output_root


ROOT = Path(__file__).resolve().parents[1]
OFFICE_SCENE = ROOT / "stretch_mujoco" / "models" / "office_scene.xml"
CAMERAS = [
    StretchCameras.cam_d435i_rgb,
    StretchCameras.cam_d435i_depth,
    StretchCameras.office_overview_rgb,
]
OBJECT_BASE_X = {
    "soda_can": -0.34,
    "cereal_box": -0.20,
    "bread_snack": 0.15,
    "lemon": 0.38,
}
ROBOT_START_Y = -0.34


def json_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def normalize_direction_vector(direction: Any, name: str = "direction") -> np.ndarray:
    """Return a normalized 3D direction vector (matches real-robot validation)."""
    vec = np.asarray(direction, dtype=np.float64).reshape(-1)
    if vec.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vec.shape}")
    if not np.isfinite(vec).all():
        raise ValueError(f"{name} contains NaN or infinity")
    norm = np.linalg.norm(vec)
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return vec / norm


def object_position(snapshot: dict, object_id: str) -> np.ndarray:
    return np.asarray(snapshot["objects"][object_id]["position"], dtype=float)


class GraspVideoRecorder:
    def __init__(self, sim: StretchMujocoSimulator, output: Path, fps: int) -> None:
        self.sim = sim
        self.output = output
        self.fps = fps
        self.stage = "STARTING"
        self.detail = ""
        self.inset_rgb: np.ndarray | None = None
        self.bbox: list[int] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: cv2.VideoWriter | None = None

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_stage(self, stage: str, detail: str = "") -> None:
        self.stage = stage
        self.detail = detail

    def set_detection(self, rgb: np.ndarray, bbox: list[int] | None) -> None:
        self.inset_rgb = np.asarray(rgb).copy()
        self.bbox = bbox

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._writer is not None:
            self._writer.release()

    def _run(self) -> None:
        started = time.monotonic()
        written = 0
        while not self._stop.is_set() and self.sim.is_running():
            expected = int((time.monotonic() - started) * self.fps) + 1
            if written >= expected:
                time.sleep(0.005)
                continue
            try:
                cameras = self.sim.pull_camera_data()
                frame = cameras.get_camera_data(StretchCameras.office_overview_rgb)
                self.inset_rgb = cameras.get_camera_data(
                    StretchCameras.cam_d435i_rgb,
                    auto_rotate=True,
                    auto_correct_rgb=False,
                ).copy()
            except ValueError:
                time.sleep(0.01)
                continue
            annotated = self._annotate(frame)
            if self._writer is None:
                height, width = annotated.shape[:2]
                self._writer = cv2.VideoWriter(
                    str(self.output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (width, height),
                )
                if not self._writer.isOpened():
                    raise RuntimeError(f"Cannot open MP4 writer: {self.output}")
            while written < expected:
                self._writer.write(annotated)
                written += 1

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        cv2.rectangle(result, (18, 16), (610, 92), (20, 24, 28), thickness=-1)
        cv2.putText(
            result,
            f"GRASPGEN  |  {self.stage}",
            (34, 49),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            self.detail[:70],
            (34, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (80, 210, 255),
            1,
            cv2.LINE_AA,
        )
        if self.inset_rgb is not None:
            inset = cv2.cvtColor(self.inset_rgb, cv2.COLOR_RGB2BGR)
            scale = min(220 / inset.shape[0], 180 / inset.shape[1])
            inset = cv2.resize(inset, None, fx=scale, fy=scale)
            show_detection = self.stage in {
                "INFERENCE",
                "PLAN",
                "PREGRASP",
                "APPROACH",
                "ALIGN",
                "CLOSE",
            }
            if self.bbox is not None and show_detection:
                x1, y1, x2, y2 = self.bbox
                cv2.rectangle(
                    inset,
                    (round(x1 * scale), round(y1 * scale)),
                    (round(x2 * scale), round(y2 * scale)),
                    (0, 255, 255),
                    2,
                )
            x0 = result.shape[1] - inset.shape[1] - 20
            y0 = 18
            result[y0 : y0 + inset.shape[0], x0 : x0 + inset.shape[1]] = inset
            cv2.rectangle(
                result,
                (x0 - 1, y0 - 1),
                (x0 + inset.shape[1], y0 + inset.shape[0]),
                (230, 230, 230),
                1,
            )
        return result


def move_joints(
    sim: StretchMujocoSimulator, joints: np.ndarray, *, arm: float | None = None
) -> None:
    sim.move_to(Actuators.lift, float(joints[0]))
    if not sim.wait_until_at_setpoint(Actuators.lift, timeout=45.0, position_tolerance=0.025):
        raise RuntimeError("Lift did not reach the pregrasp height")
    wrist_commands = {
        Actuators.wrist_yaw: float(joints[2]),
        Actuators.wrist_pitch: float(joints[3]),
        Actuators.wrist_roll: float(joints[4]),
    }
    for actuator, value in wrist_commands.items():
        sim.move_to(actuator, value)
    for actuator in wrist_commands:
        if not sim.wait_until_at_setpoint(actuator, timeout=45.0, position_tolerance=0.025):
            raise RuntimeError(f"{actuator.name} did not reach the pregrasp pose")
    sim.move_to(Actuators.arm, float(joints[1] if arm is None else arm))
    if not sim.wait_until_at_setpoint(Actuators.arm, timeout=45.0, position_tolerance=0.025):
        raise RuntimeError("Arm did not reach the pregrasp extension")


@click.command()
@click.option(
    "--host",
    default=lambda: os.environ.get("STRETCH_MUJOCO_GRASPGEN_HOST", "127.0.0.1"),
    show_default="127.0.0.1",
    help="GraspGen server host (or set STRETCH_MUJOCO_GRASPGEN_HOST).",
)
@click.option("--port", type=int, default=5557, show_default=True)
@click.option("--object-id", default="cereal_box", show_default=True)
@click.option("--prompt", default="cereal box", show_default=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=lambda: output_root() / "office_graspgen" / "run.mp4",
    show_default="outputs/office_graspgen/run.mp4",
)
@click.option(
    "--artifacts-dir",
    type=click.Path(path_type=Path),
    default=lambda: output_root() / "office_graspgen" / "artifacts",
    show_default="outputs/office_graspgen/artifacts",
)
@click.option("--fps", type=click.IntRange(1, 60), default=24, show_default=True)
@click.option("--motion-speed", type=click.FloatRange(min=0.1), default=6.0, show_default=True)
@click.option("--dry-run", is_flag=True, help="Stop after inference and IK planning.")
@click.option(
    "--approach-direction-base",
    type=float,
    nargs=3,
    default=(0.0, 0.0, -1.0),
    show_default=True,
    help="Desired approach direction in base_link, e.g. 0 0 -1 for downward or 1 0 0 for horizontal +X",
)
@click.option(
    "--max-approach-angle-deg",
    type=float,
    default=30.0,
    show_default=True,
    help="Maximum allowed angle from the desired approach direction for candidate grasps",
)
@click.option(
    "--approach-allow-opposite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether to also accept grasps from the opposite direction (180° flipped)",
)
def main(
    host: str,
    port: int,
    object_id: str,
    prompt: str,
    output: Path,
    artifacts_dir: Path,
    fps: int,
    motion_speed: float,
    dry_run: bool,
    approach_direction_base: tuple[float, float, float],
    max_approach_angle_deg: float,
    approach_allow_opposite: bool,
) -> None:
    """Run an RGB-D GraspGen grasp in the office scene."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if object_id not in OBJECT_BASE_X:
        raise click.BadParameter(
            f"Unsupported office snack {object_id!r}; choose one of {sorted(OBJECT_BASE_X)}"
        )
    sim = StretchMujocoSimulator(
        scene_xml_path=str(OFFICE_SCENE), # 场景
        camera_hz=min(fps, 8), # 摄像头帧率
        cameras_to_use=CAMERAS, # 摄像头列表
        start_translation=[OBJECT_BASE_X[object_id], ROBOT_START_Y, 0.0], # 机器人初始位置
    )
    sim.set_robot_motion_speed(motion_speed)
    recorder = GraspVideoRecorder(sim, output.resolve(), fps)
    result: dict[str, Any] = {"object_id": object_id, "prompt": prompt}
    try:
        sim.start(headless=True)
        recorder.start()
        recorder.set_stage("OBSERVE", "Turn head toward the snack counter")
        sim.move_to(Actuators.head_pan, -1.57)
        sim.move_to(Actuators.head_tilt, -0.65)
        sim.move_to(Actuators.gripper, 0.56)
        for actuator in (Actuators.head_pan, Actuators.head_tilt):
            sim.wait_until_at_setpoint(actuator, timeout=8.0)
        open_started = time.monotonic()
        gripper_opening = float(sim.pull_status().gripper.pos)
        while gripper_opening < 0.50 and time.monotonic() - open_started < 30.0:
            time.sleep(0.1)
            gripper_opening = float(sim.pull_status().gripper.pos)
        result["pregrasp_gripper_opening"] = gripper_opening
        if gripper_opening < 0.50:
            raise RuntimeError(f"Gripper did not open enough for approach: {gripper_opening:.3f}")
        time.sleep(1.0)

        cameras = sim.pull_camera_data()
        rgb = cameras.get_camera_data(
            StretchCameras.cam_d435i_rgb,
            auto_rotate=True,
            auto_correct_rgb=False,
        )
        depth = cameras.get_camera_data(StretchCameras.cam_d435i_depth, auto_rotate=True)
        camera_k = np.asarray(cameras.cam_d435i_K, dtype=np.float32)
        if rgb.shape[:2] != depth.shape or camera_k.shape != (3, 3):
            raise RuntimeError("D435i RGB, depth, and intrinsics are inconsistent")
        np.savez_compressed(
            artifacts_dir / "observation.npz", rgb=rgb, depth=depth, camera_K=camera_k
        )
        initial_object_position = object_position(sim.pull_semantic_state(), object_id)

        recorder.set_detection(rgb, None)
        recorder.set_stage("INFERENCE", f"Requesting {prompt!r} grasps from {host}:{port}")

        # Convert the desired approach direction from base_link to camera frame
        # (matches the real-robot grasp_object_use_graspgen.py pipeline).
        approach_direction_base_vec = normalize_direction_vector(
            approach_direction_base, name="approach_direction_base"
        )
        raw_camera_pose = sim.get_link_pose("camera_color_optical_frame")
        world_camera_pose = rotated_d435i_optical_pose(raw_camera_pose)
        T_base_camera = world_to_base_pose(sim.get_base_pose()) @ world_camera_pose
        R_base_camera = T_base_camera[:3, :3]
        desired_approach_camera = R_base_camera.T @ approach_direction_base_vec
        desired_approach_camera /= np.linalg.norm(desired_approach_camera)
        print(
            "[GRASPGEN APP] Desired approach direction in base frame:",
            approach_direction_base_vec,
        )
        print(
            "[GRASPGEN APP] Desired approach direction expressed in camera frame:",
            desired_approach_camera,
        )

        with RGBDGraspClient(host, port) as client:
            health = client.health()
            if not health.get("ready"):
                raise RuntimeError(f"GraspGen service is not ready: {health}")
            response = client.infer(
                rgb,
                depth,
                camera_k,
                prompt,
                candidate_top_k=200,
                desired_approach_direction=tuple(desired_approach_camera),
                max_approach_angle_deg=max_approach_angle_deg,
                approach_allow_opposite=approach_allow_opposite,
            )
            result["request_bytes"] = client.last_request_bytes
        (artifacts_dir / "response.json").write_text(
            json.dumps(json_compatible(response), indent=2), encoding="utf-8"
        )
        if response.get("status") != "ok" or not response.get("num_grasps"):
            raise RuntimeError(f"GraspGen did not return grasps: {response.get('status')}")
        bbox = [int(value) for value in response["bbox_xyxy"]]
        recorder.set_detection(rgb, bbox)

        camera_tcp_poses = np.asarray(response["grasp_tcp_poses"], dtype=float)
        world_tcp_poses = np.asarray([world_camera_pose @ pose for pose in camera_tcp_poses])
        base_tcp_poses = np.asarray(
            [world_to_base_pose(sim.get_base_pose()) @ pose for pose in world_tcp_poses]
        )
        confidences = np.asarray(response["confidences"], dtype=float)
        symmetry = np.diag([-1.0, -1.0, 1.0, 1.0])
        symmetric_base_poses = np.asarray(
            [variant for pose in base_tcp_poses for variant in (pose, pose @ symmetry)]
        )
        symmetric_confidences = np.repeat(confidences, 2)
        recorder.set_stage(
            "PLAN",
            f"Checking {len(symmetric_base_poses)} symmetric candidates with constrained IK",
        )
        candidates = StretchGraspIK(sim.urdf_model).rank_candidates(
            symmetric_base_poses, symmetric_confidences
        )
        candidates = [candidate for candidate in candidates if abs(candidate.joints[3]) <= 0.35]
        if not candidates:
            raise RuntimeError("No horizontal GraspGen candidate is reachable within 15 mm")
        candidate = candidates[0]
        source_candidate_index = candidate.index // 2
        symmetric_variant = candidate.index % 2 == 1
        result.update(
            selected_candidate=source_candidate_index,
            symmetric_variant=symmetric_variant,
            confidence=candidate.confidence,
            position_error_m=candidate.position_error,
            orientation_error_rad=candidate.orientation_error,
            target_world_pose=world_tcp_poses[source_candidate_index],
            target_base_pose=candidate.target_pose,
            target_joints=candidate.joints,
        )
        time.sleep(1.0)

        if not dry_run:
            pregrasp_arm = max(float(candidate.joints[1]) - 0.12, 0.0)
            recorder.set_stage("PREGRASP", "Open gripper and move behind the selected grasp")
            move_joints(sim, candidate.joints, arm=pregrasp_arm)
            time.sleep(0.6)

            recorder.set_stage("APPROACH", "Extend to the GraspGen TCP target")
            sim.request_grasp_metrics(object_id)
            sim.move_to(Actuators.arm, float(candidate.joints[1]))
            if not sim.wait_until_at_setpoint(
                Actuators.arm, timeout=45.0, position_tolerance=0.015
            ):
                raise RuntimeError("Arm did not reach the GraspGen target")
            time.sleep(0.6)

            # --- ALIGN stage disabled: rely purely on GraspGen planning accuracy ---
            # metrics_before_alignment = sim.pull_grasp_metrics() # 返回当前抓取状态的度量数据，包含物体位置
            # object_xyz = np.asarray(metrics_before_alignment["object_position"], dtype=float) # 物体位置
            # grasp_xyz = np.asarray(metrics_before_alignment["grasp_center_position"], dtype=float) # 抓取中心位置
            # status = sim.pull_status()
            # aligned_arm = float(np.clip(status.arm.pos + grasp_xyz[1] - object_xyz[1], 0.0, 0.52))
            # aligned_lift = float(np.clip(status.lift.pos + object_xyz[2] - grasp_xyz[2], 0.0, 1.1))
            # recorder.set_stage(
            #     "ALIGN",
            #     f"Correct residual Y/Z error before closing: {metrics_before_alignment['center_distance_m']:.3f} m",
            # )
            # sim.move_to(Actuators.arm, aligned_arm)
            # sim.move_to(Actuators.lift, aligned_lift)
            # if not sim.wait_until_at_setpoint(
            #     Actuators.arm, timeout=30.0, position_tolerance=0.008
            # ):
            #     raise RuntimeError("Arm did not complete final grasp alignment")
            # if not sim.wait_until_at_setpoint(
            #     Actuators.lift, timeout=30.0, position_tolerance=0.008
            # ):
            #     raise RuntimeError("Lift did not complete final grasp alignment")
            # time.sleep(0.3)
            # metrics_after_alignment = sim.pull_grasp_metrics()
            # result.update(
            #     grasp_metrics_before_alignment=metrics_before_alignment,
            #     grasp_metrics_after_alignment=metrics_after_alignment,
            #     aligned_arm=aligned_arm,
            #     aligned_lift=aligned_lift,
            # )
            # ----------------------------------------------------------------

            recorder.set_stage("CLOSE", f"Close gripper around {object_id}")
            sim.move_to(Actuators.gripper, -0.15)
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
            gripper_position = float(sim.pull_status().gripper.pos)
            result["gripper_contact_position"] = gripper_position
            result["grasp_metrics_at_close"] = grasp_metrics
            if not grasp_metrics.get("bilateral_contact", False):
                result["success"] = False
                print(
                    "[GRASPGEN APP] WARNING: Physical grasp validation failed: "
                    f"left={grasp_metrics.get('left_finger_contacts', 0)}, "
                    f"right={grasp_metrics.get('right_finger_contacts', 0)}, "
                    f"center_distance={grasp_metrics.get('center_distance_m')}"
                )

            if result.get("success") is not False:
                pre_attach_object_position = object_position(sim.pull_semantic_state(), object_id)
                pre_attach_grasp_center = sim.get_ee_pose()[:3, 3]
                pre_attach_distance = float(
                    np.linalg.norm(pre_attach_object_position - pre_attach_grasp_center)
                )
                result.update(
                    pre_attach_object_position=pre_attach_object_position,
                    pre_attach_grasp_center=pre_attach_grasp_center,
                    pre_attach_distance_m=pre_attach_distance,
                )
                time.sleep(0.4)
                sim.attach_object_to_gripper(object_id)
                time.sleep(0.3)

                recorder.set_stage("RETREAT", "Retract beyond the table edge before lifting")
                sim.move_to(Actuators.arm, 0.06)
                if not sim.wait_until_at_setpoint(
                    Actuators.arm, timeout=45.0, position_tolerance=0.025
                ):
                    raise RuntimeError("Arm did not clear the table before lifting")
                time.sleep(0.5)

                recorder.set_stage("ORIENT", "Rotate the cleared gripper vertically downward")
                sim.move_to(Actuators.wrist_pitch, -1.50)
                sim.move_to(Actuators.wrist_roll, 0.0)
                if not sim.wait_until_at_setpoint(
                    Actuators.wrist_pitch, timeout=45.0, position_tolerance=0.03
                ):
                    raise RuntimeError("Gripper did not reach the vertical carry orientation")
                if not sim.wait_until_at_setpoint(
                    Actuators.wrist_roll, timeout=45.0, position_tolerance=0.03
                ):
                    raise RuntimeError("Gripper roll did not settle before lifting")
                time.sleep(0.5)
                oriented_object_position = object_position(sim.pull_semantic_state(), object_id)

                recorder.set_stage("LIFT", "Raise with the gripper vertical and clear of the table")
                sim.move_to(Actuators.lift, min(float(candidate.joints[0]) + 0.45, 1.1))
                if not sim.wait_until_at_setpoint(
                    Actuators.lift, timeout=45.0, position_tolerance=0.015
                ):
                    raise RuntimeError("Lift did not reach the post-grasp height")
                time.sleep(1.0)

                final_object_position = object_position(sim.pull_semantic_state(), object_id)
                lift_delta = float(final_object_position[2] - initial_object_position[2])
                carry_lift_delta = float(final_object_position[2] - oriented_object_position[2])
                success = carry_lift_delta >= 0.25 and final_object_position[2] >= 0.88
                result.update(
                    success=success,
                    initial_object_position=initial_object_position,
                    oriented_object_position=oriented_object_position,
                    final_object_position=final_object_position,
                    lift_delta_m=lift_delta,
                    carry_lift_delta_m=carry_lift_delta,
                )
                recorder.set_stage(
                    "SUCCESS" if success else "FAILED",
                    f"Vertical carry: {carry_lift_delta:.3f} m  |  final z: {final_object_position[2]:.3f} m",
                )
                time.sleep(2.0)
            else:
                recorder.set_stage(
                    "FAILED",
                    f"Gripper did not contact object: center_distance={grasp_metrics.get('center_distance_m'):.3f} m",
                )
                time.sleep(2.0)
        else:
            result["success"] = None
            recorder.set_stage("DRY RUN", "Inference and IK succeeded; motion skipped")
            time.sleep(2.0)
    finally:
        recorder.stop()
        if sim.is_running():
            sim.stop()
        (artifacts_dir / "result.json").write_text(
            json.dumps(json_compatible(result), indent=2), encoding="utf-8"
        )
    if result.get("success") is False:
        print("[GRASPGEN APP] WARNING: The object did not reach the required lift height")
    click.echo(f"Result: {artifacts_dir.resolve() / 'result.json'}")
    click.echo(f"MP4: {output.resolve()}")


if __name__ == "__main__":
    main()


"""
.venv/bin/python examples/office_graspgen.py \
  --host 10.29.150.95 \
  --port 5557 \
  --object-id cereal_box \
  --prompt "cereal box" \
  --approach-direction-base 0 0 -1 \
  --max-approach-angle-deg 30 \
  --fps 24 \
  --motion-speed 6 \
  --output office_graspgen.mp4 \
  --artifacts-dir output/office_graspgen
"""
