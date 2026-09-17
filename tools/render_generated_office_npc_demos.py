#!/usr/bin/env python3
"""Render receipt-gated NPC route/action demos for every generated office.

Each demo composes the three-NPC population into one generated Office MJCF,
checks all configured Alex route contracts, executes those collision-checked
routes, then plays an approved gesture clip.  Robot control is intentionally
absent: ``robot_mode`` is recorded as ``mock`` so the artifacts do not claim a
physical Stretch navigation, grasp, or delivery execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import cv2
import mujoco

from stretch_mujoco.npc import CommandStatus, NpcCommand, NpcCommandKind
from stretch_mujoco.npc.composition import load_composed_npc_runtime
from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "stretch_mujoco" / "models"
SCENES = MODELS / "assets" / "office_scenes"
GENERATED = MODELS / "generated_office_npc"
DEFAULT_OUTPUT = ROOT / "aaa_workspace" / "demo" / "npc_in_new_scenes"
DEMO_NPC_ID = "npc_alex_chen"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scene_ids() -> tuple[str, ...]:
    return tuple(path.stem.removesuffix("_npc") for path in sorted(SCENES.glob("office_*_npc.xml")))


def _parse_scene_ids(value: str) -> tuple[str, ...]:
    available = _scene_ids()
    if value == "all":
        return available
    requested = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"Unknown generated office scene(s): {', '.join(sorted(unknown))}")
    return requested


def _transcode_h264(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(destination),
        ],
        check=True,
    )


def _annotate(
    rgb: Any,
    *,
    scene_id: str,
    activity: str,
    route_index: int,
    route_count: int,
) -> Any:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (12, 12), (image.shape[1] - 12, 84), (18, 25, 35), thickness=-1)
    cv2.putText(
        image,
        f"GENERATED OFFICE NPC | {scene_id}",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (239, 244, 248),
        2,
    )
    cv2.putText(
        image,
        f"{activity} | route {route_index}/{route_count} | robot=mock (not controlled)",
        (28, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (150, 226, 177),
        1,
    )
    return image


def _set_source_pose(
    model: mujoco.MjModel, data: mujoco.MjData, controller: Any, site: str
) -> None:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if site_id < 0:
        raise ValueError(f"route_source_site_missing:{site}")
    data.mocap_pos[controller.binding.mocap_id, :2] = data.site_xpos[site_id, :2]
    data.mocap_pos[controller.binding.mocap_id, 2] = data.mocap_pos[controller.binding.mocap_id, 2]
    mujoco.mju_mat2Quat(data.mocap_quat[controller.binding.mocap_id], data.site_xmat[site_id])
    mujoco.mj_forward(model, data)


def _render_scene(
    scene_id: str,
    output_dir: Path,
    *,
    width: int,
    height: int,
    fps: int,
    speed: float,
) -> dict[str, object]:
    population_path = GENERATED / "populations" / f"{scene_id}.population.json"
    semantic_path = GENERATED / "semantics" / f"{scene_id}.semantic.json"
    composition_path = output_dir / "composed" / f"{scene_id}.xml"
    loaded = load_composed_npc_runtime(
        population_path,
        composition_path,
        semantic_world_path=semantic_path,
        simulation_seed=20260914,
    )
    population = NpcPopulation.from_json(population_path)
    profile_path = population.resolve_path(population.trajectory_profile or "")
    profile = NpcTrajectoryProfile.from_json(profile_path)
    model = loaded.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    profile.validate_scene(loaded.composition.base_scene_path)
    preflight = profile.preflight(model, data)
    clearance = profile.audit_npc_clearance(model, data, DEMO_NPC_ID, sample_period=0.1)
    controller = loaded.npc_system.controllers[DEMO_NPC_ID]
    output_path = output_dir / f"{scene_id}_route_action_mock.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    route_reports: list[dict[str, object]] = []
    sim_time = 0.0
    temporary_path: Path | None = None
    writer: cv2.VideoWriter | None = None

    def write_frame(activity: str, route_index: int) -> None:
        # Keep the moving actor legible while retaining nearby furniture and
        # the other configured NPCs in frame.  The receipt still identifies
        # the complete generated scene and its source hashes.
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = data.xpos[controller.binding.body_id]
        camera.lookat[2] += 0.8
        camera.distance = 6.0
        camera.azimuth = 42.0
        camera.elevation = -38.0
        renderer.update_scene(data, camera=camera)
        writer.write(
            _annotate(
                renderer.render(),
                scene_id=scene_id,
                activity=activity,
                route_index=route_index,
                route_count=len(profile.routes),
            )
        )

    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, prefix=f".{output_path.stem}.", suffix=".mp4", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        writer = cv2.VideoWriter(
            str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"video_writer_unavailable:{output_path}")
        for route_index, route in enumerate(profile.routes, start=1):
            source = profile.anchors[route.source]
            destination = profile.anchors[route.destination]
            _set_source_pose(model, data, controller, source.site)
            for _ in range(max(1, round(fps * 0.4))):
                write_frame(f"prepare {route.route_id}", route_index)
            command = NpcCommand(
                f"{scene_id}:{route.route_id}",
                route_index - 1,
                DEMO_NPC_ID,
                NpcCommandKind.MOVE_TO,
                {"site": destination.site, "speed": speed, "max_replans": 1},
                sim_time,
            )
            accepted = loaded.npc_system.submit(command)
            if accepted.status == CommandStatus.FAILED:
                raise RuntimeError(f"route_rejected:{route.route_id}:{accepted.reason}")
            for frame_count in range(1, 1201):
                loaded.npc_system.step(model, data, sim_time)
                mujoco.mj_forward(model, data)
                write_frame(f"walk {route.route_id}", route_index)
                state = loaded.npc_system.states(data, sim_time)[DEMO_NPC_ID]
                receipt = state.last_receipt
                if (
                    receipt is not None
                    and receipt.command_id == command.command_id
                    and receipt.status.terminal
                ):
                    route_reports.append(
                        {
                            "route_id": route.route_id,
                            "frames": frame_count,
                            "status": receipt.status.value,
                            "reason": receipt.reason,
                        }
                    )
                    if receipt.status is not CommandStatus.SUCCEEDED:
                        raise RuntimeError(f"route_failed:{route.route_id}:{receipt.reason}")
                    break
                sim_time += 1.0 / fps
            else:
                raise RuntimeError(f"route_timeout:{route.route_id}")

        gesture = NpcCommand(
            f"{scene_id}:gesture_wave",
            len(profile.routes),
            DEMO_NPC_ID,
            NpcCommandKind.PLAY_ANIMATION,
            # This presentation cue has no interaction side effect.  A bounded
            # duration intentionally exercises the real clip without claiming
            # a social-marker transaction such as a handover.
            {"clip": "gesture_wave", "duration": 3.0, "arrival_clip": "idle"},
            sim_time,
        )
        accepted = loaded.npc_system.submit(gesture)
        if accepted.status == CommandStatus.FAILED:
            raise RuntimeError(f"gesture_rejected:{accepted.reason}")
        gesture_resolved_observed = False
        gesture_sampled_observed = False
        for frame_count in range(1, 301):
            loaded.npc_system.step(model, data, sim_time)
            mujoco.mj_forward(model, data)
            state = loaded.npc_system.states(data, sim_time)[DEMO_NPC_ID]
            receipt = state.last_receipt
            submitted_gesture = state.active_command_id == gesture.command_id or (
                receipt is not None and receipt.command_id == gesture.command_id
            )
            if submitted_gesture:
                gesture_resolved_observed |= state.resolved_clip == "gesture_wave"
                gesture_sampled_observed |= controller.animation.last_sampled_clip == "gesture_wave"
            gesture_active = gesture_resolved_observed and gesture_sampled_observed
            write_frame(
                (
                    "gesture_wave active+sampled"
                    if gesture_active
                    else "gesture_wave waiting for activation"
                ),
                len(profile.routes),
            )
            if (
                receipt is not None
                and receipt.command_id == gesture.command_id
                and receipt.status.terminal
            ):
                gesture_report = {
                    "clip": "gesture_wave",
                    "frames": frame_count,
                    "status": receipt.status.value,
                    "reason": receipt.reason,
                    "resolved_observed": gesture_resolved_observed,
                    "sampled_observed": gesture_sampled_observed,
                    "activation_observed": gesture_active,
                }
                if not gesture_active:
                    raise RuntimeError("gesture_not_resolved_and_sampled")
                if receipt.status is not CommandStatus.SUCCEEDED:
                    raise RuntimeError(f"gesture_failed:{receipt.reason}")
                break
            sim_time += 1.0 / fps
        else:
            raise RuntimeError("gesture_timeout")
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
    if temporary_path is None:
        raise RuntimeError("temporary_video_missing")
    try:
        _transcode_h264(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "scene_id": scene_id,
        "video": output_path.name,
        "population": str(population_path.relative_to(ROOT)),
        "semantic_world": str(semantic_path.relative_to(ROOT)),
        "population_sha256": _sha256(population_path),
        "semantic_world_sha256": _sha256(semantic_path),
        "scene_sha256": _sha256(loaded.composition.base_scene_path),
        "robot_mode": "mock",
        "robot_control_executed": False,
        "configured_npcs": sorted(population.npcs),
        "preflight_routes": [route.route_id for route in preflight],
        "clearance_audit": [audit.route_id for audit in clearance],
        "route_receipts": route_reports,
        "gesture_receipt": gesture_report,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scenes", default="all", help="'all' or comma-separated generated scene IDs"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--speed", type=float, default=1.35)
    args = parser.parse_args()
    if min(args.width, args.height, args.fps, args.speed) <= 0:
        parser.error("width, height, fps, and speed must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = args.output_dir.resolve()
    records: list[dict[str, object]] = []
    for scene_id in _parse_scene_ids(args.scenes):
        try:
            records.append(
                _render_scene(
                    scene_id,
                    output_dir,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    speed=args.speed,
                )
            )
        except Exception as error:  # Keep successful scene evidence when one scene fails.
            records.append(
                {"scene_id": scene_id, "passed": False, "error": f"{type(error).__name__}: {error}"}
            )
    manifest = {
        "purpose": "generated-office NPC route/action demonstration",
        "robot_mode": "mock",
        "robot_control_executed": False,
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    failed = [record for record in records if not record["passed"]]
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "passed": len(records) - len(failed),
                "failed": len(failed),
            }
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
