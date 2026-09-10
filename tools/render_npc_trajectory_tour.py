#!/usr/bin/env python3
"""Render every preflighted NPC route through the unchanged MuJoCo office scene.

The renderer refuses to create a video until navigation preflight and the
controller-driven collision/turning audit both pass.  Route resets are labelled
as static source placements; only submitted ``MOVE_TO`` commands produce motion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import mujoco
import numpy as np

from stretch_mujoco.npc import CommandStatus, NpcCommand, NpcCommandKind
from stretch_mujoco.npc.system import NpcSystem
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile


def _configure_follow_camera(
    camera: mujoco.MjvCamera,
    root_position: np.ndarray,
) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = root_position
    camera.lookat[2] += 0.82
    camera.distance = 5.4
    camera.azimuth = 42.0
    camera.elevation = -42.0


def _transcode_h264(source: Path, destination: Path) -> None:
    try:
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
    except FileNotFoundError as error:
        raise RuntimeError("Trajectory tour rendering requires the 'ffmpeg' executable") from error


def _annotate(
    rgb: np.ndarray,
    *,
    route_index: int,
    route_count: int,
    route_id: str,
    source_role: str,
    destination_role: str,
    lifecycle: str,
    clip: str,
    audit_samples: int,
    static_placement: bool,
) -> np.ndarray:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    title = "NPC trajectory tour | collision + turning audit passed"
    if static_placement:
        status = "Static source placement — no movement in this transition"
    else:
        status = f"runtime={lifecycle}  clip={clip}"
    lines = (
        title,
        f"Route {route_index}/{route_count}: {route_id} ({source_role} -> {destination_role})",
        status,
        f"MuJoCo proxy-contact audit: clear across {audit_samples} sampled move/turn poses",
    )
    cv2.rectangle(image, (18, 18), (1040, 164), (20, 24, 32), thickness=-1)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (34, 48 + 34 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (235, 239, 244),
            1,
        )
    return image


def render_trajectory_tour(
    scene_path: Path,
    profile_path: Path,
    output_path: Path,
    *,
    npc_id: str = "employee_01",
    fps: int = 12,
    width: int = 1280,
    height: int = 720,
    speed: float = 1.15,
) -> dict[str, object]:
    """Render all profile routes using real marker-gated runtime commands."""
    if fps <= 0 or width <= 0 or height <= 0 or speed <= 0:
        raise ValueError("fps, width, height, and speed must be positive")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    profile = NpcTrajectoryProfile.from_json(profile_path)
    profile.validate_scene(scene_path)
    preflight = profile.preflight(model, data)
    clearance = profile.audit_npc_clearance(model, data, npc_id)
    audit_by_route = {audit.route_id: audit for audit in clearance}

    system = NpcSystem.from_model(model, simulation_seed=7, scene_path=scene_path)
    controller = system.controllers.get(npc_id)
    if controller is None:
        raise ValueError(f"Unknown NPC '{npc_id}'")
    root_z = float(data.mocap_pos[controller.binding.mocap_id, 2])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = False
    temporary_path: Path | None = None
    writer: cv2.VideoWriter | None = None
    reports: list[dict[str, object]] = []
    sim_time = 0.0
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, prefix=f".{output_path.stem}.", suffix=".mp4", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        writer = cv2.VideoWriter(
            str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for '{output_path}'")

        def write_frame(
            route_index: int,
            route_id: str,
            source_role: str,
            destination_role: str,
            *,
            static_placement: bool,
        ) -> None:
            state = system.states(data, sim_time)[npc_id]
            _configure_follow_camera(
                camera,
                data.xpos[controller.binding.body_id],
            )
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            writer.write(
                _annotate(
                    renderer.render(),
                    route_index=route_index,
                    route_count=len(profile.routes),
                    route_id=route_id,
                    source_role=source_role,
                    destination_role=destination_role,
                    lifecycle=state.animation_lifecycle,
                    clip=state.resolved_clip,
                    audit_samples=audit_by_route[route_id].sampled_poses,
                    static_placement=static_placement,
                )
            )

        for route_index, route in enumerate(profile.routes, start=1):
            source = profile.anchors[route.source]
            destination = profile.anchors[route.destination]
            source_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, source.site)
            data.mocap_pos[controller.binding.mocap_id, :2] = data.site_xpos[source_id, :2]
            data.mocap_pos[controller.binding.mocap_id, 2] = root_z
            mujoco.mju_mat2Quat(
                data.mocap_quat[controller.binding.mocap_id], data.site_xmat[source_id]
            )
            mujoco.mj_forward(model, data)
            for _ in range(max(1, round(fps * 0.7))):
                write_frame(
                    route_index,
                    route.route_id,
                    source.role,
                    destination.role,
                    static_placement=True,
                )
                sim_time += 1.0 / fps

            command = NpcCommand(
                f"trajectory_tour_{route.route_id}",
                route_index - 1,
                npc_id,
                NpcCommandKind.MOVE_TO,
                {
                    "site": destination.site,
                    "speed": speed,
                    "progress_timeout": 3.0,
                    "max_replans": 1,
                },
                sim_time,
            )
            accepted = system.submit(command)
            if accepted.status == CommandStatus.FAILED:
                raise RuntimeError(f"Route '{route.route_id}' was rejected: {accepted.reason}")
            frame_count = 0
            while frame_count < 1200:
                system.step(model, data, sim_time)
                mujoco.mj_forward(model, data)
                write_frame(
                    route_index,
                    route.route_id,
                    source.role,
                    destination.role,
                    static_placement=False,
                )
                frame_count += 1
                state = system.states(data, sim_time)[npc_id]
                if (
                    state.last_receipt is not None
                    and state.last_receipt.command_id == command.command_id
                    and state.last_receipt.status.terminal
                ):
                    reports.append(
                        {
                            "route_id": route.route_id,
                            "frames": frame_count,
                            "status": state.last_receipt.status.value,
                            "reason": state.last_receipt.reason,
                            "clearance": {
                                "sampled_poses": audit_by_route[route.route_id].sampled_poses,
                                "collision_free": True,
                            },
                        }
                    )
                    if state.last_receipt.status != CommandStatus.SUCCEEDED:
                        raise RuntimeError(
                            f"Route '{route.route_id}' did not complete: {state.last_receipt.reason}"
                        )
                    break
                sim_time += 1.0 / fps
            else:
                raise RuntimeError(f"Route '{route.route_id}' exceeded its rendering time budget")
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
    if temporary_path is None:
        raise RuntimeError("Trajectory tour writer was not initialized")
    try:
        _transcode_h264(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    report = {
        "scene": str(scene_path),
        "profile": str(profile_path),
        "npc_id": npc_id,
        "navigation_preflight_routes": [route.route_id for route in preflight],
        "route_reports": reports,
        "output": str(output_path),
        "passed": len(reports) == len(profile.routes)
        and all(route["status"] == CommandStatus.SUCCEEDED.value for route in reports),
    }
    output_path.with_suffix(".trajectory-tour.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npc-id", default="employee_01")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--speed", type=float, default=1.15)
    args = parser.parse_args()
    report = render_trajectory_tour(
        args.scene.resolve(),
        args.profile.resolve(),
        args.output.resolve(),
        npc_id=args.npc_id,
        fps=args.fps,
        width=args.width,
        height=args.height,
        speed=args.speed,
    )
    print(f"Trajectory tour saved -> {args.output} (passed={report['passed']})")


if __name__ == "__main__":
    main()
