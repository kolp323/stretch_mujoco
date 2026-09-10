#!/usr/bin/env python3
"""Render a controller-driven MOVE_TO trace in an unchanged office MJCF.

Unlike the acceptance renderer, this script submits a real ``MOVE_TO`` command
to :class:`NpcSystem`.  Its HUD exposes the animation lifecycle, sampled clip,
foot markers, and terminal receipt so the video is evidence of runtime control
rather than a direct mesh-frame replay.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

import cv2
import mujoco
import numpy as np

from stretch_mujoco.npc import CommandStatus, NpcCommand, NpcCommandKind, NpcRuntimeState
from stretch_mujoco.npc.system import NpcSystem


@dataclass(frozen=True)
class MoveTrace:
    """Observable boundaries of one marker-synchronised MOVE_TO command."""

    initial_position: tuple[float, float, float]
    first_foot_marker: str | None
    first_foot_marker_time: float | None
    first_movement_time: float | None
    stop_marker: str | None
    terminal_status: str | None
    terminal_reason: str | None
    terminal_time: float | None

    @property
    def marker_gated(self) -> bool:
        return (
            self.first_foot_marker_time is not None
            and self.first_movement_time is not None
            and self.first_movement_time >= self.first_foot_marker_time
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_position": list(self.initial_position),
            "first_foot_marker": self.first_foot_marker,
            "first_foot_marker_time": self.first_foot_marker_time,
            "first_movement_time": self.first_movement_time,
            "stop_marker": self.stop_marker,
            "terminal_status": self.terminal_status,
            "terminal_reason": self.terminal_reason,
            "terminal_time": self.terminal_time,
            "marker_gated": self.marker_gated,
        }


def run_move_trace(
    system: NpcSystem,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    npc_id: str,
    target_site: str,
    seconds: float,
    fps: int,
    on_frame: Callable[[NpcRuntimeState], None] | None = None,
) -> MoveTrace:
    """Run one MOVE_TO command and report when visual and root-motion boundaries occur."""
    if seconds <= 0 or fps <= 0:
        raise ValueError("seconds and fps must be positive")
    controller = system.controllers.get(npc_id)
    if controller is None:
        raise ValueError(f"Unknown NPC '{npc_id}'")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, target_site) < 0:
        raise ValueError(f"Unknown target site '{target_site}'")

    initial_position = cast(
        tuple[float, float, float],
        tuple(float(value) for value in data.mocap_pos[controller.binding.mocap_id]),
    )
    command = NpcCommand(
        "runtime_demo_move",
        0,
        npc_id,
        NpcCommandKind.MOVE_TO,
        {"site": target_site, "speed": 1.0, "progress_timeout": 2.0, "max_replans": 1},
        0.0,
    )
    receipt = system.submit(command)
    if receipt.status == CommandStatus.FAILED:
        raise RuntimeError(f"MOVE_TO was rejected: {receipt.reason}")

    first_foot_marker: str | None = None
    first_foot_marker_time: float | None = None
    first_movement_time: float | None = None
    terminal_status: str | None = None
    terminal_reason: str | None = None
    terminal_time: float | None = None
    frame_count = max(1, round(seconds * fps))
    for frame_index in range(frame_count):
        sim_time = frame_index / fps
        system.step(model, data, sim_time)
        mujoco.mj_forward(model, data)
        state = system.states(data, sim_time)[npc_id]
        if state.last_marker in {"left_foot", "right_foot"} and first_foot_marker is None:
            first_foot_marker = state.last_marker
            first_foot_marker_time = sim_time
        current_position = data.mocap_pos[controller.binding.mocap_id]
        if first_movement_time is None and not np.allclose(
            current_position, initial_position, atol=1e-7
        ):
            first_movement_time = sim_time
        if on_frame is not None:
            on_frame(state)
        if state.last_receipt is not None and state.last_receipt.status.terminal:
            terminal_status = state.last_receipt.status.value
            terminal_reason = state.last_receipt.reason
            terminal_time = state.last_receipt.finished_at
            break

    return MoveTrace(
        initial_position,
        first_foot_marker,
        first_foot_marker_time,
        first_movement_time,
        system.states(data)[npc_id].stop_marker,
        terminal_status,
        terminal_reason,
        terminal_time,
    )


def _configure_camera(camera: mujoco.MjvCamera, target: np.ndarray, yaw: float) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    camera.distance = 3.3
    camera.azimuth = math.degrees(yaw) + 55.0
    camera.elevation = -9.0


def _yaw_from_xmat(xmat: np.ndarray) -> float:
    return math.atan2(float(xmat[3]), float(xmat[0]))


def _annotate(rgb: np.ndarray, state: NpcRuntimeState) -> np.ndarray:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lines = (
        "Runtime command: MOVE_TO (not direct clip replay)",
        f"lifecycle={state.animation_lifecycle}  clip={state.resolved_clip}  "
        f"phase={state.clip_phase:.2f}",
        f"marker={state.last_marker or '-'}  stop_marker={state.stop_marker or '-'}  "
        f"replan={state.replan_attempt}",
    )
    cv2.rectangle(image, (18, 18), (880, 128), (20, 24, 32), thickness=-1)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (32, 47 + index * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 239, 244),
            1,
        )
    return image


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
        raise RuntimeError("Runtime demo rendering requires the 'ffmpeg' executable") from error


def render_runtime_demo(
    scene_path: Path,
    output_path: Path,
    *,
    npc_id: str = "employee_01",
    target_site: str = "meeting_human_stand_site",
    starting_position: tuple[float, float, float] = (-1.5, -0.5, 0.0),
    seconds: float = 8.0,
    fps: int = 12,
    width: int = 1280,
    height: int = 720,
) -> dict[str, object]:
    """Render a native-office, controller-driven locomotion trace and JSON evidence."""
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model, simulation_seed=7, scene_path=scene_path)
    controller = system.controllers.get(npc_id)
    if controller is None:
        raise ValueError(f"Unknown NPC '{npc_id}'")
    data.mocap_pos[controller.binding.mocap_id] = starting_position
    mujoco.mj_forward(model, data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    temporary_path: Path | None = None
    writer: cv2.VideoWriter | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, prefix=f".{output_path.stem}.", suffix=".mp4", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        writer = cv2.VideoWriter(
            str(temporary_path),
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for '{output_path}'")

        def render_frame(state: NpcRuntimeState) -> None:
            target = data.xpos[controller.binding.body_id].copy()
            target[2] += 0.85
            _configure_camera(camera, target, _yaw_from_xmat(data.xmat[controller.binding.body_id]))
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            writer.write(_annotate(renderer.render(), state))

        trace = run_move_trace(
            system,
            model,
            data,
            npc_id=npc_id,
            target_site=target_site,
            seconds=seconds,
            fps=fps,
            on_frame=render_frame,
        )
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
    if temporary_path is None:
        raise RuntimeError("Runtime demo writer was not initialized")
    try:
        _transcode_h264(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    report = {
        "scene": str(scene_path),
        "npc_id": npc_id,
        "target_site": target_site,
        "starting_position": list(starting_position),
        "lighting": "scene_native_only",
        "trace": trace.to_dict(),
        "passed": trace.marker_gated and trace.terminal_status == CommandStatus.SUCCEEDED.value,
        "output": str(output_path),
    }
    output_path.with_suffix(".runtime.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npc-id", default="employee_01")
    parser.add_argument("--target-site", default="meeting_human_stand_site")
    parser.add_argument(
        "--starting-position",
        type=float,
        nargs=3,
        default=(-1.5, -0.5, 0.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    report = render_runtime_demo(
        args.scene.resolve(),
        args.output.resolve(),
        npc_id=args.npc_id,
        target_site=args.target_site,
        starting_position=tuple(args.starting_position),
        seconds=args.seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(f"Runtime MP4 saved -> {args.output} (passed={report['passed']})")


if __name__ == "__main__":
    main()
