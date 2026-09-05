"""Offline 2-D and MuJoCo 3-D renderers for JSONL office snapshots."""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import cv2
import mujoco
import numpy as np

from stretch_mujoco.agents.mp4_recorder import OfficeMp4Recorder
from stretch_mujoco.npc.naming import parse_frame_geom_name


def format_clock(minute_of_day: float) -> str:
    minute = int(minute_of_day) % (24 * 60)
    return f"{minute // 60:02d}:{minute % 60:02d}"


def event_lines(snapshot: dict[str, Any]) -> list[str]:
    """Convert portable event dictionaries into compact subtitle lines."""
    lines = []
    for event in snapshot.get("events", [])[-6:]:
        details = event.get("details", {})
        detail = details.get("action", details.get("goal", ""))
        lines.append(
            f"[{format_clock(float(event.get('time', 0)))}] "
            f"{event.get('agent_id', '?')}: {event.get('event', '?')} {detail}".rstrip()
        )
    return lines


def annotate_3d_frame(rgb: np.ndarray, snapshot: dict[str, Any]) -> np.ndarray:
    """Add a readable inspection HUD without changing simulation state."""
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]
    panel_width = min(330, width // 3)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, height), (17, 25, 35), thickness=-1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(
        frame,
        "OFFICE NPC / 3D PLAYBACK",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 244, 248),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        format_clock(float(snapshot.get("minute_of_day", 0.0))),
        (24, 69),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        (99, 210, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "NPC STATUS",
        (24, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (163, 184, 203),
        1,
        cv2.LINE_AA,
    )
    y = 140
    for agent_id, state in sorted(snapshot["agents"].items()):
        action = str(state.get("action", "idle")).replace("_", " ").upper()
        location = str(state.get("location", "unknown")).replace("_", " ")
        target = state.get("target")
        display_name = agent_id.replace("employee_", "NPC ").replace("_", " ")
        cv2.rectangle(frame, (18, y - 22), (panel_width - 18, y + 39), (38, 55, 70), thickness=-1)
        cv2.putText(
            frame,
            display_name,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (245, 247, 250),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            action,
            (30, y + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (115, 222, 172),
            1,
            cv2.LINE_AA,
        )
        detail = f"at {location}" + (f" -> {str(target).replace('_', ' ')}" if target else "")
        cv2.putText(
            frame,
            detail,
            (30, y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (194, 207, 220),
            1,
            cv2.LINE_AA,
        )
        y += 76
    cv2.putText(
        frame,
        "ROBOT TASKS",
        (24, y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (163, 184, 203),
        1,
        cv2.LINE_AA,
    )
    robot_tasks = snapshot.get("robot_tasks", [])
    if not robot_tasks:
        cv2.putText(
            frame,
            "No robot task in this clip",
            (30, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (194, 207, 220),
            1,
            cv2.LINE_AA,
        )
    else:
        for task in robot_tasks[-2:]:
            status = str(task.get("status", "unknown")).upper()
            object_id = str(task.get("object", "item")).replace("_", " ")
            destination = str(task.get("destination", "destination")).replace("_", " ")
            cv2.putText(
                frame,
                f"{status}: {object_id} -> {destination}",
                (30, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (115, 222, 172) if status == "SUCCEEDED" else (255, 212, 112),
                1,
                cv2.LINE_AA,
            )
            y += 20
    y = height - 18
    for line in reversed(event_lines(snapshot)[-4:]):
        cv2.putText(
            frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (215, 225, 232), 1, cv2.LINE_AA
        )
        y -= 19
    return frame


def render_topdown_video(
    snapshots: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    fps: int = 10,
    scene_manifest: str | Path | None = None,
) -> int:
    """Render a 2-D office MP4 from snapshots without starting MuJoCo."""
    recorder = OfficeMp4Recorder(output_path, scene_manifest, fps=fps)
    recorder.start()
    try:
        for snapshot in snapshots:
            topdown_agents = {
                agent_id: {
                    **state,
                    "position": tuple(state.get("position", (0.0, 0.0))[:2]),
                }
                for agent_id, state in snapshot["agents"].items()
            }
            recorder.record_frame(
                float(snapshot.get("minute_of_day", 0.0)),
                topdown_agents,
                events=event_lines(snapshot),
            )
    finally:
        recorder.close()
    if recorder._frame_count == 0:
        raise ValueError("Cannot render a video from an empty snapshot recording")
    return recorder._frame_count


class MujocoSnapshotRenderer:
    """Render visual NPC state from snapshots into an offscreen MuJoCo image."""

    def __init__(self, scene_path: str | Path, *, width: int, height: int) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.scene_option)
        # The bundled Stretch model contains 360 lidar rangefinders.  They
        # are useful during interactive debugging but obscure a narrative
        # office recording with bright rays.
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = (0.0, 0.0, 0.8)
        if Path(scene_path).name.startswith("native_office_"):
            overview_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "office_overview"
            )
            if overview_id >= 0:
                self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.camera.fixedcamid = overview_id
            else:
                self.camera.distance = 9.5
                self.camera.azimuth = 45.0
                self.camera.elevation = -26.0
        else:
            self.camera.distance = 19.0
            self.camera.azimuth = 45.0
            self.camera.elevation = -32.0
        self._frames, self._npc_geoms = self._discover_npc_frames()

    def close(self) -> None:
        self.renderer.close()

    def _discover_npc_frames(
        self,
    ) -> tuple[dict[str, dict[str, dict[int, list[int]]]], dict[str, set[int]]]:
        frames: dict[str, dict[str, dict[int, list[int]]]] = {}
        npc_geoms: dict[str, set[int]] = {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name is None:
                continue
            parsed = parse_frame_geom_name(name)
            if parsed is None:
                continue
            agent_id, clip, frame, _ = parsed
            frames.setdefault(agent_id, {}).setdefault(clip, {}).setdefault(frame, []).append(
                geom_id
            )
            npc_geoms.setdefault(agent_id, set()).add(geom_id)
        return frames, npc_geoms

    @staticmethod
    def _yaw_quaternion(yaw: float) -> np.ndarray:
        return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])

    def _set_pose(self, agent_id: str, state: dict[str, Any]) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{agent_id}_body")
        if body_id < 0:
            return
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            return
        self.data.mocap_pos[mocap_id] = np.asarray(state["position"], dtype=float)
        self.data.mocap_quat[mocap_id] = self._yaw_quaternion(float(state.get("yaw", math.pi)))

    def _set_animation(self, agent_id: str, state: dict[str, Any], frame_index: int) -> None:
        clips = self._frames.get(agent_id)
        if not clips:
            return
        clip = state.get("animation", "idle")
        if clip not in clips:
            clip = "idle" if "idle" in clips else next(iter(clips))
        frame_numbers = sorted(clips[clip])
        phase = state.get("animation_phase")
        if phase is None:
            chosen_frame = frame_numbers[frame_index % len(frame_numbers)]
        else:
            phase_index = min(int(float(phase) * len(frame_numbers)), len(frame_numbers) - 1)
            chosen_frame = frame_numbers[phase_index]
        for geom_id in self._npc_geoms[agent_id]:
            self.model.geom_rgba[geom_id, 3] = 0.0
        for geom_id in clips[clip][chosen_frame]:
            self.model.geom_rgba[geom_id, 3] = 1.0

    def render_snapshot(self, snapshot: dict[str, Any], frame_index: int) -> np.ndarray:
        for agent_id, state in snapshot["agents"].items():
            self._set_pose(agent_id, state)
            self._set_animation(agent_id, state, frame_index)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        return self.renderer.render()


def render_mujoco_video(
    snapshots: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    scene_path: str | Path,
    fps: int = 10,
    width: int = 1280,
    height: int = 720,
) -> int:
    """Render a H.264 3-D MuJoCo MP4 that browser-based viewers can play."""
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("fps, width, and height must be positive")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=".mp4",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    writer = cv2.VideoWriter(
        str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not open video writer for '{destination}'")

    renderer: MujocoSnapshotRenderer | None = None
    frame_count = 0
    try:
        renderer = MujocoSnapshotRenderer(scene_path, width=width, height=height)
        for frame_count, snapshot in enumerate(snapshots, start=1):
            rgb = renderer.render_snapshot(snapshot, frame_count - 1)
            writer.write(annotate_3d_frame(rgb, snapshot))
    finally:
        if renderer is not None:
            renderer.close()
        writer.release()
    if frame_count == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("Cannot render a video from an empty snapshot recording")
    try:
        _transcode_h264(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return frame_count


def _transcode_h264(source: Path, destination: Path) -> None:
    """Convert OpenCV's intermediate video to a VS Code/browser-compatible MP4."""
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
        raise RuntimeError("3-D MP4 rendering requires the 'ffmpeg' executable") from error
