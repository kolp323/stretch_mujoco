#!/usr/bin/env python3
"""Render a receipt-gated office NPC walk, sit/stand, and conversation demo.

The renderer uses the in-process MuJoCo NPC runtime rather than directly
placing mesh frames.  Every displayed stage must therefore end in a successful
protocol receipt.  Stretch is visible as scene furniture only; no robot control
is requested.  MuJoCo diagnostic visibility (especially rangefinder rays) is
explicitly disabled for presentation output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable

import cv2
import mujoco
import numpy as np

from stretch_mujoco.npc import CommandStatus, NpcCommand, NpcCommandKind
from stretch_mujoco.npc.composition import load_composed_npc_runtime


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POPULATION = ROOT / "aaa_workspace/demo/new_demo/production_two_npc_clear_population.json"
DEFAULT_OUTPUT = ROOT / "aaa_workspace/demo/npc_regenerated_acceptance/01_office_walk_sit_talk.mp4"
ALEX = "npc_alex_chen"
MORGAN = "npc_morgan_lee"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _transcode_h264(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
            str(destination),
        ],
        check=True,
    )


def _annotate(rgb: np.ndarray, stage: str) -> np.ndarray:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (12, 12), (image.shape[1] - 12, 84), (18, 25, 35), -1)
    cv2.putText(image, "OFFICE NPC | RECEIPT-GATED DEMO", (28, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (239, 244, 248), 2)
    cv2.putText(image, f"{stage} | robot=mock (not controlled)", (28, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (150, 226, 177), 1)
    return image


def _presentation_option() -> mujoco.MjvOption:
    """Return a deliberately clean render option, with no sensor/debug rays."""
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = False
    return option


def _camera_frame(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    controllers: dict[str, Any],
    focus: Iterable[str],
    option: mujoco.MjvOption,
) -> np.ndarray:
    positions = [data.xpos[controllers[npc_id].binding.body_id] for npc_id in focus]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.mean(positions, axis=0)
    camera.lookat[2] += 0.72
    # Social role sites are separated by the meeting table.  A steep, wider
    # view avoids hiding either participant behind its tabletop.
    camera.distance = 5.2 if len(positions) > 1 else 3.4
    camera.azimuth = 52.0
    camera.elevation = -72.0 if len(positions) > 1 else -38.0
    renderer.update_scene(data, camera=camera, scene_option=option)
    return renderer.render()


def render(output: Path, population_path: Path, *, width: int, height: int, fps: int) -> dict[str, object]:
    """Execute the demonstrated protocol commands and write an MP4 plus receipt JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    composed_path = output.parent / "composed" / "office_walk_sit_talk.xml"
    loaded = load_composed_npc_runtime(population_path, composed_path, simulation_seed=20260914)
    model, data, system = loaded.model, mujoco.MjData(loaded.model), loaded.npc_system
    mujoco.mj_forward(model, data)
    controllers = system.controllers
    if not {ALEX, MORGAN} <= set(controllers):
        raise ValueError("demo_population_requires_alex_and_morgan")
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, width=width, height=height)
    option = _presentation_option()
    reports: list[dict[str, object]] = []
    sequence = {ALEX: 0, MORGAN: 0}
    sim_time = 0.0
    writer: cv2.VideoWriter | None = None
    temporary_path: Path | None = None

    def write_frame(stage: str, focus: Iterable[str]) -> None:
        assert writer is not None
        writer.write(_annotate(_camera_frame(renderer, data, controllers, focus, option), stage))

    def run_one(npc_id: str, kind: NpcCommandKind, payload: dict[str, object], stage: str,
                focus: Iterable[str] = (MORGAN,)) -> None:
        nonlocal sim_time
        command = NpcCommand(f"office_demo:{npc_id}:{sequence[npc_id]}", sequence[npc_id], npc_id,
                             kind, payload, sim_time)
        sequence[npc_id] += 1
        accepted = system.submit(command)
        if accepted.status is CommandStatus.FAILED:
            raise RuntimeError(f"command_rejected:{stage}:{accepted.reason}")
        for frame in range(1, fps * 90 + 1):
            system.step(model, data, sim_time)
            mujoco.mj_forward(model, data)
            write_frame(stage, focus)
            receipt = system.states(data, sim_time)[npc_id].last_receipt
            if receipt is not None and receipt.command_id == command.command_id and receipt.status.terminal:
                reports.append({"stage": stage, "command_id": command.command_id,
                                "status": receipt.status.value, "reason": receipt.reason, "frames": frame})
                if receipt.status is not CommandStatus.SUCCEEDED:
                    raise RuntimeError(f"command_failed:{stage}:{receipt.reason}")
                return
            sim_time += 1.0 / fps
        raise RuntimeError(f"command_timeout:{stage}")

    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.stem}.", suffix=".mp4", delete=False) as handle:
            temporary_path = Path(handle.name)
        writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"video_writer_unavailable:{output}")
        for _ in range(max(1, fps // 2)):
            write_frame("Alex and Morgan at distinct office locations", (ALEX, MORGAN))
        # This is the repaired chair workflow: collision-safe approach, then
        # only the declared ingress-to-seat segment, alignment, and marker-gated sit.
        run_one(MORGAN, NpcCommandKind.MOVE_TO, {
            "site": "chair_right_sit", "navigation_site": "chair_right_approach_site",
            "allow_final_ingress": True, "max_replans": 1,
        }, "Morgan walks along the chair approach route")
        run_one(MORGAN, NpcCommandKind.ALIGN_TO, {"yaw": math.pi, "target_site": "chair_right_sit"},
                "Morgan aligns with the chair")
        run_one(MORGAN, NpcCommandKind.PLAY_ANIMATION,
                {"clip": "sit", "completion_marker": "seated", "target_site": "chair_right_sit"},
                "Morgan sits (seated marker reached)")
        for _ in range(max(1, fps)):
            write_frame("Morgan remains seated at the chair", (MORGAN,))
        run_one(MORGAN, NpcCommandKind.PLAY_ANIMATION,
                {"clip": "stand_up", "completion_marker": "standing", "arrival_clip": "idle",
                 "target_site": "chair_right_sit"}, "Morgan stands (standing marker reached)")
        for _ in range(max(1, fps // 2)):
            write_frame("Morgan is standing after the chair transition", (MORGAN,))
        # Use the configured, mutually-facing conversation role sites.  Each
        # route has its own physical receipt before the social clips are accepted.
        run_one(ALEX, NpcCommandKind.MOVE_TO, {"site": "meeting_conversation_alex_site", "max_replans": 1},
                "Alex walks to the meeting conversation position", (ALEX, MORGAN))
        run_one(MORGAN, NpcCommandKind.MOVE_TO, {"site": "meeting_conversation_morgan_site", "max_replans": 1},
                "Morgan walks to the meeting conversation position", (ALEX, MORGAN))
        for npc_id, target_id in ((ALEX, MORGAN), (MORGAN, ALEX)):
            run_one(npc_id, NpcCommandKind.PLAY_ANIMATION, {
                "clip": "talk", "completion_marker": "talk_cycle", "arrival_clip": "idle",
                "interaction_target": target_id, "interaction_distance_min": 0.45,
                "interaction_distance_max": 0.95, "interaction_yaw_tolerance": 0.30,
            }, f"{npc_id.removeprefix('npc_').replace('_', ' ').title()} talks face-to-face", (ALEX, MORGAN))
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
    if temporary_path is None:
        raise RuntimeError("temporary_video_missing")
    try:
        _transcode_h264(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    report = {
        "purpose": "receipt-gated office NPC walk, sit/stand, and conversation demonstration",
        "video": output.name, "robot_mode": "mock", "robot_control_executed": False,
        "population": str(population_path.relative_to(ROOT)), "population_sha256": _sha256(population_path),
        "scene": str(loaded.composition.base_scene_path.relative_to(ROOT)),
        "scene_sha256": _sha256(loaded.composition.base_scene_path),
        "diagnostic_visuals": {"rangefinder": False, "contact": False, "constraint": False},
        "receipts": reports, "passed": True,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if min(args.width, args.height, args.fps) <= 0:
        parser.error("width, height, and fps must be positive")
    report = render(args.output.resolve(), args.population.resolve(), width=args.width, height=args.height, fps=args.fps)
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))


if __name__ == "__main__":
    main()
