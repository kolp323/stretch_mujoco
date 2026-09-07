#!/usr/bin/env python3
"""Render a short, headless review video for a validated NPC population."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from stretch_mujoco.npc.naming import body_name, parse_frame_geom_name
from stretch_mujoco.npc.scene_builder import build_npc_scene
from stretch_mujoco.npc.schema import NpcPopulation


@dataclass(frozen=True)
class ReviewShot:
    npc_id: str
    view: str
    clip: str


def review_shots(npc_ids: tuple[str, ...]) -> tuple[ReviewShot, ...]:
    """Return front, side and seated acceptance shots for every configured NPC."""
    return tuple(
        ReviewShot(npc_id, view, clip)
        for npc_id in npc_ids
        for view, clip in (("front", "idle"), ("side", "idle"), ("sit", "sit"))
    )


def _add_presentation_environment(scene_path: Path) -> None:
    """Add temporary review-only lighting and floor to a standalone NPC scene."""
    root = ET.parse(scene_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Generated NPC scene has no worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 -2 4", "dir": "0 1 -2", "diffuse": "0.9 0.9 0.9"})
    ET.SubElement(
        worldbody,
        "geom",
        {"type": "plane", "size": "8 8 0.1", "rgba": "0.16 0.18 0.22 1"},
    )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(scene_path, encoding="unicode", xml_declaration=False)


def _shot_camera(camera: mujoco.MjvCamera, target: np.ndarray, shot: ReviewShot) -> str:
    """Set a stable front/side/seated acceptance camera for one NPC."""
    azimuth = {"front": 90.0, "side": 0.0, "sit": 90.0}[shot.view]
    target = target.copy()
    target[2] += 0.35
    camera.lookat[:] = target
    camera.distance = 1.0
    camera.azimuth = azimuth
    camera.elevation = -12.0
    return f"{shot.npc_id} — {shot.view} — {shot.clip}"


def _annotate(image: np.ndarray, label: str) -> np.ndarray:
    annotated = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(annotated, (24, 24), (600, 76), (20, 24, 32), thickness=-1)
    cv2.putText(annotated, label, (42, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 239, 244), 2)
    return annotated


def frame_visibility(
    geom_names: tuple[str, ...], npc_id: str, clip: str, frame_index: int
) -> dict[str, bool]:
    """Select one body-and-accessory frame without alpha overlap or flicker."""
    frames: dict[int, list[int]] = {}
    for geom_id, name in enumerate(geom_names):
        parsed = parse_frame_geom_name(name)
        if parsed is not None and parsed[0] == npc_id:
            if parsed[1] == clip:
                frames.setdefault(parsed[2], []).append(geom_id)
    if not frames:
        raise ValueError(f"NPC '{npc_id}' has no '{clip}' review frames")
    selected = sorted(frames)[frame_index % len(frames)]
    selected_ids = set(frames[selected])
    return {
        name: geom_id in selected_ids
        for geom_id, name in enumerate(geom_names)
        if (parsed := parse_frame_geom_name(name)) is not None and parsed[0] == npc_id
    }


def _sample_npc_clip(model: mujoco.MjModel, npc_id: str, clip: str, frame_index: int) -> None:
    names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        for geom_id in range(model.ngeom)
    )
    visible = frame_visibility(names, npc_id, clip, frame_index)
    for geom_id, name in enumerate(names):
        if name in visible:
            model.geom_rgba[geom_id, 3] = float(visible[name])


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
        raise RuntimeError("Persona video rendering requires the 'ffmpeg' executable") from error


def render_persona_video(
    population_path: Path, output_path: Path, *, seconds: float, fps: int, width: int, height: int
) -> int:
    """Build a validated population projection and render a review MP4."""
    if seconds <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("seconds, fps, width, and height must be positive")
    population = NpcPopulation.from_json(population_path)
    if len(population.npcs) < 2:
        raise ValueError("Persona review video requires at least two NPCs")
    frame_count = round(seconds * fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="npc-persona-review-") as temporary_directory:
        scene_path = Path(temporary_directory) / "npc_personas.xml"
        _add_presentation_environment(build_npc_scene(population_path, scene_path))
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        targets: dict[str, np.ndarray] = {}
        for npc_id in population.npcs:
            identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name(npc_id))
            if identifier < 0:
                raise RuntimeError(f"Generated scene is missing NPC body '{npc_id}'")
            target = data.xpos[identifier].copy()
            target[2] = 1.25
            targets[npc_id] = target
        plan = review_shots(tuple(population.npcs))
        renderer = mujoco.Renderer(model, width=width, height=height)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        temporary_video = output_path.with_suffix(".review.tmp.mp4")
        video_writer_fourcc = getattr(cv2, "VideoWriter_fourcc")
        writer = cv2.VideoWriter(
            str(temporary_video), video_writer_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            renderer.close()
            raise RuntimeError(f"Could not open video writer for '{output_path}'")
        try:
            for frame_index in range(frame_count):
                shot = plan[min(frame_index * len(plan) // frame_count, len(plan) - 1)]
                label = _shot_camera(camera, targets[shot.npc_id], shot)
                _sample_npc_clip(model, shot.npc_id, shot.clip, frame_index)
                renderer.update_scene(data, camera=camera)
                writer.write(_annotate(renderer.render(), label))
        finally:
            writer.release()
            renderer.close()
        try:
            _transcode_h264(temporary_video, output_path)
        finally:
            temporary_video.unlink(missing_ok=True)
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    frames = render_persona_video(
        args.population.resolve(),
        args.output.resolve(),
        seconds=args.seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(f"Persona MP4 saved -> {args.output} ({frames} frames)")


if __name__ == "__main__":
    main()
