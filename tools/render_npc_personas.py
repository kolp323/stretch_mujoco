#!/usr/bin/env python3
"""Render a short, headless review video for a validated NPC population."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np

from stretch_mujoco.npc.naming import body_name, parse_frame_geom_name
from stretch_mujoco.npc.scene_builder import build_npc_scene
from stretch_mujoco.npc.schema import NpcPopulation


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


def _shot_camera(
    camera: mujoco.MjvCamera,
    targets: list[np.ndarray],
    frame_index: int,
    frame_count: int,
) -> tuple[int, str]:
    """Select two close identity shots followed by an overview shot."""
    fraction = frame_index / max(frame_count - 1, 1)
    shot, label = 0, "Alex Chen — OBJ cap follows head"
    distance, azimuth = 1.0, 90.0
    target = targets[shot].copy()
    target[2] += 0.35
    camera.lookat[:] = target
    camera.distance = distance
    camera.azimuth = azimuth + 8.0 * np.sin(fraction * np.pi * 4.0)
    camera.elevation = -12.0
    return shot, label


def _annotate(image: np.ndarray, label: str) -> np.ndarray:
    annotated = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(annotated, (24, 24), (600, 76), (20, 24, 32), thickness=-1)
    cv2.putText(annotated, label, (42, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 239, 244), 2)
    return annotated


def _sample_primary_clip(
    model: mujoco.MjModel, npc_id: str, frame_index: int, frame_count: int
) -> str:
    clips = ("idle", "walk", "sit")
    clip = clips[min(frame_index * len(clips) // frame_count, len(clips) - 1)]
    frames: dict[int, list[int]] = {}
    all_ids: list[int] = []
    for geom_id in range(model.ngeom):
        parsed = parse_frame_geom_name(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        )
        if parsed is not None and parsed[0] == npc_id:
            all_ids.append(geom_id)
            if parsed[1] == clip:
                frames.setdefault(parsed[2], []).append(geom_id)
    for geom_id in all_ids:
        model.geom_rgba[geom_id, 3] = 0.0
    selected = sorted(frames)[frame_index % len(frames)]
    for geom_id in frames[selected]:
        model.geom_rgba[geom_id, 3] = 1.0
    return clip


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
        targets: list[np.ndarray] = []
        for npc_id in population.npcs:
            identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name(npc_id))
            if identifier < 0:
                raise RuntimeError(f"Generated scene is missing NPC body '{npc_id}'")
            target = data.xpos[identifier].copy()
            target[2] = 1.25
            targets.append(target)
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
                _, label = _shot_camera(camera, targets, frame_index, frame_count)
                clip = _sample_primary_clip(
                    model, next(iter(population.npcs)), frame_index, frame_count
                )
                label = f"{label} — {clip}"
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
