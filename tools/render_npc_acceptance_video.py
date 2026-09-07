#!/usr/bin/env python3
"""Render a close, office-lit NPC animation acceptance video and JSON report.

The input MJCF is loaded unchanged: its own lights, headlight, skybox and
materials are the only visual environment used for review. The clip contains
front, side and seated shots so reviewers can inspect full-body animation,
textures and any frame-synchronised OBJ accessory geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict

import cv2
import mujoco
import numpy as np

from stretch_mujoco.humanoid.mesh_animator import SIT_ROOT_TO_SEAT_HEIGHT
from stretch_mujoco.npc.naming import candidate_body_names, parse_frame_geom_name


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "stretch_mujoco" / "models" / "office_scene.xml"


@dataclass(frozen=True)
class AcceptanceShot:
    """A required close inspection angle and the animation it exercises."""

    name: str
    clip: str
    azimuth_offset: float
    seated: bool = False


ACCEPTANCE_SHOTS = (
    AcceptanceShot("front", "walk", 90.0),
    AcceptanceShot("side", "walk", 0.0),
    AcceptanceShot("seated", "work", -45.0, seated=True),
)


FrameGroups = dict[str, dict[int, list[int]]]


def _discover_frames(
    model: mujoco.MjModel, npc_id: str, *, body_id: int | None = None
) -> FrameGroups:
    """Return all visual frame geoms for one NPC, including accessory slots."""
    frames: DefaultDict[str, DefaultDict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for geom_id in range(model.ngeom):
        if body_id is not None and int(model.geom_bodyid[geom_id]) != body_id:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        parsed = parse_frame_geom_name(name or "")
        if parsed is None or parsed[0] != npc_id:
            continue
        _, clip, frame, _ = parsed
        frames[clip][frame].append(geom_id)
    return {clip: dict(clip_frames) for clip, clip_frames in frames.items()}


def _body_id(model: mujoco.MjModel, npc_id: str) -> int:
    for name in candidate_body_names(npc_id):
        identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if identifier >= 0:
            return identifier
    raise ValueError(f"NPC '{npc_id}' has no body; tried {candidate_body_names(npc_id)}")


def _show_frame(
    model: mujoco.MjModel, frame_groups: FrameGroups, clip: str, frame: int
) -> list[int]:
    """Make exactly one mesh-sequence frame visible and return its geom IDs."""
    if clip not in frame_groups:
        raise ValueError(f"NPC has no '{clip}' clip; available: {', '.join(sorted(frame_groups))}")
    if frame not in frame_groups[clip]:
        raise ValueError(f"NPC clip '{clip}' has no frame {frame}")
    for clip_frames in frame_groups.values():
        for geom_ids in clip_frames.values():
            model.geom_rgba[geom_ids, 3] = 0.0
    active = frame_groups[clip][frame]
    model.geom_rgba[active, 3] = 1.0
    return active


def _yaw_from_xmat(xmat: np.ndarray) -> float:
    """Read the world yaw from MuJoCo's row-major rotation matrix."""
    return math.atan2(float(xmat[3]), float(xmat[0]))


def _configure_camera(
    camera: mujoco.MjvCamera,
    target: np.ndarray,
    yaw: float,
    shot: AcceptanceShot,
) -> None:
    """Frame a full body at close range, relative to the NPC's facing direction."""
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    camera.distance = 3.15 if not shot.seated else 3.4
    camera.azimuth = math.degrees(yaw) + shot.azimuth_offset
    camera.elevation = -8.0 if not shot.seated else -10.0


def _annotate(rgb: np.ndarray, shot: AcceptanceShot, clip_frame: int) -> np.ndarray:
    """Label the review angle without altering the rendered scene itself."""
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    label = f"NPC acceptance | {shot.name} | {shot.clip} frame {clip_frame:03d}"
    cv2.rectangle(image, (18, 18), (660, 62), (20, 24, 32), thickness=-1)
    cv2.putText(image, label, (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 239, 244), 1)
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
        raise RuntimeError("Acceptance video rendering requires the 'ffmpeg' executable") from error


def _frame_material_checks(
    model: mujoco.MjModel, frame_groups: FrameGroups
) -> dict[str, list[str]]:
    """Check frame transparency toggles and material identity before rendering."""
    transparency_failures: list[str] = []
    color_failures: list[str] = []
    for clip, clip_frames in sorted(frame_groups.items()):
        reference_by_slot: dict[str, tuple[int, tuple[float, float, float]]] = {}
        for frame, geom_ids in sorted(clip_frames.items()):
            if not geom_ids:
                transparency_failures.append(f"{clip}/{frame}: no visible geometry")
                continue
            for geom_id in geom_ids:
                material_id = int(model.geom_matid[geom_id])
                rgba = tuple(float(value) for value in model.geom_rgba[geom_id, :3])
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or str(geom_id)
                slot = parse_frame_geom_name(name)
                slot_name = slot[3] if slot is not None else name
                previous = reference_by_slot.setdefault(slot_name, (material_id, rgba))
                if previous != (material_id, rgba):
                    color_failures.append(
                        f"{clip}/{frame}/{slot_name}: material or RGB differs from prior frame"
                    )
    return {"transparency_failures": transparency_failures, "color_failures": color_failures}


def render_acceptance_video(
    scene_path: Path,
    output_path: Path,
    *,
    npc_id: str = "employee_01",
    standing_position: tuple[float, float, float] = (-1.5, -0.5, 0.0),
    sit_site: str = "chair_right_sit",
    seconds_per_shot: float = 2.0,
    fps: int = 20,
    width: int = 1280,
    height: int = 720,
) -> dict[str, object]:
    """Render required inspection shots and return their machine-readable checks.

    A scene-load failure is intentionally not hidden: MuJoCo reports missing
    mesh, texture and XML assets directly, which is the acceptance failure the
    caller needs to fix.
    """
    if seconds_per_shot <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("seconds_per_shot, fps, width and height must be positive")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_id = _body_id(model, npc_id)
    frame_groups = _discover_frames(model, npc_id, body_id=body_id)
    missing_clips = [shot.clip for shot in ACCEPTANCE_SHOTS if shot.clip not in frame_groups]
    if missing_clips:
        raise ValueError(
            f"NPC '{npc_id}' is missing required acceptance clips: {sorted(set(missing_clips))}"
        )

    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"NPC '{npc_id}' body is not mocap-controlled")
    sit_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sit_site)
    if sit_site_id < 0:
        raise ValueError(f"Required sit site '{sit_site}' is missing from '{scene_path}'")
    initial_position = data.mocap_pos[mocap_id].copy()
    initial_quaternion = data.mocap_quat[mocap_id].copy()
    frame_count = max(1, round(seconds_per_shot * fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    material_checks = _frame_material_checks(model, frame_groups)
    report: dict[str, object] = {
        "scene": str(scene_path),
        "npc_id": npc_id,
        "standing_position": list(standing_position),
        "lighting": "scene_native_only",
        "shots": [],
        **material_checks,
    }

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
            str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for '{output_path}'")

        for shot in ACCEPTANCE_SHOTS:
            if shot.seated:
                data.mocap_pos[mocap_id] = data.site_xpos[sit_site_id]
                data.mocap_pos[mocap_id, 2] -= SIT_ROOT_TO_SEAT_HEIGHT
                mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site_xmat[sit_site_id])
            else:
                data.mocap_pos[mocap_id] = standing_position
                data.mocap_quat[mocap_id] = initial_quaternion
            mujoco.mj_forward(model, data)
            target = data.xpos[body_id].copy()
            target[2] += 0.86 if not shot.seated else 0.7
            _configure_camera(camera, target, _yaw_from_xmat(data.xmat[body_id]), shot)
            visible_pixels: list[int] = []
            for video_frame in range(frame_count):
                clip_frames = sorted(frame_groups[shot.clip])
                selected_frame = clip_frames[video_frame % len(clip_frames)]
                active = _show_frame(model, frame_groups, shot.clip, selected_frame)
                if not np.allclose(model.geom_rgba[active, 3], 1.0):
                    material_checks["transparency_failures"].append(
                        f"{shot.clip}/{selected_frame}: selected geometry is not opaque"
                    )
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                rgb = renderer.render()
                writer.write(_annotate(rgb, shot, selected_frame))
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                segmentation = renderer.render()
                renderer.disable_segmentation_rendering()
                visible_pixels.append(int(np.isin(segmentation[..., 0], active).sum()))
            minimum_visible_pixels = int(width * height * 0.012)
            report["shots"].append(
                {
                    "name": shot.name,
                    "clip": shot.clip,
                    "seated": shot.seated,
                    "frames": frame_count,
                    "minimum_visible_npc_pixels": min(visible_pixels),
                    "minimum_required_npc_pixels": minimum_visible_pixels,
                    "occlusion_passed": min(visible_pixels) >= minimum_visible_pixels,
                }
            )
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
        data.mocap_pos[mocap_id] = initial_position
        data.mocap_quat[mocap_id] = initial_quaternion
    if temporary_path is None:
        raise RuntimeError("Acceptance video writer was not initialized")
    try:
        _transcode_h264(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    report["output"] = str(output_path)
    report["frames"] = frame_count * len(ACCEPTANCE_SHOTS)
    report["passed"] = (
        not material_checks["transparency_failures"]
        and not material_checks["color_failures"]
        and all(shot["occlusion_passed"] for shot in report["shots"])
    )
    report_path = output_path.with_suffix(".acceptance.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npc-id", default="employee_01")
    parser.add_argument(
        "--standing-position",
        type=float,
        nargs=3,
        default=(-1.5, -0.5, 0.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--sit-site", default="chair_right_sit")
    parser.add_argument("--seconds-per-shot", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    report = render_acceptance_video(
        args.scene.resolve(),
        args.output.resolve(),
        npc_id=args.npc_id,
        standing_position=tuple(args.standing_position),
        sit_site=args.sit_site,
        seconds_per_shot=args.seconds_per_shot,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(
        f"Acceptance MP4 saved -> {args.output} ({report['frames']} frames; passed={report['passed']})"
    )


if __name__ == "__main__":
    main()
