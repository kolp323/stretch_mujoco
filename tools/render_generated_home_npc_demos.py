#!/usr/bin/env python3
"""Render route, action, and NPC-to-NPC conversation demos in home scenes.

The videos exercise the real mocap NPC transport and animation clips.  They
intentionally do not issue Stretch commands or claim physical robot closure.
The source home scenes are composed with the three-NPC population at runtime;
the HSSD converted cache must therefore be present before rendering.
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
from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.npc.scene_site_config import load_scene_site_plans
from stretch_mujoco.humanoid.navigation import NavigationPathError, OfficeNavigationMesh


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "stretch_mujoco" / "models"
SCENES = MODELS / "assets" / "home_scenes"
GENERATED = MODELS / "generated_home_npc"
DEFAULT_OUTPUT = ROOT / "aaa_workspace" / "demo" / "npc_in_home"
SITE_CONFIG = SCENES / "npc_slot_overrides.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scene_ids() -> tuple[str, ...]:
    return tuple(path.stem.removesuffix("_npc") for path in sorted(SCENES.glob("home_*_npc.xml")))


def _parse_scene_ids(value: str) -> tuple[str, ...]:
    available = _scene_ids()
    if value == "all":
        return available
    requested = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"Unknown generated home scene(s): {', '.join(sorted(unknown))}")
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


def _yaw_toward(source: np.ndarray, target: np.ndarray) -> float:
    delta = target[:2] - source[:2]
    return math.atan2(float(delta[0]), float(-delta[1]))


def _set_world_site_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site: str,
    position: np.ndarray,
) -> np.ndarray:
    """Move a transparent worldbody site used as a runtime navigation target."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if site_id < 0:
        raise ValueError(f"missing_home_npc_site:{site}")
    if int(model.site_bodyid[site_id]) != 0:
        raise ValueError(f"home_npc_navigation_site_must_be_worldbody_attached:{site}")
    model.site_pos[site_id, :3] = np.asarray(position, dtype=float)[:3]
    mujoco.mj_forward(model, data)
    return data.site_xpos[site_id].copy()


def _annotate(rgb: Any, scene_id: str, activity: str) -> Any:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (12, 12), (image.shape[1] - 12, 84), (18, 25, 35), thickness=-1)
    cv2.putText(
        image,
        f"GENERATED HOME NPC | {scene_id}",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (239, 244, 248),
        2,
    )
    cv2.putText(
        image,
        f"{activity} | robot=mock (not controlled)",
        (28, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (150, 226, 177),
        1,
    )
    return image


def _site_position(model: mujoco.MjModel, data: mujoco.MjData, site: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if site_id < 0:
        raise ValueError(f"missing_home_npc_site:{site}")
    return data.site_xpos[site_id].copy()


def _ensure_reachable_walk_site(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: Any,
    site: str,
    *,
    include_mocap_obstacles: bool = True,
) -> np.ndarray:
    """Keep the presentation target on a reachable floor patch.

    HSSD homes can contain disconnected rooms.  The generated semantic walk
    cue is therefore screened against the actual collision grid at runtime;
    if its static point is unreachable, move only this transparent demo site
    to the nearest free patch around Alex.  No source XML or population data
    is changed.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if site_id < 0:
        raise ValueError(f"missing_home_npc_site:{site}")
    start = data.mocap_pos[controller.binding.mocap_id].copy()
    target = data.site_xpos[site_id].copy()
    root = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, controller.binding.body_id)
    mesh = OfficeNavigationMesh.from_model(
        model,
        data,
        exclude_body_roots=(() if root is None else (root,)),
        include_mocap_obstacles=include_mocap_obstacles,
    )
    try:
        mesh.plan(start, target)
        return target
    except NavigationPathError:
        pass
    for radius in np.arange(0.55, 2.05, 0.15):
        for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
            candidate = start.copy()
            candidate[:2] += radius * np.array((math.cos(angle), math.sin(angle)))
            try:
                mesh.plan(start, candidate)
            except NavigationPathError:
                continue
            if int(model.site_bodyid[site_id]) != 0:
                raise ValueError("home_npc_walk_site_must_be_worldbody_attached")
            model.site_pos[site_id, :2] = candidate[:2]
            mujoco.mj_forward(model, data)
            return data.site_xpos[site_id].copy()
    raise NavigationPathError(f"no_reachable_home_walk_site:{start[:2].tolist()}")


def _render_scene(
    scene_id: str, output_dir: Path, *, width: int, height: int, fps: int, speed: float
) -> dict[str, object]:
    try:
        plan = load_scene_site_plans(SITE_CONFIG)[scene_id]
    except KeyError as error:
        raise ValueError(f"home_demo_missing_scene_site_plan:{scene_id}") from error
    if plan.demo is None:
        raise ValueError(f"home_demo_missing_demo_contract:{scene_id}")
    npc_ids = tuple(entry.npc_id for entry in plan.roster)
    walker_id = plan.demo.walker_npc_id
    partner_id = plan.demo.conversation_partner_npc_id
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
    model = loaded.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    controllers = loaded.npc_system.controllers
    if set(controllers) != set(npc_ids):
        raise ValueError(f"home_demo_population_roster_mismatch:{scene_id}")
    walker = controllers[walker_id]
    partner = controllers[partner_id]
    # NPCs are moving collision proxies in this demo.  Their locomotion
    # planners must include the other NPC roots so a conversation approach
    # cannot visually pass through a stationary participant.
    for controller in controllers.values():
        controller.locomotion.dynamic_obstacles = True
    walk_site = plan.demo.activity_site
    target_position = _ensure_reachable_walk_site(model, data, walker, walk_site)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "office_floor")
    if floor_id < 0:
        raise ValueError("home_npc_navigation_floor_missing")
    output_path = output_dir / f"{scene_id}_route_action_talk.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    scene_option = mujoco.MjvOption()
    # Home XMLs include Stretch rangefinder sensors.  They are useful for
    # control, but their yellow debug rays obscure an NPC-only demo.
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = False
    # Use one free camera for the complete clip.  Its state is filtered below
    # so focus changes never reset the camera pose in a single frame.
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    initial_positions = [data.xpos[controllers[npc_id].binding.body_id] for npc_id in npc_ids]
    camera.lookat[:] = np.mean(initial_positions, axis=0)
    camera.lookat[2] += 0.75
    # HSSD homes have many interior partition walls but no authored cinematic
    # rail.  A near top-down, slightly wider free camera avoids tracking
    # through a wall as actors cross a doorway while retaining one continuous
    # view for the complete demo.
    camera.distance = 4.5
    camera.azimuth = 270.0
    camera.elevation = -82.0
    route_receipts: list[dict[str, object]] = []
    sim_time = 0.0
    sequence = {npc_id: 0 for npc_id in npc_ids}
    temporary_path: Path | None = None
    writer: cv2.VideoWriter | None = None

    def write_frame(
        activity: str,
        focus: Iterable[str] = (),
        *,
        overview: bool = False,
    ) -> None:
        del overview  # retained as a call-site compatibility argument
        positions = [
            data.xpos[controllers[npc_id].binding.body_id]
            for npc_id in (tuple(focus) or (walker_id,))
        ]
        desired_lookat = np.mean(positions, axis=0)
        desired_lookat[2] += 0.75
        # Exponential smoothing is frame-rate independent and keeps the
        # camera continuous when the focus set changes between phases.
        blend = 1.0 - math.exp(-5.0 / max(float(fps), 1.0))
        camera.lookat[:] += blend * (desired_lookat - camera.lookat)
        camera.distance += blend * (4.5 - camera.distance)
        camera.azimuth += blend * (270.0 - camera.azimuth)
        camera.elevation += blend * (-82.0 - camera.elevation)
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        writer.write(_annotate(renderer.render(), scene_id, activity))

    def run_one(
        command: NpcCommand,
        activity: str,
        *,
        max_frames: int = 900,
        focus: Iterable[str] = (),
    ) -> dict[str, object]:
        nonlocal sim_time
        accepted = loaded.npc_system.submit(command)
        if accepted.status == CommandStatus.FAILED:
            raise RuntimeError(f"command_rejected:{command.command_id}:{accepted.reason}")
        for frame_count in range(1, max_frames + 1):
            loaded.npc_system.step(model, data, sim_time)
            mujoco.mj_forward(model, data)
            write_frame(activity, focus)
            state = loaded.npc_system.states(data, sim_time)[command.npc_id]
            receipt = state.last_receipt
            if (
                receipt is not None
                and receipt.command_id == command.command_id
                and receipt.status.terminal
            ):
                report = {
                    "command_id": command.command_id,
                    "status": receipt.status.value,
                    "reason": receipt.reason,
                    "frames": frame_count,
                }
                if receipt.status is not CommandStatus.SUCCEEDED:
                    raise RuntimeError(f"command_failed:{command.command_id}:{receipt.reason}")
                return report
            sim_time += 1.0 / fps
        raise RuntimeError(f"command_timeout:{command.command_id}")

    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir, prefix=f".{output_path.stem}.", suffix=".mp4", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        writer = cv2.VideoWriter(
            str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"video_writer_unavailable:{output_path}")

        for _ in range(max(1, round(fps * 1.0))):
            write_frame(f"{len(npc_ids)} NPCs at configured spawn sites", npc_ids, overview=True)
        move_payload: dict[str, object] = {
            "site": walk_site,
            "speed": speed,
            "max_replans": 3,
        }
        move = NpcCommand(
            f"{scene_id}:{walker_id.removeprefix('npc_')}_walk",
            sequence[walker_id],
            walker_id,
            NpcCommandKind.MOVE_TO,
            move_payload,
            sim_time,
        )
        sequence[walker_id] += 1
        route_receipts.append(
            run_one(
                move,
                f"{walker_id.removeprefix('npc_')} walking to configured activity area",
            )
        )
        for _ in range(max(1, round(fps * 0.6))):
            write_frame(f"{walker_id.removeprefix('npc_')} arrived at activity area", npc_ids)

        action = NpcCommand(
            f"{scene_id}:{walker_id.removeprefix('npc_')}_wave",
            sequence[walker_id],
            walker_id,
            NpcCommandKind.PLAY_ANIMATION,
            {
                "clip": "gesture_wave",
                "duration": 3.0,
                "arrival_clip": "idle",
                "target_site": walk_site,
            },
            sim_time,
        )
        sequence[walker_id] += 1
        action_receipt = run_one(
            action, f"{walker_id.removeprefix('npc_')} waves at the activity area"
        )

        # Keep the transition continuous: use two transparent worldbody sites
        # as temporary navigation targets instead of teleporting either NPC
        # into the conversation pose.
        conversation_center = target_position.copy()
        walker_target_site = next(entry.site for entry in plan.roster if entry.npc_id == walker_id)
        partner_target_site = next(entry.site for entry in plan.roster if entry.npc_id == partner_id)
        walker_target = conversation_center.copy()
        partner_target = conversation_center.copy()
        # Keep final conversation roots farther apart than the inflated NPC
        # collision proxies.  The former 0.70 m root-to-root separation could
        # make Jordan's target unavailable in a narrow HSSD room after Alex
        # had arrived; 1.20 m still reads as an in-room conversation while
        # preserving a collision-free approach for both actors.
        walker_target[0] += 0.60
        partner_target[0] -= 0.60
        _set_world_site_position(model, data, walker_target_site, walker_target)
        _set_world_site_position(model, data, partner_target_site, partner_target)
        walker_target = _ensure_reachable_walk_site(
            model, data, walker, walker_target_site, include_mocap_obstacles=False
        )
        partner_target = _ensure_reachable_walk_site(
            model, data, partner, partner_target_site, include_mocap_obstacles=False
        )
        conversation_moves = (
            (
                walker,
                walker_id,
                walker_target_site,
                f"{walker_id.removeprefix('npc_')} walks to the conversation position",
            ),
            (
                partner,
                partner_id,
                partner_target_site,
                f"{partner_id.removeprefix('npc_')} walks to the conversation position",
            ),
        )
        for controller, npc_id, target_site, activity in conversation_moves:
            conversation_move = NpcCommand(
                f"{scene_id}:{npc_id.removeprefix('npc_')}_conversation_walk",
                sequence[npc_id],
                npc_id,
                NpcCommandKind.MOVE_TO,
                {"site": target_site, "speed": speed, "max_replans": 3},
                sim_time,
            )
            sequence[npc_id] += 1
            route_receipts.append(run_one(conversation_move, activity, focus=npc_ids))
        talk_receipts: dict[str, dict[str, object]] = {}
        walker_pose = data.mocap_pos[walker.binding.mocap_id, :2]
        partner_pose = data.mocap_pos[partner.binding.mocap_id, :2]
        conversation_distance = float(np.linalg.norm(walker_pose - partner_pose))
        if not 0.45 <= conversation_distance <= 0.95:
            # Some source homes contain disconnected rooms.  Keep the video
            # continuous and honest: do not teleport or claim a conversation
            # when navigation cannot bring both actors into the interaction
            # gate.  The route/action portion remains a valid demo.
            for command_id in (
                f"{scene_id}:{walker_id.removeprefix('npc_')}_talk",
                f"{scene_id}:{partner_id.removeprefix('npc_')}_talk",
            ):
                talk_receipts[command_id] = {
                    "command_id": command_id,
                    "status": "skipped",
                    "reason": "no_common_reachable_conversation_area",
                    "distance": conversation_distance,
                }
            for _ in range(max(1, round(fps * 1.0))):
                write_frame(
                    "NPCs continue separately (scene rooms are disconnected)",
                    npc_ids,
                )
        else:
            for controller, npc_id, target_site, target in (
                (walker, walker_id, walker_target_site, partner_target),
                (partner, partner_id, partner_target_site, walker_target),
            ):
                align = NpcCommand(
                    f"{scene_id}:{npc_id.removeprefix('npc_')}_conversation_align",
                    sequence[npc_id],
                    npc_id,
                    NpcCommandKind.ALIGN_TO,
                    {
                        "yaw": _yaw_toward(data.mocap_pos[controller.binding.mocap_id], target),
                        "target_site": target_site,
                    },
                    sim_time,
                )
                sequence[npc_id] += 1
                route_receipts.append(run_one(align, f"{npc_id} faces the conversation partner"))
            for _ in range(max(1, round(fps * 0.8))):
                write_frame(
                    "Configured NPCs take opposed conversation positions",
                    npc_ids,
                )

            walker_talk = NpcCommand(
                f"{scene_id}:{walker_id.removeprefix('npc_')}_talk",
                sequence[walker_id],
                walker_id,
                NpcCommandKind.PLAY_ANIMATION,
                {
                    "clip": "talk",
                    "duration": 3.0,
                    "interaction_target": partner_id,
                    "interaction_distance_min": 0.45,
                    "interaction_distance_max": 0.95,
                    "interaction_yaw_tolerance": 0.30,
                },
                sim_time,
            )
            partner_talk = NpcCommand(
                f"{scene_id}:{partner_id.removeprefix('npc_')}_talk",
                sequence[partner_id],
                partner_id,
                NpcCommandKind.PLAY_ANIMATION,
                {
                    "clip": "talk",
                    "duration": 3.0,
                    "interaction_target": walker_id,
                    "interaction_distance_min": 0.45,
                    "interaction_distance_max": 0.95,
                    "interaction_yaw_tolerance": 0.30,
                },
                sim_time,
            )
            sequence[walker_id] += 1
            sequence[partner_id] += 1
            accepted_walker = loaded.npc_system.submit(walker_talk)
            accepted_partner = loaded.npc_system.submit(partner_talk)
            if (
                accepted_walker.status == CommandStatus.FAILED
                or accepted_partner.status == CommandStatus.FAILED
            ):
                raise RuntimeError(
                    f"conversation_rejected:{accepted_walker.reason}:{accepted_partner.reason}"
                )
            for frame_count in range(1, max(1, round(fps * 6.0)) + 1):
                loaded.npc_system.step(model, data, sim_time)
                mujoco.mj_forward(model, data)
                write_frame(
                    "Configured NPCs face each other and talk",
                    npc_ids,
                )
                states = loaded.npc_system.states(data, sim_time)
                for command in (walker_talk, partner_talk):
                    if command.command_id in talk_receipts:
                        continue
                    receipt = states[command.npc_id].last_receipt
                    if (
                        receipt is not None
                        and receipt.command_id == command.command_id
                        and receipt.status.terminal
                    ):
                        talk_receipts[command.command_id] = {
                            "command_id": command.command_id,
                            "status": receipt.status.value,
                            "reason": receipt.reason,
                            "frames": frame_count,
                        }
                        if receipt.status is not CommandStatus.SUCCEEDED:
                            raise RuntimeError(
                                f"conversation_failed:{command.command_id}:{receipt.reason}"
                            )
                if len(talk_receipts) == 2:
                    break
                sim_time += 1.0 / fps
            else:
                raise RuntimeError(f"conversation_timeout:distance={conversation_distance:.3f}")
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
    if temporary_path is None:
        raise RuntimeError("temporary_video_missing")
    _transcode_h264(temporary_path, output_path)
    temporary_path.unlink(missing_ok=True)
    return {
        "scene_id": scene_id,
        "video": output_path.name,
        "population": str(population_path.relative_to(ROOT)),
        "semantic_world": str(semantic_path.relative_to(ROOT)),
        "scene_sha256": _sha256(loaded.composition.base_scene_path),
        "population_sha256": _sha256(population_path),
        "semantic_world_sha256": _sha256(semantic_path),
        "configured_npcs": sorted(population.npcs),
        "robot_mode": "mock",
        "robot_control_executed": False,
        "navigation_mode": "dynamic_npc_collision",
        "route_receipts": route_receipts,
        "action_receipt": action_receipt,
        "conversation_receipts": sorted(
            talk_receipts.values(), key=lambda item: str(item["command_id"])
        ),
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
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--speed", type=float, default=1.25)
    args = parser.parse_args()
    if min(args.width, args.height, args.fps, args.speed) <= 0:
        parser.error("width, height, fps, and speed must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for scene_id in _parse_scene_ids(args.scenes):
        try:
            records.append(
                _render_scene(
                    scene_id,
                    args.output_dir.resolve(),
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    speed=args.speed,
                )
            )
        except Exception as error:
            records.append(
                {"scene_id": scene_id, "passed": False, "error": f"{type(error).__name__}: {error}"}
            )
    manifest = {
        "purpose": "generated home NPC route, gesture action, and NPC conversation demonstration",
        "robot_mode": "mock",
        "robot_control_executed": False,
        "navigation_mode": "dynamic_npc_collision",
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# NPC demos in generated home scenes\n\n"
        "Each video composes the scene-configured NPC roster into one HSSD home scene and demonstrates "
        "a collision-aware walk, a `gesture_wave` action at the destination, and, "
        "when the source-room geometry permits, a gated configured-pair conversation. "
        "The home demo intentionally omits the sit animation; bed semantics and "
        "bed-edge interaction sites remain available for later scenarios. "
        "Conversation positioning is reached through continuous navigation and "
        "alignment; no NPC pose teleport is used. "
        "If a source home has disconnected rooms, the manifest records the "
        "conversation as `skipped` and the video keeps the NPCs on separate "
        "continuous routes. "
        "The named-site config selects the roster, spawn sites, and optional demo activity "
        "site; the demo route includes live NPC collision proxies and up to three "
        "replans for dynamic obstacle avoidance. Rangefinder debug rays are hidden. These are "
        "NPC-only mock demos; no Stretch control is executed.\n\n"
        "Regenerate after preparing the HSSD cache with:\n\n"
        "```bash\nMUJOCO_GL=egl .venv/bin/python tools/render_generated_home_npc_demos.py \\\n"
        "  --scenes all --output-dir aaa_workspace/demo/npc_in_home\n```\n\n"
        "`manifest.json` is the authoritative per-scene receipt. The source home "
        "mesh cache is ignored by Git and must be available locally.\n",
        encoding="utf-8",
    )
    failed = [record for record in records if not record["passed"]]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "passed": len(records) - len(failed),
                "failed": len(failed),
            }
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
