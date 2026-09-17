#!/usr/bin/env python3
"""Render ten real production NPCs converging on distinct snack-area slots.

The generated base scene and population live below the requested demo output
directory.  They are disposable, reproducible composition inputs: this tool
never edits the authored office XML or the production roster.  Three roster
members receive concurrent, receipt-polled ``MOVE_TO`` commands; the remaining
seven stay at distinct, collision-free initial positions so the global view
can show the complete roster throughout the demonstration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from stretch_mujoco.agents.drivers import LocationSlotAllocator
from stretch_mujoco.npc import CommandStatus, NpcCommand, NpcCommandKind
from stretch_mujoco.npc.assets import NpcAssetManifest
from stretch_mujoco.npc.composition import compose_npc_scene
from stretch_mujoco.npc.naming import body_name
from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.npc.system import NpcSystem


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "stretch_mujoco" / "models"
PRODUCTION_POPULATION = MODELS / "office_population.production.example.json"
BASE_SCENE = MODELS / "assets" / "office_scenes" / "office_10_social_core_npc.xml"
DEFAULT_OUTPUT = ROOT / "aaa_workspace" / "demo" / "npc_area_slots"

# These table centres are stable authored scene locations.  Each initial NPC
# pose derives its yaw from one of them, rather than facing an arbitrary world
# axis in the top-down demo.
TABLE_FACING_TARGETS = {
    "work_bench": (-5.5, 0.0),
    "meeting_table": (4.5, 3.5),
    "lounge_table": (2.0, -2.5),
    "snack_counter": (7.5, -2.5),
}

# These positions were chosen on office_10's collision-inflated navigation
# grid.  They are intentionally spread across work, lounge, circulation, and
# perimeter space rather than using the three-NPC scene's limited spawn sites.
# The final field names the table that the stationary initial pose faces.
INITIAL_POSITIONS: tuple[tuple[str, float, float, str], ...] = (
    # Each mover approaches the snack area from a different open circulation
    # branch.  This avoids the central meeting table's shared corridor while
    # still exercising a three-slot semantic region.
    ("npc_alex_chen", 9.5, -4.0, "snack_counter"),
    ("npc_morgan_lee", 5.0, -4.0, "snack_counter"),
    ("npc_jordan_patell", 9.5, 0.0, "snack_counter"),
    ("npc_priya_narayanan", -9.0, -5.0, "work_bench"),
    ("npc_marco_silva", 9.0, 5.0, "meeting_table"),
    ("npc_olivia_bennett", 0.5, -4.5, "lounge_table"),
    ("npc_daniel_kim", 7.0, -5.0, "snack_counter"),
    ("npc_samira_haddad", 0.0, 1.5, "meeting_table"),
    ("npc_wei_zhang", -8.5, 0.0, "work_bench"),
    ("npc_lena_fischer", -6.5, -2.5, "work_bench"),
)
MOVING_NPCS = ("npc_alex_chen", "npc_morgan_lee", "npc_jordan_patell")
AREA_SLOTS = {
    "snack_01": "npc_snack_site",
    "snack_02": "npc_snack_slot_02_site",
    "snack_03": "npc_snack_slot_03_site",
}
TARGET_REGION = "snack_counter"


def _facing_yaw(origin: tuple[float, float], target: tuple[float, float]) -> float:
    """Return the NPC root yaw whose forward axis points from origin to target."""
    return math.atan2(target[0] - origin[0], origin[1] - target[1])


def _yaw_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _angle_delta(source: float, target: float) -> float:
    return (target - source + math.pi) % (2.0 * math.pi) - math.pi


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _write_demo_inputs(output_dir: Path) -> tuple[Path, Path]:
    """Create an output-local base scene and population with ten spawn sites."""
    inputs = output_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    derived_scene = inputs / "office_10_social_core_ten_npc_base.xml"
    population_path = inputs / "ten_npc_population.json"

    tree = ET.parse(BASE_SCENE)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    # The output-local copy retains the real scene's assets without copying
    # them.  Its root assets are made absolute so the included Stretch model
    # retains its own compiler assetdir (a root assetdir would override it
    # when MuJoCo flattens nested includes).
    compiler.attrib.pop("assetdir", None)
    for include in root.findall("include"):
        source = (BASE_SCENE.parent / include.attrib["file"]).resolve()
        include.set("file", str(source))
    assets = root.find("asset")
    if assets is not None:
        for asset in assets:
            source = asset.get("file")
            if source is not None and not Path(source).is_absolute():
                # Generated office XML inherits ``assetdir=..`` from its
                # Stretch include, placing office props under ``assets/``.
                asset.set("file", str((BASE_SCENE.parent.parent / source).resolve()))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"demo_source_scene_missing_worldbody:{BASE_SCENE}")
    existing_sites = {site.get("name"): site for site in worldbody.findall("site")}
    for npc_id, x, y, _ in INITIAL_POSITIONS:
        site = f"demo_spawn_{npc_id}_site"
        if site in existing_sites:
            raise ValueError(f"demo_spawn_site_conflict:{site}")
        ET.SubElement(
            worldbody,
            "site",
            {
                "name": site,
                "pos": f"{x:.3f} {y:.3f} 0.025",
                "size": "0.015",
                "rgba": "0 0 0 0",
            },
        )
    # The generic generated scene's snack sites have a zero yaw.  This demo
    # needs all arriving NPCs to face the counter, so orient only the
    # output-local scene copy toward the authored snack-counter centre.
    snack_target = TABLE_FACING_TARGETS["snack_counter"]
    for site in AREA_SLOTS.values():
        node = existing_sites.get(site)
        if node is None:
            raise ValueError(f"demo_area_slot_missing_from_base_scene:{site}")
        position = tuple(float(value) for value in node.attrib["pos"].split()[:2])
        node.set("euler", f"0 0 {_facing_yaw(position, snack_target):.9g}")
    ET.indent(tree, space="  ")
    tree.write(derived_scene, encoding="unicode", xml_declaration=False)
    derived_scene.write_text(derived_scene.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    payload = json.loads(PRODUCTION_POPULATION.read_text(encoding="utf-8"))
    payload["scene"] = derived_scene.name
    payload["asset_manifest"] = str(
        (PRODUCTION_POPULATION.parent / payload["asset_manifest"]).resolve()
    )
    if payload.get("appearance_catalog") is not None:
        payload["appearance_catalog"] = str(
            (PRODUCTION_POPULATION.parent / payload["appearance_catalog"]).resolve()
        )
    # This demonstration submits direct, unique slot targets; it intentionally
    # has no single-NPC trajectory profile or interaction-station contract to
    # inherit.  The production roster's office_scene-specific role stations do
    # not exist in this different generated office, and this demo performs no
    # dialogue or handover.
    payload.pop("trajectory_profile", None)
    payload.pop("interaction_templates", None)
    for npc_id, x, y, table_id in INITIAL_POSITIONS:
        yaw = _facing_yaw((x, y), TABLE_FACING_TARGETS[table_id])
        payload["npcs"][npc_id]["spawn"] = {
            "location": "demo_initial_distribution",
            "site": f"demo_spawn_{npc_id}_site",
            "yaw": yaw,
        }
    population_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return derived_scene, population_path


def _validate_initial_distribution(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    positions: dict[str, list[float]] = {}
    facing: dict[str, dict[str, object]] = {}
    for npc_id, x, y, table_id in INITIAL_POSITIONS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name(npc_id))
        if body_id < 0:
            raise ValueError(f"demo_composed_scene_missing_npc:{npc_id}")
        positions[npc_id] = [float(value) for value in data.xpos[body_id, :2]]
        expected_yaw = _facing_yaw((x, y), TABLE_FACING_TARGETS[table_id])
        actual_yaw = _yaw_from_quaternion(data.xquat[body_id])
        if abs(_angle_delta(actual_yaw, expected_yaw)) > 1e-5:
            raise ValueError(f"demo_initial_facing_mismatch:{npc_id}:{table_id}")
        facing[npc_id] = {"table": table_id, "yaw": actual_yaw}
    distances = [
        float(np.linalg.norm(np.subtract(positions[left], positions[right])))
        for index, left in enumerate(positions)
        for right in tuple(positions)[index + 1 :]
    ]
    minimum = min(distances)
    if minimum < 0.5:
        raise ValueError(f"demo_initial_positions_overlap:min_distance={minimum:.3f}")
    return {
        "positions_xy": positions,
        "minimum_pairwise_distance": minimum,
        "facing_tables": facing,
    }


def _annotate(image: np.ndarray, *, phase: str, elapsed: float, completed: int) -> np.ndarray:
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(result, (14, 14), (626, 94), (18, 25, 35), thickness=-1)
    cv2.putText(
        result,
        "OFFICE 10 | 10 production NPCs | global overview",
        (28, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (239, 244, 248),
        1,
    )
    cv2.putText(
        result,
        f"{phase} | 3 concurrent MOVE_TO commands | snack slots {completed}/3",
        (28, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (150, 226, 177),
        1,
    )
    cv2.putText(
        result,
        f"simulation {elapsed:.1f}s | targets are distinct physical slot sites",
        (28, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (190, 204, 219),
        1,
    )
    return result


def _write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        """# Ten-NPC concurrent snack-slot demo

`ten_npc_concurrent_snack_slots.mp4` is a real MuJoCo render of the ten
identities in `office_population.production.example.json`, composed into the
real `office_10_social_core_npc.xml` office.  All ten start at distinct,
collision-free standing locations facing their assigned work, meeting, lounge,
or snack table.  Alex Chen, Morgan Lee, and Jordan Patell
then receive concurrent NPC `MOVE_TO` commands and occupy the three different
configured `snack_counter` slot sites, each facing the snack counter on arrival.

`manifest.json` is the receipt: it records source hashes, the output-local
derived composition inputs, initial positions, slot allocation, accepted and
terminal command receipts, and the video parameters.  This demo exercises NPC
locomotion only; it does not command Stretch or claim a robot interaction
closure.  It independently records the minimum observed NPC root separation
and fails if any pair would overlap.

Recreate from the repository root:

```bash
MUJOCO_GL=egl .venv/bin/python tools/render_ten_npc_area_slots_demo.py
```
""",
        encoding="utf-8",
    )


def render_demo(
    output_dir: Path,
    *,
    width: int,
    height: int,
    fps: int,
    speed: float,
) -> dict[str, object]:
    if width <= 0 or height <= 0 or fps <= 0 or speed <= 0:
        raise ValueError("width, height, fps, and speed must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    derived_scene, population_path = _write_demo_inputs(output_dir)
    composition_path = output_dir / "composed" / "ten_npc_office_10.xml"
    composition = compose_npc_scene(population_path, composition_path, reuse=False)
    population = NpcPopulation.from_json(population_path)
    model = mujoco.MjModel.from_xml_path(str(composition.scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    distribution = _validate_initial_distribution(model, data)
    system = NpcSystem.from_population(
        model,
        population,
        # The composition routine already applies the same manifest validation.
        # NpcSystem needs this object only to bind each production animation graph.
        NpcAssetManifest.from_json(Path(population.asset_manifest)),
        simulation_seed=20260914,
        scene_path=derived_scene,
    )
    allocator = LocationSlotAllocator({TARGET_REGION: AREA_SLOTS})
    assignments = {npc_id: allocator.acquire(TARGET_REGION, npc_id) for npc_id in MOVING_NPCS}
    if len(set(assignments.values())) != len(MOVING_NPCS):
        raise ValueError("demo_area_slot_assignment_not_unique")
    for site in assignments.values():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) < 0:
            raise ValueError(f"demo_missing_area_slot_site:{site}")

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.4)
    camera.distance = 22.0
    camera.azimuth = 90.0
    camera.elevation = -89.0
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    # MuJoCo defaults some diagnostic overlays (notably rangefinder cones) to
    # visible.  They look like opaque yellow fans in a top-down EGL render and
    # obscure the actual office, so retain only ordinary static geometry and
    # texture rendering.
    scene_option.flags[:] = 0
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = True
    hidden_sequence_group = 5
    visible_sequence_group = 2
    scene_option.geomgroup[hidden_sequence_group] = False
    frame_geom_ids = {
        npc_id: tuple(
            geom_id
            for clip in controller.binding.frame_geom_ids.values()
            for frame in clip.values()
            for geom_id in frame.values()
        )
        for npc_id, controller in system.controllers.items()
    }
    # MeshSequenceBackend changes frame alpha, but MuJoCo's transparent mesh
    # draw path is not a reliable visibility gate in a global top-down view.
    # Use the same disabled sequence group convention as the production visual
    # acceptance renderer, while retaining the backend-selected active frame.
    for geom_ids in frame_geom_ids.values():
        model.geom_group[list(geom_ids)] = hidden_sequence_group
    output_path = output_dir / "ten_npc_concurrent_snack_slots.mp4"
    writer: cv2.VideoWriter | None = None
    temporary_path: Path | None = None
    accepted: dict[str, dict[str, object]] = {}
    terminal: dict[str, dict[str, object]] = {}
    sim_time = 0.0
    frame_count = 0
    body_ids = {
        npc_id: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name(npc_id))
        for npc_id in population.npcs
    }
    observed_minimum_separation = math.inf
    observed_closest_pair: tuple[str, str] | None = None

    def render_frame(phase: str) -> None:
        nonlocal frame_count, observed_minimum_separation, observed_closest_pair
        closest = min(
            (
                float(np.linalg.norm(data.xpos[left_id, :2] - data.xpos[right_id, :2])),
                left,
                right,
            )
            for index, (left, left_id) in enumerate(body_ids.items())
            for right, right_id in tuple(body_ids.items())[index + 1 :]
        )
        if closest[0] < observed_minimum_separation:
            observed_minimum_separation = closest[0]
            observed_closest_pair = (closest[1], closest[2])
        for npc_id, controller in system.controllers.items():
            active = tuple(controller.animation.backend._active)
            model.geom_group[list(frame_geom_ids[npc_id])] = hidden_sequence_group
            model.geom_group[list(active)] = visible_sequence_group
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        writer.write(
            _annotate(renderer.render(), phase=phase, elapsed=sim_time, completed=len(terminal))
        )
        frame_count += 1

    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir,
            prefix=f".{output_path.stem}.",
            suffix=".mp4",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        writer = cv2.VideoWriter(
            str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"demo_video_writer_unavailable:{output_path}")
        for _ in range(max(1, round(fps * 2.0))):
            system.step(model, data, sim_time)
            mujoco.mj_forward(model, data)
            render_frame("initial distributed roster")
            sim_time += 1.0 / fps

        for sequence, npc_id in enumerate(MOVING_NPCS):
            command = NpcCommand(
                command_id=f"ten-npc-snack:{npc_id}",
                sequence=sequence,
                npc_id=npc_id,
                kind=NpcCommandKind.MOVE_TO,
                payload={
                    "site": assignments[npc_id],
                    "speed": speed,
                    "progress_timeout": 6.0,
                    "max_replans": 12,
                },
                issued_at=sim_time,
                deadline=sim_time + 45.0,
            )
            receipt = system.submit(command)
            accepted[npc_id] = receipt.to_dict()
            if receipt.status is CommandStatus.FAILED:
                raise RuntimeError(f"demo_command_rejected:{npc_id}:{receipt.reason}")

        for _ in range(900):
            system.step(model, data, sim_time)
            mujoco.mj_forward(model, data)
            states = system.states(data, sim_time)
            for npc_id in MOVING_NPCS:
                receipt = states[npc_id].last_receipt
                if (
                    receipt is not None
                    and receipt.command_id == f"ten-npc-snack:{npc_id}"
                    and receipt.status.terminal
                ):
                    terminal[npc_id] = receipt.to_dict()
            render_frame("concurrent approach to snack area")
            if len(terminal) == len(MOVING_NPCS):
                break
            sim_time += 1.0 / fps
        else:
            raise RuntimeError("demo_concurrent_move_timeout")
        failed = {
            npc_id: receipt
            for npc_id, receipt in terminal.items()
            if receipt["status"] != CommandStatus.SUCCEEDED.value
        }
        if failed:
            raise RuntimeError(f"demo_concurrent_move_failed:{failed}")
        if observed_minimum_separation < 0.34:
            raise RuntimeError(
                "demo_npc_root_overlap:"
                f"minimum_pairwise_distance={observed_minimum_separation:.3f};"
                f"pair={observed_closest_pair}"
            )
        arrival_facing: dict[str, float] = {}
        for npc_id, site in assignments.items():
            body_yaw = _yaw_from_quaternion(data.xquat[body_ids[npc_id]])
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
            site_quaternion = np.empty(4)
            mujoco.mju_mat2Quat(site_quaternion, data.site_xmat[site_id])
            target_yaw = _yaw_from_quaternion(site_quaternion)
            if abs(_angle_delta(body_yaw, target_yaw)) > 0.03:
                raise RuntimeError(f"demo_arrival_facing_mismatch:{npc_id}:{site}")
            arrival_facing[npc_id] = body_yaw
        for _ in range(max(1, round(fps * 2.0))):
            system.step(model, data, sim_time)
            mujoco.mj_forward(model, data)
            render_frame("all three snack slots occupied")
            sim_time += 1.0 / fps
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
    if temporary_path is None:
        raise RuntimeError("demo_video_writer_not_initialized")
    try:
        _transcode_h264(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"demo_video_missing:{output_path}")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "demo": "ten_npc_concurrent_snack_slots",
        "scene": str(BASE_SCENE),
        "production_population": str(PRODUCTION_POPULATION),
        "sources": {
            "scene_sha256": _sha256(BASE_SCENE),
            "population_sha256": _sha256(PRODUCTION_POPULATION),
            "derived_scene": str(derived_scene),
            "derived_scene_sha256": _sha256(derived_scene),
            "population_input": str(population_path),
            "population_input_sha256": _sha256(population_path),
            "composition": str(composition.scene_path),
            "composition_receipt": str(composition.receipt_path),
        },
        "population": {"npc_count": len(population.npcs), "npc_ids": list(population.npcs)},
        "initial_distribution": distribution,
        "observed_minimum_pairwise_root_separation": observed_minimum_separation,
        "observed_closest_pair": observed_closest_pair,
        "target": {
            "semantic_region": TARGET_REGION,
            "moving_npc_ids": list(MOVING_NPCS),
            "slot_assignments": assignments,
            "allocator_owners": allocator.snapshot(),
            "arrival_yaws_facing_snack_counter": arrival_facing,
        },
        "receipts": {"accepted": accepted, "terminal": terminal},
        "render": {
            "output": str(output_path),
            "frames": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "camera": "global_free_top_down",
        },
        "robot_control_executed": False,
        "npc_dynamic_obstacle_replanning": True,
        "passed": len(population.npcs) == 10
        and len(assignments) == 3
        and len(set(assignments.values())) == 3
        and all(
            receipt["status"] == CommandStatus.SUCCEEDED.value for receipt in terminal.values()
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_readme(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--speed", type=float, default=1.6)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        result = render_demo(
            args.output_dir.resolve(),
            width=args.width,
            height=args.height,
            fps=args.fps,
            speed=args.speed,
        )
    except Exception as error:
        print(f"ten_npc_area_slots_demo_failed:{error}", file=sys.stderr)
        raise
    print(json.dumps({"output": result["render"]["output"], "passed": result["passed"]}))


if __name__ == "__main__":
    main()
