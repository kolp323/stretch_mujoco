#!/usr/bin/env python3
"""Run a source-compiled, auditable office/home NPC showcase."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np

from aaa_workspace.demo_new.npc_in_scenes.render_active_scene_npc_demo import (
    _audit_trace_against_profile,
    _portable_demo_base,
)
from stretch_mujoco.agents.actions import (
    ActionCommand,
    ActionExecution,
    ActionType,
    ConversationRequest,
    ExecutionStatus,
)
from stretch_mujoco.agents.conversation import DialogueAct, DialogueCandidate
from stretch_mujoco.agents.simulated_robot_executor import SimulatedRobotExecutor
from stretch_mujoco.agents.runtime import OfficeAgentRuntime
from stretch_mujoco.agents.simulation_bridge import create_mujoco_action_driver
from stretch_mujoco.npc import CommandStatus, NpcCommand, NpcCommandKind
from stretch_mujoco.npc.assets import NpcAssetManifest
from stretch_mujoco.npc.provenance import load_active_catalog, resolve_active_artifact, sha256_file, validate_active_catalog
from stretch_mujoco.npc.composition import _bind_population_semantics
from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.npc.scene_builder import build_npc_scene
from stretch_mujoco.npc.system import NpcSystem
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile
from stretch_mujoco.semantics import SemanticWorld


def load_scenario(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not payload.get("scenario_id"):
        raise ValueError("showcase_scenario_schema_invalid")
    if not isinstance(payload.get("phases"), list) or not payload["phases"]:
        raise ValueError("showcase_scenario_requires_phases")
    return payload


def validate_scenario(scenario: dict, profile_path: Path, population_path: Path) -> dict:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    population = json.loads(population_path.read_text(encoding="utf-8"))
    routes = {item["route_id"]: item for item in profile["routes"]}
    npcs = population["npcs"]
    capabilities = population.get("interaction_capabilities", {})
    traffic = population.get("traffic_policy", {})
    if set(capabilities) != {"conversation", "robot_handover"}:
        raise ValueError("showcase_population_interaction_projection_missing")
    if traffic.get("policy") != "sequential_route_reservation" or traffic.get("no_direct_fallback") is not True:
        raise ValueError("showcase_population_traffic_projection_missing")
    if any(capabilities[key].get("status") != "supported" for key in capabilities):
        raise ValueError("showcase_requires_supported_interaction_capabilities")
    receipts = []
    for index, phase in enumerate(scenario["phases"]):
        kind = phase.get("kind")
        participants = phase.get("participants", [])
        if kind == "move":
            route = routes.get(phase.get("route"))
            if phase.get("npc") not in npcs or route is None or "move_to" not in route["actions"]:
                raise ValueError(f"showcase_route_action_invalid:{phase.get('route')}")
        elif kind == "environment":
            if phase.get("npc") not in npcs or phase.get("clip") not in {"sit", "stand_up", "eat", "use_computer"}:
                raise ValueError(f"showcase_environment_action_invalid:{index}")
        elif kind in {"conversation", "robot_handover"}:
            if len(participants) != 2 or not all(item in npcs for item in participants):
                raise ValueError(f"showcase_participants_invalid:{index}")
        else:
            raise ValueError(f"showcase_phase_kind_invalid:{kind}")
        receipts.append({"phase": index, "kind": kind, "status": "declared", "reason": "awaiting_runtime_execution"})
    return {"profile_id": profile["profile_id"], "phase_receipts": receipts}


def _encode(source: Path, destination: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(destination)], check=True)


def _display_label(value: object) -> str:
    raw = str(value)
    parts = raw.split(":")
    if parts and parts[0].isdigit():
        raw = ":".join((f"Phase {int(parts[0]) + 1}", *parts[1:]))
    text = raw.replace("npc_", "").replace("_", " ").replace(":", "  |  ")
    return " ".join(text.split()).title()


def _put_fitted_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fitted = scale
    while fitted > 0.28:
        width = cv2.getTextSize(text, font, fitted, thickness)[0][0]
        if width <= max_width:
            break
        fitted -= 0.03
    cv2.putText(image, text, origin, font, fitted, color, thickness, cv2.LINE_AA)


class _Transport:
    def __init__(self, model, data, system, runtime, world, capture, observe, *, capture_stride: int):
        if capture_stride <= 0:
            raise ValueError("capture_stride_must_be_positive")
        self.model, self.data, self.system = model, data, system
        self.runtime, self.world, self.capture, self.observe = runtime, world, capture, observe
        self.capture_stride = capture_stride
        self.time = 0.0
        self._steps = 0
        self._receipts = []

    def pull_status(self):
        return type("Status", (), {"time": self.time})()

    def submit_npc_command(self, command):
        result = self.system.submit(command)
        self._receipts.extend(self.system.drain_receipts())
        if result.status is CommandStatus.FAILED:
            raise ValueError(result.reason or "npc_command_rejected")
        return command.command_id

    def pull_npc_receipts(self):
        receipts, self._receipts = tuple(self._receipts), []
        return receipts

    def cancel_npc_command(self, npc_id, command_id):
        self.system.submit(NpcCommand(f"cancel:{command_id}", 100000, npc_id, NpcCommandKind.CANCEL, {"command_id": command_id}, self.time))
        self._receipts.extend(self.system.drain_receipts())

    def advance(self, phase: str):
        dt = 1.0 / 12.0
        self.time += dt
        self.data.time = self.time
        self.system.step(self.model, self.data, self.time)
        mujoco.mj_forward(self.model, self.data)
        self.runtime.tick(dt, self.world.pose_snapshot(self.model, self.data))
        self._receipts.extend(self.system.drain_receipts())
        self._steps += 1
        self.observe()
        if self._steps % self.capture_stride == 0:
            self.capture(phase)


def _wait_action(transport, driver, execution, phase):
    result = driver.start(execution)
    if result.status is ExecutionStatus.FAILED:
        raise RuntimeError(f"action_start_failed:{result.error}")
    execution.driver_handle, execution.phase = result.handle, result.phase
    for _ in range(1800):
        transport.advance(phase)
        result = driver.poll(execution)
        if result.status is not ExecutionStatus.RUNNING:
            if result.status is not ExecutionStatus.SUCCEEDED:
                raise RuntimeError(f"action_terminal_failed:{result.error}")
            return result
    raise RuntimeError("action_timeout")


def _run_conversation(transport, runtime, participants, texts, phase):
    session_id = f"showcase-conversation-{phase}"
    request = ConversationRequest(session_id, tuple(participants), "daily coordination", timeout=120.0, max_turns=len(texts), semantic_snapshot=runtime.world.pose_snapshot(transport.model, transport.data))
    receipt = runtime.begin_conversation(request)
    if not receipt.accepted:
        raise RuntimeError(f"conversation_rejected:{receipt.error}")
    for _ in range(1800):
        transport.advance(phase)
        if runtime.conversation(session_id).phase.value == "waiting_for_turn":
            break
    else:
        session = runtime.conversation(session_id)
        raise RuntimeError(
            f"conversation_alignment_timeout:status={session.status.value}:"
            f"phase={session.phase.value}:error={session.error or session.failure_reason}:"
            f"driver_receipts={[(item.command_id, item.status.value, item.reason) for item in runtime.interaction_driver.command_receipts() if item.command_id.startswith(session_id)]}"
        )
    committed = []
    if len(participants) != 2 or len(texts) != 2:
        raise RuntimeError("conversation_requires_exactly_two_participants_and_two_turns")
    alignment_receipt = None
    for _ in range(1):
        alignment_receipt = runtime.conversation(session_id).phase.value
    if alignment_receipt != "waiting_for_turn":
        raise RuntimeError(f"conversation_alignment_not_committed:{alignment_receipt}")
    driver = runtime.interaction_driver
    alignment_receipt_ids = sorted(
        receipt.command_id
        for receipt in driver.command_receipts()
        if receipt.command_id.startswith(session_id + ":conversation_approach_")
        or receipt.command_id.startswith(session_id + ":conversation_align_")
    )
    if not alignment_receipt_ids:
        raise RuntimeError("conversation_physical_alignment_receipts_missing")
    committed = []
    for index, text in enumerate(texts):
        speaker = participants[index % 2]
        listener = participants[(index + 1) % 2]
        turn_id = f"{session_id}:turn:{index}"
        candidate = DialogueCandidate(f"{session_id}:candidate:{index}", session_id, turn_id, speaker, listener, DialogueAct.GREETING if index == 0 else DialogueAct.STATEMENT, text, runtime.elapsed_minutes)
        accepted = runtime.submit_dialogue_candidate(candidate)
        if not accepted.valid:
            raise RuntimeError(f"dialogue_rejected:{accepted.errors}")
        for _ in range(900):
            transport.advance(phase)
            turn = runtime.conversation(session_id).dialogue_turns[turn_id]
            if turn.status.value == "committed":
                committed.append(turn_id)
                break
        else:
            session = runtime.conversation(session_id)
            raise RuntimeError(
                f"dialogue_commit_timeout:phase={session.phase.value}:"
                f"status={session.status.value}:turn={session.dialogue_turns[turn_id].status.value}:"
                f"error={session.dialogue_turns[turn_id].error or session.error or session.failure_reason}:"
                f"driver_receipts={[(item.command_id, item.status.value, item.reason) for item in runtime.interaction_driver.command_receipts() if item.command_id.startswith(session_id)]}"
            )
    session = runtime.conversation(session_id)
    interaction_receipt_ids = sorted(
        receipt.command_id
        for receipt in driver.command_receipts()
        if receipt.command_id.startswith(session_id + ":")
        and receipt.command_id.endswith(":talk")
        and receipt.status is CommandStatus.SUCCEEDED
    )
    if len(interaction_receipt_ids) != len(committed):
        raise RuntimeError(
            "conversation_physical_interaction_receipts_missing:"
            f"{[(item.command_id, item.status.value, item.reason) for item in driver.command_receipts() if item.command_id.startswith(session_id)]}"
        )
    if session.status.value != "completed":
        raise RuntimeError(f"conversation_not_terminal_success:{session.status.value}:{session.error}")
    return {
        "session_id": session_id,
        "alignment_receipt": alignment_receipt,
        "alignment_receipt_ids": alignment_receipt_ids,
        "interaction_receipt_ids": interaction_receipt_ids,
        "committed_turns": committed,
        "committed_turn_receipts": [
            {
                "turn_id": item,
                "status": session.dialogue_turns[item].status.value,
                "execution_id": session.dialogue_turns[item].execution_id,
                "text": session.dialogue_turns[item].text,
            }
            for item in committed
        ],
        "transcript": [session.dialogue_turns[item].text for item in committed],
        "terminal_status": session.status.value,
    }


def _execution_receipt(result):
    return {
        "status": result.status.value,
        "phase": result.phase,
        "handle": result.handle,
        "error": result.error,
    }


def render_demo(
    catalog_path: Path,
    scenario_path: Path,
    output_path: Path,
    *,
    fps: int = 12,
    width: int = 640,
    height: int = 360,
    capture_stride: int = 24,
    frame_repeat: int = 2,
    min_duration: float = 15.0,
) -> dict:
    if width <= 0 or height <= 0 or fps <= 0 or frame_repeat <= 0 or min_duration <= 0:
        raise ValueError("showcase_video_parameters_must_be_positive")
    if abs(width / height - 16 / 9) > 0.02:
        raise ValueError("showcase_video_aspect_ratio_must_be_16_9")
    if output_path.exists():
        raise FileExistsError(f"refusing_to_overwrite:{output_path}")
    scenario = load_scenario(scenario_path)
    catalog = load_active_catalog(catalog_path)
    validate_active_catalog(catalog_path)
    record = catalog["scenes"].get(scenario["scene_id"])
    if not isinstance(record, dict):
        raise ValueError(f"showcase_scene_missing_from_active_catalog:{scenario['scene_id']}")
    artifacts = {
        key: resolve_active_artifact(catalog_path, scenario["scene_id"], key)
        for key in record
    }
    validation = validate_scenario(scenario, artifacts["trajectory_profile"], artifacts["population"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composition = output_path.parent / "composed" / f"{scenario['scene_id']}.xml"
    portable = _portable_demo_base(artifacts["scene"], composition.with_name(f"{scenario['scene_id']}_base.xml"))
    population = NpcPopulation.from_json(artifacts["population"])
    manifest = NpcAssetManifest.from_json(population.resolve_path(population.asset_manifest))
    # The portable base includes the authored showcase-only interaction sites.
    # Keep the population's relative paths valid by materializing a sibling
    # copy whose scene field points to that base before composition.
    runtime_population_payload = json.loads(artifacts["population"].read_text(encoding="utf-8"))
    source_population_root = artifacts["population"].parent
    runtime_population_payload["scene"] = portable.name
    for path_field in ("asset_manifest", "trajectory_profile"):
        value = runtime_population_payload.get(path_field)
        if isinstance(value, str) and not Path(value).is_absolute():
            runtime_population_payload[path_field] = str((source_population_root / value).resolve())
    if isinstance(runtime_population_payload.get("appearance_catalog"), str):
        value = runtime_population_payload["appearance_catalog"]
        if not Path(value).is_absolute():
            runtime_population_payload["appearance_catalog"] = str((source_population_root / value).resolve())
    runtime_population_path = composition.with_suffix(".population.json")
    runtime_population_path.write_text(
        json.dumps(runtime_population_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    build_npc_scene(runtime_population_path, composition, include_base_scene=True, _base_scene_path=portable)
    model = mujoco.MjModel.from_xml_path(str(composition))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    world = SemanticWorld.from_json(artifacts["semantic_v2"])
    # v2 uses stable semantic IDs (for example ``object.075_bread_022``),
    # while the action contract carries the source manifest IDs.  Preserve the
    # v2 graph but expose only these action-facing aliases at the boundary.
    legacy_world = SemanticWorld.from_v1_json_payload(
        json.loads(artifacts["semantic_v1"].read_text(encoding="utf-8"))
    )
    for object_id, semantic_object in legacy_world.objects.items():
        if object_id not in world.objects:
            world.objects[object_id] = semantic_object
    for relation in legacy_world.relations:
        if relation.subject in world.objects and relation.object in world.objects:
            world.relations.add(relation)
    # ``stretch_3`` and the capability object are execution-facing IDs. They
    # deliberately remain aliases during the v1→v2 schema migration so the
    # public robot-task contract does not leak semantic storage identifiers.
    for object_id in ("stretch_3", *(
        capability.get("object")
        for capability in runtime_population_payload.get("interaction_capabilities", {}).values()
        if isinstance(capability, dict) and isinstance(capability.get("object"), str)
    )):
        if object_id in legacy_world.objects:
            world.objects[object_id] = legacy_world.objects[object_id]
    for relation in legacy_world.relations:
        if relation.subject in world.objects and relation.object in world.objects:
            world.relations.add(relation)
    # Schema-v2 points are the runtime fact source; showcase capability sites
    # are explicitly authored production sites but are not semantic entities.
    # Register them before population schema validation instead of falling
    # back to the v1 semantic projection.
    for capability in runtime_population_payload.get("interaction_capabilities", {}).values():
        if not isinstance(capability, dict):
            continue
        capability_sites = list(capability.get("sites", []))
        if isinstance(capability.get("robot_approach"), str):
            capability_sites.append(capability["robot_approach"])
        for site in capability_sites:
            if isinstance(site, str) and site not in {point.site for point in world.interaction_points.values()}:
                owner = next(iter(world.objects), None)
                if owner is not None:
                    from stretch_mujoco.semantics import InteractionPoint, InteractionRole
                    world.interaction_points[f"showcase.{site}"] = InteractionPoint(
                        f"showcase.{site}", InteractionRole.HANDOVER, owner, site
                    )
    _bind_population_semantics(world, population)
    world.validate_model(model)
    system = NpcSystem.from_population(model, population, manifest, simulation_seed=20260915, scene_path=artifacts["scene"])
    # The scenario is the sole scheduler for this auditable showcase.  Disable
    # autonomous daily events so an unrelated generated snack event cannot
    # reserve/consume the registered handover object before its phase.
    runtime = OfficeAgentRuntime.from_json(world, runtime_population_path, auto_plan=False, daily_events=False)
    profile = NpcTrajectoryProfile.from_json(artifacts["trajectory_profile"])
    profile.validate_scene(artifacts["scene"])
    profile.preflight(model, data)
    traces = {npc_id: [] for npc_id in system.controllers}
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    # Keep the delivery render free of MuJoCo debug overlays. Tendons and flex
    # edges are especially noisy in composite humanoid scenes. Physics and
    # collision audits are unaffected by visualization flags.
    option.flags[:] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 1
    option.geomgroup[3] = 0
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 5.3 if scenario["scene_id"].startswith("office") else 5.8
    camera.azimuth = 42 if scenario["scene_id"].startswith("office") else 270
    camera.elevation = -34 if scenario["scene_id"].startswith("office") else -66
    temporary = output_path.with_name(f".{output_path.stem}.tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("showcase_video_writer_open_failed")
    written_frames = 0
    def observe():
        for npc_id, controller in system.controllers.items():
            traces[npc_id].append(data.mocap_pos[controller.binding.mocap_id].copy())

    def capture(phase, *, repeats=None):
        nonlocal written_frames
        center = np.mean([data.xpos[c.binding.body_id] for c in system.controllers.values()], axis=0)
        camera.lookat[:] = center
        camera.lookat[2] += 0.7
        renderer.update_scene(data, camera=camera, scene_option=option)
        image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        margin = max(12, round(width * 0.025))
        panel_height = max(62, round(height * 0.19))
        panel_top = height - margin - panel_height
        panel = image.copy()
        cv2.rectangle(panel, (margin, panel_top), (width - margin, height - margin), (10, 17, 26), -1)
        cv2.addWeighted(panel, 0.82, image, 0.18, 0, image)
        title_scale = max(0.46, width / 1250.0)
        subtitle_scale = max(0.42, width / 1450.0)
        max_text_width = width - 4 * margin
        _put_fitted_text(
            image,
            f"NPC SHOWCASE  |  {_display_label(scenario['scene_id'])}",
            (2 * margin, panel_top + round(panel_height * 0.42)),
            max_text_width,
            title_scale,
            (245, 248, 250),
            1,
        )
        _put_fitted_text(
            image,
            _display_label(phase),
            (2 * margin, panel_top + round(panel_height * 0.76)),
            max_text_width,
            subtitle_scale,
            (151, 231, 184),
            1,
        )
        for _ in range(frame_repeat if repeats is None else repeats):
            writer.write(image)
            written_frames += 1
    transport = _Transport(
        model, data, system, runtime, world, capture, observe, capture_stride=capture_stride
    )
    observe()
    capture("initial")
    # This showcase uses explicit capability stations and does not claim a
    # workstation/desk action.  Leaving the optional furniture binding out
    # avoids making an unrelated incomplete desk semantic block the demo.
    driver = create_mujoco_action_driver(
        transport,
        npc_ids=population.npcs,
        trajectory_profile=profile,
        interaction_templates=population.interaction_templates,
    )
    # The request is itself an embodied approach-plus-talk workflow. Its
    # timeout must cover the generated route and marker, not the short generic
    # action-recipe deadline used by interactive UI commands.
    driver.timeout_seconds = 600.0
    # The complete request workflow is guarded above by its successful
    # approach receipt.  The generic talking clip cannot use an unmodelled
    # robot gaze pose as a second acceptance gate in this headless simulator.
    driver.supported_actions = frozenset(
        action for action in driver.supported_actions if action is not ActionType.REQUEST_ROBOT
    )
    robot_capability = json.loads(artifacts["population"].read_text(encoding="utf-8"))["interaction_capabilities"]["robot_handover"]
    driver.robot_request_sites["stretch_3"] = robot_capability["robot_approach"]
    for anchor in profile.anchors.values():
        driver.location_sites.setdefault(anchor.site, anchor.site)
    runtime.action_driver = driver
    runtime.interaction_driver = driver
    handover = population.interaction_templates["handover"]
    handover_site = handover.roles["receiver"].site
    if scenario["scene_id"].startswith("home_"):
        receiver_id = scenario["phases"][-1]["participants"][1]
        spawn_id = {
            "npc_alex_chen": "point.home.alex_spawn",
            "npc_jordan_patell": "point.home.jordan_spawn",
            "npc_morgan_lee": "point.home.morgan_spawn",
        }[receiver_id]
        handover_site = profile.anchors[spawn_id].site
    robot = SimulatedRobotExecutor(
        npc_transport=transport,
        handover_sites={"stretch_3": handover_site},
        handover_yaws={"stretch_3": 3.141592653589793},
        timeout_seconds=180.0,
    )
    if scenario["scene_id"].startswith("home_"):
        # The handover receiver station is an authored component point.  The
        # compile-time topology contract is the route authority here; keep
        # stale parked mocap actors out of its local raster replan.
        for controller in system.controllers.values():
            controller.locomotion.dynamic_obstacles = False
    runtime.robot_task_driver = robot
    phase_details = []
    failure = None
    current_detail = None
    try:
        for index, phase in enumerate(scenario["phases"]):
            kind = phase["kind"]
            capture(f"Phase {index + 1}  |  {kind}  |  Starting", repeats=fps)
            detail = {"phase": index, "kind": kind, "status": "executed"}
            current_detail = detail
            if kind == "move":
                route = next(item for item in profile.routes if item.route_id == phase["route"])
                site = profile.anchors[route.destination].site
                execution = ActionExecution(
                    f"showcase:{index}:move",
                    ActionCommand(
                        phase["npc"],
                        ActionType.MOVE_TO,
                        site,
                        {"trajectory_route": route.route_id, "trajectory_source": route.source},
                    ),
                    ExecutionStatus.RUNNING,
                )
                # The scenario contract authorizes these declared business
                # routes.  Preserve their identity in the receipt even when
                # sequential prior phases leave a later home route with a
                # stale dynamic occupancy raster.
                driver.agent_locations[phase["npc"]] = route.source
                if scenario["scene_id"].startswith("home_"):
                    # Home component routes are validated at compile time;
                    # retaining prior NPC mocap bodies as planner obstacles
                    # can make a later sequential route falsely unavailable
                    # after its owner has already released its reservation.
                    system.controllers[phase["npc"]].locomotion.dynamic_obstacles = False
                result = _wait_action(transport, driver, execution, f"{index}:move:{phase['npc']}")
                detail["terminal_receipt"] = _execution_receipt(result)
                detail["passed"] = result.status is ExecutionStatus.SUCCEEDED
            elif kind == "environment":
                npc_id = phase["npc"]
                sequence = driver._sequences.get(npc_id, -1) + 1
                driver._sequences[npc_id] = sequence
                command = NpcCommand(f"showcase:{index}:environment", sequence, npc_id, NpcCommandKind.PLAY_ANIMATION, {"clip": phase["clip"], "duration": float(phase.get("duration", 2.0)), "arrival_clip": "idle"}, transport.time, transport.time + 30.0)
                transport.submit_npc_command(command)
                for _ in range(300):
                    transport.advance(f"{index}:environment:{phase['clip']}")
                    receipt = system.states(data, transport.time)[npc_id].last_receipt
                    if receipt is not None and receipt.command_id == command.command_id and receipt.status.terminal:
                        if receipt.status is not CommandStatus.SUCCEEDED:
                            raise RuntimeError(f"environment_failed:{receipt.reason}")
                        detail["terminal_receipt"] = receipt.to_dict()
                        detail["passed"] = receipt.status is CommandStatus.SUCCEEDED and receipt.status.terminal
                        break
                else:
                    raise RuntimeError("environment_timeout")
            elif kind == "conversation":
                detail.update(_run_conversation(transport, runtime, phase["participants"], phase["transcript"], f"{index}:conversation"))
                detail["passed"] = (
                    detail["terminal_status"] == "completed"
                    and len(detail["committed_turn_receipts"]) == 2
                    and all(item["status"] == "committed" and item["execution_id"] for item in detail["committed_turn_receipts"])
                    and detail["alignment_receipt"] == "waiting_for_turn"
                    and detail["alignment_receipt_ids"]
                    and len(detail["interaction_receipt_ids"]) == 2
                )
            else:
                giver, receiver = phase["participants"]
                capability = json.loads(artifacts["population"].read_text(encoding="utf-8"))["interaction_capabilities"]["robot_handover"]
                destination = "zone.snack" if scenario["scene_id"].startswith("office") else "zone.home"
                accepted = runtime.submit_action(ActionCommand(giver, ActionType.REQUEST_ROBOT, "stretch_3", {"task": "robot_to_npc_handover", "object": capability["object"], "destination": destination, "recipient": receiver}))
                if not accepted.valid:
                    raise RuntimeError(f"robot_request_rejected:{accepted.errors}")
                # The request action is already receipt-gated.  In generated
                # scenes the visible robot base is not a locomotion participant,
                # so its authored approach point is the physical acceptance
                # boundary rather than a second gaze-distance dependency.
                # Once its movement receipt succeeds, submit the robot task
                # through the runtime's normal semantic commit path.
                for _ in range(2400):
                    transport.advance(f"{index}:robot_request")
                    requester = runtime.agents[giver]
                    if requester.executor.status is ExecutionStatus.SUCCEEDED:
                        break
                    if requester.executor.status in {
                        ExecutionStatus.FAILED,
                        ExecutionStatus.CANCELLED,
                        ExecutionStatus.TIMED_OUT,
                    }:
                        raise RuntimeError(
                            f"robot_request_action_failed:{requester.executor.error}"
                        )
                else:
                    raise RuntimeError("robot_request_acceptance_timeout")
                for _ in range(2400):
                    transport.advance(f"{index}:robot_handover")
                    tasks = tuple(runtime.robot_tasks.values())
                    # REQUEST_ROBOT must first pass through the normal NPC
                    # action completion gate. Report the pre-task wait
                    # explicitly; never count it as handover progress.
                    if not tasks:
                        requester = runtime.agents[giver]
                        if requester.executor.status in {
                            ExecutionStatus.FAILED,
                            ExecutionStatus.CANCELLED,
                            ExecutionStatus.TIMED_OUT,
                        }:
                            raise RuntimeError(
                                f"robot_request_action_failed:{requester.executor.error}"
                            )
                        continue
                    if tasks and tasks[-1].status.value in {"succeeded", "failed"}:
                        task = tasks[-1]
                        if task.status.value != "succeeded":
                            raise RuntimeError(f"robot_task_failed:{task.error}")
                        detail["robot_task_id"] = task.task_id
                        detail["robot_receipt_ids"] = sorted(task.receipt_ids)
                        required = {
                            "task_terminal_success": task.status.value == "succeeded",
                            "robot_release_confirmed": task.robot_release_confirmed and any(item.endswith(":release_confirmed") for item in task.receipt_ids),
                            "npc_attachment_confirmed": any(item.endswith(":npc_attachment_confirmed") for item in task.receipt_ids),
                            "interaction_completed": any(item.endswith(":interaction_completed") for item in task.receipt_ids),
                            "simulated_mujoco_terminal_receipt": any(item.endswith(":succeeded") for item in task.receipt_ids),
                        }
                        detail["robot_evidence"] = required
                        detail["robot_receipts"] = {
                            receipt_id: receipt.evidence
                            for receipt_id, receipt in robot.receipts.items()
                            if receipt.task_id == task.task_id
                        }
                        detail["passed"] = all(required.values())
                        if not detail["passed"]:
                            raise RuntimeError("robot_handover_required_receipt_missing")
                        break
                else:
                    requester = runtime.agents[giver]
                    raise RuntimeError(
                        "robot_handover_timeout:"
                        f"request_status={requester.executor.status.value}:"
                        f"request_phase={requester.executor.phase}:"
                        f"request_error={requester.executor.error}"
                    )
            phase_details.append(detail)
        capture("Showcase Complete", repeats=2 * fps)
        minimum_frames = math.ceil(fps * min_duration)
        if written_frames < minimum_frames:
            capture("Showcase Complete", repeats=minimum_frames - written_frames)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        if current_detail is not None:
            current_detail["status"] = "failed"
            current_detail["passed"] = False
            current_detail["error"] = failure
            if current_detail.get("kind") == "robot_handover":
                current_detail["robot_debug"] = {
                    "tasks": {
                        task_id: {
                            "status": task.status.value,
                            "error": task.error,
                            "receipt_ids": sorted(task.receipt_ids),
                        }
                        for task_id, task in runtime.robot_tasks.items()
                    },
                    "stages": dict(robot._stages),
                    "sessions": {
                        task_id: {
                            "phase": bridge._handovers[task_id].session.phase,
                            "status": bridge._handovers[task_id].session.status.value,
                            "error": bridge._handovers[task_id].session.error,
                        }
                        for task_id, bridge in robot._bridges.items()
                    },
                }
            if not phase_details or phase_details[-1] is not current_detail:
                phase_details.append(current_detail)
    finally:
        writer.release()
        renderer.close()
    audits = {npc_id: _audit_trace_against_profile(model, data, system.controllers[npc_id], profile, positions) for npc_id, positions in traces.items()}
    overlap_violations = []
    npc_ids = tuple(traces)
    for left_index, left_id in enumerate(npc_ids):
        for right_id in npc_ids[left_index + 1 :]:
            for frame, (left, right) in enumerate(zip(traces[left_id], traces[right_id])):
                distance = float(np.linalg.norm(np.asarray(left)[:2] - np.asarray(right)[:2]))
                if distance < 2.0 * profile.agent_radius:
                    overlap_violations.append({"frame": frame, "npc_ids": [left_id, right_id], "distance_m": distance})
    for npc_id in npc_ids:
        own_overlaps = [item for item in overlap_violations if npc_id in item["npc_ids"]]
        audits[npc_id]["npc_npc_overlap_violations"] = own_overlaps
        audits[npc_id]["passed"] = audits[npc_id]["passed"] and not own_overlaps
    showcase_passed = (
        len(phase_details) == len(scenario["phases"])
        and all(item.get("passed") is True for item in phase_details)
        and all(item.get("passed") is True and not item.get("violations") and not item.get("npc_npc_overlap_violations") for item in audits.values())
    )
    if failure is not None or not showcase_passed:
        report = {
            "schema_version": 2,
            "generator": "render_active_scene_showcase.py",
            "build_id": catalog["build_id"],
            "active_catalog_sha256": sha256_file(catalog_path),
            "scene_id": scenario["scene_id"],
            "simulator_robot": True,
            "executor_kind": "simulated_mujoco",
            "output": str(output_path.resolve()),
            "phase_receipts": phase_details,
            "collision_trace_audit": audits,
            "showcase_passed": False,
            "passed": False,
            "error": failure or "showcase_acceptance_failed",
        }
        output_path.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        temporary.unlink(missing_ok=True)
        raise RuntimeError(report["error"])
    _encode(temporary, output_path)
    temporary.unlink(missing_ok=True)
    report = {"schema_version": 2, "generator": "render_active_scene_showcase.py", "build_id": catalog["build_id"], "active_catalog_sha256": sha256_file(catalog_path), "scene_id": scenario["scene_id"], "simulator_robot": True, "executor_kind": "simulated_mujoco", "output": str(output_path.resolve()), "video": {"width": width, "height": height, "fps": fps, "frame_count": written_frames, "duration_seconds": written_frames / fps}, "phase_receipts": phase_details, "collision_trace_audit": audits, "showcase_passed": True, "passed": True}
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--capture-stride", type=int, default=24)
    parser.add_argument("--frame-repeat", type=int, default=2)
    parser.add_argument("--min-duration", type=float, default=15.0)
    args = parser.parse_args()
    report = render_demo(
        args.catalog.resolve(), args.scenario.resolve(), args.output.resolve(),
        fps=args.fps, width=args.width, height=args.height, capture_stride=args.capture_stride,
        frame_repeat=args.frame_repeat, min_duration=args.min_duration,
    )
    print(json.dumps({"output": report["output"], "showcase_passed": report["showcase_passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
