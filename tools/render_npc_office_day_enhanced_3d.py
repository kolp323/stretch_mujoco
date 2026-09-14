#!/usr/bin/env python3
"""Render a control sequence; business commands are lowered only by the action driver."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
import cv2
from stretch_mujoco.agents.control_sequences import (
    ControlSequenceError,
    SequenceCompiler,
    SequenceExecutor,
    StrictSequenceLoader,
)
from stretch_mujoco.agents.control_sequences.contracts import validate_llm_contract
from stretch_mujoco.agents.control_sequences.preflight import discover_capabilities
from stretch_mujoco.agents.control_sequences.sources import LlmPlanSource
from stretch_mujoco.agents.simulation_bridge import create_mujoco_action_driver
from stretch_mujoco.agents.renderer_transport import ServerBackedRendererTransport
from stretch_mujoco.agents.runtime import OfficeAgentRuntime
from stretch_mujoco.agents.mock_robot import MockRobotExecutor
from stretch_mujoco.agents.robot_task_driver import RobotTaskExecutor
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile
from stretch_mujoco.semantics import SemanticWorld
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

ROOT = Path(__file__).resolve().parents[1]
WIDTH = int(os.environ.get("CONTROL_RENDER_WIDTH", "1280"))
HEIGHT = int(os.environ.get("CONTROL_RENDER_HEIGHT", "720"))
FPS = int(os.environ.get("CONTROL_RENDER_FPS", "60"))
POLL_INTERVAL = float(os.environ.get("CONTROL_RENDER_POLL_INTERVAL", "0.05"))
POST_ROLL_SECONDS = float(os.environ.get("CONTROL_RENDER_POST_ROLL_SECONDS", "1.5"))
ROBOT_MOTION_SPEED = float(os.environ.get("CONTROL_ROBOT_MOTION_SPEED", "4.0"))
MOCK_ROBOT_DELAY_MINUTES = float(
    os.environ.get("CONTROL_MOCK_ROBOT_DELAY_MINUTES", "0.5")
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _annotate(
    image: Any,
    executor: SequenceExecutor,
    runtime: OfficeAgentRuntime,
    sim_time: float,
    npc_states: dict[str, Any] | None = None,
) -> Any:
    height, width = image.shape[:2]
    sx, sy = width / 1280.0, height / 720.0

    def point(x: int, y: int) -> tuple[int, int]:
        return int(x * sx), int(y * sy)

    def font(size: float) -> float:
        return max(0.28, size * min(sx, sy))

    active = next((x for x in reversed(executor.audit.records) if x["event"] == "step_started"), {})
    cv2.rectangle(image, point(24, 22), point(1256, 122), (15, 23, 32), -1)
    cv2.putText(
        image,
        "NATIVE MUJOCO 3D | CONTROL SEQUENCE",
        point(46, 58),
        0,
        font(0.78),
        (239, 244, 248),
        max(1, int(2 * sx)),
    )
    cv2.putText(
        image,
        f"mode={executor.compiled.control_mode.value} step={active.get('step_id','-')}",
        point(46, 91),
        0,
        font(0.55),
        (108, 209, 255),
        1,
    )
    cv2.putText(
        image,
        f"corr={active.get('correlation_id','-')} sim={sim_time:05.1f}s",
        point(46, 113),
        0,
        font(0.43),
        (150, 226, 177),
        1,
    )
    active_step = getattr(executor, "_active_step", None)
    if active_step is not None and active_step.step.kind.value == "action":
        command = (
            f"COMMAND | {active_step.step.payload['actor']} "
            f"{active_step.step.payload['action']} -> {active_step.target_id or '-'}"
        )
        cv2.putText(image, command[:120], point(46, 146), 0, font(0.48), (255, 209, 112), max(1, int(sx)))
    latest_turn = None
    for session in runtime.conversations.sessions.values():
        for turn in session.dialogue_turns.values():
            if getattr(turn.status, "value", turn.status) == "committed":
                latest_turn = turn
    if latest_turn is not None:
        subtitle = f"DIALOGUE | {latest_turn.speaker}: {latest_turn.text}"
        cv2.rectangle(image, point(24, 720 - 88), point(1256, 720 - 24), (15, 23, 32), -1)
        cv2.putText(image, subtitle[:150], point(46, 720 - 48), 0, font(0.52), (239, 244, 248), max(1, int(sx)))
    tasks = tuple(runtime.robot_tasks.values())
    delivery = next((task for task in reversed(tasks) if task.status.value == "succeeded"), None)
    accepted = next(
        (task for task in reversed(tasks) if task.request_receipt_id is not None), None
    )
    if delivery is not None:
        status = (
            "DELIVERY VERIFIED | "
            f"{delivery.destination} received {delivery.object_id}"
        )
    elif accepted is not None and any(
        receipt.endswith(":task_completed") for receipt in accepted.receipt_ids
    ):
        receiver_state = (npc_states or {}).get(accepted.destination)
        resolved_clip = getattr(receiver_state, "resolved_clip", None)
        locomotion = getattr(receiver_state, "locomotion", None)
        if resolved_clip == "pick_up":
            status = f"PICK_UP IN PROGRESS | {accepted.destination} at Stretch"
        elif locomotion not in {None, "idle"}:
            status = f"NPC APPROACHING | {accepted.destination} -> Stretch"
        else:
            status = f"TASK COMPLETED | Stretch notified {accepted.destination}"
    elif accepted is not None:
        status = f"REQUEST RECEIVED | Stretch accepted {accepted.object_id}"
    else:
        status = None
    if status is not None:
        cv2.rectangle(image, point(24, 155), point(1256, 184), (15, 23, 32), -1)
        cv2.putText(image, status[:150], point(46, 176), 0, font(0.46), (150, 226, 177), max(1, int(sx)))
    return image


def _server_frame(simulator: Any) -> Any:
    """Read the image produced by the same server that emitted receipts."""
    imagery = simulator.pull_camera_data()
    rgb = imagery.office_overview_rgb
    if rgb is None:
        raise ControlSequenceError("server did not provide office_overview_rgb")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _load_replay(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, dict) for k, v in value.items()
    ):
        raise ControlSequenceError("LLM replay must map step IDs to response objects")
    return value


def render(
    scene: Path,
    population_path: Path,
    sequence_path: Path,
    output: Path,
    *,
    control_mode: str | None = None,
    llm_replay: Path | None = None,
    semantic_world_path: Path | None = None,
    robot_task_driver: RobotTaskExecutor | None = None,
    robot_mode: str = "mock",
    mock_robot_delay_minutes: float = MOCK_ROBOT_DELAY_MINUTES,
    simulator_factory: Any = StretchMujocoSimulator,
    max_seconds: float = 90.0,
    camera_hz: float | None = None,
    minutes_per_second: float | None = None,
) -> dict[str, Any]:
    effective_camera_hz = float(FPS if camera_hz is None else camera_hz)
    if effective_camera_hz <= 0:
        raise ValueError("camera_hz must be positive")
    if minutes_per_second is not None and minutes_per_second <= 0:
        raise ValueError("minutes_per_second must be positive")
    if robot_mode not in {"mock", "physical"}:
        raise ValueError("robot_mode must be 'mock' or 'physical'")
    if mock_robot_delay_minutes <= 0:
        raise ValueError("mock_robot_delay_minutes must be positive")
    if robot_mode == "mock" and robot_task_driver is not None:
        raise ValueError("robot_task_driver cannot be supplied in mock robot mode")
    if robot_mode == "physical" and robot_task_driver is None:
        raise ValueError("physical robot mode requires an external RobotTaskExecutor")
    sequence = StrictSequenceLoader().load(sequence_path)
    compiled = SequenceCompiler(discover_capabilities(sequence)).compile(sequence, control_mode)
    replay = _load_replay(llm_replay)
    if compiled.control_mode.value == "llm" and "action_proposal" not in replay:
        raise ControlSequenceError("LLM renderer mode requires --llm-replay action_proposal")
    semantic_world_path = (
        semantic_world_path or ROOT / "stretch_mujoco/models/office_semantics.json"
    )
    simulator = simulator_factory(
        scene_xml_path=str(scene),
        population_path=str(population_path),
        camera_hz=effective_camera_hz,
        cameras_to_use=[StretchCameras.office_overview_rgb],
        semantic_world_path=str(semantic_world_path),
    )
    # Preserve the authored scene pose.  A startup ``home`` command would be
    # a second, unrecorded robot action before the control sequence begins.
    simulator.start(headless=True, home_on_start=False)
    if not simulator.is_running():
        raise ControlSequenceError(
            "live MuJoCo server failed to start; no renderer evidence was produced"
        )
    # Keep the acceptance render bounded while retaining the live server's
    # physical controller/receipt sequence.  This only scales commanded robot
    # actuator motion; it does not bypass navigation, IK, attachment, release,
    # or placement verification.
    if callable(getattr(simulator, "set_robot_motion_speed", None)):
        simulator.set_robot_motion_speed(ROBOT_MOTION_SPEED)
    transport = ServerBackedRendererTransport(simulator)
    world = SemanticWorld.from_json(semantic_world_path)
    # Load schema-v2 population before constructing the bridge so its
    # interaction templates and trajectory profile are the actual runtime
    # contracts used by command lowering (not merely report metadata).
    population = NpcPopulation.from_json(population_path)
    trajectory_profile = (
        None
        if population.trajectory_profile is None
        else NpcTrajectoryProfile.from_json(population.resolve_path(population.trajectory_profile))
    )
    npc_ids = tuple(npc for npc in sequence.participants.values() if npc != "stretch_3")
    driver = create_mujoco_action_driver(
        transport,
        npc_ids=npc_ids,
        world=world,
        trajectory_profile=trajectory_profile,
        interaction_templates=population.interaction_templates,
    )
    runtime = OfficeAgentRuntime.from_json(
        world, population_path, auto_plan=False, action_driver=driver, seed=sequence.seed
    )
    if minutes_per_second is not None:
        # Presentation speed changes office-clock durations only.  Movement,
        # animation markers, collision routing, and every terminal receipt stay
        # on the live MuJoCo server clock.
        runtime.minutes_per_second = float(minutes_per_second)
    runtime.interaction_driver = driver
    mock_robot = None
    if robot_mode == "physical":
        runtime.robot_task_driver = robot_task_driver
    else:
        # The mock is installed on the runtime's normal robot-task driver
        # boundary.  This keeps request acceptance, task completion, receiver
        # notification, NPC approach, receive marker, and terminal receipts in
        # the same tick chain for demo/test callers. It is not a production
        # robot executor.
        mock_robot = MockRobotExecutor(
            completion_delay_minutes=mock_robot_delay_minutes,
            npc_transport=transport,
            receipt_source=driver,
            handover_sites={
                "stretch_3": population.interaction_templates["handover"].roles["receiver"].site
                if population.interaction_templates and "handover" in population.interaction_templates
                else "employee_01_handover_site"
            },
            handover_yaws={
                "stretch_3": population.interaction_templates["handover"].roles["receiver"].yaw
                if population.interaction_templates and "handover" in population.interaction_templates
                else 0.0
            },
        )
        runtime.robot_task_driver = mock_robot
    if compiled.control_mode.value == "llm":
        assert sequence.llm.autonomy is not None
        proposal_pending = [replay["action_proposal"]]
        source = LlmPlanSource(
            lambda _observation: proposal_pending.pop(0) if proposal_pending else None,
            allowed_actions=frozenset(sequence.llm.autonomy.allowed_actions),
            allowed_locations=frozenset(sequence.llm.autonomy.allowed_locations),
            horizon=sequence.llm.autonomy.plan_horizon_actions,
        )
        executor = SequenceExecutor(
            compiled,
            runtime,
            plan_source=source,
            segment_compiler=lambda steps: SequenceCompiler(
                discover_capabilities(sequence)
            ).compile_segment(sequence, steps),
            max_segments=sequence.llm.autonomy.max_decisions,
            observation=transport.pull_semantic_state,
        )
    else:
        executor = SequenceExecutor(compiled, runtime, observation=transport.pull_semantic_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as file:
        intermediate = Path(file.name)
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError("video_writer_unavailable")
    render_error: BaseException | None = None
    written_frames = 0
    min_npc_distance_m: float | None = None
    npc_position_samples: list[dict[str, Any]] = []
    handover_marker_geometry_samples: list[dict[str, Any]] = []
    try:
        last_now = float(transport.pull_status().time)
        deadline = last_now + max_seconds
        next_frame_at = last_now
        terminal_seen_at: float | None = None
        while last_now < deadline:
            status = transport.pull_status()
            now = float(status.time)
            elapsed_seconds = max(0.0, now - last_now)
            runtime.tick(elapsed_seconds)
            result = executor.tick(now)
            if executor.executions:
                active = executor.executions[-1]
                if active.status.value == "running" and active.step_id in replay:
                    step = next(
                        item for item in compiled.steps if item.step.step_id == active.step_id
                    )
                    if step.step.kind.value == "llm_request":
                        executor.bind_llm_result(
                            active.step_id,
                            validate_llm_contract(
                                step.step.payload["output_contract"], replay[active.step_id]
                            ),
                        )
            # Camera rendering is expensive and must not throttle the server's
            # physics/control loop.  Frames are sampled by server simulation
            # time, while runtime/executor receipt polling remains continuous.
            if now >= next_frame_at:
                frame = cv2.resize(_server_frame(simulator), (WIDTH, HEIGHT))
                npc_states = {
                    npc_id: state
                    for npc_id, state in transport.pull_npc_states().items()
                    if npc_id in population.npcs
                }
                states = list(npc_states.values())
                for index, first in enumerate(states):
                    for second in states[index + 1 :]:
                        distance = math.dist(first.position[:2], second.position[:2])
                        min_npc_distance_m = (
                            distance
                            if min_npc_distance_m is None
                            else min(min_npc_distance_m, distance)
                        )
                # This is intentionally narrower than the general navigation
                # metric: it captures only the simultaneous give/receive
                # marker window, whose commands are live-gated by distance
                # and mutual-facing checks in NpcController.
                by_interaction: dict[str, list[Any]] = {}
                for state in states:
                    if state.interaction_id is not None:
                        by_interaction.setdefault(state.interaction_id, []).append(state)
                for interaction_id, participants in by_interaction.items():
                    if len(participants) != 2 or {
                        participant.resolved_clip for participant in participants
                    } != {"give", "receive"}:
                        continue
                    first, second = participants
                    distance = math.dist(first.position[:2], second.position[:2])
                    handover_marker_geometry_samples.append(
                        {
                            "sim_time": now,
                            "interaction_id": interaction_id,
                            "participants": [first.npc_id, second.npc_id],
                            "distance_m": distance,
                            "yaws": {
                                first.npc_id: math.atan2(
                                    2 * (first.quaternion[0] * first.quaternion[3]
                                         + first.quaternion[1] * first.quaternion[2]),
                                    1 - 2 * (first.quaternion[2] ** 2 + first.quaternion[3] ** 2),
                                ),
                                second.npc_id: math.atan2(
                                    2 * (second.quaternion[0] * second.quaternion[3]
                                         + second.quaternion[1] * second.quaternion[2]),
                                    1 - 2 * (second.quaternion[2] ** 2 + second.quaternion[3] ** 2),
                                ),
                            },
                        }
                    )
                if written_frames % max(FPS, 1) == 0:
                    npc_position_samples.append(
                        {
                            "sim_time": now,
                            "positions": {
                                npc_id: list(state.position) for npc_id, state in npc_states.items()
                            },
                        }
                    )
                writer.write(
                    _annotate(
                        frame,
                        executor,
                        runtime,
                        now,
                        npc_states,
                    )
                )
                written_frames += 1
                next_frame_at += 1 / FPS
            if result.status.value in {"succeeded", "failed", "timed_out", "cancelled"}:
                if terminal_seen_at is None:
                    terminal_seen_at = now
                elif now - terminal_seen_at >= POST_ROLL_SECONDS:
                    break
            # Receipts are server-clocked and remain queued between polls; a
            # 50 ms observation cadence avoids saturating the cross-process
            # status proxy while retaining responsive action cancellation.
            time.sleep(min(POLL_INTERVAL, 1 / FPS))
            last_now = now
        else:
            executor.cancel(float(transport.pull_status().time), "render_frame_budget_exhausted")
    except BaseException as exc:
        # Keep a playable partial artifact when a long physical render is
        # stopped manually (or the simulator raises).  It is explicitly
        # marked failed below and is never an acceptance result.
        render_error = exc
    finally:
        writer.release()
        simulator.stop()
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(intermediate),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
        )
    finally:
        intermediate.unlink(missing_ok=True)
    report = {
        "passed": executor.status.value == "succeeded",
        "renderer": "native_mujoco_3d",
        "scene": str(scene),
        "sequence": str(sequence_path),
        "effective_control_mode": compiled.control_mode.value,
        "robot_mode": robot_mode,
        "mock_robot_delay_minutes": (
            mock_robot_delay_minutes if robot_mode == "mock" else None
        ),
        "sequence_sha256": sequence.source_sha256,
        "population": str(population_path),
        "population_sha256": _sha256(population_path),
        "semantic_world": str(semantic_world_path),
        "semantic_world_sha256": _sha256(semantic_world_path),
        "trajectory_profile": (
            None
            if population.trajectory_profile is None
            else str(population.resolve_path(population.trajectory_profile))
        ),
        "trajectory_profile_sha256": (
            None
            if population.trajectory_profile is None
            else _sha256(population.resolve_path(population.trajectory_profile))
        ),
        "compiled_plan_sha256": compiled.plan_sha256,
        "execution_status": executor.status.value,
        "render_error": None
        if render_error is None
        else f"{type(render_error).__name__}: {render_error}",
        "partial_artifact": render_error is not None,
        "steps": [x.__dict__ for x in executor.executions],
        "robot_tasks": [
            {
                "task_id": task.task_id,
                "requester": task.requester,
                "object": task.object_id,
                "destination": task.destination,
                "status": task.status.value,
                "error": task.error,
                "receipt_ids": sorted(task.receipt_ids),
            }
            for task in runtime.robot_tasks.values()
        ],
        "npc_command_receipts": [
            {
                "command_id": receipt.command_id,
                "npc_id": receipt.npc_id,
                "status": receipt.status.value,
                "reason": receipt.reason,
                "finished_at": receipt.finished_at,
            }
            for receipt in driver.command_receipts()
        ],
        "audit": executor.audit.records,
        "output": str(output),
        "quality": {
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "frames": written_frames,
            "encoding": "h264",
        },
        "navigation_metrics": {
            "minimum_npc_distance_m": min_npc_distance_m,
            "position_samples": npc_position_samples,
        },
        "handover_marker_geometry": {
            "sample_count": len(handover_marker_geometry_samples),
            "minimum_distance_m": (
                None
                if not handover_marker_geometry_samples
                else min(sample["distance_m"] for sample in handover_marker_geometry_samples)
            ),
            "maximum_distance_m": (
                None
                if not handover_marker_geometry_samples
                else max(sample["distance_m"] for sample in handover_marker_geometry_samples)
            ),
            "samples": handover_marker_geometry_samples,
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if render_error is not None:
        raise render_error
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument(
        "--population",
        type=Path,
        default=ROOT / "stretch_mujoco/models/office_population.production.example.json",
    )
    p.add_argument(
        "--sequence",
        type=Path,
        default=ROOT / "stretch_mujoco/models/control_sequences/npc_full_acceptance_v1.yaml",
    )
    p.add_argument("--control-mode", choices=("yaml", "llm"))
    p.add_argument(
        "--robot-mode",
        choices=("mock", "physical"),
        default="mock",
        help="Robot task backend; mock completes after a short delay, physical uses Stretch IK.",
    )
    p.add_argument("--mock-robot-delay-minutes", type=float, default=MOCK_ROBOT_DELAY_MINUTES)
    p.add_argument("--llm-replay", type=Path)
    p.add_argument("--semantic-world", type=Path)
    p.add_argument("--max-seconds", type=float, default=90.0)
    p.add_argument(
        "--camera-hz",
        type=float,
        help="Server camera sampling rate; lower values reduce renderer contention.",
    )
    p.add_argument(
        "--minutes-per-second",
        type=float,
        help="Optional presentation-only office-clock speed multiplier.",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = render(
        a.scene.resolve(),
        a.population.resolve(),
        a.sequence.resolve(),
        a.output.resolve(),
        control_mode=a.control_mode,
        llm_replay=None if a.llm_replay is None else a.llm_replay.resolve(),
        semantic_world_path=(None if a.semantic_world is None else a.semantic_world.resolve()),
        max_seconds=a.max_seconds,
        camera_hz=a.camera_hz,
        minutes_per_second=a.minutes_per_second,
        robot_mode=a.robot_mode,
        mock_robot_delay_minutes=a.mock_robot_delay_minutes,
    )
    print(json.dumps({"passed": report["passed"], "output": report["output"]}, sort_keys=True))


if __name__ == "__main__":
    main()
