"""Create a deterministic 2-D and 3-D acceptance replay for NPC conversations.

The replay exercises the P0--P3 logical contracts, rather than claiming that
the action driver performed approach, turn, or talk animations.  Both videos
are rendered from exactly the same JSONL snapshots and event stream.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import click

from stretch_mujoco.agents import (
    ConversationIntent,
    EmployeeAgent,
    MockRobotExecutor,
    OfficeAgentRuntime,
    SocialConversationProposal,
)
from stretch_mujoco.recording import (
    JsonlSnapshotWriter,
    build_native_multi_npc_scene,
    build_office_snapshot,
    read_snapshots,
    write_recording_manifest,
)
from stretch_mujoco.recording.renderers import render_mujoco_video, render_topdown_video
from stretch_mujoco.semantics import ObjectType, SemanticObject, SemanticWorld


MODELS_PATH = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"
SEMANTICS_PATH = MODELS_PATH / "office_semantics.json"

# The poses are an explicit visual staging of the logically validated
# conversation observation.  They are not a command to the action driver.
GREETING_POSES = {
    "employee_01": ((-0.75, 1.45, 0.0), 0.0),
    "employee_02": ((0.75, 1.45, 0.0), math.pi),
    "employee_03": ((0.0, 0.45, 0.0), math.pi / 2.0),
    "stretch_3": ((0.0, -1.8, 0.0), math.pi / 2.0),
}
PROGRESS_POSES = {
    "employee_01": ((-0.75, 1.45, 0.0), 0.0),
    "employee_02": ((-0.65, 0.7, 0.0), 0.0),
    "employee_03": ((0.65, 0.7, 0.0), math.pi),
    "stretch_3": ((0.0, -1.8, 0.0), math.pi / 2.0),
}
MEETING_POSES = {
    "employee_01": ((-0.65, 1.4, 0.0), 0.0),
    "employee_02": ((0.75, 1.45, 0.0), math.pi),
    "employee_03": ((0.65, 1.4, 0.0), math.pi),
    "stretch_3": ((0.0, -1.8, 0.0), math.pi / 2.0),
}
ROBOT_POSES = {
    "employee_01": ((-0.7, -1.8, 0.0), 0.0),
    "employee_02": ((-0.65, 1.4, 0.0), 0.0),
    "employee_03": ((0.65, 1.4, 0.0), math.pi),
    "stretch_3": ((0.3, -1.8, 0.0), math.pi),
}

LOCATION_POSITIONS = {
    "meeting_table": (0.0, 1.5),
    "snack_counter": (0.0, -1.97),
    "workstation_left": (-2.35, 0.65),
    "workstation_right": (2.25, 0.65),
}


@dataclass(frozen=True)
class ConversationAcceptanceArtifacts:
    """Paths and evidence produced by one deterministic replay."""

    snapshot_path: Path
    manifest_path: Path
    report_path: Path
    scene_path: Path
    snapshots: int
    event_counts: dict[str, int]


def _employee(employee_id: str, location: str) -> EmployeeAgent:
    return EmployeeAgent.from_dict(
        employee_id,
        {
            "profile": {"role": "Conversation QA", "department": "QA"},
            "initial_location": location,
        },
    )


def build_demo_runtime(
    employee_ids: tuple[str, ...] = ("employee_01", "employee_02", "employee_03"),
) -> OfficeAgentRuntime:
    """Build the no-LLM, no-visual-asset runtime used by this replay."""
    if not employee_ids or employee_ids[0] != "employee_01":
        raise ValueError("The deterministic demo requires employee_01")
    world = SemanticWorld.from_json(SEMANTICS_PATH)
    binding = world.object("employee_01").binding
    for employee_id in employee_ids[1:]:
        world.objects[employee_id] = SemanticObject(
            employee_id,
            ObjectType.EMPLOYEE,
            binding,
            {"display_name": employee_id},
        )
    return OfficeAgentRuntime(
        world,
        {employee_id: _employee(employee_id, "meeting_table") for employee_id in employee_ids},
        auto_plan=False,
        daily_events=False,
    )


def _semantic_snapshot(
    runtime: OfficeAgentRuntime, poses: dict[str, tuple[tuple[float, float, float], float]]
) -> dict[str, Any]:
    return {
        "time": runtime.elapsed_minutes,
        "objects": {
            participant: {"position": position, "yaw": yaw}
            for participant, (position, yaw) in poses.items()
        },
    }


def _visual_states(
    poses: dict[str, tuple[tuple[float, float, float], float]],
) -> dict[str, dict[str, Any]]:
    return {
        participant: {
            "position": position,
            "quaternion": [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
            "locomotion": "stationary",
            "resolved_clip": "idle",
            "clip_phase": 0.0,
            "active_command_id": None,
        }
        for participant, (position, yaw) in poses.items()
        if participant.startswith("employee_")
    }


def _capture(
    runtime: OfficeAgentRuntime,
    poses: dict[str, tuple[tuple[float, float, float], float]],
    snapshots: list[dict[str, Any]],
    events: Iterable[Any] = (),
) -> None:
    snapshot = build_office_snapshot(
        runtime,
        LOCATION_POSITIONS,
        events=events,
        npc_states=_visual_states(poses),
    )
    for agent_id, state in snapshot["agents"].items():
        state["label"] = f"NPC {agent_id.removeprefix('employee_')}"
        state["display_name"] = agent_id.replace("employee_", "NPC ").replace("_", " ")
        state["role"] = "Conversation participant"
    snapshots.append(snapshot)


def _advance(
    runtime: OfficeAgentRuntime,
    poses: dict[str, tuple[tuple[float, float, float], float]],
    snapshots: list[dict[str, Any]],
    seconds: float = 1.0,
) -> None:
    _capture(runtime, poses, snapshots, runtime.tick(seconds))


def build_deterministic_conversation_recording(
    output_dir: str | Path,
) -> ConversationAcceptanceArtifacts:
    """Exercise social and robot dialogue contracts and write portable evidence."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    runtime = build_demo_runtime()
    # Agent construction may enqueue ordinary planning signals. They are not
    # part of this no-provider acceptance story, so retain dialogue evidence only.
    runtime.drain_events()
    snapshots: list[dict[str, Any]] = []

    # 1. NPC-NPC greeting.
    session = runtime.start_conversation(
        "employee_01",
        "employee_02",
        "morning greeting",
        _semantic_snapshot(runtime, GREETING_POSES),
        timeout=20,
    )
    runtime.record_conversation_candidate(
        session.session_id, "employee_01", ConversationIntent.GREETING, None
    )
    _capture(runtime, GREETING_POSES, snapshots, runtime.drain_events())
    _advance(runtime, GREETING_POSES, snapshots)
    runtime.record_conversation_candidate(
        session.session_id, "employee_02", ConversationIntent.GREETING, None
    )
    runtime.complete_conversation(session.session_id)
    _capture(runtime, GREETING_POSES, snapshots, runtime.drain_events())
    _advance(runtime, GREETING_POSES, snapshots)

    # 2. NPC-NPC progress inquiry.
    session = runtime.start_conversation(
        "employee_02",
        "employee_03",
        "release status",
        _semantic_snapshot(runtime, PROGRESS_POSES),
        timeout=20,
    )
    runtime.record_conversation_candidate(
        session.session_id, "employee_02", ConversationIntent.PROGRESS_INQUIRY, None
    )
    _capture(runtime, PROGRESS_POSES, snapshots, runtime.drain_events())
    _advance(runtime, PROGRESS_POSES, snapshots)
    runtime.record_conversation_candidate(
        session.session_id, "employee_03", ConversationIntent.PROGRESS_INQUIRY, None
    )
    runtime.complete_conversation(session.session_id)
    _capture(runtime, PROGRESS_POSES, snapshots, runtime.drain_events())
    _advance(runtime, PROGRESS_POSES, snapshots)

    # 3. Scheduler-owned meeting invitation, including the response record.
    invitation = SocialConversationProposal(
        "employee_01",
        "employee_03",
        ConversationIntent.MEETING_INVITATION,
        "release planning",
        runtime.elapsed_minutes,
        {"location": "meeting_table"},
    )
    decision = runtime.schedule_npc_conversations(
        [invitation], _semantic_snapshot(runtime, MEETING_POSES)
    )[0]
    if not decision.accepted:
        raise RuntimeError(f"Deterministic meeting invitation was rejected: {decision.reason}")
    _capture(runtime, MEETING_POSES, snapshots, runtime.drain_events())
    active = next(
        session
        for session in runtime.conversations.sessions.values()
        if not session.status.terminal
    )
    _advance(runtime, MEETING_POSES, snapshots)
    runtime.record_conversation_candidate(
        active.session_id, "employee_03", ConversationIntent.MEETING_INVITATION, None
    )
    runtime.complete_conversation(active.session_id)
    invitation_id = next(reversed(runtime.social_conversations.invitations))
    runtime.social_conversations.respond_invitation(invitation_id, True, runtime.elapsed_minutes)
    _capture(runtime, MEETING_POSES, snapshots, runtime.drain_events())
    _advance(runtime, MEETING_POSES, snapshots)

    # 4. NPC-NPC conflict resolution.
    session = runtime.start_conversation(
        "employee_03",
        "employee_02",
        "desk conflict",
        _semantic_snapshot(runtime, PROGRESS_POSES),
        timeout=20,
    )
    runtime.record_conversation_candidate(
        session.session_id, "employee_03", ConversationIntent.CONFLICT_RESOLUTION, None
    )
    _capture(runtime, PROGRESS_POSES, snapshots, runtime.drain_events())
    _advance(runtime, PROGRESS_POSES, snapshots)
    runtime.record_conversation_candidate(
        session.session_id, "employee_02", ConversationIntent.CONFLICT_RESOLUTION, None
    )
    runtime.complete_conversation(session.session_id)
    _capture(runtime, PROGRESS_POSES, snapshots, runtime.drain_events())
    _advance(runtime, PROGRESS_POSES, snapshots)

    # 5. NPC-robot request/clarify/acknowledge/handover-confirm sequence.
    session = runtime.start_conversation(
        "employee_01",
        "stretch_3",
        "deliver report",
        _semantic_snapshot(runtime, ROBOT_POSES),
        timeout=20,
    )
    validation = runtime.request_robot_task(
        session.session_id,
        "employee_01",
        task="deliver",
        object_id="document_report",
        destination="workstation_right",
    )
    if not validation.valid or session.robot_task_id is None:
        raise RuntimeError(f"Deterministic robot request was rejected: {validation.errors}")
    runtime.record_conversation_candidate(
        session.session_id, "employee_01", ConversationIntent.REQUEST, None
    )
    _capture(runtime, ROBOT_POSES, snapshots, runtime.drain_events())
    _advance(runtime, ROBOT_POSES, snapshots)
    runtime.record_conversation_candidate(
        session.session_id, "stretch_3", ConversationIntent.CLARIFY, None
    )
    _capture(runtime, ROBOT_POSES, snapshots, runtime.drain_events())
    _advance(runtime, ROBOT_POSES, snapshots)
    runtime.record_conversation_candidate(
        session.session_id, "employee_01", ConversationIntent.ACKNOWLEDGE, None
    )
    robot = MockRobotExecutor(
        completion_delay_minutes=0.01, handover_ready_task_ids={session.robot_task_id}
    )
    robot.tick(runtime, 0.01)
    _capture(runtime, ROBOT_POSES, snapshots, runtime.drain_events())
    _advance(runtime, ROBOT_POSES, snapshots)
    runtime.record_conversation_candidate(
        session.session_id, "stretch_3", ConversationIntent.HANDOVER_CONFIRM, None
    )
    runtime.complete_conversation(session.session_id)
    _capture(runtime, ROBOT_POSES, snapshots, runtime.drain_events())

    snapshot_path = directory / "conversation_snapshots.jsonl"
    with JsonlSnapshotWriter(snapshot_path) as writer:
        for snapshot in snapshots:
            writer.write(snapshot)
    scene_path = build_native_multi_npc_scene(
        directory / "native_office_three_npc.xml", employee_numbers=(1, 2, 3)
    )
    manifest_path = write_recording_manifest(
        directory,
        snapshot_file=snapshot_path.name,
        scene_xml=scene_path.name,
        snapshot_fps=10,
    )
    all_events = [event for snapshot in snapshots for event in snapshot["events"]]
    event_counts = dict(sorted(Counter(event["event"] for event in all_events).items()))
    report_path = directory / "conversation_acceptance.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deterministic": True,
                "rendering_claim": "logical conversation replay; no embodied approach, turn, or talk receipt",
                "scenarios": [
                    "greeting",
                    "progress_inquiry",
                    "meeting_invitation_accepted",
                    "conflict_resolution",
                    "robot_request_clarify_acknowledge_handover_confirm",
                ],
                "snapshot_file": snapshot_path.name,
                "scene_xml": scene_path.name,
                "event_counts": event_counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ConversationAcceptanceArtifacts(
        snapshot_path, manifest_path, report_path, scene_path, len(snapshots), event_counts
    )


def build_deterministic_robot_task_recording(
    output_dir: str | Path,
) -> ConversationAcceptanceArtifacts:
    """Write the release-scoped single-NPC-to-mock-robot acceptance replay.

    This is intentionally narrower than the full conversation replay: one NPC
    submits one validated delivery request to Stretch, receives the mock
    executor's receipt, and closes the session only after handover evidence.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    runtime = build_demo_runtime(("employee_01",))
    runtime.drain_events()
    snapshots: list[dict[str, Any]] = []

    session = runtime.start_conversation(
        "employee_01",
        "stretch_3",
        "deliver report",
        _semantic_snapshot(runtime, ROBOT_POSES),
        timeout=20,
    )
    validation = runtime.request_robot_task(
        session.session_id,
        "employee_01",
        task="deliver",
        object_id="document_report",
        destination="workstation_right",
    )
    if not validation.valid or session.robot_task_id is None:
        raise RuntimeError(f"Deterministic robot request was rejected: {validation.errors}")
    task = runtime.robot_tasks[session.robot_task_id]
    runtime.record_conversation_candidate(
        session.session_id, "employee_01", ConversationIntent.REQUEST, None
    )
    _capture(runtime, ROBOT_POSES, snapshots, runtime.drain_events())
    _advance(runtime, ROBOT_POSES, snapshots)

    runtime.record_conversation_candidate(
        session.session_id, "stretch_3", ConversationIntent.CLARIFY, None
    )
    _capture(runtime, ROBOT_POSES, snapshots, runtime.drain_events())
    _advance(runtime, ROBOT_POSES, snapshots)

    runtime.record_conversation_candidate(
        session.session_id, "employee_01", ConversationIntent.ACKNOWLEDGE, None
    )
    robot = MockRobotExecutor(completion_delay_minutes=0.01, handover_ready_task_ids={task.task_id})
    if robot.tick(runtime, 0.01) != (task.task_id,):
        raise RuntimeError("Mock robot did not complete the deterministic task")
    _capture(runtime, ROBOT_POSES, snapshots, runtime.drain_events())
    _advance(runtime, ROBOT_POSES, snapshots)

    runtime.record_conversation_candidate(
        session.session_id, "stretch_3", ConversationIntent.HANDOVER_CONFIRM, None
    )
    runtime.complete_conversation(session.session_id)
    _capture(runtime, ROBOT_POSES, snapshots, runtime.drain_events())

    snapshot_path = directory / "npc_robot_task_snapshots.jsonl"
    with JsonlSnapshotWriter(snapshot_path) as writer:
        for snapshot in snapshots:
            writer.write(snapshot)
    scene_path = build_native_multi_npc_scene(
        directory / "native_office_one_npc.xml", employee_numbers=(1,)
    )
    manifest_path = write_recording_manifest(
        directory,
        snapshot_file=snapshot_path.name,
        scene_xml=scene_path.name,
        snapshot_fps=10,
    )
    all_events = [event for snapshot in snapshots for event in snapshot["events"]]
    event_counts = dict(sorted(Counter(event["event"] for event in all_events).items()))
    report_path = directory / "npc_robot_task_acceptance.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deterministic": True,
                "scope": "single_npc_to_mock_robot_task_interface",
                "rendering_claim": (
                    "logical task replay; no embodied approach, turn, or talk receipt"
                ),
                "contract": {
                    "requester": task.requester,
                    "robot_id": task.robot_id,
                    "task_id": task.task_id,
                    "task": task.task,
                    "object": task.object_id,
                    "destination": task.destination,
                    "final_status": task.status.value,
                    "handover_receipts": sorted(task.receipt_ids),
                },
                "snapshot_file": snapshot_path.name,
                "scene_xml": scene_path.name,
                "event_counts": event_counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ConversationAcceptanceArtifacts(
        snapshot_path, manifest_path, report_path, scene_path, len(snapshots), event_counts
    )


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Ignored local artifact directory",
)
@click.option("--fps", type=int, default=10, show_default=True)
@click.option("--width", type=int, default=960, show_default=True)
@click.option("--height", type=int, default=540, show_default=True)
@click.option("--render", "render_mode", type=click.Choice(["2d", "3d", "both"]), default="both")
@click.option(
    "--scenario",
    type=click.Choice(["full", "robot-task"]),
    default="full",
    show_default=True,
    help="Use robot-task for the release-scoped single-NPC interface demo.",
)
def main(
    output_dir: Path, fps: int, width: int, height: int, render_mode: str, scenario: str
) -> None:
    """Write deterministic conversation evidence and optional 2-D/3-D MP4s."""
    if fps <= 0 or width <= 0 or height <= 0:
        raise click.BadParameter("fps, width, and height must be positive")
    if scenario == "robot-task":
        artifacts = build_deterministic_robot_task_recording(output_dir)
        output_prefix = "npc_robot_task"
        title = "NPC -> MOCK ROBOT TASK / 2D"
        subtitle = "Single-NPC task-interface acceptance replay"
    else:
        artifacts = build_deterministic_conversation_recording(output_dir)
        output_prefix = "conversation"
        title = "NPC CONVERSATION / 2D"
        subtitle = "Deterministic logical acceptance replay"
    click.echo(f"Snapshots -> {artifacts.snapshot_path} ({artifacts.snapshots} frames)")
    click.echo(f"Report -> {artifacts.report_path}")
    if render_mode in {"2d", "both"}:
        output = output_dir / f"{output_prefix}_2d.mp4"
        frames = render_topdown_video(
            read_snapshots(artifacts.snapshot_path),
            output,
            fps=fps,
            title=title,
            subtitle=subtitle,
        )
        click.echo(f"2-D logical replay -> {output} ({frames} frames)")
    if render_mode in {"3d", "both"}:
        output = output_dir / f"{output_prefix}_3d.mp4"
        frames = render_mujoco_video(
            read_snapshots(artifacts.snapshot_path),
            output,
            scene_path=artifacts.scene_path,
            fps=fps,
            width=width,
            height=height,
        )
        click.echo(f"3-D logical replay -> {output} ({frames} frames)")


if __name__ == "__main__":
    main()
