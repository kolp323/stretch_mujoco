"""Run a multi-NPC office day in Scene 2 with event-driven LLM guidance.

Each employee agent follows their own schedule, makes independent utility
decisions, and receives LLM assistance for planning and unexpected events.
The runtime already supports multiple agents natively — this example simply
loads a multi-employee configuration and processes all LLM requests in the
main loop.

Pass --mp4 to generate a 2-D top-down animation of the working day.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import click

from stretch_mujoco.agents import LLMProviderError, MockRobotExecutor, OfficeAgentRuntime
from stretch_mujoco.agents.mp4_recorder import OfficeMp4Recorder
from stretch_mujoco.recording import (
    JsonlSnapshotWriter,
    build_native_multi_npc_scene,
    build_office_snapshot,
    write_recording_manifest,
)
from stretch_mujoco.semantics import SemanticWorld

MODELS_PATH = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"
DEFAULT_SEMANTICS = MODELS_PATH / "office_scene2_semantics.json"
DEFAULT_AGENTS = MODELS_PATH / "office_scene2_agents.json"

# World-coordinate positions for semantic locations (Scene 2 layout).
# These drive the 2-D agent markers in the MP4 recording.
LOCATION_POSITIONS: dict[str, tuple[float, float]] = {
    "workstation_left": (-3.0, -3.0),
    "workstation_right": (3.0, -3.0),
    "chair_left": (-3.0, -3.6),
    "chair_right": (3.0, -3.6),
    "meeting_table": (-5.5, 3.0),
    "snack_counter": (6.8, 3.0),
    "storage_cabinet": (5.2, 5.5),
}

# Equivalent semantic locations in the project-native office scene.  These
# coordinates sit at the human-facing side of the real furniture rather than
# at the furniture body's centre, so visual NPCs do not stand inside desks.
NATIVE_OFFICE_LOCATION_POSITIONS: dict[str, tuple[float, float]] = {
    "workstation_left": (-2.35, 0.65),
    "workstation_right": (2.25, 0.65),
    "chair_left": (-2.35, 0.45),
    "chair_right": (2.25, 0.45),
    "meeting_table": (0.0, 1.50),
    "snack_counter": (0.0, -1.97),
    "storage_cabinet": (-3.35, 1.93),
}


def format_clock(minute: float) -> str:
    total = int(minute) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def format_visual_event(event, day_start_minute: float) -> str:
    """Translate internal runtime events into labels suitable for an MP4 subtitle."""
    details = event.details
    actor = event.agent_id.replace("employee_", "NPC ").replace("_", " ")
    timestamp = format_clock(day_start_minute + event.time)
    if event.event == "plan_selected":
        return f"{timestamp}  {actor}: plan is {details.get('goal', 'unknown')}"
    if event.event in {"action_started", "action_succeeded", "action_failed"}:
        action = str(details.get("action", "action")).replace("_", " ")
        verb = event.event.removeprefix("action_")
        return f"{timestamp}  {actor}: {action} {verb}"
    if event.event == "robot_task_created":
        return f"{timestamp}  {actor}: asked robot for {str(details.get('object', 'item')).replace('_', ' ')}"
    if event.event == "robot_task_completed":
        return f"{timestamp}  Robot task {details.get('status', 'updated')}"
    if event.event == "daily_office_event":
        return f"{timestamp}  New office event: {str(details.get('event_type', 'task')).replace('_', ' ')}"
    return f"{timestamp}  {actor}: {event.event.replace('_', ' ')}"


def process_llm_requests(runtime, provider, records: list[dict[str, Any]]):
    """Drain pending LLM requests for all agents and apply responses.

    This is the same pattern as the single-NPC example — the runtime drains
    requests for *all* agents, so no per-agent loop is needed.
    """
    events = []
    for request in runtime.drain_llm_requests():
        record = {
            "time": format_clock(request.minute_of_day),
            "agent": request.agent_id,
            "trigger": request.trigger.value,
        }
        if provider is None:
            record.update(status="skipped", error="LLM disabled")
            records.append(record)
            continue
        try:
            response = runtime.llm.process(request, provider)
            validation = runtime.apply_llm_response(request, response)
        except (LLMProviderError, TimeoutError, OSError, RuntimeError, ValueError) as error:
            record.update(status="provider_error", error=str(error))
        else:
            record.update(
                status="applied" if validation.valid else "rejected",
                response_fields=sorted(response) if isinstance(response, dict) else [],
                errors=list(validation.errors),
            )
            if isinstance(response, dict):
                action = response.get("action")
                if isinstance(action, dict):
                    record["action"] = {
                        "action": action.get("action"),
                        "target": action.get("target"),
                        "parameters": action.get("parameters", {}),
                    }
                dialogue = response.get("dialogue")
                if isinstance(dialogue, str):
                    record["dialogue"] = dialogue
        records.append(record)
        events.extend(runtime.drain_events())
    return events


def _per_agent_report(runtime, agent_id: str, events: list) -> dict[str, Any]:
    """Build a per-agent statistics block from the completed simulation."""
    agent = runtime.agents[agent_id]
    agent_events = [e for e in events if e.agent_id == agent_id]

    action_counts = Counter(
        e.details["action"] for e in agent_events if e.event == "action_succeeded"
    )
    goal_counts = Counter(e.details["goal"] for e in agent_events if e.event == "plan_selected")
    rejection_counts: Counter = Counter()
    for e in agent_events:
        if e.event == "action_rejected":
            for err in e.details.get("errors", []):
                rejection_counts[err] += 1

    return {
        "agent": agent_id,
        "profile": {
            "role": agent.profile.role,
            "department": agent.profile.department,
            "personality": agent.profile.personality,
        },
        "llm_calls": runtime.llm.calls_for(runtime.day, agent_id),
        "schedule_items": len(agent.schedule.items),
        "actions": {
            "successful": dict(sorted(action_counts.items())),
            "goals_selected": dict(sorted(goal_counts.items())),
            "rejections": dict(sorted(rejection_counts.items())),
        },
        "final_state": {
            "location": agent.state.location,
            "held_object": agent.state.held_object,
            "hunger": round(agent.state.hunger, 3),
            "thirst": round(agent.state.thirst, 3),
            "fatigue": round(agent.state.fatigue, 3),
            "mood": round(agent.state.mood, 3),
            "last_action": agent.state.current_action,
        },
        "memory_entries": len(agent.memory.entries),
    }


@click.command()
@click.option(
    "--end", "end_hour", type=int, default=18, show_default=True, help="Simulation end hour (10-23)"
)
@click.option("--step", type=float, default=0.25, show_default=True, help="Tick step in minutes")
@click.option(
    "--output", type=click.Path(path_type=Path), default=None, help="Write JSON report to this file"
)
@click.option(
    "--native-office",
    is_flag=True,
    help="Record against the project-native furnished office, not the Scene 2 proxy layout",
)
@click.option(
    "--mp4",
    "mp4_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Generate a 2-D top-down MP4 animation of the workday",
)
@click.option(
    "--mp4-fps", type=int, default=10, show_default=True, help="Frame rate for MP4 output"
)
@click.option(
    "--snapshot-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Write portable JSONL snapshots for offline 2-D or 3-D video rendering",
)
@click.option(
    "--snapshot-fps",
    type=int,
    default=10,
    show_default=True,
    help="Snapshot sampling rate when --snapshot-dir is set",
)
@click.option(
    "--semantics",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_SEMANTICS,
    show_default=True,
)
@click.option(
    "--agents",
    "agents_path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_AGENTS,
    show_default=True,
)
@click.option(
    "--llm/--no-llm",
    default=True,
    show_default=True,
    help="Enable configured LLM requests; --no-llm keeps recording fully offline",
)
@click.option("--chat-log", is_flag=True, help="Print a live chat-style log of agent activity")
def main(
    end_hour: int,
    step: float,
    output: Path | None,
    mp4_path: Path | None,
    mp4_fps: int,
    snapshot_dir: Path | None,
    snapshot_fps: int,
    native_office: bool,
    semantics: Path,
    agents_path: Path,
    llm: bool,
    chat_log: bool,
) -> None:
    """Simulate multiple office employees from 09:00 until END."""
    if not 10 <= end_hour <= 23:
        raise click.BadParameter("end must be between 10 and 23")
    if step <= 0:
        raise click.BadParameter("step must be positive")
    if snapshot_fps <= 0:
        raise click.BadParameter("snapshot-fps must be positive")

    location_positions = NATIVE_OFFICE_LOCATION_POSITIONS if native_office else LOCATION_POSITIONS

    world = SemanticWorld.from_json(semantics)
    runtime = OfficeAgentRuntime.from_json(world, agents_path, auto_plan=True)
    provider = runtime.create_llm_provider() if llm else None
    day_start_minute = runtime.minute_of_day

    agent_ids = sorted(runtime.agents.keys())
    click.echo(f"Loaded {len(agent_ids)} agent(s): {', '.join(agent_ids)}")
    for aid in agent_ids:
        agent = runtime.agents[aid]
        click.echo(f"  {aid}: {agent.profile.role} at {agent.state.location}")

    all_events = list(runtime.drain_events())
    llm_records: list[dict[str, Any]] = []
    mock_robot = MockRobotExecutor()

    # Kick off day-start LLM events for all agents
    all_events.extend(process_llm_requests(runtime, provider, llm_records))

    # --- MP4 recorder ---
    recorder: OfficeMp4Recorder | None = None
    mp4_manifest = MODELS_PATH / "assets" / "office_scenes" / "office_02_cross_axis.json"
    if mp4_path is not None:
        recorder = OfficeMp4Recorder(
            mp4_path,
            mp4_manifest if mp4_manifest.exists() else None,
            fps=mp4_fps,
        )
        recorder.start()
        click.echo(f"Recording MP4 -> {mp4_path}")

    snapshot_writer: JsonlSnapshotWriter | None = None
    snapshot_accumulator = 0.0
    snapshot_interval = 1.0 / snapshot_fps
    if snapshot_dir is not None:
        snapshot_path = snapshot_dir / "office_snapshots.jsonl"
        snapshot_writer = JsonlSnapshotWriter(snapshot_path)
        scene_xml = "stretch_mujoco/models/office_scene2_multi_npc.xml"
        if native_office:
            scene_xml = str(
                build_native_multi_npc_scene(snapshot_dir / "native_office_multi_npc.xml")
            )
        manifest_path = write_recording_manifest(
            snapshot_dir, scene_xml=scene_xml, snapshot_fps=snapshot_fps
        )
        snapshot_writer.write(
            build_office_snapshot(runtime, location_positions, events=all_events[-6:])
        )
        click.echo(f"Recording snapshots -> {snapshot_path} (manifest: {manifest_path})")

    end_minute = end_hour * 60
    mp4_frame_interval = max(1.0, 1.0 / mp4_fps)  # real seconds between MP4 frames
    mp4_accumulator = 0.0

    while runtime.minute_of_day < end_minute:
        remaining_seconds = (end_minute - runtime.minute_of_day) / runtime.minutes_per_second
        tick_seconds = min(step, remaining_seconds)
        all_events.extend(runtime.tick(tick_seconds))

        # The mock behaves like an asynchronous robot: it consumes tasks but
        # leaves state changes to OfficeAgentRuntime.complete_robot_task().
        mock_robot.tick(runtime, tick_seconds * runtime.minutes_per_second)
        all_events.extend(runtime.drain_events())

        # Process new LLM requests (planning, dialogue, events)
        new_events = process_llm_requests(runtime, provider, llm_records)

        # --- MP4 frame ---
        if recorder is not None:
            mp4_accumulator += tick_seconds
            if mp4_accumulator >= mp4_frame_interval:
                mp4_accumulator -= mp4_frame_interval
                agents_viz = {}
                for aid, agent in runtime.agents.items():
                    pos = location_positions.get(agent.state.location, (0.0, 0.0))
                    command = agent.executor.command
                    target = command.target if command is not None else agent.state.attention_target
                    agents_viz[aid] = {
                        "position": pos,
                        "action": agent.state.current_action,
                        "label": f"NPC {aid.removeprefix('employee_')}",
                        "display_name": aid.replace("_", " ").title(),
                        "role": agent.profile.role,
                        "location": agent.state.location,
                        "target": target,
                        "target_position": location_positions.get(target),
                    }
                event_lines = [
                    format_visual_event(event, day_start_minute) for event in all_events[-6:]
                ]
                robot_tasks = [
                    {
                        "status": task.status.value,
                        "object": task.object_id,
                        "destination": task.destination,
                    }
                    for task in runtime.robot_tasks.values()
                ]
                recorder.record_frame(
                    runtime.minute_of_day,
                    agents_viz,
                    events=event_lines,
                    robot_tasks=robot_tasks,
                )

        if snapshot_writer is not None:
            snapshot_accumulator += tick_seconds
            if snapshot_accumulator >= snapshot_interval:
                snapshot_accumulator -= snapshot_interval
                snapshot_writer.write(
                    build_office_snapshot(runtime, location_positions, events=all_events[-6:])
                )

        # Live chat log
        if chat_log:
            for record in llm_records:
                if "dialogue" in record:
                    click.echo(f"[{record['time']}] {record['agent']}: " f"{record['dialogue']}")
                elif record.get("action"):
                    act = record["action"]
                    click.echo(
                        f"[{record['time']}] {record['agent']} -> "
                        f"{act.get('action','?')} "
                        f"target={act.get('target','?')} "
                        f"({record.get('status','?')})"
                    )

        all_events.extend(new_events)

    # Let in-progress actions finish
    scheduled_end = runtime.minute_of_day
    runtime.auto_plan = False
    for agent in runtime.agents.values():
        agent.planner.action_queue.clear()
    while any(agent.executor.is_busy for agent in runtime.agents.values()):
        all_events.extend(runtime.tick(step))

    # --- Finalise MP4 ---
    if recorder is not None:
        recorder.close()
        click.echo(f"MP4 saved -> {mp4_path}  ({recorder._frame_count} frames)")
    if snapshot_writer is not None:
        snapshot_writer.close()

    # --- Build report ---
    event_counts = Counter(e.event for e in all_events)
    per_agent = {aid: _per_agent_report(runtime, aid, all_events) for aid in agent_ids}

    # Dialogue summary
    dialogues = [r for r in llm_records if "dialogue" in r and r.get("status") == "applied"]

    report = {
        "scene": world.scene,
        "workday": {
            "start": "09:00",
            "scheduled_end": format_clock(scheduled_end),
            "completed_at": format_clock(runtime.minute_of_day),
        },
        "llm_config": runtime.llm_config_summary(),
        "agent_count": len(agent_ids),
        "agents": per_agent,
        "llm": {
            "total_calls": sum(runtime.llm.calls_for(runtime.day, aid) for aid in agent_ids),
            "requests": llm_records,
        },
        "dialogues": dialogues,
        "global_event_counts": dict(sorted(event_counts.items())),
        "robot_tasks": [
            {
                "task": task.task,
                "object": task.object_id,
                "destination": task.destination,
                "requester": task.requester,
                "status": task.status.value,
            }
            for task in runtime.robot_tasks.values()
        ],
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        click.echo(f"\nReport written to {output}")
    else:
        click.echo(rendered)


if __name__ == "__main__":
    main()
