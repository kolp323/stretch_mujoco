"""JSONL snapshots shared by simulation runs and offline renderers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from stretch_mujoco.agents.action_recipes import animation_for_action
from stretch_mujoco.agents.actions import RuntimeEvent

SNAPSHOT_SCHEMA_VERSION = 2

DEFAULT_LOCATION_YAWS = {
    "workstation_left": math.pi,
    "workstation_right": math.pi,
    "chair_left": math.pi,
    "chair_right": math.pi,
    "meeting_table": math.pi,
    "snack_counter": 0.0,
    "storage_cabinet": math.pi,
}


class JsonlSnapshotWriter:
    """Write independently consumable simulation snapshots without video dependencies."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

    def write(self, snapshot: Mapping[str, Any]) -> None:
        self._file.write(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "JsonlSnapshotWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _event_to_dict(event: RuntimeEvent, day_start_minute: float) -> dict[str, Any]:
    """Make runtime-relative event timestamps readable by offline renderers."""
    return {
        "time": (day_start_minute + event.time) % (24 * 60),
        "event": event.event,
        "agent_id": event.agent_id,
        "details": event.details,
    }


def _position_and_yaw(
    agent: Any,
    location_positions: Mapping[str, tuple[float, float]],
) -> tuple[tuple[float, float, float], float, str | None]:
    """Derive a visual pose from the agent's semantic location and active move command."""
    location = agent.state.location
    start = location_positions.get(location, (0.0, 0.0))
    yaw = DEFAULT_LOCATION_YAWS.get(location, math.pi)
    target = None

    command = agent.executor.command
    if (
        command is not None
        and command.action.value == "move_to"
        and command.target in location_positions
    ):
        target = command.target
        target_xy = location_positions[target]
        # MOVE_TO has a three-minute logical duration in OfficeAgentRuntime.  The
        # interpolation is visual only; it never feeds a pose back into the runtime.
        phase = 1.0 - max(0.0, min(1.0, agent.executor.remaining_minutes / 3.0))
        x = start[0] + (target_xy[0] - start[0]) * phase
        y = start[1] + (target_xy[1] - start[1]) * phase
        if target_xy != start:
            yaw = math.atan2(target_xy[0] - start[0], -(target_xy[1] - start[1]))
        return (x, y, 0.0), yaw, target

    return (start[0], start[1], 0.0), yaw, target


def build_office_snapshot(
    runtime: Any,
    location_positions: Mapping[str, tuple[float, float]],
    *,
    events: Iterable[RuntimeEvent] = (),
    npc_states: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build schema-v2 state, preferring observed simulator state when supplied."""
    agents: dict[str, dict[str, Any]] = {}
    npcs: dict[str, dict[str, Any]] = {}
    for agent_id, agent in runtime.agents.items():
        action = agent.state.current_action
        observed = None if npc_states is None else npc_states.get(agent_id)
        if observed is None:
            position, yaw, target = _position_and_yaw(agent, location_positions)
            quaternion = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
            locomotion = "walk" if target else "stationary"
            resolved_clip = animation_for_action(action)
            phase = 0.0
            active_command_id = None
        else:
            if hasattr(observed, "to_dict"):
                observed = observed.to_dict()
            position = tuple(observed["position"])
            quaternion = list(observed["quaternion"])
            yaw = math.atan2(
                2.0 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                1.0 - 2.0 * (quaternion[2] ** 2 + quaternion[3] ** 2),
            )
            target = None
            locomotion = str(observed["locomotion"])
            resolved_clip = str(observed["resolved_clip"])
            phase = float(observed["clip_phase"])
            active_command_id = observed.get("active_command_id")
        agents[agent_id] = {
            "position": list(position),
            "yaw": yaw,
            "location": agent.state.location,
            "target": target,
            "action": action,
            "animation": resolved_clip,
            "animation_phase": phase,
            "label": agent.profile.role.split()[0][:8],
        }
        npcs[agent_id] = {
            "pose": {"position": list(position), "quaternion": quaternion},
            "logical_action": action,
            "execution_id": getattr(agent.executor, "execution_id", None),
            "locomotion": locomotion,
            "animation": {"clip": resolved_clip, "phase": phase},
            "held_objects": (
                []
                if getattr(agent.state, "held_object", None) is None
                else [agent.state.held_object]
            ),
            "interaction_id": None,
            "active_command_id": active_command_id,
        }

    day_start_minute = runtime.minute_of_day - getattr(runtime, "elapsed_minutes", 0.0)
    robot_tasks = [
        {
            "status": task.status.value,
            "object": task.object_id,
            "destination": task.destination,
            "requester": task.requester,
        }
        for task in getattr(runtime, "robot_tasks", {}).values()
    ]
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "sim_time": float(getattr(runtime, "elapsed_minutes", 0.0)),
        "minute_of_day": runtime.minute_of_day,
        "npcs": npcs,
        # Transitional projection for existing 2-D tools. It is derived solely
        # from ``npcs``/runtime state and is not authoritative.
        "agents": agents,
        "robot_tasks": robot_tasks,
        "events": [_event_to_dict(event, day_start_minute) for event in events],
    }


def read_snapshots(path: str | Path, *, upgrade_v1: bool = False) -> Iterable[dict[str, Any]]:
    """Yield JSONL snapshots, optionally projecting legacy v1 records into v2."""
    source = Path(path)
    with source.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            snapshot = json.loads(line)
            if not isinstance(snapshot, dict):
                raise ValueError(f"Invalid snapshot at {source}:{line_number}")
            version = snapshot.get("schema_version", 1)
            if version == 1:
                # Accept legacy archives without mutating their payload. New
                # writers emit v2; legacy renderers can continue to replay v1.
                yield adapt_snapshot_v1(snapshot) if upgrade_v1 else snapshot
            elif version == SNAPSHOT_SCHEMA_VERSION and "npcs" in snapshot:
                snapshot.setdefault("agents", _agents_projection(snapshot["npcs"]))
                yield snapshot
            else:
                raise ValueError(
                    f"Unsupported snapshot schema_version {version!r} at {source}:{line_number}"
                )


def adapt_snapshot_v1(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Project a schema-v1 agent snapshot onto the minimal v2 NPC contract."""
    agents = dict(snapshot.get("agents", {}))
    npcs = {}
    for npc_id, state in agents.items():
        yaw = float(state.get("yaw", math.pi))
        npcs[npc_id] = {
            "pose": {
                "position": list(state.get("position", (0.0, 0.0, 0.0))),
                "quaternion": [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
            },
            "logical_action": state.get("action", "idle"),
            "execution_id": None,
            "locomotion": "walk" if state.get("animation") == "walk" else "stationary",
            "animation": {"clip": state.get("animation", "idle"), "phase": 0.0},
            "held_objects": [],
            "interaction_id": None,
        }
    return {**snapshot, "schema_version": SNAPSHOT_SCHEMA_VERSION, "npcs": npcs, "agents": agents}


def _agents_projection(npcs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    projection = {}
    for npc_id, state in npcs.items():
        pose = state["pose"]
        quaternion = pose["quaternion"]
        yaw = math.atan2(
            2.0 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
            1.0 - 2.0 * (quaternion[2] ** 2 + quaternion[3] ** 2),
        )
        projection[npc_id] = {
            "position": list(pose["position"]),
            "yaw": yaw,
            "action": state.get("logical_action", "idle"),
            "animation": state.get("animation", {}).get("clip", "idle"),
            "animation_phase": state.get("animation", {}).get("phase", 0.0),
        }
    return projection


def write_recording_manifest(
    output_dir: str | Path,
    *,
    snapshot_file: str = "office_snapshots.jsonl",
    scene_xml: str = "stretch_mujoco/models/office_scene2_multi_npc.xml",
    layout_manifest: str = "stretch_mujoco/models/assets/office_scenes/office_02_cross_axis.json",
    snapshot_fps: int = 10,
) -> Path:
    """Write recording metadata that render tools can inspect without a simulator."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "recording_manifest.json"
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_file": snapshot_file,
        "scene_xml": scene_xml,
        "layout_manifest": layout_manifest,
        "snapshot_fps": snapshot_fps,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
