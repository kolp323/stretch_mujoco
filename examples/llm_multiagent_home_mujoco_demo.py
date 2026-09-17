"""Three-NPC LLM household demo in one active generated home."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import mujoco

from examples.llm_multiagent_mujoco_benchmark import (
    DT,
    DeterministicAsyncLLM,
    TimelineVideoRecorder,
    Transport,
)
from stretch_mujoco.agents import ExecutionStatus, LLMTrigger
from stretch_mujoco.agents.simulation_bridge import create_mujoco_action_driver
from stretch_mujoco.npc.composition import load_composed_npc_runtime
from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "stretch_mujoco/models/generated_scene_npc/active/active_catalog.json"
SCENE_ID = "home_04_103997970_171031287"


def _fixture(directory: Path) -> dict[str, Any]:
    """Create an evening-clock projection without modifying active artifacts."""
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    build = CATALOG.parent / catalog["build_root"]
    entry = catalog["scenes"][SCENE_ID]
    source = build / entry["population"]
    payload = json.loads(source.read_text(encoding="utf-8"))
    for field in ("scene", "asset_manifest", "appearance_catalog", "trajectory_profile"):
        value = payload.get(field)
        if isinstance(value, str):
            payload[field] = str((source.parent / value).resolve())
    payload["clock"] = {"start": "18:00", "minutes_per_second": 0.25}
    population_path = directory / "home_demo.population.json"
    population_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "catalog": catalog,
        "entry": entry,
        "build": build,
        "population_path": population_path,
        "semantic_path": build / entry["semantic_v1"],
        "capabilities": payload["interaction_capabilities"],
    }


def run_demo(output: Path, video_output: Path, *, provider_latency: float = 0.02,
             fps: int = 10, width: int = 1280, height: int = 720) -> dict[str, Any]:
    output = output.resolve()
    video_output = video_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture = _fixture(output.parent)
    composed = load_composed_npc_runtime(
        fixture["population_path"], output.with_name("home_demo.scene.xml"),
        semantic_world_path=fixture["semantic_path"], simulation_seed=20260917,
    )
    model, system, runtime = composed.model, composed.npc_system, composed.office_runtime
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    population = NpcPopulation.from_json(fixture["population_path"])
    profile = NpcTrajectoryProfile.from_json(population.resolve_path(population.trajectory_profile))
    agents = tuple(population.npcs)
    if len(agents) != 3:
        raise ValueError(f"home_demo_requires_three_npcs:{len(agents)}")
    transport = Transport(model, data, system)
    driver = create_mujoco_action_driver(
        transport, npc_ids=agents, trajectory_profile=profile,
        agent_locations={key: value.spawn.location for key, value in population.npcs.items()},
    )
    runtime.action_driver = driver
    runtime.interaction_driver = driver

    route_specs = (
        (agents[0], "coverage_00_0086", "room.kitchen", "prepare dinner"),
        (agents[1], "coverage_01_0087", "room.living_room", "relax in living room"),
        (agents[2], "coverage_02_0085", "room.bedroom", "evening rest"),
    )
    routes = {route.route_id: route for route in profile.routes}
    llm = DeterministicAsyncLLM(agents, provider_latency)
    llm.sessions = {agent: f"home-demo-session:{agent}" for agent in agents}
    recorder = TimelineVideoRecorder(
        video_output, model, width=width, height=height, fps=fps,
        title="3-NPC ACTIVE HOME", camera_azimuth=270.0,
    )
    timeline: dict[str, deque[str]] = {
        "llm": deque(maxlen=4), "physics": deque(maxlen=4), "receipts": deque(maxlen=4)
    }
    report: dict[str, Any] = {
        "schema": "llm_multiagent_home_demo/v1",
        "scene_id": SCENE_ID,
        "active_build_id": fixture["catalog"]["build_id"],
        "agents": list(agents),
        "sessions": dict(llm.sessions),
        "interaction_capabilities": fixture["capabilities"],
        "llm": [], "events": [], "actions": [], "terminal_command_receipts": [],
        "collisions": [], "failures": [],
    }
    events = []
    shown_receipts: set[str] = set()
    min_separation = math.inf
    physics_steps_while_llm_pending = 0
    root_bodies = {key: value.binding.body_id for key, value in system.controllers.items()}

    def owner(body_id: int) -> str | None:
        while body_id > 0:
            for agent, root in root_bodies.items():
                if body_id == root:
                    return agent
            body_id = int(model.body_parentid[body_id])
        return None

    def snapshot() -> dict[str, Any]:
        value = composed.semantic_world.pose_snapshot(model, data)
        value["time"] = runtime.elapsed_minutes
        return value

    def advance(steps: int = 1) -> None:
        nonlocal min_separation, physics_steps_while_llm_pending
        for _ in range(steps):
            if llm.pending:
                physics_steps_while_llm_pending += 1
            transport.step()
            events.extend(runtime.tick(DT, snapshot()))
            positions = {key: data.mocap_pos[value.binding.mocap_id, :2].copy()
                         for key, value in system.controllers.items()}
            min_separation = min(min_separation, *(math.dist(positions[a], positions[b])
                for index, a in enumerate(agents) for b in agents[index + 1:]))
            for index in range(int(data.ncon)):
                contact = data.contact[index]
                first = owner(int(model.geom_bodyid[contact.geom1]))
                second = owner(int(model.geom_bodyid[contact.geom2]))
                if first and second and first != second:
                    report["collisions"].append({
                        "physics_step": transport.physics_steps,
                        "pair": sorted((first, second)), "distance": float(contact.dist),
                    })
            for receipt in driver.command_receipts():
                if not receipt.status.terminal or receipt.command_id in shown_receipts:
                    continue
                shown_receipts.add(receipt.command_id)
                reason = f" ({receipt.reason})" if receipt.reason else ""
                timeline["receipts"].append(
                    f"{receipt.npc_id}: {receipt.status.value} {receipt.command_id[:18]}{reason}"
                )
            timeline["physics"].append(
                f"t={transport.time:05.2f}s step={transport.physics_steps} advanced"
            )
            if transport.physics_steps % 2 == 0:
                recorder.capture(data, clock=runtime.minute_of_day,
                                 physics_step=transport.physics_steps, timeline=timeline)

    def wait_idle(agent: str, label: str, limit: int = 3000) -> None:
        for _ in range(limit):
            advance()
            state = runtime.agents[agent]
            if not state.executor.is_busy and not state.planner.action_queue:
                return
            if state.executor.status in {
                ExecutionStatus.FAILED, ExecutionStatus.CANCELLED, ExecutionStatus.TIMED_OUT
            }:
                raise RuntimeError(f"home_demo_action_failed:{label}:{state.executor.error}")
        raise TimeoutError(f"home_demo_action_timeout:{label}")

    def queue(agent: str, plan: str, response: dict[str, Any], failures: int = 0) -> None:
        runtime.drain_llm_requests()
        request_id = f"home-demo:{agent}:{plan}"
        context = {
            "request_id": request_id, "session_id": llm.sessions[agent],
            "correlation_id": f"household:{plan}", "failures": failures,
            "response": response,
        }
        if not runtime.queue_llm_event(LLMTrigger.REINTERPRET_PLAN, agent, context):
            raise RuntimeError(f"home_demo_llm_queue_rejected:{request_id}")
        request = runtime.drain_llm_requests()[0]
        llm.submit(request, transport.physics_steps)
        timeline["llm"].append(f"{agent}: REQUEST {plan}")

    def resolve() -> None:
        replanned: set[str] = set()
        while llm.pending:
            advance()
            for request, response, error, started in llm.collect():
                item = {
                    "request_id": request.request_id, "session_id": request.session_id,
                    "correlation_id": request.correlation_id, "agent_id": request.agent_id,
                    "attempt": llm.attempts[request.request_id],
                    "physics_steps_waited": transport.physics_steps - started,
                    "status": "failed" if error else "succeeded",
                }
                if error:
                    item["error"] = error
                    report["failures"].append(dict(item))
                    timeline["llm"].append(f"{request.agent_id}: FAIL attempt={item['attempt']}")
                    if request.request_id not in replanned:
                        replanned.add(request.request_id)
                        item["recovery"] = "replan"
                        timeline["llm"].append(f"{request.agent_id}: REPLAN")
                        llm.submit(request, transport.physics_steps)
                        report["llm"].append(item)
                        continue
                    raise RuntimeError(f"home_demo_provider_failed:{request.request_id}")
                result = runtime.apply_llm_response(request, response)
                item["response"] = response
                item["validation"] = {"valid": result.valid, "errors": list(result.errors)}
                if not result.valid:
                    raise RuntimeError(f"home_demo_response_rejected:{result.errors}")
                execution = runtime.agents[request.agent_id].executor
                item["execution_id"] = execution.execution_id
                item["payload_action"] = response["action"]["action"]
                timeline["llm"].append(f"{request.agent_id}: OK attempt={item['attempt']}")
                timeline["receipts"].append(
                    f"{request.agent_id}: {item['payload_action']} -> {execution.execution_id[:14]}"
                )
                report["llm"].append(item)

    recorder.capture(data, clock=runtime.minute_of_day, physics_step=0, timeline=timeline)
    try:
        for index, (agent, route_id, target, activity) in enumerate(route_specs):
            route = routes[route_id]
            driver.agent_locations[agent] = route.source
            driver.location_sites["zone.home"] = profile.anchors[route.destination].site
            response = {"action": {"action": "move_to", "target": "zone.home", "parameters": {
                "trajectory_route": route.route_id, "trajectory_source": route.source,
            }}}
            queue(agent, activity, response, failures=1 if index == 2 else 0)
            resolve()
            wait_idle(agent, activity)
            report["actions"].append({
                "agent_id": agent, "activity": activity, "route_id": route_id,
                "source": route.source, "destination": route.destination,
                "target": target, "semantic_target": "zone.home", "status": "succeeded",
            })
            advance(round(0.8 / DT))
        advance(round(1.5 / DT))
    finally:
        llm.close()
        report["video"] = recorder.close()

    terminal = {item.command_id: item for item in driver.command_receipts() if item.status.terminal}
    report["terminal_command_receipts"] = [
        item.to_dict() for item in sorted(terminal.values(), key=lambda item: item.command_id)
    ]
    report["events"] = [asdict(event) for event in events]
    threshold = 2.0 * (profile.agent_radius + profile.clearance)
    report["collision_policy"] = {
        "agent_radius_m": profile.agent_radius, "clearance_m": profile.clearance,
        "minimum_center_separation_m": threshold,
    }
    report["min_npc_separation_m"] = min_separation
    report["collision_free"] = not report["collisions"] and min_separation >= threshold
    report["physics_steps_during_llm_wait"] = physics_steps_while_llm_pending
    report["clock"] = {
        "start": "18:00", "end_minute_of_day": runtime.minute_of_day,
        "physics_seconds": transport.time,
    }
    recoveries = {item.get("recovery") for item in report["llm"]}
    report["passed"] = bool(
        len(set(llm.sessions.values())) == 3
        and len(report["actions"]) == 3
        and all(item["status"] == "succeeded" for item in report["actions"])
        and physics_steps_while_llm_pending > 0
        and "replan" in recoveries
        and report["collision_free"]
        and terminal
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    output_root = ROOT / "aaa_workspace/experiments/llm_multiagent_home_demo"
    parser.add_argument("--output", type=Path, default=output_root / "report.json")
    parser.add_argument("--video-output", type=Path,
                        default=output_root / "three_npc_home_global_topdown.mp4")
    parser.add_argument("--provider-latency", type=float, default=0.02)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    report = run_demo(args.output, args.video_output, provider_latency=args.provider_latency,
                      fps=args.fps, width=args.width, height=args.height)
    print(json.dumps({"output": str(args.output.resolve()), "video": report["video"],
                      "passed": report["passed"]}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
