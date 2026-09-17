"""Receipt-gated three-NPC/LLM benchmark in one active generated office."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from collections import deque
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import cv2
import mujoco
import numpy as np

from stretch_mujoco.humanoid.navigation import NavigationPathError, OfficeNavigationMesh
from stretch_mujoco.agents import (
    ActionCommand,
    ActionExecution,
    ActionType,
    ConversationRequest,
    ExecutionStatus,
    LLMTrigger,
)
from stretch_mujoco.agents.simulation_bridge import create_mujoco_action_driver
from stretch_mujoco.npc.composition import load_composed_npc_runtime
from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "stretch_mujoco/models/generated_scene_npc/active/active_catalog.json"
SCENE_ID = "office_01_linear_bench"
DT = 0.05


class TimelineVideoRecorder:
    """Fixed global top-down MuJoCo recorder with three receipt timelines."""

    def __init__(self, destination: Path, model, *, width: int, height: int, fps: int,
                 title: str = "3-NPC ACTIVE OFFICE", camera_azimuth: float = 90.0) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("video width, height, and fps must be positive")
        self.destination = destination.resolve()
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height, self.fps = width, height, fps
        self.title = title
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.option)
        self.option.flags[:] = 0
        self.option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
        self.option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
        self.option.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 1
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth = camera_azimuth
        self.camera.elevation = -80.0
        self.camera.lookat[:] = model.stat.center
        self.camera.lookat[2] = 0.6
        extent = max(float(model.stat.extent), 1.0)
        self.camera.distance = max(7.0, extent * 1.7)
        temporary = tempfile.NamedTemporaryFile(
            dir=self.destination.parent, prefix=f".{self.destination.stem}.",
            suffix=".mp4", delete=False,
        )
        temporary.close()
        self.temporary = Path(temporary.name)
        self.writer = cv2.VideoWriter(
            str(self.temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not self.writer.isOpened():
            self.temporary.unlink(missing_ok=True)
            self.renderer.close()
            raise RuntimeError("benchmark_video_writer_open_failed")
        self.frames = 0

    @staticmethod
    def _line(image: np.ndarray, text: str, origin: tuple[int, int],
              color: tuple[int, int, int], max_width: int) -> None:
        clipped = text[:96]
        scale = 0.42
        measured = cv2.getTextSize(clipped, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if measured > max_width:
            scale = max(0.28, scale * max_width / measured)
        cv2.putText(image, clipped, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)

    def capture(self, data, *, clock: float, physics_step: int,
                timeline: dict[str, deque[str]]) -> None:
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.option)
        image = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        panel_height = min(205, max(170, self.height // 4))
        panel_top = self.height - panel_height
        overlay = image.copy()
        cv2.rectangle(overlay, (0, panel_top), (self.width, self.height), (8, 14, 22), -1)
        cv2.addWeighted(overlay, 0.86, image, 0.14, 0, image)
        clock_minutes = int(round(clock)) % (24 * 60)
        workday_clock = f"{clock_minutes // 60:02d}:{clock_minutes % 60:02d}"
        self._line(image, f"{self.title} | GLOBAL TOP-DOWN | clock={workday_clock} | physics={physics_step}",
                   (20, 30), (245, 248, 250), self.width - 40)
        columns = (("LLM TIMELINE", "llm", (113, 207, 255)),
                   ("PHYSICS TIMELINE", "physics", (163, 231, 184)),
                   ("ACTION RECEIPTS", "receipts", (255, 213, 122)))
        col_width = self.width // 3
        for index, (title, key, color) in enumerate(columns):
            x = index * col_width + 20
            self._line(image, title, (x, panel_top + 28), color, col_width - 40)
            lines = list(timeline[key])[-4:]
            for row, line in enumerate(lines):
                self._line(image, line, (x, panel_top + 55 + row * 28),
                           (224, 232, 238), col_width - 40)
        self.writer.write(image)
        self.frames += 1

    def close(self) -> dict[str, Any]:
        self.writer.release()
        self.renderer.close()
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(self.temporary),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-an", str(self.destination),
        ], check=True)
        self.temporary.unlink(missing_ok=True)
        return {"path": str(self.destination), "frames": self.frames,
                "fps": self.fps, "width": self.width, "height": self.height,
                "camera": {"type": "free", "azimuth": float(self.camera.azimuth),
                           "elevation": -80.0,
                           "mode": "global_top_down"}}


class Transport:
    """The only NpcCommand transport; its single driver owns every sequence."""

    def __init__(self, model, data, system) -> None:
        self.model, self.data, self.system = model, data, system
        self.time = 0.0
        self.physics_steps = 0
        self.receipts = []

    def pull_status(self):
        return SimpleNamespace(time=self.time)

    def submit_npc_command(self, command):
        self.system.submit(command)
        self.receipts.extend(self.system.drain_receipts())
        return command.command_id

    def pull_npc_receipts(self):
        result = tuple(self.receipts)
        self.receipts.clear()
        return result

    def cancel_npc_command(self, npc_id, command_id) -> None:
        from stretch_mujoco.npc import NpcCommand, NpcCommandKind

        command = NpcCommand(
            f"cancel:{command_id}", self.system.next_sequence(npc_id), npc_id,
            NpcCommandKind.CANCEL, {"command_id": command_id}, self.time,
        )
        self.system.submit(command)
        self.receipts.extend(self.system.drain_receipts())

    def step(self, seconds: float = DT) -> None:
        self.time += seconds
        self.data.time = self.time
        self.system.step(self.model, self.data, self.time)
        mujoco.mj_forward(self.model, self.data)
        self.receipts.extend(self.system.drain_receipts())
        self.physics_steps += 1


class DeterministicAsyncLLM:
    """Injectable threaded provider with one session per NPC."""

    def __init__(self, agents: tuple[str, ...], latency: float) -> None:
        self.sessions = {agent: f"benchmark-session:{agent}" for agent in agents}
        self.latency = latency
        self.pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="npc-llm")
        self.pending: dict[str, tuple[Any, Future, int]] = {}
        self.attempts: dict[str, int] = {}

    def submit(self, request, physics_step: int) -> None:
        attempt = self.attempts.get(request.request_id, 0) + 1
        self.attempts[request.request_id] = attempt

        def invoke() -> dict[str, Any]:
            time.sleep(self.latency)
            if attempt <= int(request.context.get("failures", 0)):
                raise RuntimeError("injected_provider_failure")
            return dict(request.context["response"])

        self.pending[request.request_id] = (request, self.pool.submit(invoke), physics_step)

    def collect(self):
        done = []
        for request_id, (request, future, started) in tuple(self.pending.items()):
            if not future.done():
                continue
            self.pending.pop(request_id)
            try:
                done.append((request, future.result(), None, started))
            except Exception as error:
                done.append((request, None, str(error), started))
        return done

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)


def _active_files():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    build = CATALOG.parent / catalog["build_root"]
    entry = catalog["scenes"][SCENE_ID]
    return catalog, build, entry


def _site(model, data, name: str) -> tuple[float, float]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"benchmark_fixture_missing_site:{name}")
    return tuple(float(value) for value in data.site_xpos[site_id, :2])


def _fixture(directory: Path) -> dict[str, Any]:
    """Create a temporary, model-validated projection; never mutate active files."""
    catalog, build, entry = _active_files()
    source_population_path = build / entry["population"]
    source_semantic_path = build / entry["semantic_v1"]
    scene_path = build / entry["scene"]
    population = json.loads(source_population_path.read_text(encoding="utf-8"))
    agent_ids = tuple(population["npcs"])
    source_capability = dict(population["interaction_capabilities"]["conversation"])
    semantics = json.loads(source_semantic_path.read_text(encoding="utf-8"))
    semantic_v2 = json.loads((build / entry["semantic_v2"]).read_text(encoding="utf-8"))
    entities, points = semantic_v2["entities"], semantic_v2["points"]
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def entity_points(entity_id: str, kind: str) -> list[dict[str, Any]]:
        return [points[item] for item in entities[entity_id]["points"][kind]]

    profile = NpcTrajectoryProfile.from_json(build / entry["trajectory_profile"])
    navigation = OfficeNavigationMesh.from_model(
        model, data, resolution=profile.resolution,
        agent_radius=profile.agent_radius + profile.clearance,
        floor_geom_name=profile.navigation_surface,
        exclude_body_roots=profile.exclude_body_roots,
    )
    semantic_sites = {point["site"] for point in semantics["interaction_points"].values()}
    spawn_positions = {
        npc_id: _site(model, data, definition["spawn"]["site"])
        for npc_id, definition in population["npcs"].items()
    }

    def route_is_clear(start, target) -> bool:
        try:
            path = navigation.plan(start, target)
        except NavigationPathError:
            return False
        for source, destination in zip(path, path[1:]):
            distance = math.dist(source[:2], destination[:2])
            samples = max(2, int(math.ceil(distance / (navigation.resolution / 2.0))) + 1)
            for index in range(samples):
                ratio = index / (samples - 1)
                point = (
                    source[0] + ratio * (destination[0] - source[0]),
                    source[1] + ratio * (destination[1] - source[1]),
                )
                if not navigation.is_world_free(point, component_id=navigation.primary_component_id):
                    return False
        return True

    authored_points = []
    for point_id, point in points.items():
        if not point.get("site"):
            continue
        try:
            authored_points.append((point_id, point, _site(model, data, point["site"])))
        except ValueError:
            continue
    feasible_pairs = []
    for speaker_id, speaker_point, speaker_xy in authored_points:
        if not route_is_clear(spawn_positions[agent_ids[1]], speaker_xy):
            continue
        for listener_id, listener_point, listener_xy in authored_points:
            role_distance = math.dist(listener_xy, speaker_xy)
            if listener_id == speaker_id or not 0.45 <= role_distance <= 0.95:
                continue
            if route_is_clear(spawn_positions[agent_ids[2]], listener_xy):
                feasible_pairs.append(
                    (role_distance, speaker_id, speaker_point, speaker_xy,
                     listener_id, listener_point, listener_xy)
                )
    if not feasible_pairs:
        failure = {"status": "blocked", "reason": "no_route_feasible_conversation_pair",
                   "scene_id": SCENE_ID,
                   "candidates": [point_id for point_id, _, _ in authored_points],
                   "semantic_sites": sorted(semantic_sites)}
        (directory / "benchmark.failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise AssertionError(json.dumps(failure, sort_keys=True))
    # Prefer the widest valid envelope: the shorter authored pairs terminate
    # at furniture ingress points whose final segment is statically occupied;
    # this remains a preflighted choice, not a relaxed runtime threshold.
    best_distance = max(item[0] for item in feasible_pairs)
    role_distance, speaker_point_id, speaker, speaker_xy, meeting_point_id, meeting, meeting_xy = min(
        (item for item in feasible_pairs if item[0] == best_distance),
        key=lambda item: (item[1], item[4]),
    )
    bearing = math.atan2(meeting_xy[0] - speaker_xy[0], -(meeting_xy[1] - speaker_xy[1]))
    population["interaction_templates"] = {
        "conversation": {
            "speaker": {"site": speaker["site"], "yaw": bearing},
            "listener": {"site": meeting["site"], "yaw": math.remainder(bearing + math.pi, 2 * math.pi)},
        }
    }
    population["interaction_capabilities"]["conversation"] = {
        "status": "benchmark_fixture", "reason": "model_validated_existing_role_sites"
    }
    agents = tuple(population["npcs"])
    for agent in agents:
        capabilities = population["npcs"][agent]["capabilities"]
        if "conversation" not in capabilities:
            capabilities.append("conversation")
    if "object_handover" not in population["npcs"][agent_ids[1]]["capabilities"]:
        population["npcs"][agent_ids[1]]["capabilities"].append("object_handover")

    chairs = sorted(key for key, value in entities.items() if value["semantic_class"] == "furniture.seat")
    for chair in chairs:
        action = entity_points(chair, "action")[0]
        approach = entity_points(chair, "navigation")[0]
        _site(model, data, action["site"]); _site(model, data, approach["site"])
        semantics["interaction_points"][f"benchmark.{chair}.sit"] = {
            "owner": chair, "role": "chair_sit_site", "site": action["site"],
            "attributes": {"yaw": float(action["yaw"])},
        }
        semantics["interaction_points"][f"benchmark.{chair}.approach"] = {
            "owner": chair, "role": "human_stand_site", "site": approach["site"],
            "attributes": {"binding": "seat_navigation", "target": chair, "yaw": float(approach["yaw"])},
        }

    workstation = "object.new_cb_desk_2400.0"
    workstation_point = entity_points(workstation, "navigation")[0]
    work_xy = _site(model, data, workstation_point["site"])
    work_chair = min(chairs, key=lambda chair: math.dist(
        work_xy, _site(model, data, entity_points(chair, "action")[0]["site"])
    ))
    rest_chair = next(chair for chair in chairs if chair != work_chair)
    # The production bridge deliberately requires one work site for every
    # semantic Workstation, including the work zone.  Project each one onto an
    # existing, model-validated approach site; do not weaken that contract.
    for owner, semantic_object in sorted(semantics["objects"].items()):
        if semantic_object["type"] != "Workstation":
            continue
        candidates_for_owner = [
            point for point in semantics["interaction_points"].values()
            if point["owner"] == owner and point["role"] == "human_stand_site"
        ]
        if not candidates_for_owner:
            raise ValueError(f"benchmark_fixture_missing_workstation_approach:{owner}")
        point = candidates_for_owner[-1] if owner == "zone.work" else candidates_for_owner[0]
        _site(model, data, point["site"])
        semantics["interaction_points"][f"benchmark.{owner}.work"] = {
            "owner": owner, "role": "desk_work_site", "site": point["site"],
            "attributes": {"binding": "location", "target": owner,
                           "yaw": float(point["attributes"].get("yaw", 0.0))},
        }
    semantics["relations"].append({"subject": work_chair, "relation": "NEAR", "object": workstation})

    snack_point = next(point for point in semantics["interaction_points"].values()
                       if point["owner"] == "zone.snack" and point["role"] == "human_stand_site")
    _site(model, data, snack_point["site"])
    snack_point["attributes"].update({"binding": "location", "target": "zone.snack"})

    bottle_entity = next(key for key, value in sorted(entities.items())
                         if "bottle" in value["labels"].get("canonical", "").lower())
    bottle = entities[bottle_entity]["source"]["xml_binding"]["name"]
    bottle_point = entity_points(bottle_entity, "navigation")[0]
    _site(model, data, bottle_point["site"])
    semantics["objects"][bottle] = {
        "type": "Snack", "binding": {"kind": "body", "name": bottle},
        "attributes": {"graspable": True, "available": True, "consumed": False},
    }
    semantics["interaction_points"]["benchmark.bottle.approach"] = {
        "owner": bottle, "role": "human_stand_site", "site": bottle_point["site"],
        "attributes": {"binding": "object_approach", "target": bottle, "yaw": float(bottle_point["yaw"])},
    }

    for field in ("scene", "asset_manifest", "appearance_catalog", "trajectory_profile"):
        if population.get(field):
            population[field] = str((source_population_path.parent / population[field]).resolve())
    population["clock"] = {"start": "09:00", "minutes_per_second": 0.01}
    population_path = directory / "benchmark.population.json"
    semantic_path = directory / "benchmark.semantic.json"
    population_path.write_text(json.dumps(population, indent=2), encoding="utf-8")
    semantic_path.write_text(json.dumps(semantics, indent=2), encoding="utf-8")
    return {
        "catalog": catalog, "population_path": population_path, "semantic_path": semantic_path,
        "source_capability": source_capability, "agents": agents, "workstation": workstation,
        "work_chair": work_chair, "rest_chair": rest_chair, "bottle": bottle,
        "conversation_point_ids": [speaker_point_id, meeting_point_id],
        "conversation_sites": [speaker["site"], meeting["site"]], "conversation_distance": role_distance,
    }


def run_benchmark(output: Path, *, provider_latency: float = 0.04,
                  video_output: Path | None = None, video_fps: int = 10,
                  video_width: int = 1280, video_height: int = 720) -> dict[str, Any]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture = _fixture(output.parent)
    composed = load_composed_npc_runtime(
        fixture["population_path"], output.with_name("benchmark.scene.xml"),
        semantic_world_path=fixture["semantic_path"],
    )
    model, system, runtime = composed.model, composed.npc_system, composed.office_runtime
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    population = NpcPopulation.from_json(fixture["population_path"])
    profile = NpcTrajectoryProfile.from_json(population.resolve_path(population.trajectory_profile))
    transport = Transport(model, data, system)
    agents = fixture["agents"]
    driver = create_mujoco_action_driver(
        transport, npc_ids=agents, trajectory_profile=profile,
        agent_locations={key: value.spawn.location for key, value in population.npcs.items()},
        world=composed.semantic_world, interaction_templates=runtime.population_interaction_templates,
    )
    runtime.action_driver = driver
    runtime.interaction_driver = driver
    for index, site in enumerate(fixture["conversation_sites"]):
        driver.location_sites[f"benchmark_conversation_{index}"] = site
    llm = DeterministicAsyncLLM(agents, provider_latency)
    report: dict[str, Any] = {
        "schema": "llm_multiagent_mujoco_benchmark/v2", "scene_id": SCENE_ID,
        "active_build_id": fixture["catalog"]["build_id"], "agents": list(agents),
        "fixture": {
            "used": True, "active_source_conversation_capability": fixture["source_capability"],
            "reason": "active source lacks registered conversation role sites",
            "role_point_ids": fixture["conversation_point_ids"],
            "model_validated_sites": fixture["conversation_sites"],
            "role_distance_m": fixture["conversation_distance"],
        },
        "events": [], "llm": [], "actions": [], "terminal_command_receipts": [],
        "conversation_receipts": {}, "collisions": [], "failures": [],
        "semantic_to_receipt": [], "orphans": [],
    }
    recorder = None
    timeline: dict[str, deque[str]] = {
        "llm": deque(maxlen=4),
        "physics": deque(maxlen=4),
        "receipts": deque(maxlen=4),
    }
    if video_output is not None:
        recorder = TimelineVideoRecorder(
            video_output, model, width=video_width, height=video_height, fps=video_fps
        )
        report["video"] = {"requested": True, "path": str(video_output.resolve())}
    events = []
    min_separation = math.inf
    physics_steps_while_llm_pending = 0
    shown_receipts: set[str] = set()
    root_bodies = {key: value.binding.body_id for key, value in system.controllers.items()}

    def owner(body_id: int) -> str | None:
        while body_id > 0:
            for agent, root in root_bodies.items():
                if body_id == root:
                    return agent
            body_id = int(model.body_parentid[body_id])
        return None

    def sample() -> None:
        nonlocal min_separation
        positions = {key: data.mocap_pos[value.binding.mocap_id, :2].copy()
                     for key, value in system.controllers.items()}
        min_separation = min(min_separation, *(math.dist(positions[a], positions[b])
            for index, a in enumerate(agents) for b in agents[index + 1:]))
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            first = owner(int(model.geom_bodyid[contact.geom1])); second = owner(int(model.geom_bodyid[contact.geom2]))
            if first and second and first != second:
                report["collisions"].append({"physics_step": transport.physics_steps,
                                              "pair": sorted((first, second)), "distance": float(contact.dist)})

    def runtime_snapshot() -> dict[str, Any]:
        snapshot = composed.semantic_world.pose_snapshot(model, data)
        # Conversation observations are timestamped in the office-runtime clock
        # (minutes), while MuJoCo data.time is physical seconds.
        snapshot["time"] = runtime.elapsed_minutes
        return snapshot

    def advance(steps: int = 1) -> None:
        nonlocal physics_steps_while_llm_pending
        for _ in range(steps):
            if llm.pending:
                physics_steps_while_llm_pending += 1
            transport.step()
            events.extend(runtime.tick(DT, runtime_snapshot()))
            sample()
            for receipt in driver.command_receipts():
                status = receipt.status.value
                if not receipt.status.terminal or receipt.command_id in shown_receipts:
                    continue
                shown_receipts.add(receipt.command_id)
                reason = f" ({receipt.reason})" if receipt.reason else ""
                timeline["receipts"].append(
                    f"{receipt.npc_id}: {status} {receipt.command_id[:18]}{reason}"
                )
            timeline["physics"].append(
                f"t={transport.time:05.2f}s step={transport.physics_steps} advanced"
            )
            if recorder is not None:
                recorder.capture(
                    data,
                    clock=runtime.minute_of_day,
                    physics_step=transport.physics_steps,
                    timeline=timeline,
                )

    def wait(predicate, label: str, limit: int = 2400) -> None:
        for _ in range(limit):
            advance()
            if predicate():
                return
        raise AssertionError(f"benchmark_timeout:{label}")

    requests = {}

    def queue(agent: str, plan: str, response: dict[str, Any], failures: int = 0, trigger: LLMTrigger = LLMTrigger.REINTERPRET_PLAN):
        # The composed office emits day-start/automatic dialogue requests.
        # This benchmark owns its deterministic request schedule, so discard
        # those pending mailbox entries before selecting the request below;
        # otherwise a response can be applied to the wrong session.
        runtime.drain_llm_requests()
        request_id = f"benchmark:{agent}:{plan}"
        context = {"request_id": request_id, "session_id": llm.sessions[agent],
                   "correlation_id": f"workday:{plan}", "failures": failures,
                   "response": response}
        assert runtime.queue_llm_event(trigger, agent, context)
        request = runtime.drain_llm_requests()[0]
        requests[request_id] = request
        llm.submit(request, transport.physics_steps)
        timeline["llm"].append(f"{request.agent_id}: REQUEST {plan}")
        return request

    def resolve() -> None:
        replanned = set()
        while llm.pending:
            advance()
            for request, response, error, started in llm.collect():
                item = {"request_id": request.request_id, "session_id": request.session_id,
                        "correlation_id": request.correlation_id, "agent_id": request.agent_id,
                        "attempt": llm.attempts[request.request_id],
                        "physics_steps_waited": transport.physics_steps - started,
                        "status": "failed" if error else "succeeded"}
                if error:
                    item["error"] = error
                    report["failures"].append(dict(item))
                    timeline["llm"].append(
                        f"{request.agent_id}: FAIL attempt={item['attempt']}"
                    )
                    if request.request_id not in replanned:
                        replanned.add(request.request_id); item["recovery"] = "replan"
                        timeline["llm"].append(f"{request.agent_id}: REPLAN")
                        llm.submit(request, transport.physics_steps)
                    else:
                        item["recovery"] = "safe_idle"
                        timeline["llm"].append(f"{request.agent_id}: SAFE_IDLE")
                        response = {"action": {"action": "idle", "parameters": {"duration_seconds": 0.3}}}
                        result = runtime.apply_llm_response(request, response)
                        item["fallback_response"] = response
                        item["validation"] = {"valid": result.valid, "errors": list(result.errors)}
                else:
                    result = runtime.apply_llm_response(request, response)
                    item["response"] = response
                    item["validation"] = {"valid": result.valid, "errors": list(result.errors)}
                    timeline["llm"].append(
                        f"{request.agent_id}: OK attempt={item['attempt']}"
                    )
                applied = item.get("fallback_response", item.get("response"))
                if item.get("validation", {}).get("valid") and isinstance(applied, dict) and "action" in applied:
                    execution = runtime.agents[request.agent_id].executor
                    item["execution_id"] = execution.execution_id
                    item["payload_action"] = applied["action"]["action"]
                    timeline["receipts"].append(
                        f"{request.agent_id}: {item['payload_action']} -> {execution.execution_id[:14]}"
                    )
                report["llm"].append(item)

    clock_start = {"day": runtime.day, "minute_of_day": runtime.minute_of_day,
                   "elapsed_minutes": runtime.elapsed_minutes, "physics_seconds": transport.time}
    conversation_id = "benchmark:conversation"
    conversation = runtime.begin_conversation(ConversationRequest(
        conversation_id, (agents[1], agents[2]), "workday check-in", timeout=180.0,
        max_turns=1, semantic_snapshot=runtime_snapshot(),
    ))
    assert conversation.accepted, asdict(conversation)

    def conversation_diagnostic() -> dict[str, Any]:
        session = runtime.conversation(conversation_id)
        workflow = driver._conversations.get(conversation_id)
        receipt_by_id = {item.command_id: item for item in driver.command_receipts()}
        receipt_by_id.update({item.command_id: item for item in transport.receipts})
        command_ids = sorted(command_id for command_id in system._commands
                             if command_id.startswith(conversation_id))
        role_sites = ({} if workflow is None else dict(workflow.sites))
        if not role_sites:
            pair = driver.conversation_role_sites.get((agents[1], agents[2]))
            if pair:
                role_sites = dict(zip((agents[1], agents[2]), pair))
        participants = {}
        for participant in (agents[1], agents[2]):
            controller = system.controllers[participant]
            position = [float(value) for value in data.mocap_pos[controller.binding.mocap_id, :3]]
            quaternion = [float(value) for value in data.mocap_quat[controller.binding.mocap_id, :4]]
            site = role_sites.get(participant)
            site_pose = None
            if site:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
                rotation = data.site_xmat[site_id].reshape(3, 3)
                site_pose = {"site": site,
                             "position": [float(value) for value in data.site_xpos[site_id, :3]],
                             "yaw": math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))}
            participants[participant] = {
                "position": position, "quaternion": quaternion, "role_site": site_pose,
                "distance_to_role_site": None if site_pose is None else math.dist(position[:2], site_pose["position"][:2]),
            }
        return {
            "runtime": {"phase": session.phase.value, "status": session.status.value,
                        "failure_reason": session.failure_reason, "error": session.error},
            "driver_workflow_phase": None if workflow is None else workflow.stage,
            "receipts": [{**receipt_by_id[command_id].to_dict(),
                          "kind": system._commands[command_id].kind.value,
                          "payload": system._commands[command_id].payload}
                         if command_id in receipt_by_id else {
                             "command_id": command_id, "status": "missing_receipt",
                             "kind": system._commands[command_id].kind.value,
                             "payload": system._commands[command_id].payload}
                         for command_id in command_ids],
            "participants": participants,
            "participant_distance": math.dist(
                participants[agents[1]]["position"][:2], participants[agents[2]]["position"][:2]),
            "executions": {agent: {"execution_id": runtime.agents[agent].executor.execution_id,
                                    "status": runtime.agents[agent].executor.status.value,
                                    "phase": runtime.agents[agent].executor.phase}
                           for agent in (agents[1], agents[2])},
            "reservations": dict(runtime._participant_reservations),
            "active_commands": {agent: (None if system.controllers[agent].active_command is None
                                          else system.controllers[agent].active_command.command.to_dict())
                                for agent in (agents[1], agents[2])},
        }

    for _ in range(600):
        advance()
        session = runtime.conversation(conversation_id)
        if session.phase.value == "waiting_for_turn":
            break
        if session.status.terminal:
            raise AssertionError("conversation_terminal:" + json.dumps(conversation_diagnostic(), sort_keys=True))
    else:
        raise AssertionError("conversation_ready_timeout:" + json.dumps(conversation_diagnostic(), sort_keys=True))
    dialogue = {"dialogue": {"request_id": "benchmark:dialogue", "session_id": conversation_id,
        "turn_id": "benchmark:turn:1", "speaker": agents[1], "listener": agents[2],
        "act": "greeting", "text": "Workday benchmark check-in."}}
    queue(agents[1], "dialogue", dialogue, trigger=LLMTrigger.DIALOGUE)
    resolve()
    for _ in range(600):
        advance()
        if runtime.conversation(conversation_id).status.terminal:
            break
    else:
        raise AssertionError("conversation_turn_timeout:" + json.dumps({
            "conversation": asdict(runtime.conversation(conversation_id)),
            "llm": report["llm"][-3:],
            "events": [asdict(event) for event in events[-20:]],
        }, sort_keys=True, default=str))

    # These responses are the sole source of business action intent. They run
    # after the social receipt chain so the conversation participants begin
    # from their authored spawn poses rather than from a seated/occupied task.
    queue(agents[0], "work", {"action": {"action": "work", "target": fixture["workstation"],
                                           "parameters": {"duration_seconds": 2.0}}})
    queue(agents[1], "drink_move", {"action": {"action": "move_to", "target": "zone.snack"}}, failures=1)
    queue(agents[2], "rest_sit", {"action": {"action": "sit", "target": fixture["rest_chair"]}})
    resolve()
    wait(lambda: not runtime.agents[agents[1]].executor.is_busy, "drink_move")
    wait(lambda: not runtime.agents[agents[2]].executor.is_busy, "rest_sit")
    queue(agents[1], "drink_pickup", {"action": {"action": "pick_up", "target": fixture["bottle"]}})
    queue(agents[2], "persistent_failure", {"action": {"action": "move_to", "target": "zone.meeting"}}, failures=2)
    resolve()
    wait(lambda: not runtime.agents[agents[1]].executor.is_busy, "drink_pickup")
    wait(lambda: not runtime.agents[agents[2]].executor.is_busy, "safe_idle")
    queue(agents[1], "drink", {"action": {"action": "drink", "target": fixture["bottle"]}})
    resolve(); wait(lambda: not runtime.agents[agents[1]].executor.is_busy, "drink")
    # REST is a logical activity whose physical proof is sit -> timed idle -> stand-up.
    queue(agents[2], "rest_idle", {"action": {"action": "idle", "parameters": {"duration_seconds": 0.6}}})
    resolve(); wait(lambda: not runtime.agents[agents[2]].executor.is_busy, "rest_idle")
    queue(agents[2], "rest_stand", {"action": {"action": "stand_up", "target": fixture["rest_chair"]}})
    resolve(); wait(lambda: not runtime.agents[agents[2]].executor.is_busy, "rest_stand")
    wait(lambda: not runtime.agents[agents[0]].executor.is_busy
         and not runtime.agents[agents[0]].planner.action_queue, "work")

    # Advance the real runtime clock to 18:00 without pretending physics ran for nine hours.
    target = 18 * 60.0
    remaining = target - runtime.minute_of_day
    acceleration_seconds = 2.0
    runtime.minutes_per_second = remaining / acceleration_seconds
    advance(round(acceleration_seconds / DT))
    llm.close()
    if recorder is not None:
        report["video"] = recorder.close()

    terminal = {item.command_id: item for item in driver.command_receipts() if item.status.terminal}
    report["terminal_command_receipts"] = [item.to_dict() for item in sorted(terminal.values(), key=lambda x: x.command_id)]
    report["events"] = [asdict(event) for event in events]
    commits = [event for event in events if event.event == "semantic_commit"]
    for event in commits:
        execution_id = str(event.details["execution_id"])
        receipt_ids = sorted(key for key in terminal if key == execution_id or key.startswith(execution_id + ":"))
        trace = {"event_id": event.event_id, "execution_id": execution_id,
                 "action": event.details["action"], "receipt_ids": receipt_ids}
        report["semantic_to_receipt"].append(trace)
        if not receipt_ids or any(terminal[key].status.value != "succeeded" for key in receipt_ids):
            report["orphans"].append(trace)

    starts = [event for event in events if event.event == "action_started"]
    for event in starts:
        execution_id = str(event.details["execution_id"])
        causal = [item for item in report["llm"] if item.get("execution_id") == execution_id]
        if not causal:
            # Compound embodied actions emit child physical stages with fresh
            # execution IDs (for example work -> sit/work/stand).  Preserve
            # the LLM causation on those child receipts when the agent has a
            # single valid business decision in this benchmark segment.
            agent_decisions = [
                item for item in report["llm"]
                if item.get("agent_id") == event.agent_id
                and item.get("status") == "succeeded"
                and item.get("payload_action")
            ]
            if len(agent_decisions) == 1:
                causal = agent_decisions
        report["actions"].append({"event_id": event.event_id, "agent_id": event.agent_id,
            **event.details, "causal_request_ids": [item["request_id"] for item in causal],
            "causal_correlations": [item["correlation_id"] for item in causal],
            "causal_sessions": [item["session_id"] for item in causal],
            "receipt_ids": sorted(key for key in terminal if key == execution_id or key.startswith(execution_id + ":"))})

    conversation_receipts = {}
    for stage in ("conversation_approach", "conversation_align", "talk"):
        conversation_receipts[stage] = [item.to_dict() for key, item in terminal.items()
                                         if key.startswith(conversation_id) and stage in key]
    report["conversation_receipts"] = conversation_receipts
    threshold = 2.0 * (profile.agent_radius + profile.clearance)
    report["collision_policy"] = {"source": str(profile.source_path),
        "agent_radius_m": profile.agent_radius, "clearance_m": profile.clearance,
        "minimum_center_separation_m": threshold}
    report["min_npc_separation_m"] = min_separation
    report["collision_free"] = not report["collisions"] and min_separation >= threshold
    report["physics_steps_during_llm_wait"] = physics_steps_while_llm_pending
    report["clock"] = {"start": clock_start,
        "end": {"day": runtime.day, "minute_of_day": runtime.minute_of_day,
                "elapsed_minutes": runtime.elapsed_minutes, "physics_seconds": transport.time},
        "accelerated_minutes_per_physics_second": remaining / acceleration_seconds}
    report["activities"] = {
        "work": {"llm_request": f"benchmark:{agents[0]}:work",
                 "semantic_commits": [item for item in report["semantic_to_receipt"] if item["action"] == "work"]},
        "drink": {"llm_request": f"benchmark:{agents[1]}:drink", "object": fixture["bottle"],
                  "consumed": composed.semantic_world.object(fixture["bottle"]).get("consumed", False)},
        "rest": {"logical_activity": "rest", "physical_actions": ["sit", "idle", "stand_up"],
                 "llm_requests": [f"benchmark:{agents[2]}:{name}" for name in ("rest_sit", "rest_idle", "rest_stand")]},
        "conversation": {"session_id": conversation_id, "status": runtime.conversation(conversation_id).status.value},
    }
    recoveries = {item.get("recovery") for item in report["llm"]}
    action_names = {item["action"] for item in report["actions"]}
    report["passed"] = bool(
        len(set(llm.sessions.values())) == 3 and physics_steps_while_llm_pending > 0
        and {"replan", "safe_idle"} <= recoveries and not report["orphans"]
        and report["collision_free"] and abs(runtime.minute_of_day - target) < 1e-6
        and {"work", "drink", "sit", "idle", "stand_up"} <= action_names
        and report["activities"]["drink"]["consumed"]
        and runtime.conversation(conversation_id).status.value == "completed"
        and all(conversation_receipts[stage] and all(x["status"] == "succeeded" for x in conversation_receipts[stage])
                for stage in conversation_receipts)
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / "aaa_workspace/experiments/llm_multiagent_benchmark/report.json")
    parser.add_argument("--provider-latency", type=float, default=0.04)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    args = parser.parse_args()
    report = run_benchmark(
        args.output,
        provider_latency=args.provider_latency,
        video_output=args.video_output,
        video_fps=args.video_fps,
        video_width=args.video_width,
        video_height=args.video_height,
    )
    print(json.dumps({"output": str(args.output.resolve()), "video": report.get("video"),
                      "passed": report["passed"]}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
