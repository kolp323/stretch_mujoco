"""Registry, routing, idempotency, and observations for all NPC controllers."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Iterable, Mapping

import mujoco

from .animation import AnimationGraph
from .binding import NpcBinding, discover_npc_ids
from .controller import NpcController, TrajectoryRouteContract
from .locomotion import NavigationGeometryContract
from .trajectory_profile import NpcTrajectoryProfile
from .traffic import TrafficManager
from .journal import CommandJournal, JsonlCommandJournal
from .protocol import CommandStatus, NpcCommand, NpcCommandKind, NpcCommandReceipt, NpcRuntimeState
from .spawn_pool import SpawnAnchorPool


class NpcSystem:
    def __init__(
        self,
        model: mujoco.MjModel,
        npc_ids: Iterable[str],
        animation_graphs: Mapping[str, AnimationGraph] | None = None,
        simulation_seed: int = 0,
        trajectory_routes: Mapping[str, TrajectoryRouteContract] | None = None,
        navigation_geometry: NavigationGeometryContract | None = None,
        traffic_manager: TrafficManager | None = None,
        journal: CommandJournal | None = None,
    ) -> None:
        graphs = dict(animation_graphs or {})
        self.controllers = {
            npc_id: NpcController(
                model,
                NpcBinding.from_model(model, npc_id),
                graphs.get(npc_id),
                simulation_seed,
                trajectory_routes,
            )
            for npc_id in npc_ids
        }
        for controller in self.controllers.values():
            controller.locomotion.configure_navigation(navigation_geometry)
        self._receipts: dict[str, NpcCommandReceipt] = {}
        self._pending_receipts: deque[NpcCommandReceipt] = deque()
        self._last_sequence: dict[str, int] = {}
        self._attachment_claims: dict[str, str] = {}
        self._commands: dict[str, NpcCommand] = {}
        self._pending_cancel_targets: dict[str, set[str]] = {}
        self.traffic = traffic_manager or TrafficManager()
        self.spawn_pool = None
        self.journal = journal
        if self.journal is not None:
            for record in self.journal.records():
                self._last_sequence[record.npc_id] = max(self._last_sequence.get(record.npc_id, -1), record.sequence)

    def next_sequence(self, npc_id: str) -> int:
        # Reserve the value for external bridges, but do not advance the
        # accepted-command boundary until ``submit`` sees that command.
        return self._last_sequence.get(npc_id, -1) + 1

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        *,
        simulation_seed: int = 0,
        scene_path: str | Path | None = None,
    ) -> "NpcSystem":
        routes, navigation_geometry = cls._scene_trajectory_contract(model, scene_path)
        return cls(
            model,
            discover_npc_ids(model),
            simulation_seed=simulation_seed,
            trajectory_routes=routes,
            navigation_geometry=navigation_geometry,
        )

    @classmethod
    def from_population(
        cls,
        model: mujoco.MjModel,
        population,
        manifest,
        *,
        simulation_seed: int = 0,
        scene_path: str | Path | None = None,
    ) -> "NpcSystem":
        """Create a system whose clip contract comes from validated population assets."""
        manifest.validate_population(population)
        graphs = {
            npc_id: manifest.animation_graph(
                definition.embodiment.bundle, definition.embodiment.animation_graph
            )
            for npc_id, definition in population.npcs.items()
        }
        profile_scene = scene_path
        if profile_scene is None and getattr(population, "source_path", None) is not None:
            profile_scene = population.resolve_path(population.scene)
        # Schema-v2 population is authoritative.  XML custom text is retained
        # only for from_model() legacy scenes.
        routes, navigation_geometry = cls._population_trajectory_contract(
            model, population, profile_scene
        )
        system = cls(
            model,
            population.npcs,
            graphs,
            simulation_seed,
            routes,
            navigation_geometry,
        )
        policy = getattr(population, "spawn_policy", None)
        if policy is not None:
            system.spawn_pool = SpawnAnchorPool(model, policy)
        return system

    @staticmethod
    def _profile_contract(
        profile: NpcTrajectoryProfile,
    ) -> tuple[dict[str, TrajectoryRouteContract], NavigationGeometryContract]:
        return (
            {
                route.route_id: TrajectoryRouteContract(
                    route.route_id,
                    route.source,
                    profile.anchors[route.destination].site,
                    frozenset(route.actions),
                )
                for route in profile.routes
            },
            NavigationGeometryContract(
                surface=profile.navigation_surface,
                agent_radius=profile.agent_radius,
                clearance=profile.clearance,
                resolution=profile.resolution,
                exclude_body_roots=profile.exclude_body_roots,
            ),
        )

    @staticmethod
    def _population_trajectory_contract(model, population, scene_path):
        profile_name = getattr(population, "trajectory_profile", None)
        if profile_name is None:
            return {}, None
        if scene_path is None:
            raise ValueError("population_trajectory_profile_requires_scene_path")
        profile = NpcTrajectoryProfile.from_json(population.resolve_path(profile_name))
        profile.validate_scene(NpcSystem._profile_scene_path(Path(scene_path), profile.scene))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        profile.preflight(model, data)
        return NpcSystem._profile_contract(profile)

    @staticmethod
    def _scene_trajectory_contract(
        model: mujoco.MjModel, scene_path: str | Path | None
    ) -> tuple[dict[str, TrajectoryRouteContract], NavigationGeometryContract | None]:
        profile_id = NpcSystem._custom_text(model, "npc_trajectory_profile")
        if profile_id is None:
            return {}, None
        if scene_path is None:
            raise ValueError("npc_trajectory_profile_requires_scene_path")
        profile_path = Path(__file__).with_name("trajectory_profiles") / f"{profile_id}.json"
        if not profile_path.is_file():
            raise ValueError(f"npc_trajectory_profile_missing:{profile_id}")
        profile = NpcTrajectoryProfile.from_json(profile_path)
        profile.validate_scene(NpcSystem._profile_scene_path(Path(scene_path), profile.scene))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        profile.preflight(model, data)
        return NpcSystem._profile_contract(profile)

    @staticmethod
    def _custom_text(model: mujoco.MjModel, name: str) -> str | None:
        text_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, name)
        if text_id < 0:
            return None
        start = int(model.text_adr[text_id])
        size = int(model.text_size[text_id])
        return bytes(model.text_data[start : start + size]).rstrip(b"\0").decode("utf-8")

    @staticmethod
    def _profile_scene_path(scene_path: Path, expected_name: str) -> Path:
        """Find the profile's source scene through a generated wrapper include."""
        scene_path = scene_path.resolve()
        if scene_path.name == expected_name:
            return scene_path
        import xml.etree.ElementTree as ET

        root = ET.parse(scene_path).getroot()
        for include in root.findall("include"):
            candidate = Path(include.attrib.get("file", ""))
            if not candidate.is_absolute():
                candidate = scene_path.parent / candidate
            if candidate.name == expected_name and candidate.is_file():
                return candidate.resolve()
        # Population composition keeps the authored base scene byte-for-byte
        # behind generated wrappers.  Its receipt is the authoritative source
        # link when the wrapper no longer exposes the original include name.
        receipt_path = scene_path.with_suffix(".composition.json")
        if receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                candidate = Path(str(receipt.get("base_scene", ""))).resolve()
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                candidate = Path()
            if candidate.name == expected_name and candidate.is_file():
                return candidate
        raise ValueError(f"npc_trajectory_profile_scene_source_missing:{expected_name}")

    def submit(self, command: NpcCommand) -> NpcCommandReceipt:
        previous = self._receipts.get(command.command_id)
        if previous is not None:
            if self._commands.get(command.command_id) == command:
                return previous
            return NpcCommandReceipt(
                command.command_id,
                command.npc_id,
                CommandStatus.FAILED,
                reason="command_id_conflict",
                finished_at=command.issued_at,
            )
        controller = self.controllers.get(command.npc_id)
        if controller is None:
            receipt = NpcCommandReceipt(
                command.command_id,
                command.npc_id,
                CommandStatus.FAILED,
                reason="unknown_npc",
                finished_at=command.issued_at,
            )
            return self._record(receipt)
        if command.sequence <= self._last_sequence.get(command.npc_id, -1):
            receipt = NpcCommandReceipt(
                command.command_id,
                command.npc_id,
                CommandStatus.FAILED,
                reason="stale_sequence",
                finished_at=command.issued_at,
            )
            return self._record(receipt)
        self._last_sequence[command.npc_id] = command.sequence
        if self.journal is not None and hasattr(self.journal, "append_command"):
            self.journal.append_command(command, event="accepted")
        if command.kind == NpcCommandKind.MOVE_TO:
            route_id = command.payload.get("route_id")
            if route_id:
                revision = int(command.payload.get("route_revision", 0))
                if self.traffic.acquire(str(route_id), command.npc_id, revision) is None:
                    return self._record(NpcCommandReceipt(command.command_id, command.npc_id, CommandStatus.FAILED, reason="route_reserved", finished_at=command.issued_at))
        if command.kind == NpcCommandKind.ATTACH_OBJECT:
            object_name = str(command.payload.get("object", ""))
            owner = self._attachment_claims.get(object_name)
            if owner not in {None, command.npc_id}:
                return self._record(
                    NpcCommandReceipt(
                        command.command_id,
                        command.npc_id,
                        CommandStatus.FAILED,
                        reason=f"object_attached_by:{owner}",
                        finished_at=command.issued_at,
                    )
                )
            self._attachment_claims[object_name] = command.npc_id
        elif command.kind == NpcCommandKind.DETACH_OBJECT:
            object_name = str(command.payload.get("object", ""))
            if self._attachment_claims.get(object_name) != command.npc_id:
                return self._record(
                    NpcCommandReceipt(
                        command.command_id,
                        command.npc_id,
                        CommandStatus.FAILED,
                        reason="object_not_owned_by_npc",
                        finished_at=command.issued_at,
                    )
                )
        if command.kind == NpcCommandKind.CANCEL:
            target_id = str(command.payload.get("command_id", ""))
            target_receipt = controller.cancel(target_id, command.issued_at)
            target_command = self._commands.get(target_id)
            if target_command is not None and target_receipt.status == CommandStatus.CANCELLED:
                self._release_failed_claim(target_command)
            # A walking controller acknowledges cancellation at the next foot
            # marker.  Retain the cancel command's own receipt lifecycle until
            # that target command reaches its terminal receipt.
            self._commands[command.command_id] = command
            self._pending_cancel_targets.setdefault(target_id, set()).add(command.command_id)
            self._record(target_receipt)
            if target_receipt.status.terminal:
                return self._receipts[command.command_id]
            receipt = NpcCommandReceipt(
                command.command_id,
                command.npc_id,
                target_receipt.status,
                reason=target_receipt.reason,
                started_at=target_receipt.started_at,
                finished_at=target_receipt.finished_at,
            )
        else:
            receipt = controller.accept(command)
        self._commands[command.command_id] = command
        if receipt.status == CommandStatus.FAILED:
            self._release_failed_claim(command)
        return self._record(receipt)

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, sim_time: float) -> None:
        had_attachments = any(
            controller.attachments.held for controller in self.controllers.values()
        )
        for controller in self.controllers.values():
            receipt = controller.step(data, sim_time)
            if receipt is not None:
                command = self._commands.get(receipt.command_id)
                if command is not None:
                    if receipt.status != CommandStatus.SUCCEEDED:
                        self._release_failed_claim(command)
                    elif command.kind == NpcCommandKind.DETACH_OBJECT:
                        self._attachment_claims.pop(str(command.payload["object"]), None)
                    if receipt.status.terminal and command.kind == NpcCommandKind.MOVE_TO:
                        self.traffic.release(
                            command.npc_id,
                            None if command.payload.get("route_id") is None else str(command.payload["route_id"]),
                            int(command.payload.get("route_revision", 0)),
                        )
                self._record(receipt)
        has_attachments = any(
            controller.attachments.held for controller in self.controllers.values()
        )
        if had_attachments or has_attachments:
            # AttachmentController writes free-joint qpos after mj_step.  Refresh
            # derived body/site transforms before the camera or semantic state
            # reads them; otherwise the visible object can remain at its former
            # world pose even though ownership and qpos already changed.
            mujoco.mj_forward(model, data)

    def states(
        self, data: mujoco.MjData, sim_time: float | None = None
    ) -> dict[str, NpcRuntimeState]:
        timestamp = float(data.time if sim_time is None else sim_time)
        return {
            npc_id: controller.state(data, timestamp)
            for npc_id, controller in self.controllers.items()
        }

    def drain_receipts(self) -> tuple[NpcCommandReceipt, ...]:
        receipts = tuple(self._pending_receipts)
        self._pending_receipts.clear()
        return receipts

    def fail_active_commands(self, sim_time: float, reason: str) -> None:
        for controller in self.controllers.values():
            active = controller.active_command
            if active is None:
                continue
            receipt = controller.cancel(active.command.command_id, sim_time, reason)
            receipt = NpcCommandReceipt(
                receipt.command_id,
                receipt.npc_id,
                CommandStatus.FAILED,
                reason=reason,
                started_at=receipt.started_at,
                finished_at=sim_time,
            )
            controller.last_receipt = receipt
            command = self._commands.get(receipt.command_id)
            if command is not None:
                self._release_failed_claim(command)
            self._record(receipt)

    def _record(self, receipt: NpcCommandReceipt) -> NpcCommandReceipt:
        self._receipts[receipt.command_id] = receipt
        self._pending_receipts.append(receipt)
        if self.journal is not None and receipt.status.terminal:
            command = self._commands.get(receipt.command_id)
            if command is not None and hasattr(self.journal, "append_command"):
                self.journal.append_command(command, event="terminal", receipt=receipt.to_dict())
        if receipt.status.terminal:
            for cancel_id in self._pending_cancel_targets.pop(receipt.command_id, set()):
                cancel = self._commands.get(cancel_id)
                if cancel is None:
                    continue
                self._record(
                    NpcCommandReceipt(
                        cancel_id,
                        cancel.npc_id,
                        receipt.status,
                        reason=receipt.reason,
                        started_at=receipt.started_at,
                        finished_at=receipt.finished_at,
                    )
                )
        return receipt

    def _release_failed_claim(self, command: NpcCommand) -> None:
        if command.kind == NpcCommandKind.MOVE_TO:
            self.traffic.release(
                command.npc_id,
                None if command.payload.get("route_id") is None else str(command.payload["route_id"]),
                int(command.payload.get("route_revision", 0)),
            )
        if command.kind == NpcCommandKind.ATTACH_OBJECT:
            object_name = str(command.payload.get("object", ""))
            if self._attachment_claims.get(object_name) == command.npc_id:
                self._attachment_claims.pop(object_name, None)
