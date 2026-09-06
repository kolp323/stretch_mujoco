"""Registry, routing, idempotency, and observations for all NPC controllers."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Mapping

import mujoco

from .animation import AnimationGraph
from .binding import NpcBinding, discover_npc_ids
from .controller import NpcController
from .protocol import CommandStatus, NpcCommand, NpcCommandKind, NpcCommandReceipt, NpcRuntimeState


class NpcSystem:
    def __init__(
        self,
        model: mujoco.MjModel,
        npc_ids: Iterable[str],
        animation_graphs: Mapping[str, AnimationGraph] | None = None,
    ) -> None:
        graphs = dict(animation_graphs or {})
        self.controllers = {
            npc_id: NpcController(model, NpcBinding.from_model(model, npc_id), graphs.get(npc_id))
            for npc_id in npc_ids
        }
        self._receipts: dict[str, NpcCommandReceipt] = {}
        self._pending_receipts: deque[NpcCommandReceipt] = deque()
        self._last_sequence: dict[str, int] = {}
        self._attachment_claims: dict[str, str] = {}
        self._commands: dict[str, NpcCommand] = {}

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "NpcSystem":
        return cls(model, discover_npc_ids(model))

    @classmethod
    def from_population(cls, model: mujoco.MjModel, population, manifest) -> "NpcSystem":
        """Create a system whose clip contract comes from validated population assets."""
        manifest.validate_population(population)
        graphs = {
            npc_id: manifest.animation_graph(
                definition.embodiment.bundle, definition.embodiment.animation_graph
            )
            for npc_id, definition in population.npcs.items()
        }
        return cls(model, population.npcs, graphs)

    def submit(self, command: NpcCommand) -> NpcCommandReceipt:
        previous = self._receipts.get(command.command_id)
        if previous is not None:
            return previous
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
            receipt = controller.cancel(target_id, command.issued_at)
            target_command = self._commands.get(target_id)
            if target_command is not None and receipt.status == CommandStatus.CANCELLED:
                self._release_failed_claim(target_command)
        else:
            receipt = controller.accept(command)
        self._commands[command.command_id] = command
        if receipt.status == CommandStatus.FAILED:
            self._release_failed_claim(command)
        return self._record(receipt)

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, sim_time: float) -> None:
        del model
        for controller in self.controllers.values():
            receipt = controller.step(data, sim_time)
            if receipt is not None:
                command = self._commands.get(receipt.command_id)
                if command is not None:
                    if receipt.status != CommandStatus.SUCCEEDED:
                        self._release_failed_claim(command)
                    elif command.kind == NpcCommandKind.DETACH_OBJECT:
                        self._attachment_claims.pop(str(command.payload["object"]), None)
                self._record(receipt)

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
                finished_at=receipt.finished_at,
            )
            controller.last_receipt = receipt
            command = self._commands.get(receipt.command_id)
            if command is not None:
                self._release_failed_claim(command)
            self._record(receipt)

    def _record(self, receipt: NpcCommandReceipt) -> NpcCommandReceipt:
        self._receipts[receipt.command_id] = receipt
        self._pending_receipts.append(receipt)
        return receipt

    def _release_failed_claim(self, command: NpcCommand) -> None:
        if command.kind == NpcCommandKind.ATTACH_OBJECT:
            object_name = str(command.payload.get("object", ""))
            if self._attachment_claims.get(object_name) == command.npc_id:
                self._attachment_claims.pop(object_name, None)
