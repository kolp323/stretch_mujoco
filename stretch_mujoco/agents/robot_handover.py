"""Bridge a robot release receipt into the shared interaction protocol."""

from __future__ import annotations

from dataclasses import dataclass

from stretch_mujoco.npc import CommandStatus, NpcCommand, NpcCommandKind

from .drivers import NpcSimulatorClient
from .interactions import InteractionCoordinator, InteractionSession


@dataclass
class RobotHandover:
    session: InteractionSession
    command_id: str


class RobotToNpcHandoverBridge:
    """Attach an object only after a robot executor confirms its physical release.

    The explicit ``robot_release_confirmed`` input is intentional: changing the
    robot grasp proxy is not itself a physical receipt. A real or mock robot
    executor must verify release before crossing this boundary.
    """

    def __init__(self, simulator: NpcSimulatorClient) -> None:
        self.simulator = simulator
        self.coordinator = InteractionCoordinator()
        self._sequences: dict[str, int] = {}
        self._handovers: dict[str, RobotHandover] = {}

    def accept_released_object(
        self,
        robot_id: str,
        npc_id: str,
        object_name: str,
        *,
        robot_release_confirmed: bool,
        session_id: str,
    ) -> RobotHandover:
        if not robot_release_confirmed:
            raise ValueError("Robot release must be physically confirmed before NPC attachment")
        session = self.coordinator.start(
            "handover",
            (robot_id, npc_id),
            object_name,
            (("released", (robot_id,)), ("received", (npc_id,))),
            session_id=session_id,
        )
        self.coordinator.acknowledge(session.session_id, robot_id, "released")
        sequence = self._sequences.get(npc_id, -1) + 1
        self._sequences[npc_id] = sequence
        issued_at = float(self.simulator.pull_status().time)
        command = NpcCommand(
            f"{session_id}:receive",
            sequence,
            npc_id,
            NpcCommandKind.ATTACH_OBJECT,
            {"object": object_name, "interaction_id": session_id},
            issued_at,
            issued_at + 30.0,
        )
        handover = RobotHandover(session, self.simulator.submit_npc_command(command))
        self._handovers[session_id] = handover
        return handover

    def poll(self, session_id: str) -> InteractionSession:
        handover = self._handovers[session_id]
        for receipt in self.simulator.pull_npc_receipts():
            if receipt.command_id != handover.command_id:
                continue
            if receipt.status == CommandStatus.SUCCEEDED:
                self.coordinator.acknowledge(
                    session_id, handover.session.participants[1], "received"
                )
            elif receipt.status.terminal:
                self.coordinator.fail(session_id, receipt.reason or receipt.status.value)
        return handover.session
