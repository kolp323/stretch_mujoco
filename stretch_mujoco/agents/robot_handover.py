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
    stage: str = "receive"
    npc_id: str | None = None
    object_name: str | None = None
    target_site: str | None = None
    yaw: float | None = None


class RobotToNpcHandoverBridge:
    """Attach an object only after a robot executor confirms its physical release.

    The explicit ``robot_release_confirmed`` input is intentional: changing the
    robot grasp proxy is not itself a physical receipt. A real or mock robot
    executor must verify release before crossing this boundary.
    """

    def __init__(
        self,
        simulator: NpcSimulatorClient,
        *,
        handover_sites: dict[str, str] | None = None,
        interaction_yaws: dict[str, float] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.simulator = simulator
        self.coordinator = InteractionCoordinator()
        self._sequences: dict[str, int] = {}
        self._handovers: dict[str, RobotHandover] = {}
        self.handover_sites = dict(handover_sites or {})
        self.interaction_yaws = dict(interaction_yaws or {})
        self.timeout_seconds = timeout_seconds

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

    def begin_receive(
        self,
        robot_id: str,
        npc_id: str,
        object_name: str,
        *,
        session_id: str,
    ) -> RobotHandover:
        """Prepare the NPC before accepting a separately confirmed release.

        The robot remains the sole authority for physical release.  This bridge
        owns only NPC movement, orientation, receive-marker evidence, and the
        subsequent attachment receipt.
        """
        target_site = self.handover_sites.get(robot_id)
        yaw = self.interaction_yaws.get(robot_id)
        if target_site is None or yaw is None:
            raise ValueError("Robot handover requires a configured NPC site and yaw")
        session = self.coordinator.start(
            "handover",
            (robot_id, npc_id),
            object_name,
            (
                ("rendezvous", (npc_id,)),
                ("aligned", (npc_id,)),
                ("receiver_ready", (npc_id,)),
                ("released", (robot_id,)),
                ("received", (npc_id,)),
            ),
            deadline=float(self.simulator.pull_status().time) + self.timeout_seconds,
            session_id=session_id,
        )
        command_id = self._submit(
            npc_id,
            session_id,
            "rendezvous",
            NpcCommandKind.MOVE_TO,
            {"site": target_site, "arrival_clip": "idle", "max_replans": 1},
        )
        handover = RobotHandover(
            session,
            command_id,
            "rendezvous",
            npc_id,
            object_name,
            target_site,
            yaw,
        )
        self._handovers[session_id] = handover
        return handover

    def confirm_robot_release(self, session_id: str) -> InteractionSession:
        handover = self._handovers[session_id]
        if handover.stage != "released":
            raise ValueError("NPC must reach the receive marker before robot release")
        self.coordinator.acknowledge(session_id, handover.session.participants[0], "released")
        assert handover.npc_id is not None and handover.object_name is not None
        handover.stage = "received"
        handover.command_id = self._submit(
            handover.npc_id,
            session_id,
            "receive",
            NpcCommandKind.ATTACH_OBJECT,
            {"object": handover.object_name, "interaction_id": session_id},
        )
        return handover.session

    def poll(self, session_id: str) -> InteractionSession:
        handover = self._handovers[session_id]
        self.coordinator.check_deadlines(float(self.simulator.pull_status().time))
        if handover.session.status.terminal:
            return handover.session
        for receipt in self.simulator.pull_npc_receipts():
            if receipt.command_id != handover.command_id:
                continue
            if receipt.status == CommandStatus.SUCCEEDED:
                self._advance(handover)
            elif receipt.status.terminal:
                self.coordinator.fail(session_id, receipt.reason or receipt.status.value)
        return handover.session

    def _advance(self, handover: RobotHandover) -> None:
        if handover.stage in {"receive", "received"}:
            self.coordinator.acknowledge(
                handover.session.session_id, handover.session.participants[1], "received"
            )
            return
        assert handover.npc_id is not None
        if handover.stage == "rendezvous":
            self.coordinator.acknowledge(
                handover.session.session_id, handover.npc_id, "rendezvous"
            )
            assert handover.yaw is not None and handover.target_site is not None
            handover.stage = "aligned"
            handover.command_id = self._submit(
                handover.npc_id,
                handover.session.session_id,
                "aligned",
                NpcCommandKind.ALIGN_TO,
                {"yaw": handover.yaw, "target_site": handover.target_site},
            )
        elif handover.stage == "aligned":
            self.coordinator.acknowledge(
                handover.session.session_id, handover.npc_id, "aligned"
            )
            handover.stage = "receiver_ready"
            handover.command_id = self._submit(
                handover.npc_id,
                handover.session.session_id,
                "receiver_ready",
                NpcCommandKind.PLAY_ANIMATION,
                {
                    "clip": "receive",
                    "completion_marker": "handover_ready",
                    "interaction_id": handover.session.session_id,
                    "arrival_clip": "idle",
                    "target_site": handover.target_site,
                },
            )
        elif handover.stage == "receiver_ready":
            self.coordinator.acknowledge(
                handover.session.session_id, handover.npc_id, "receiver_ready"
            )
            handover.stage = "released"

    def _submit(
        self,
        npc_id: str,
        session_id: str,
        stage: str,
        kind: NpcCommandKind,
        payload: dict[str, object],
    ) -> str:
        sequence = self._sequences.get(npc_id, -1) + 1
        self._sequences[npc_id] = sequence
        issued_at = float(self.simulator.pull_status().time)
        command = NpcCommand(
            f"{session_id}:{stage}",
            sequence,
            npc_id,
            kind,
            payload,
            issued_at,
            issued_at + self.timeout_seconds,
        )
        return self.simulator.submit_npc_command(command)
