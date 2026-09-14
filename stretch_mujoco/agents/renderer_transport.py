"""Server-backed transport used by control-sequence renderers.

The renderer must not bolt robot methods onto its private ``MjData`` copy.  A
``StretchMujocoSimulator`` owns a separate server process and that process is
the only place where robot control, NPC commands, attachment state, and their
receipts are advanced together.  This small adapter deliberately forwards the
two existing public protocols without adding a local ``step`` method.
"""

from __future__ import annotations

from typing import Any

from stretch_mujoco.npc import NpcCommand, NpcCommandReceipt, NpcRuntimeState


class ServerBackedRendererTransport:
    """Expose one live Stretch server to NPC drivers and observations.

    The server advances time asynchronously.  In particular, callers must use
    ``pull_status().time`` as their clock rather than calling a synthetic local
    step, which would produce receipts for a different MuJoCo scene.
    """

    def __init__(self, simulator: Any) -> None:
        required = (
            "pull_status",
            "submit_npc_command",
            "pull_npc_receipts",
            "cancel_npc_command",
            "pull_semantic_state",
        )
        missing = [name for name in required if not callable(getattr(simulator, name, None))]
        if missing:
            raise TypeError(
                "server-backed renderer requires a live StretchMujocoSimulator transport; "
                f"missing: {', '.join(missing)}"
            )
        self._simulator = simulator

    def pull_status(self) -> Any:
        return self._simulator.pull_status()

    def submit_npc_command(self, command: NpcCommand) -> str:
        return self._simulator.submit_npc_command(command)

    def pull_npc_receipts(self) -> tuple[NpcCommandReceipt, ...]:
        return self._simulator.pull_npc_receipts()

    def pull_npc_states(self) -> dict[str, NpcRuntimeState]:
        return self._simulator.pull_npc_states()

    def cancel_npc_command(self, npc_id: str, command_id: str) -> None:
        self._simulator.cancel_npc_command(npc_id, command_id)

    def pull_semantic_state(self) -> dict[str, Any]:
        """Return semantic objects plus the live NPC poses from this server.

        Conversation validation consumes the same observation boundary as the
        sequence executor.  The MuJoCo server publishes semantic object poses
        and NPC controller states through separate proxies, so keep them
        together here instead of making the renderer invent a second local
        observation.
        """
        snapshot = dict(self._simulator.pull_semantic_state())
        npc_states = self._simulator.pull_npc_states()
        if npc_states:
            agents: dict[str, dict[str, Any]] = {}
            for npc_id, state in npc_states.items():
                payload = state.to_dict() if hasattr(state, "to_dict") else dict(state)
                agents[str(npc_id)] = payload
            snapshot["agents"] = agents
            if "sim_time" not in snapshot:
                first = next(iter(npc_states.values()))
                snapshot["sim_time"] = float(getattr(first, "sim_time", 0.0))
        return snapshot
