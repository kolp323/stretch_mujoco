"""Portable desk-work session contract derived from the semantic world."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from stretch_mujoco.semantics import ObjectType, RelationType, SemanticWorld

from .actions import ActionCommand, ActionType

if TYPE_CHECKING:
    from .employee import EmployeeAgent


WORK_SESSION_SEAT_PARAMETER = "_desk_work_seat"
WORK_DURATION_SECONDS_PARAMETER = "_desk_work_duration_seconds"
WORK_SESSION_ID_PARAMETER = "_desk_work_session_id"
# Public request parameter.  The underscored duration above remains runtime-owned
# and is attached only after the session contract has been validated.
REQUESTED_WORK_DURATION_SECONDS_PARAMETER = "duration_seconds"


def chair_for_workstation(world: SemanticWorld, workstation_id: str) -> str:
    """Return the one chair assigned to a workstation by the scene contract.

    The relationship is deliberately directional: a portable scene declares
    ``Chair --NEAR--> Workstation``.  Guessing from names or nearest geometry
    would make a migrated scene silently choose the wrong chair.
    """
    workstation = world.object(workstation_id)
    if workstation.object_type != ObjectType.WORKSTATION:
        raise ValueError(f"Target '{workstation_id}' must be Workstation")
    chairs = sorted(
        relation.subject
        for relation in world.find_relations(relation=RelationType.NEAR, object_id=workstation_id)
        if world.object(relation.subject).object_type == ObjectType.CHAIR
    )
    if len(chairs) != 1:
        raise ValueError(
            f"Workstation '{workstation_id}' requires exactly one Chair --NEAR--> "
            f"Workstation relation; found {len(chairs)}"
        )
    return chairs[0]


def is_desk_work_session(command: ActionCommand) -> bool:
    return (
        command.action == ActionType.WORK
        and isinstance(command.parameters.get(WORK_SESSION_SEAT_PARAMETER), str)
        and isinstance(command.parameters.get(WORK_SESSION_ID_PARAMETER), str)
    )


def desk_work_session_actions(
    agent: "EmployeeAgent",
    world: SemanticWorld,
    workstation_id: str,
    *,
    duration_seconds: float,
    session_id: str,
) -> tuple[ActionCommand, ...]:
    """Build the only valid idle-to-desk-work-to-idle transition.

    Child actions remain ordinary commands so each physical receipt commits its
    own semantic effect.  This keeps chair occupancy correct during work and
    releases it only after the stand-up receipt succeeds.
    """
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Desk work duration must be a positive finite number")
    if not session_id.strip():
        raise ValueError("Desk work session requires a runtime-issued session ID")
    chair_id = chair_for_workstation(world, workstation_id)
    seated = agent.state.location == chair_id and bool(
        world.find_relations(
            subject=chair_id,
            relation=RelationType.OCCUPIED_BY,
            object_id=agent.agent_id,
        )
    )
    session_metadata = {WORK_SESSION_ID_PARAMETER: session_id}
    actions: list[ActionCommand] = []
    if not seated:
        if agent.state.location != chair_id:
            actions.append(
                ActionCommand(agent.agent_id, ActionType.MOVE_TO, chair_id, session_metadata)
            )
        actions.append(ActionCommand(agent.agent_id, ActionType.SIT, chair_id, session_metadata))
    session_parameters = {
        WORK_SESSION_SEAT_PARAMETER: chair_id,
        WORK_DURATION_SECONDS_PARAMETER: duration_seconds,
        WORK_SESSION_ID_PARAMETER: session_id,
    }
    actions.extend(
        (
            ActionCommand(agent.agent_id, ActionType.WORK, workstation_id, session_parameters),
            ActionCommand(agent.agent_id, ActionType.STAND_UP, chair_id, session_metadata),
            ActionCommand(agent.agent_id, ActionType.IDLE, parameters=session_metadata),
        )
    )
    return tuple(actions)
