"""Assembly helpers connecting the agent runtime to the NPC protocol."""

from __future__ import annotations

from collections.abc import Iterable

from .action_recipes import (
    OFFICE_LOCATION_SITES,
    OFFICE_OBJECT_APPROACH_SITES,
    OFFICE_PLACEMENT_SITES,
    OFFICE_SEAT_YAWS,
    OFFICE_HANDOVER_SITES,
    OFFICE_HANDOVER_ROLE_SITES,
    OFFICE_INTERACTION_YAWS,
    production_handover_bindings,
)
from .drivers import MujocoNpcActionDriver, NpcSimulatorClient
from stretch_mujoco.npc.animation import OFFICE_CLIPS
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile


def create_mujoco_action_driver(
    simulator: NpcSimulatorClient,
    *,
    npc_ids: Iterable[str] | None = None,
    trajectory_profile: NpcTrajectoryProfile | None = None,
    agent_locations: dict[str, str] | None = None,
) -> MujocoNpcActionDriver:
    handover_sites = OFFICE_HANDOVER_SITES
    handover_role_sites = OFFICE_HANDOVER_ROLE_SITES
    interaction_yaws = dict(OFFICE_INTERACTION_YAWS)
    if npc_ids is not None:
        handover_sites, handover_role_sites, production_yaws = production_handover_bindings(npc_ids)
        interaction_yaws.update(production_yaws)
    return MujocoNpcActionDriver(
        simulator,
        OFFICE_LOCATION_SITES,
        OFFICE_PLACEMENT_SITES,
        object_approach_sites=OFFICE_OBJECT_APPROACH_SITES,
        handover_sites=handover_sites,
        handover_role_sites=handover_role_sites,
        seat_yaws=OFFICE_SEAT_YAWS,
        interaction_yaws=interaction_yaws,
        available_clips=set(OFFICE_CLIPS),
        trajectory_profile=trajectory_profile,
        agent_locations=agent_locations,
    )
