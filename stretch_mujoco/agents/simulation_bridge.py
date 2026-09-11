"""Assembly helpers connecting the agent runtime to the NPC protocol."""

from __future__ import annotations

from collections.abc import Iterable
import math

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
from stretch_mujoco.semantics import InteractionRole, ObjectType, SemanticWorld


def _desk_work_bindings(world: SemanticWorld) -> tuple[dict[str, str], dict[str, float]]:
    """Read portable desk affordances from semantic interaction points.

    A scene migration must make its chair/desk contract explicit.  Retaining a
    stale office object-name map would otherwise let a valid logical plan issue
    a physical command at an unrelated site.
    """
    location_sites: dict[str, str] = {}
    seat_yaws: dict[str, float] = {}
    for semantic_object in world.objects_of_type(ObjectType.WORKSTATION):
        points = world.interaction_points_for(
            role=InteractionRole.DESK_WORK, owner=semantic_object.object_id
        )
        if len(points) != 1:
            raise ValueError(
                f"Workstation '{semantic_object.object_id}' requires exactly one desk_work_site"
            )
        location_sites[semantic_object.object_id] = points[0].site
    for semantic_object in world.objects_of_type(ObjectType.CHAIR):
        points = world.interaction_points_for(
            role=InteractionRole.CHAIR_SIT, owner=semantic_object.object_id
        )
        if len(points) != 1:
            raise ValueError(
                f"Chair '{semantic_object.object_id}' requires exactly one chair_sit_site"
            )
        yaw = points[0].attributes.get("yaw")
        if not isinstance(yaw, (int, float)) or isinstance(yaw, bool) or not math.isfinite(yaw):
            raise ValueError(
                f"Chair '{semantic_object.object_id}' chair_sit_site requires a finite yaw"
            )
        location_sites[semantic_object.object_id] = points[0].site
        seat_yaws[semantic_object.object_id] = float(yaw)
    return location_sites, seat_yaws


def create_mujoco_action_driver(
    simulator: NpcSimulatorClient,
    *,
    npc_ids: Iterable[str] | None = None,
    trajectory_profile: NpcTrajectoryProfile | None = None,
    agent_locations: dict[str, str] | None = None,
    world: SemanticWorld | None = None,
) -> MujocoNpcActionDriver:
    handover_sites = OFFICE_HANDOVER_SITES
    handover_role_sites = OFFICE_HANDOVER_ROLE_SITES
    interaction_yaws = dict(OFFICE_INTERACTION_YAWS)
    location_sites = dict(OFFICE_LOCATION_SITES)
    seat_yaws = dict(OFFICE_SEAT_YAWS)
    if world is not None:
        scene_sites, scene_seat_yaws = _desk_work_bindings(world)
        location_sites.update(scene_sites)
        seat_yaws.update(scene_seat_yaws)
    if npc_ids is not None:
        handover_sites, handover_role_sites, production_yaws = production_handover_bindings(npc_ids)
        interaction_yaws.update(production_yaws)
    return MujocoNpcActionDriver(
        simulator,
        location_sites,
        OFFICE_PLACEMENT_SITES,
        object_approach_sites=OFFICE_OBJECT_APPROACH_SITES,
        handover_sites=handover_sites,
        handover_role_sites=handover_role_sites,
        seat_yaws=seat_yaws,
        interaction_yaws=interaction_yaws,
        available_clips=set(OFFICE_CLIPS),
        trajectory_profile=trajectory_profile,
        agent_locations=agent_locations,
    )
