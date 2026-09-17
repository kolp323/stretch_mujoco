"""Assembly helpers connecting the agent runtime to the NPC protocol."""

from __future__ import annotations

from collections.abc import Iterable
import math

from .action_recipes import (
    OFFICE_LOCATION_SITES,
    OFFICE_OBJECT_APPROACH_SITES,
    OFFICE_PLACEMENT_SITES,
    OFFICE_SEAT_YAWS,
    OFFICE_SEAT_NAVIGATION_SITES,
    OFFICE_HANDOVER_SITES,
    OFFICE_HANDOVER_ROLE_SITES,
    OFFICE_INTERACTION_YAWS,
    OFFICE_ROBOT_REQUEST_SITES,
    production_handover_bindings,
)
from .drivers import MujocoNpcActionDriver, NpcSimulatorClient
from stretch_mujoco.npc.animation import OFFICE_CLIPS
from stretch_mujoco.npc.schema import NpcInteractionTemplate
from stretch_mujoco.npc.naming import interaction_site_name
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile
from stretch_mujoco.semantics import InteractionRole, ObjectType, SemanticWorld
from .interaction_stations import InteractionStationAllocator, SeatSlotAllocator


def _desk_work_bindings(
    world: SemanticWorld,
) -> tuple[dict[str, str], dict[str, float], dict[str, str]]:
    """Read portable desk affordances from semantic interaction points.

    A scene migration must make its chair/desk contract explicit.  Retaining a
    stale office object-name map would otherwise let a valid logical plan issue
    a physical command at an unrelated site.
    """
    location_sites: dict[str, str] = {}
    seat_yaws: dict[str, float] = {}
    seat_ingress_sites: dict[str, str] = {}
    for semantic_object in world.objects_of_type(ObjectType.WORKSTATION):
        points = world.interaction_points_for(
            role=InteractionRole.DESK_WORK, owner=semantic_object.object_id
        )
        if len(points) != 1:
            raise ValueError(
                f"Workstation '{semantic_object.object_id}' requires exactly one desk_work_site"
            )
        location_sites[semantic_object.object_id] = points[0].site
    for semantic_object in (
        *world.objects_of_type(ObjectType.CHAIR),
        *world.objects_of_type(ObjectType.BED),
        *world.objects_of_type(ObjectType.SEAT_SLOT),
    ):
        points = world.interaction_points_for(
            role=InteractionRole.CHAIR_SIT, owner=semantic_object.object_id
        )
        if len(points) != 1:
            raise ValueError(
                f"Seatable '{semantic_object.object_id}' requires exactly one chair_sit_site"
            )
        yaw = points[0].attributes.get("yaw")
        if not isinstance(yaw, (int, float)) or isinstance(yaw, bool) or not math.isfinite(yaw):
            raise ValueError(
                f"Seatable '{semantic_object.object_id}' chair_sit_site requires a finite yaw"
            )
        location_sites[semantic_object.object_id] = points[0].site
        seat_yaws[semantic_object.object_id] = float(yaw)
        if semantic_object.object_type == ObjectType.SEAT_SLOT:
            ingress = world.interaction_points_for(
                role=InteractionRole.SEAT_INGRESS, owner=semantic_object.object_id
            )
            if len(ingress) != 1:
                raise ValueError(
                    f"SeatSlot '{semantic_object.object_id}' requires exactly one seat_ingress_site"
                )
            seat_ingress_sites[semantic_object.object_id] = ingress[0].site
    return location_sites, seat_yaws, seat_ingress_sites


def _semantic_action_bindings(world: SemanticWorld) -> dict[str, dict]:
    """Read optional portable action bindings from interaction-point attributes.

    ``binding`` is intentionally declarative so a new scene does not need a
    Python object-name map.  Existing office constants remain the schema-v1
    migration fallback; semantic entries override them one-for-one.
    """
    result = {
        name: {}
        for name in (
            "location",
            "seat_navigation",
            "placement",
            "object_approach",
            "robot_request",
            "handover",
            "handover_roles",
            "handover_role_yaws",
            "conversation_roles",
            "conversation_yaws",
            "location_slots",
            "yaws",
        )
    }
    for point in world.interaction_points.values():
        binding = point.attributes.get("binding")
        if binding in {"handover_role", "conversation_role"}:
            participants = point.attributes.get("participants")
            participant = point.attributes.get("participant")
            if (
                not isinstance(participants, list)
                or len(participants) != 2
                or not all(isinstance(item, str) for item in participants)
            ):
                raise ValueError(
                    f"Interaction point '{point.point_id}' role binding requires two participants"
                )
            if participant not in participants:
                raise ValueError(
                    f"Interaction point '{point.point_id}' role binding requires participant"
                )
            pair = tuple(participants)
            role_key = "handover_roles" if binding == "handover_role" else "conversation_roles"
            result[role_key].setdefault(pair, {})[participant] = point.site
            yaw = point.attributes.get("yaw")
            if yaw is not None:
                if (
                    isinstance(yaw, bool)
                    or not isinstance(yaw, (int, float))
                    or not math.isfinite(yaw)
                ):
                    raise ValueError(f"Interaction point '{point.point_id}' has invalid yaw")
                if binding == "conversation_role":
                    result["conversation_yaws"][point.site] = float(yaw)
                else:
                    result["handover_role_yaws"].setdefault(pair, {})[participant] = float(yaw)
            continue
        if binding not in result or binding == "yaws":
            continue
        target = point.attributes.get("target", point.owner)
        if not isinstance(target, str) or not target:
            raise ValueError(f"Interaction point '{point.point_id}' binding requires string target")
        if binding == "location":
            slot_id = point.attributes.get("slot_id")
            if slot_id is None:
                if target in result["location_slots"]:
                    raise ValueError(
                        f"Location '{target}' mixes legacy location binding with slot bindings"
                    )
                result["location"][target] = point.site
            else:
                if not isinstance(slot_id, str) or not slot_id:
                    raise ValueError(
                        f"Interaction point '{point.point_id}' location slot_id must be a non-empty string"
                    )
                if target in result["location"]:
                    raise ValueError(
                        f"Location '{target}' mixes legacy location binding with slot bindings"
                    )
                slots = result["location_slots"].setdefault(target, {})
                if slot_id in slots:
                    raise ValueError(f"Location '{target}' declares duplicate slot_id '{slot_id}'")
                slots[slot_id] = point.site
        else:
            result[binding][target] = point.site
        yaw = point.attributes.get("yaw")
        if yaw is not None:
            if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
                raise ValueError(f"Interaction point '{point.point_id}' has invalid yaw")
            result["yaws"][target] = float(yaw)
    return result


def create_mujoco_action_driver(
    simulator: NpcSimulatorClient,
    *,
    npc_ids: Iterable[str] | None = None,
    trajectory_profile: NpcTrajectoryProfile | None = None,
    agent_locations: dict[str, str] | None = None,
    world: SemanticWorld | None = None,
    interaction_templates: dict[str, NpcInteractionTemplate] | None = None,
    interaction_station_allocator: InteractionStationAllocator | None = None,
) -> MujocoNpcActionDriver:
    roster_ids = None if npc_ids is None else tuple(sorted(set(npc_ids)))
    handover_sites = OFFICE_HANDOVER_SITES
    handover_role_sites = OFFICE_HANDOVER_ROLE_SITES
    interaction_yaws = dict(OFFICE_INTERACTION_YAWS)
    conversation_role_sites: dict[tuple[str, str], tuple[str, str]] = {}
    conversation_site_yaws = {
        "meeting_conversation_alex_site": math.pi / 2,
        "meeting_conversation_morgan_site": -math.pi / 2,
    }
    handover_role_yaws: dict[tuple[str, str], tuple[float, float]] = {}
    if roster_ids is not None and len(roster_ids) >= 2:
        if interaction_templates:
            # Generated hand sites remain an NPC-body contract; only the
            # shared role stations are scene configuration.
            handover_sites = {
                npc_id: interaction_site_name(npc_id, "handover") for npc_id in roster_ids
            }
            if "handover" not in interaction_templates:
                _, handover_role_sites, production_yaws = production_handover_bindings(roster_ids)
                interaction_yaws.update(production_yaws)
            if "conversation" not in interaction_templates:
                conversation_role_sites = {
                    (first, second): (
                        "meeting_conversation_alex_site",
                        "meeting_conversation_morgan_site",
                    )
                    for first in roster_ids
                    for second in roster_ids
                    if first != second
                }
        else:
            handover_sites, handover_role_sites, production_yaws = production_handover_bindings(
                roster_ids
            )
            interaction_yaws.update(production_yaws)
            conversation_role_sites = {
                (first, second): (
                    "meeting_conversation_alex_site",
                    "meeting_conversation_morgan_site",
                )
                for first in roster_ids
                for second in roster_ids
                if first != second
            }
    for kind, template in (interaction_templates or {}).items():
        if roster_ids is None:
            continue
        for first in roster_ids:
            for second in roster_ids:
                if first == second:
                    continue
                if kind == "conversation":
                    speaker, listener = template.roles["speaker"], template.roles["listener"]
                    conversation_role_sites[(first, second)] = (speaker.site, listener.site)
                    if speaker.yaw is not None:
                        conversation_site_yaws[speaker.site] = speaker.yaw
                    if listener.yaw is not None:
                        conversation_site_yaws[listener.site] = listener.yaw
                elif kind == "handover":
                    giver, receiver = template.roles["giver"], template.roles["receiver"]
                    handover_role_sites[(first, second)] = (giver.site, receiver.site)
                    # The action contract validates its recipient against this
                    # participant-indexed map before using the ordered role
                    # sites. Keep custom population templates authoritative.
                    handover_sites.setdefault(first, giver.site)
                    handover_sites.setdefault(second, receiver.site)
                    if giver.yaw is not None:
                        interaction_yaws.setdefault(first, giver.yaw)
                    if receiver.yaw is not None:
                        interaction_yaws.setdefault(second, receiver.yaw)
                    giver_yaw = (
                        giver.yaw if giver.yaw is not None else interaction_yaws.get(first, 0.0)
                    )
                    receiver_yaw = (
                        receiver.yaw
                        if receiver.yaw is not None
                        else math.remainder(giver_yaw + math.pi, 2 * math.pi)
                    )
                    handover_role_yaws[(first, second)] = (giver_yaw, receiver_yaw)
    cue_sites = (
        {
            npc_id: (
                (interaction_templates or {}).get("handover", None).roles["giver"].site
                if (interaction_templates or {}).get("handover") is not None
                else "npc_handover_giver_stand_site"
            )
            for npc_id in roster_ids
        }
        if roster_ids is not None
        else {}
    )
    location_sites = dict(OFFICE_LOCATION_SITES)
    seat_yaws = dict(OFFICE_SEAT_YAWS)
    if world is not None:
        scene_sites, scene_seat_yaws, scene_seat_ingress = _desk_work_bindings(world)
        location_sites.update(scene_sites)
        seat_yaws.update(scene_seat_yaws)
        bindings = _semantic_action_bindings(world)
        location_sites.update(bindings["location"])
        interaction_yaws.update(bindings["yaws"])
    else:
        bindings = {
            name: {}
            for name in (
                "seat_navigation",
                "placement",
                "object_approach",
                "robot_request",
                "handover",
                "handover_roles",
                "handover_role_yaws",
                "conversation_roles",
                "conversation_yaws",
                "location_slots",
            )
        }
        scene_seat_ingress = {}
    for pair, participant_sites in bindings["handover_roles"].items():
        if set(pair) == set(participant_sites):
            handover_role_sites[pair] = (participant_sites[pair[0]], participant_sites[pair[1]])
            explicit_yaws = bindings["handover_role_yaws"].get(pair, {})
            fallback_yaws = handover_role_yaws.get(pair)
            if fallback_yaws is None:
                giver_yaw = interaction_yaws.get(pair[0], 0.0)
                fallback_yaws = (
                    giver_yaw,
                    math.remainder(giver_yaw + math.pi, 2 * math.pi),
                )
            handover_role_yaws[pair] = (
                explicit_yaws.get(pair[0], fallback_yaws[0]),
                explicit_yaws.get(pair[1], fallback_yaws[1]),
            )
    for pair, participant_sites in bindings["conversation_roles"].items():
        if set(pair) == set(participant_sites):
            conversation_role_sites[pair] = (participant_sites[pair[0]], participant_sites[pair[1]])
    handover_sites = {**handover_sites, **bindings["handover"]}
    conversation_site_yaws.update(bindings["conversation_yaws"])
    return MujocoNpcActionDriver(
        simulator,
        location_sites,
        {**OFFICE_PLACEMENT_SITES, **bindings["placement"]},
        object_approach_sites={**OFFICE_OBJECT_APPROACH_SITES, **bindings["object_approach"]},
        handover_sites=handover_sites,
        handover_role_sites=handover_role_sites,
        handover_role_yaws=handover_role_yaws,
        conversation_role_sites=conversation_role_sites,
        conversation_site_yaws=conversation_site_yaws,
        interaction_station_allocator=interaction_station_allocator,
        seat_slot_allocator=(
            SeatSlotAllocator(interaction_station_allocator.catalog)
            if interaction_station_allocator is not None
            else None
        ),
        seat_verifier=getattr(simulator, "verify_seat_contact", None),
        cue_sites=cue_sites,
        seat_yaws=seat_yaws,
        seat_navigation_sites={
            **OFFICE_SEAT_NAVIGATION_SITES,
            **scene_seat_ingress,
            **bindings["seat_navigation"],
        },
        interaction_yaws=interaction_yaws,
        robot_request_sites={**OFFICE_ROBOT_REQUEST_SITES, **bindings["robot_request"]},
        location_slot_sites=bindings["location_slots"],
        available_clips=set(OFFICE_CLIPS),
        trajectory_profile=trajectory_profile,
        agent_locations=agent_locations,
    )
