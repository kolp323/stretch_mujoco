"""Central mapping from validated business actions to execution intent."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .actions import ActionType


@dataclass(frozen=True)
class ActionRecipe:
    animation: str
    completion: str


ACTION_RECIPES = {
    ActionType.IDLE: ActionRecipe("idle", "duration"),
    ActionType.MOVE_TO: ActionRecipe("walk", "pose_and_yaw"),
    ActionType.SIT: ActionRecipe("sit", "seat_pose_and_marker"),
    ActionType.WORK: ActionRecipe("work", "minimum_duration"),
    ActionType.USE_COMPUTER: ActionRecipe("work", "minimum_duration"),
    ActionType.EAT: ActionRecipe("eat", "attachment_and_duration"),
    ActionType.DRINK: ActionRecipe("eat", "attachment_and_duration"),
    ActionType.PICK_UP: ActionRecipe("idle", "attachment_exists"),
    ActionType.PUT_DOWN: ActionRecipe("idle", "object_on_affordance"),
    ActionType.HANDOVER: ActionRecipe("idle", "interaction_session"),
    ActionType.REQUEST_ROBOT: ActionRecipe("idle", "robot_task_and_session"),
    ActionType.ATTEND_MEETING: ActionRecipe("sit", "interaction_session"),
    ActionType.REST: ActionRecipe("sit", "minimum_duration"),
    ActionType.OPEN_CABINET: ActionRecipe("idle", "cabinet_open"),
}


OFFICE_LOCATION_SITES = {
    "workstation_left": "desk_left_work_site",
    "workstation_right": "desk_right_work_site",
    "chair_left": "chair_left_sit",
    "chair_right": "chair_right_sit",
    "meeting_table": "meeting_human_stand_site",
    "storage_cabinet": "cabinet_human_stand_site",
    "snack_counter": "snack_human_stand_site",
    "coffee_machine": "coffee_human_stand_site",
}

OFFICE_PLACEMENT_SITES = {
    "snack_counter": "snack_place_site",
    "meeting_table": "meeting_table_place_site",
    "storage_cabinet": "cabinet_open_site",
    "workstation_left": "desk_left_work_site",
    "workstation_right": "desk_right_work_site",
}

# These values mirror the authoritative seat-site orientation in office_scene.xml.
# They are intentionally configuration, not a controller default: a new scene must
# provide its own affordance yaw instead of silently facing the wrong direction.
OFFICE_SEAT_YAWS = {
    "chair_left": math.pi,
    "chair_right": math.pi,
}


def animation_for_action(action: str) -> str:
    try:
        return ACTION_RECIPES[ActionType(action)].animation
    except ValueError:
        return "idle"
