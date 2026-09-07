"""Central mapping from validated business actions to execution intent."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .actions import ActionType


@dataclass(frozen=True)
class ActionRecipe:
    animation: str
    completion: str
    target_kind: str = "logical"
    interaction_site_key: str | None = None
    yaw_source: str | None = None
    approach_required: bool = False
    completion_marker: str | None = None
    observation_condition: str = "marker_only"
    timeout_seconds: float | None = None
    interrupt_policy: str = "safe_marker"
    recovery_policy: str = "idle"
    base_clip_after_completion: str | None = None

    def validate(
        self,
        *,
        target: str | None,
        location_sites: dict[str, str],
        yaws: dict[str, float],
        available_clips: set[str] | None = None,
    ) -> str | None:
        """Return a stable, machine-readable failure reason before any command is sent."""
        if self.target_kind == "logical":
            return "recipe_not_embodied"
        if self.approach_required and (target is None or target not in location_sites):
            return "recipe_missing_site"
        if self.yaw_source is not None and (target is None or target not in yaws):
            return "recipe_missing_yaw"
        if self.completion_marker is None:
            return "recipe_missing_marker"
        if self.timeout_seconds is None or self.timeout_seconds <= 0:
            return "recipe_missing_timeout"
        if available_clips is not None and self.animation not in available_clips:
            return "recipe_clip_unavailable"
        return None


ACTION_RECIPES = {
    ActionType.IDLE: ActionRecipe("idle", "duration"),
    ActionType.MOVE_TO: ActionRecipe("walk", "pose_and_yaw"),
    ActionType.SIT: ActionRecipe("sit_down", "seat_pose_and_marker"),
    ActionType.STAND_UP: ActionRecipe("stand_up", "standing_marker"),
    ActionType.WORK: ActionRecipe("work", "minimum_duration"),
    ActionType.USE_COMPUTER: ActionRecipe(
        "use_computer",
        "computer_cycle",
        "location",
        "workstation",
        "site",
        True,
        "computer_cycle",
        "marker_only",
        30.0,
        "safe_marker",
        "idle",
        "idle",
    ),
    ActionType.EAT: ActionRecipe("eat", "attachment_and_duration"),
    ActionType.DRINK: ActionRecipe("eat", "attachment_and_duration"),
    ActionType.PICK_UP: ActionRecipe("pick_up", "grasp_and_attachment"),
    ActionType.PUT_DOWN: ActionRecipe("place", "release_and_detachment"),
    ActionType.HANDOVER: ActionRecipe("give", "interaction_markers"),
    ActionType.REQUEST_ROBOT: ActionRecipe("idle", "robot_task_and_session"),
    ActionType.ATTEND_MEETING: ActionRecipe("sit_down", "interaction_session"),
    ActionType.REST: ActionRecipe("sit_down", "minimum_duration"),
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
