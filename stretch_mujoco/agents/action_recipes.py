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
    ActionType.IDLE: ActionRecipe("idle", "duration", timeout_seconds=60.0),
    ActionType.MOVE_TO: ActionRecipe(
        "walk", "pose_and_yaw", target_kind="location", approach_required=True, timeout_seconds=30.0
    ),
    ActionType.SIT: ActionRecipe(
        "sit_down",
        "seat_pose_and_marker",
        target_kind="chair",
        interaction_site_key="seat",
        yaw_source="site",
        approach_required=True,
        completion_marker="seated",
        timeout_seconds=30.0,
    ),
    ActionType.STAND_UP: ActionRecipe(
        "stand_up",
        "standing_marker",
        target_kind="chair",
        completion_marker="standing",
        timeout_seconds=15.0,
    ),
    ActionType.WORK: ActionRecipe("work", "minimum_duration", timeout_seconds=900.0),
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
    ActionType.TALK: ActionRecipe(
        "talk",
        "talk_cycle",
        "participant",
        "conversation_approach",
        "site",
        True,
        "talk_cycle",
        "marker_only",
        30.0,
    ),
    ActionType.GESTURE_POINT: ActionRecipe(
        "gesture_point",
        "gesture_point_complete",
        "participant",
        "conversation_approach",
        "site",
        True,
        "gesture_point_complete",
        "marker_only",
        15.0,
    ),
    ActionType.GESTURE_WAVE: ActionRecipe(
        "gesture_wave",
        "gesture_wave_complete",
        "participant",
        "conversation_approach",
        "site",
        True,
        "gesture_wave_complete",
        "marker_only",
        15.0,
    ),
    ActionType.EAT: ActionRecipe("eat", "attachment_and_duration", timeout_seconds=240.0),
    ActionType.DRINK: ActionRecipe("eat", "attachment_and_duration", timeout_seconds=120.0),
    ActionType.PICK_UP: ActionRecipe(
        "pick_up",
        "grasp_and_attachment",
        target_kind="object",
        interaction_site_key="object_approach",
        yaw_source="site",
        approach_required=True,
        completion_marker="grasp",
        timeout_seconds=30.0,
    ),
    ActionType.PUT_DOWN: ActionRecipe(
        "place",
        "release_and_detachment",
        target_kind="location",
        interaction_site_key="placement_approach",
        yaw_source="site",
        approach_required=True,
        completion_marker="release",
        timeout_seconds=30.0,
    ),
    ActionType.HANDOVER: ActionRecipe(
        "give",
        "interaction_markers",
        target_kind="participant",
        interaction_site_key="handover_approach",
        yaw_source="site",
        approach_required=True,
        completion_marker="handover_ready",
        timeout_seconds=45.0,
    ),
    ActionType.REQUEST_ROBOT: ActionRecipe("idle", "robot_task_and_session", timeout_seconds=30.0),
    ActionType.ATTEND_MEETING: ActionRecipe(
        "sit_down", "interaction_session", timeout_seconds=900.0
    ),
    ActionType.REST: ActionRecipe("sit_down", "minimum_duration", timeout_seconds=600.0),
    ActionType.OPEN_CABINET: ActionRecipe("idle", "cabinet_open", timeout_seconds=30.0),
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

# Object sites are human root approach points, not the object's contact point.
# The attachment receipt remains the authoritative evidence of a pickup.
OFFICE_OBJECT_APPROACH_SITES = {
    "document_report": "cabinet_human_stand_site",
    "document_invoice": "cabinet_human_stand_site",
    "soda_can": "snack_human_stand_site",
    "cereal_box": "snack_human_stand_site",
    "bread_snack": "snack_human_stand_site",
    "lemon": "snack_human_stand_site",
}

# These values mirror the authoritative seat-site orientation in office_scene.xml.
# They are intentionally configuration, not a controller default: a new scene must
# provide its own affordance yaw instead of silently facing the wrong direction.
OFFICE_SEAT_YAWS = {
    "chair_left": math.pi,
    "chair_right": math.pi,
}

# Embodied interaction approaches must name their physical facing direction.
# Object and workstation entries mirror their human-stand sites in office_scene.xml;
# handover entries are counterpart-facing directions for the native two-NPC scene.
OFFICE_INTERACTION_YAWS = {
    "document_report": math.pi,
    "document_invoice": math.pi,
    "soda_can": math.pi,
    "cereal_box": math.pi,
    "bread_snack": math.pi,
    "lemon": math.pi,
    "snack_counter": math.pi,
    "meeting_table": math.pi,
    "storage_cabinet": math.pi,
    "workstation_left": math.pi,
    "workstation_right": math.pi,
    "employee_01": 0.0,
    "employee_02": math.pi,
}

OFFICE_HANDOVER_SITES = {
    "employee_01": "employee_01_handover_site",
    "employee_02": "employee_02_handover_site",
}

# A handover has two ground-level rendezvous sites.  They are deliberately
# distinct from the hand sites above: locomotion must never navigate either
# participant into the other's body just to make the attachment point coincide.
OFFICE_HANDOVER_ROLE_SITES = {
    ("employee_01", "employee_02"): (
        "employee_01_handover_stand_site",
        "employee_02_handover_stand_site",
    ),
    ("employee_02", "employee_01"): (
        "employee_02_handover_stand_site",
        "employee_01_handover_stand_site",
    ),
}


def animation_for_action(action: str) -> str:
    try:
        return ACTION_RECIPES[ActionType(action)].animation
    except ValueError:
        return "idle"
