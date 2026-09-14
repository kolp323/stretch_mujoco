"""Local validators for the closed LLM response contracts used by sequences."""

from __future__ import annotations

from typing import Any, Mapping

from stretch_mujoco.agents.actions import ActionType

from .models import ControlSequenceError


CONTRACT_FIELDS: dict[str, frozenset[str]] = {
    "robot_task_v1": frozenset({"robot_id", "task", "object", "destination", "recipient", "utterance"}),
    "dialogue_turn_v1": frozenset({"session_id", "turn_id", "speaker", "listener", "act", "text"}),
    "action_proposal_v1": frozenset({"proposal_id", "actions", "conversations", "post_actions"}),
}


def _object(payload: object, contract: str) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        raise ControlSequenceError(f"{contract} response must be an object")
    return payload


def validate_llm_contract(contract: str, payload: object) -> dict[str, Any]:
    """Reject extra fields as well as malformed values; remote JSON is never trusted."""
    if contract not in CONTRACT_FIELDS:
        raise ControlSequenceError(f"Unsupported LLM output contract '{contract}'")
    value = _object(payload, contract)
    unknown = set(value) - CONTRACT_FIELDS[contract]
    if unknown:
        raise ControlSequenceError(f"{contract} has unknown fields: {', '.join(sorted(unknown))}")
    if contract == "robot_task_v1":
        required = {"robot_id", "task", "object", "destination", "utterance"}
        if not required <= set(value) or not all(
            isinstance(value[key], str) and value[key] for key in required
        ):
            raise ControlSequenceError(
                "robot_task_v1 requires non-empty robot/task/object/destination/utterance"
            )
        if "recipient" in value and (not isinstance(value["recipient"], str) or not value["recipient"]):
            raise ControlSequenceError("robot_task_v1 recipient must be a non-empty string")
    elif contract == "dialogue_turn_v1":
        required = {"session_id", "turn_id", "speaker", "listener", "act", "text"}
        if set(value) != required or not all(
            isinstance(value[key], str) and value[key] for key in required
        ):
            raise ControlSequenceError("dialogue_turn_v1 requires non-empty string fields")
        if len(value["text"]) > 160:
            raise ControlSequenceError("dialogue_turn_v1 text exceeds 160 characters")
    else:
        if set(value) - {
            "proposal_id",
            "actions",
            "conversations",
            "post_actions",
        } or not isinstance(value.get("proposal_id"), str):
            raise ControlSequenceError("action_proposal_v1 requires proposal_id and actions")
        actions = value["actions"]
        post_actions = value.get("post_actions", [])
        if (
            not isinstance(actions, list)
            or not actions
            or not isinstance(post_actions, list)
            or len(actions) + len(post_actions) > 12
        ):
            raise ControlSequenceError("action_proposal_v1 actions must contain 1 to 12 actions")
        for action in [*actions, *post_actions]:
            if not isinstance(action, dict) or set(action) - {
                "actor",
                "action",
                "target",
                "reason",
                "parameters",
            }:
                raise ControlSequenceError("action_proposal_v1 action has unknown fields")
            if not all(
                isinstance(action.get(key), str) and action[key]
                for key in ("actor", "action", "target")
            ):
                raise ControlSequenceError("action_proposal_v1 action requires actor/action/target")
            try:
                ActionType(action["action"])
            except ValueError as error:
                raise ControlSequenceError(
                    f"LLM proposed unknown action '{action['action']}'"
                ) from error
            if "parameters" in action and not isinstance(action["parameters"], dict):
                raise ControlSequenceError("LLM action parameters must be an object")
        conversations = value.get("conversations", [])
        if not isinstance(conversations, list) or len(conversations) > 3:
            raise ControlSequenceError(
                "action_proposal_v1 conversations must contain at most 3 items"
            )
        for conversation in conversations:
            if not isinstance(conversation, dict) or set(conversation) != {
                "participants",
                "topic",
                "turns",
            }:
                raise ControlSequenceError("LLM conversation requires participants/topic/turns")
            participants = conversation["participants"]
            turns = conversation["turns"]
            if (
                not isinstance(participants, list)
                or len(participants) != 2
                or not all(isinstance(item, str) and item for item in participants)
                or not isinstance(conversation["topic"], str)
                or not conversation["topic"]
                or not isinstance(turns, list)
                or not 1 <= len(turns) <= 4
            ):
                raise ControlSequenceError("LLM conversation has invalid participants/topic/turns")
            for turn in turns:
                if not isinstance(turn, dict) or set(turn) != {
                    "speaker",
                    "listener",
                    "act",
                    "text",
                }:
                    raise ControlSequenceError("LLM conversation turn has invalid fields")
                if not all(isinstance(turn.get(key), str) and turn[key] for key in turn):
                    raise ControlSequenceError("LLM conversation turn requires non-empty strings")
                if len(turn["text"]) > 160:
                    raise ControlSequenceError("LLM conversation text exceeds 160 characters")
    return dict(value)


def binding_fields(contract: str) -> frozenset[str]:
    try:
        return CONTRACT_FIELDS[contract]
    except KeyError as error:
        raise ControlSequenceError(f"Unsupported LLM output contract '{contract}'") from error
