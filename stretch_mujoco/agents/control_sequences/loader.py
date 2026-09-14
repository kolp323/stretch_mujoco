"""Strict, safe YAML loader for control-sequence v1."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import yaml
from yaml.tokens import AliasToken, AnchorToken

from .models import (
    BehaviorSequence,
    ControlMode,
    ControlSequenceError,
    FailureHandling,
    FailurePolicy,
    LlmAutonomy,
    LlmSettings,
    SequencePurpose,
    SequenceStep,
    StepKind,
)


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ControlSequenceError(f"Duplicate YAML key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "sequence_id",
    "description",
    "scene",
    "purpose",
    "default_control_mode",
    "seed",
    "clock",
    "participants",
    "locations",
    "llm",
    "defaults",
    "setup",
    "steps",
    "acceptance",
}
_STEP_FIELDS = {
    "id",
    "kind",
    "timeout_s",
    "on_failure",
    "caption",
    "tags",
    "actor",
    "action",
    "target",
    "parameters",
    "clip",
    "duration_s",
    "completion_marker",
    "join",
    "cancel_remaining_on_any",
    "steps",
    "participants",
    "topic",
    "meeting",
    "max_turns",
    "turn_timeout_s",
    "turns",
    "trigger",
    "output_contract",
    "save_as",
    "context",
    "duration_s",
    "condition",
    "source",
    "options",
    "max_iterations",
    "until",
    "weight",
    "when",
    "allowed_acts",
    "speaker",
    "listener",
    "gesture",
    "listener_gesture",
}


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ControlSequenceError(f"{context} must be a mapping with string keys")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ControlSequenceError(f"{context} has unknown fields: {', '.join(sorted(unknown))}")


def _positive(value: Any, context: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ControlSequenceError(f"{context} must be a finite number")
    result = float(value)
    if result < 0 or (result == 0 and not allow_zero):
        raise ControlSequenceError(f"{context} must be positive")
    return result


def _resolve_path(value: Any, base: Path, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ControlSequenceError(f"{context} must be a non-empty relative path")
    raw = Path(value)
    if raw.is_absolute():
        raise ControlSequenceError(f"{context} must be relative to the YAML file")
    resolved = (base / raw).resolve()
    # A sequence may refer to a sibling model/config file, never outside its repository root.
    repository_root = next(
        (parent for parent in (base, *base.parents) if (parent / "pyproject.toml").exists()), base
    )
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise ControlSequenceError(f"{context} escapes the repository root") from error
    return resolved


def _failure(value: Any, default: FailureHandling) -> FailureHandling:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return FailureHandling(policy=FailurePolicy(value))
        except ValueError as error:
            raise ControlSequenceError(f"Unknown failure policy {value!r}") from error
    payload = _mapping(value, "on_failure")
    _reject_unknown(payload, {"policy", "max_attempts", "backoff_s", "fallback_step"}, "on_failure")
    try:
        policy = FailurePolicy(payload.get("policy", default.policy.value))
    except ValueError as error:
        raise ControlSequenceError("Unknown failure policy") from error
    attempts = payload.get("max_attempts", default.max_attempts)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ControlSequenceError("on_failure.max_attempts must be a positive integer")
    fallback = payload.get("fallback_step")
    if fallback is not None and not isinstance(fallback, str):
        raise ControlSequenceError("on_failure.fallback_step must be a step ID")
    if policy is FailurePolicy.FALLBACK and not fallback:
        raise ControlSequenceError("fallback policy requires on_failure.fallback_step")
    return FailureHandling(
        policy=policy,
        max_attempts=attempts,
        backoff_seconds=_positive(
            payload.get("backoff_s", 0.0), "on_failure.backoff_s", allow_zero=True
        ),
        fallback_step=fallback,
    )


def _step(
    value: Any, defaults: Mapping[str, Any], known_ids: set[str], depth: int = 0
) -> SequenceStep:
    if depth > 4:
        raise ControlSequenceError("Control sequence nesting exceeds the maximum depth of 4")
    payload = _mapping(value, "step")
    _reject_unknown(payload, _STEP_FIELDS, "step")
    step_id = payload.get("id")
    if not isinstance(step_id, str) or not step_id:
        raise ControlSequenceError("Every step requires a non-empty id")
    if step_id in known_ids:
        raise ControlSequenceError(f"Duplicate step id '{step_id}'")
    known_ids.add(step_id)
    try:
        kind = StepKind(payload["kind"])
    except KeyError as error:
        raise ControlSequenceError(f"Step '{step_id}' is missing kind") from error
    except ValueError as error:
        raise ControlSequenceError(
            f"Step '{step_id}' has unknown kind {payload.get('kind')!r}"
        ) from error
    timeout = _positive(
        payload.get("timeout_s", defaults.get("timeout_s", 30)), f"step '{step_id}' timeout_s"
    )
    default_failure = _failure(defaults.get("on_failure"), FailureHandling())
    failure = _failure(payload.get("on_failure"), default_failure)
    children_data = payload.get("steps", [])
    if kind in {StepKind.PARALLEL, StepKind.CHOOSE, StepKind.REPEAT}:
        if not isinstance(children_data, list) or not children_data:
            raise ControlSequenceError(f"{kind.value} step '{step_id}' requires non-empty steps")
    elif "steps" in payload:
        raise ControlSequenceError(f"Step '{step_id}' of kind {kind.value} cannot have child steps")
    if kind is StepKind.REPEAT:
        iterations = payload.get("max_iterations")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ControlSequenceError(
                f"repeat step '{step_id}' requires finite max_iterations >= 1"
            )
    if kind is StepKind.WAIT:
        _positive(payload.get("duration_s"), f"wait step '{step_id}' duration_s")
    if kind is StepKind.ACTION:
        if not isinstance(payload.get("actor"), str) or not isinstance(payload.get("action"), str):
            raise ControlSequenceError(f"action step '{step_id}' requires actor and action")
    if kind is StepKind.ANIMATION:
        if not isinstance(payload.get("actor"), str) or not isinstance(payload.get("clip"), str):
            raise ControlSequenceError(f"animation step '{step_id}' requires actor and clip")
        _positive(payload.get("duration_s"), f"animation step '{step_id}' duration_s")
    if kind is StepKind.LLM_REQUEST:
        required = ("trigger", "actor", "output_contract", "save_as", "context")
        if any(not isinstance(payload.get(key), str) for key in required[:-1]) or not isinstance(
            payload.get("context"), dict
        ):
            raise ControlSequenceError(f"llm_request step '{step_id}' has an invalid contract")
    children = tuple(_step(child, defaults, known_ids, depth + 1) for child in children_data)
    tags = payload.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ControlSequenceError(f"step '{step_id}' tags must be a list of strings")
    return SequenceStep(
        step_id=step_id,
        kind=kind,
        timeout_seconds=timeout,
        failure=failure,
        payload=dict(payload),
        caption=payload.get("caption"),
        tags=tuple(tags),
        children=children,
    )


def _llm(value: Any, base: Path) -> LlmSettings:
    if value is None:
        return LlmSettings()
    payload = _mapping(value, "llm")
    allowed = {
        "enabled",
        "provider_config",
        "daily_budget",
        "failure_policy",
        "record_responses",
        "replay_file",
        "autonomy",
    }
    _reject_unknown(payload, allowed, "llm")
    autonomy = None
    if "autonomy" in payload:
        data = _mapping(payload["autonomy"], "llm.autonomy")
        _reject_unknown(
            data,
            {
                "allowed_actions",
                "allowed_locations",
                "decision_interval_s",
                "plan_horizon_actions",
                "max_decisions",
                "max_consecutive_failures",
                "max_replans_per_segment",
                "stop",
            },
            "llm.autonomy",
        )
        required = {"allowed_actions", "allowed_locations"}
        if required - data.keys():
            raise ControlSequenceError(
                "llm.autonomy requires allowed_actions and allowed_locations"
            )
        actions, locations = data["allowed_actions"], data["allowed_locations"]
        if (
            not isinstance(actions, list)
            or not isinstance(locations, list)
            or not all(isinstance(item, str) for item in actions)
            or not all(isinstance(item, str) for item in locations)
        ):
            raise ControlSequenceError("LLM allowlists must contain only strings")
        interval = data.get("decision_interval_s", [5.0, 15.0])
        if not isinstance(interval, list) or len(interval) != 2:
            raise ControlSequenceError("llm.autonomy.decision_interval_s must contain [min, max]")
        minimum, maximum = (_positive(item, "llm autonomy decision interval") for item in interval)
        if maximum < minimum:
            raise ControlSequenceError("llm autonomy decision interval max is smaller than min")
        horizon = data.get("plan_horizon_actions", 1)
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ControlSequenceError("llm autonomy plan_horizon_actions must be positive")
        autonomy = LlmAutonomy(
            tuple(actions),
            tuple(locations),
            (minimum, maximum),
            horizon,
            data.get("max_decisions"),
            int(data.get("max_consecutive_failures", 3)),
            int(data.get("max_replans_per_segment", 1)),
            dict(data.get("stop", {})),
        )
    try:
        policy = FailurePolicy(payload.get("failure_policy", FailurePolicy.FAIL_SEQUENCE.value))
    except ValueError as error:
        raise ControlSequenceError("llm.failure_policy is invalid") from error
    return LlmSettings(
        enabled=bool(payload.get("enabled", False)),
        provider_config=(
            _resolve_path(payload["provider_config"], base, "llm.provider_config")
            if "provider_config" in payload
            else None
        ),
        daily_budget=payload.get("daily_budget"),
        failure_policy=policy,
        record_responses=bool(payload.get("record_responses", False)),
        replay_file=(
            _resolve_path(payload["replay_file"], base, "llm.replay_file")
            if isinstance(payload.get("replay_file"), str)
            else None
        ),
        autonomy=autonomy,
    )


def load_control_sequence(path: str | Path) -> BehaviorSequence:
    """Load v1 YAML without constructing a simulator, provider, or output artifact."""
    source = Path(path).resolve()
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise ControlSequenceError(
            f"Could not read control sequence '{source}': {error}"
        ) from error
    try:
        if any(isinstance(token, (AliasToken, AnchorToken)) for token in yaml.scan(raw)):
            raise ControlSequenceError("YAML anchors and aliases are not allowed")
        document = yaml.load(raw, Loader=_StrictLoader)
    except yaml.YAMLError as error:
        raise ControlSequenceError(f"Invalid control sequence YAML: {error}") from error
    payload = _mapping(document, "control sequence")
    _reject_unknown(payload, _TOP_LEVEL_FIELDS, "control sequence")
    if payload.get("schema_version") != 1:
        raise ControlSequenceError("Only control sequence schema_version 1 is supported")
    sequence_id = payload.get("sequence_id")
    if not isinstance(sequence_id, str) or not sequence_id:
        raise ControlSequenceError("sequence_id is required")
    try:
        purpose = SequencePurpose(payload.get("purpose", "runtime"))
        mode = ControlMode(payload.get("default_control_mode", "yaml"))
    except ValueError as error:
        raise ControlSequenceError("purpose or default_control_mode is invalid") from error
    seed = payload.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ControlSequenceError("seed must be an integer")
    scene_payload = _mapping(payload.get("scene", {}), "scene")
    scene = {
        key: _resolve_path(value, source.parent, f"scene.{key}")
        for key, value in scene_payload.items()
    }
    participants = _mapping(payload.get("participants", {}), "participants")
    if not participants or not all(
        isinstance(alias, str) and isinstance(npc, str) for alias, npc in participants.items()
    ):
        raise ControlSequenceError("participants must map non-empty aliases to IDs")
    defaults = _mapping(payload.get("defaults", {}), "defaults")
    known_ids: set[str] = set()
    raw_steps = payload.get("steps", [])
    if not isinstance(raw_steps, list):
        raise ControlSequenceError("steps must be a list")
    steps = tuple(_step(step, defaults, known_ids) for step in raw_steps)
    if len(known_ids) > 64:
        raise ControlSequenceError("Control sequence exceeds the maximum of 64 steps")
    llm = _llm(payload.get("llm"), source.parent)
    if purpose is SequencePurpose.ACCEPTANCE and (
        seed is None or not (llm.record_responses or llm.replay_file)
    ):
        raise ControlSequenceError("acceptance sequences require seed and LLM recording or replay")
    if mode is ControlMode.YAML and not steps:
        raise ControlSequenceError("YAML control mode requires non-empty steps")
    if mode is ControlMode.LLM:
        if not llm.enabled or llm.autonomy is None or not llm.autonomy.stop:
            raise ControlSequenceError(
                "LLM control mode requires enabled autonomy with a finite stop condition"
            )
    return BehaviorSequence(
        schema_version=1,
        sequence_id=sequence_id,
        purpose=purpose,
        default_control_mode=mode,
        seed=seed,
        source_path=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        scene=scene,
        participants=dict(participants),
        locations=dict(_mapping(payload.get("locations", {}), "locations")),
        defaults=dict(defaults),
        setup=dict(_mapping(payload.get("setup", {}), "setup")),
        steps=steps,
        acceptance=dict(_mapping(payload.get("acceptance", {}), "acceptance")),
        llm=llm,
    )


class StrictSequenceLoader:
    """Small object wrapper for callers that prefer an explicit loader dependency."""

    def load(self, path: str | Path) -> BehaviorSequence:
        return load_control_sequence(path)
