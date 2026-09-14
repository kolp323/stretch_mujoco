"""Static binding and policy checks for loaded control sequences."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from stretch_mujoco.agents.actions import ActionType

from .contracts import binding_fields
from .models import (
    BehaviorSequence,
    CompiledSequence,
    CompiledStep,
    ControlMode,
    ControlSequenceError,
    SequencePurpose,
    SequenceStep,
    StepKind,
    resolve_control_mode,
)


_BINDING = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\}$")


@dataclass(frozen=True)
class ControlCapabilities:
    """Read-only scene/runtime facts supplied by the bridge, never by YAML."""

    npc_ids: frozenset[str]
    robot_ids: frozenset[str] = frozenset()
    object_ids: frozenset[str] = frozenset()
    site_ids: frozenset[str] = frozenset()
    clips_by_npc: Mapping[str, frozenset[str]] = field(default_factory=dict)
    reachable_locations: frozenset[str] = frozenset()

    @classmethod
    def from_runtime(cls, runtime: Any) -> "ControlCapabilities":
        world = runtime.world
        return cls(
            npc_ids=frozenset(runtime.agents),
            robot_ids=frozenset(
                object_id
                for object_id, value in world.objects.items()
                if getattr(value.object_type, "value", value.object_type) == "StretchRobot"
            ),
            object_ids=frozenset(world.objects),
            site_ids=frozenset(point.site for point in world.interaction_points.values()),
            reachable_locations=frozenset(world.objects),
        )


class SequenceCompiler:
    """Compile only closed IDs. It deliberately has no MuJoCo write capability."""

    def __init__(self, capabilities: ControlCapabilities) -> None:
        self.capabilities = capabilities

    def compile(
        self, sequence: BehaviorSequence, requested_control_mode: str | ControlMode | None = None
    ) -> CompiledSequence:
        mode = resolve_control_mode(requested_control_mode, sequence.default_control_mode)
        self._validate_mode(sequence, mode)
        self._validate_participants(sequence)
        bindings = self._binding_contracts(sequence.steps)
        # LLM mode treats YAML as policy/configuration, not a second hidden action plan.
        # Its finite segments are compiled by ``compile_segment`` before execution.
        source_steps = sequence.steps if mode is ControlMode.YAML else ()
        compiled = tuple(self._compile_step(sequence, step, bindings) for step in source_steps)
        self._validate_fallbacks(source_steps)
        actions, clips = self._coverage(compiled)
        required_actions = (
            set(sequence.acceptance.get("require_actions", []))
            if mode is ControlMode.YAML
            else set()
        )
        missing = required_actions - actions
        if missing:
            raise ControlSequenceError(
                f"Sequence cannot cover required actions: {', '.join(sorted(missing))}"
            )
        digest_payload = {
            "sequence_sha256": sequence.source_sha256,
            "mode": mode.value,
            "steps": [self._canonical_step(item) for item in compiled],
        }
        plan_sha = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return CompiledSequence(
            sequence, mode, compiled, plan_sha, frozenset(actions), frozenset(clips)
        )

    def compile_segment(
        self, sequence: BehaviorSequence, steps: tuple[SequenceStep, ...]
    ) -> tuple[CompiledStep, ...]:
        """Apply exactly the same checks to a source-produced short plan."""
        bindings = self._binding_contracts(sequence.steps)
        return tuple(self._compile_step(sequence, step, bindings) for step in steps)

    def _validate_mode(self, sequence: BehaviorSequence, mode: ControlMode) -> None:
        if mode is ControlMode.YAML and not sequence.steps:
            raise ControlSequenceError("YAML control mode requires steps")
        if mode is ControlMode.LLM:
            if not sequence.llm.enabled or sequence.llm.autonomy is None:
                raise ControlSequenceError("LLM control mode requires enabled LLM autonomy")
            if not sequence.llm.autonomy.stop:
                raise ControlSequenceError("LLM control mode requires a finite stop condition")

    def _validate_participants(self, sequence: BehaviorSequence) -> None:
        aliases = sequence.participants
        if len(set(aliases.values())) != len(aliases):
            raise ControlSequenceError("Each participant alias must bind a distinct runtime ID")
        unknown = set(aliases.values()) - self.capabilities.npc_ids - self.capabilities.robot_ids
        if unknown:
            raise ControlSequenceError(f"Unknown participants: {', '.join(sorted(unknown))}")
        for alias, location in sequence.locations.items():
            if isinstance(location, str):
                if (
                    location not in self.capabilities.site_ids
                    and location not in self.capabilities.object_ids
                ):
                    raise ControlSequenceError(
                        f"Location '{alias}' binds unknown site/object '{location}'"
                    )
            elif isinstance(location, dict):
                for key in ("target_site", "navigation_site"):
                    value = location.get(key)
                    if value is not None and value not in self.capabilities.site_ids:
                        raise ControlSequenceError(
                            f"Location '{alias}' has unknown {key} '{value}'"
                        )
            else:
                raise ControlSequenceError(
                    f"Location '{alias}' must be a site/object ID or mapping"
                )

    def _compile_step(
        self, sequence: BehaviorSequence, step: SequenceStep, bindings: Mapping[str, frozenset[str]]
    ) -> CompiledStep:
        payload = step.payload
        actor_id: str | None = None
        if "actor" in payload:
            actor_id = self._resolve_actor(sequence, payload["actor"], step.step_id)
        target_id = self._resolve_target(sequence, payload.get("target"), bindings, step.step_id)
        if step.kind is StepKind.ACTION:
            try:
                action = ActionType(payload["action"])
            except ValueError as error:
                raise ControlSequenceError(f"Step '{step.step_id}' has unknown action") from error
            if actor_id is None or actor_id not in self.capabilities.npc_ids:
                raise ControlSequenceError(f"Action step '{step.step_id}' requires an NPC actor")
            if (
                target_id is not None
                and not _BINDING.match(target_id)
                and target_id not in self.capabilities.object_ids
                and target_id not in self.capabilities.npc_ids
                and target_id not in self.capabilities.robot_ids
                and not (action is ActionType.MOVE_TO and target_id in self.capabilities.site_ids)
            ):
                raise ControlSequenceError(
                    f"Step '{step.step_id}' targets unknown object '{target_id}'"
                )
            if (
                action is ActionType.MOVE_TO
                and target_id is not None
                and target_id not in self.capabilities.reachable_locations
            ):
                raise ControlSequenceError(
                    f"Step '{step.step_id}' targets unreachable location '{target_id}'"
                )
        elif step.kind is StepKind.ANIMATION:
            if sequence.purpose is not SequencePurpose.ACCEPTANCE:
                raise ControlSequenceError("animation steps are restricted to acceptance sequences")
            if actor_id is None:
                raise ControlSequenceError(f"Animation step '{step.step_id}' requires an actor")
            clip = payload["clip"]
            if clip not in self.capabilities.clips_by_npc.get(actor_id, frozenset()):
                raise ControlSequenceError(
                    f"Animation step '{step.step_id}' uses unregistered clip '{clip}'"
                )
        elif step.kind is StepKind.PARALLEL:
            actors = [
                child.payload.get("actor") for child in step.children if child.payload.get("actor")
            ]
            if len(actors) != len(set(actors)):
                raise ControlSequenceError(
                    f"Parallel step '{step.step_id}' reserves a participant twice"
                )
            join = payload.get("join", "all")
            if join not in {"all", "any"}:
                raise ControlSequenceError(
                    f"Parallel step '{step.step_id}' join must be all or any"
                )
        elif step.kind is StepKind.CONVERSATION:
            participants = payload.get("participants")
            if not isinstance(participants, list) or len(participants) != 2:
                raise ControlSequenceError(
                    f"Conversation step '{step.step_id}' requires exactly two participants"
                )
            for participant in participants:
                self._resolve_actor(sequence, participant, step.step_id)
        elif step.kind is StepKind.LLM_REQUEST:
            contract = payload["output_contract"]
            binding_fields(contract)
            self._resolve_actor(sequence, payload["actor"], step.step_id)
        self._validate_bindings(payload, bindings, step.step_id)
        return CompiledStep(
            step=step,
            actor_id=actor_id,
            target_id=target_id,
            children=tuple(
                self._compile_step(sequence, child, bindings) for child in step.children
            ),
        )

    def _resolve_actor(self, sequence: BehaviorSequence, value: Any, step_id: str) -> str:
        if not isinstance(value, str):
            raise ControlSequenceError(f"Step '{step_id}' references unknown participant '{value}'")
        if value in sequence.participants.values():
            return value
        if value not in sequence.participants:
            raise ControlSequenceError(f"Step '{step_id}' references unknown participant '{value}'")
        return sequence.participants[value]

    def _resolve_target(
        self,
        sequence: BehaviorSequence,
        value: Any,
        bindings: Mapping[str, frozenset[str]],
        step_id: str,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ControlSequenceError(f"Step '{step_id}' target must be a string")
        if _BINDING.match(value):
            self._validate_binding(value, bindings, step_id)
            return value
        if value in sequence.participants:
            return sequence.participants[value]
        location = sequence.locations.get(value)
        if isinstance(location, str):
            return location
        # Seat mapping is an affordance. The action driver receives the named semantic target,
        # while compiler separately verifies its ingress/target sites above.
        return value

    def _binding_contracts(self, steps: tuple[SequenceStep, ...]) -> dict[str, frozenset[str]]:
        bindings: dict[str, frozenset[str]] = {}

        def visit(items: tuple[SequenceStep, ...]) -> None:
            for item in items:
                if item.kind is StepKind.LLM_REQUEST:
                    name = item.payload["save_as"]
                    if name in bindings:
                        raise ControlSequenceError(f"Duplicate LLM binding '{name}'")
                    bindings[name] = binding_fields(item.payload["output_contract"])
                visit(item.children)

        visit(steps)
        return bindings

    def _validate_bindings(
        self, value: Any, bindings: Mapping[str, frozenset[str]], step_id: str
    ) -> None:
        if isinstance(value, str):
            if value.startswith("${"):
                self._validate_binding(value, bindings, step_id)
        elif isinstance(value, dict):
            for child in value.values():
                self._validate_bindings(child, bindings, step_id)
        elif isinstance(value, list):
            for child in value:
                self._validate_bindings(child, bindings, step_id)

    @staticmethod
    def _validate_binding(value: str, bindings: Mapping[str, frozenset[str]], step_id: str) -> None:
        match = _BINDING.match(value)
        if match is None:
            raise ControlSequenceError(f"Step '{step_id}' has illegal variable reference '{value}'")
        binding, field = match.groups()
        if field not in bindings.get(binding, frozenset()):
            raise ControlSequenceError(f"Step '{step_id}' references unknown variable '{value}'")

    @staticmethod
    def _validate_fallbacks(steps: tuple[SequenceStep, ...]) -> None:
        all_ids: set[str] = set()

        def visit(items: tuple[SequenceStep, ...]) -> None:
            for item in items:
                all_ids.add(item.step_id)
                visit(item.children)

        visit(steps)

        def check(items: tuple[SequenceStep, ...]) -> None:
            for item in items:
                if item.failure.fallback_step and item.failure.fallback_step not in all_ids:
                    raise ControlSequenceError(f"Step '{item.step_id}' has unknown fallback step")
                check(item.children)

        check(steps)

    @staticmethod
    def _coverage(steps: tuple[CompiledStep, ...]) -> tuple[set[str], set[str]]:
        actions: set[str] = set()
        clips: set[str] = set()

        def visit(items: tuple[CompiledStep, ...]) -> None:
            for item in items:
                if item.step.kind is StepKind.ACTION:
                    actions.add(item.step.payload["action"])
                elif item.step.kind is StepKind.ANIMATION:
                    clips.add(item.step.payload["clip"])
                visit(item.children)

        visit(steps)
        return actions, clips

    @staticmethod
    def _canonical_step(step: CompiledStep) -> dict[str, Any]:
        return {
            "id": step.step.step_id,
            "kind": step.step.kind.value,
            "actor": step.actor_id,
            "target": step.target_id,
            "children": [SequenceCompiler._canonical_step(child) for child in step.children],
        }
