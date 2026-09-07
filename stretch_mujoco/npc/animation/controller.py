"""Animation graph and marker owner independent from NPC root locomotion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .backend import AnimationBackend
from .graph import OFFICE_ANIMATION_GRAPH, AnimationGraph, ClipDefinition
from .state import AnimationLifecycle, AnimationState, InterruptPolicy


@dataclass(frozen=True)
class AnimationEvent:
    name: str
    clip: str
    phase: float
    cycle: int = 0
    sim_time: float = 0.0


class AnimationController:
    def __init__(
        self,
        backend: AnimationBackend,
        graph: AnimationGraph | None = None,
        fps: float | None = None,
        clips: dict[str, ClipDefinition] | None = None,
        phase_seed: int | None = None,
        npc_id: str = "npc",
    ) -> None:
        self.backend = backend
        if graph is not None and clips is not None:
            raise ValueError("Specify either animation graph or clips, not both")
        self.graph = graph or AnimationGraph(
            "inline", "idle", "idle", dict(OFFICE_ANIMATION_GRAPH.clips if clips is None else clips)
        )
        self.fps = fps
        self.clips = self.graph.clips
        self.requested_clip = self.graph.initial_clip
        self.resolved_clip = self.graph.initial_clip
        self.phase = 0.0
        self.transition: str | None = None
        self.fallback_event: AnimationEvent | None = None
        self._fallback_request: str | None = None
        self._last_time: float | None = None
        self._pending_reset = False
        self.lifecycle = AnimationLifecycle.REQUESTED
        self.phase_seed = 0 if phase_seed is None else phase_seed
        self.npc_id = npc_id
        self.phase_offset = 0.0
        self._execution_id = "initial"
        self._cycle = 0
        self._deferred_clip: str | None = None
        self._pending_events: list[AnimationEvent] = []

    def request(self, clip: str, *, force: bool = False) -> None:
        """Request a clip without interrupting an unsafe mesh frame sequence."""
        current = self.clips.get(self.resolved_clip, ClipDefinition())
        if clip != self.resolved_clip and not force:
            if current.interrupt_policy == InterruptPolicy.UNINTERRUPTIBLE:
                self._defer_interrupt(clip)
                return
            if (
                current.interrupt_policy == InterruptPolicy.SAFE_MARKER
                and current.safe_marker is not None
            ):
                self._defer_interrupt(clip)
                return
        if clip != self.requested_clip:
            self._pending_reset = True
            self._fallback_request = None
        self.requested_clip = clip
        self.lifecycle = AnimationLifecycle.REQUESTED

    def set_execution(self, execution_id: str) -> None:
        """Set the command scope used for deterministic loop phase offsets."""
        self._execution_id = execution_id

    @property
    def pending_clip(self) -> str | None:
        return self._deferred_clip

    def _phase_offset(self, clip: str, cycle: int = 0) -> float:
        material = f"{self.phase_seed}:{self.npc_id}:{self._execution_id}:{clip}:{cycle}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64

    def recover_to_idle(self) -> None:
        """Cancel-safe recovery used after a deadline or an explicit cancellation."""
        self._deferred_clip = None
        self.request("idle", force=True)

    def _defer_interrupt(self, clip: str) -> None:
        if self._deferred_clip != clip:
            self._pending_events.append(
                AnimationEvent("interrupt_deferred", self.resolved_clip, self.phase)
            )
        self._deferred_clip = clip

    @property
    def state(self) -> AnimationState:
        definition = self.clips.get(self.resolved_clip, ClipDefinition())
        return AnimationState(
            self.resolved_clip,
            self.phase,
            definition.speed,
            definition.loop,
            upper_body_overlay=self.resolved_clip if definition.upper_body_overlay else None,
            lifecycle=self.lifecycle,
        )

    def settle_completed_clip(self) -> str | None:
        """Request the graph-defined stable pose after a successful marker action."""
        completion_clip = self.clips.get(self.resolved_clip, ClipDefinition()).completion_clip
        if completion_clip is not None:
            self.request(completion_clip)
        return completion_clip

    def step(
        self, sim_time: float, *, locomotion: str = "stationary"
    ) -> tuple[AnimationEvent, ...]:
        requested = "walk" if locomotion == "walk" else self.requested_clip
        clips = self.backend.available_clips
        resolved = requested if requested in clips else self.graph.fallback_clip
        if resolved not in clips:
            resolved = "idle" if "idle" in clips else clips[0]
        events = self._pending_events
        self._pending_events = []
        self.fallback_event = None
        if resolved != requested and self._fallback_request != requested:
            self.fallback_event = AnimationEvent("clip_fallback", requested, self.phase)
            events.append(self.fallback_event)
            self._fallback_request = requested
        elif resolved == requested:
            self._fallback_request = None
        if resolved != self.resolved_clip or self._pending_reset:
            self.transition = f"{self.resolved_clip}->{resolved}"
            self.resolved_clip = resolved
            definition = self.clips.get(resolved, ClipDefinition())
            self._cycle = 0
            self.phase_offset = (
                self._phase_offset(resolved) if definition.loop and resolved != "idle" else 0.0
            )
            self.phase = self.phase_offset
            self._pending_reset = False
        else:
            self.transition = None
        dt = 0.0 if self._last_time is None else max(sim_time - self._last_time, 0.0)
        self._last_time = sim_time
        previous_phase = self.phase
        clip_fps = self.fps if self.fps is not None else self.clips[self.resolved_clip].fps
        if self.lifecycle not in {AnimationLifecycle.COMPLETED, AnimationLifecycle.FAILED}:
            self.lifecycle = AnimationLifecycle.PLAYING
        definition = self.clips.get(self.resolved_clip, ClipDefinition())
        next_phase = self.phase + dt * clip_fps * definition.speed / max(
            _frame_count(self.backend, self.resolved_clip), 1
        )
        if definition.loop:
            self._cycle += int(next_phase)
            self.phase = next_phase % 1.0
        else:
            self.phase = min(next_phase, 1.0)
        events.extend(self._crossed_markers(definition, previous_phase, next_phase, sim_time))
        if self._deferred_clip is not None and (
            any(event.name == definition.safe_marker for event in events)
            or (
                definition.interrupt_policy == InterruptPolicy.UNINTERRUPTIBLE
                and not definition.loop
                and self.phase >= 1.0
            )
        ):
            deferred = self._deferred_clip
            self._deferred_clip = None
            self.request(deferred, force=True)
        self.backend.sample(self.resolved_clip, self.phase)
        return tuple(events)

    def _crossed_markers(
        self,
        definition: ClipDefinition,
        previous_phase: float,
        next_phase: float,
        sim_time: float,
    ) -> list[AnimationEvent]:
        if not definition.markers or next_phase <= previous_phase:
            return []
        if definition.loop:
            start_cycle = int(previous_phase)
            end_cycle = int(next_phase)
            return [
                AnimationEvent(name, self.resolved_clip, phase, cycle, sim_time)
                for cycle in range(start_cycle, end_cycle + 1)
                for name, phase in definition.markers
                if previous_phase < cycle + phase <= next_phase
            ]
        return [
            AnimationEvent(name, self.resolved_clip, phase, 0, sim_time)
            for name, phase in definition.markers
            if previous_phase < phase <= min(next_phase, 1.0)
        ]


def _frame_count(backend: AnimationBackend, clip: str) -> int:
    binding = getattr(backend, "binding", None)
    if binding is None:
        return 1
    return len(binding.frame_geom_ids[clip])
