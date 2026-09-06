"""Animation graph and marker owner independent from NPC root locomotion."""

from __future__ import annotations

from dataclasses import dataclass

from .backend import AnimationBackend
from .graph import OFFICE_ANIMATION_GRAPH, AnimationGraph, ClipDefinition


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

    def request(self, clip: str) -> None:
        if clip != self.requested_clip:
            self._pending_reset = True
            self._fallback_request = None
        self.requested_clip = clip

    def step(
        self, sim_time: float, *, locomotion: str = "stationary"
    ) -> tuple[AnimationEvent, ...]:
        requested = "walk" if locomotion == "walk" else self.requested_clip
        clips = self.backend.available_clips
        resolved = requested if requested in clips else self.graph.fallback_clip
        if resolved not in clips:
            resolved = "idle" if "idle" in clips else clips[0]
        events: list[AnimationEvent] = []
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
            self.phase = 0.0
            self._pending_reset = False
        else:
            self.transition = None
        dt = 0.0 if self._last_time is None else max(sim_time - self._last_time, 0.0)
        self._last_time = sim_time
        previous_phase = self.phase
        clip_fps = self.fps if self.fps is not None else self.clips[self.resolved_clip].fps
        next_phase = self.phase + dt * clip_fps / max(
            _frame_count(self.backend, self.resolved_clip), 1
        )
        definition = self.clips.get(self.resolved_clip, ClipDefinition())
        if definition.loop:
            self.phase = next_phase % 1.0
        else:
            self.phase = min(next_phase, 1.0)
        events.extend(self._crossed_markers(definition, previous_phase, next_phase, sim_time))
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
