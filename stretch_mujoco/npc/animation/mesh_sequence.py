"""Compatibility backend for pre-baked OBJ frame sequences."""

from __future__ import annotations

import mujoco

from ..binding import NpcBinding


class MeshSequenceBackend:
    capabilities = frozenset({"mesh_sequence", "phase_switch"})
    _REVERSED_CLIP_SOURCES = {"stand_up": ("sit", "sit_down")}

    def __init__(self, model: mujoco.MjModel, binding: NpcBinding) -> None:
        self.model = model
        self.binding = binding
        self._active: set[int] = {
            geom_id
            for clip in binding.frame_geom_ids.values()
            for frame in clip.values()
            for geom_id in frame.values()
            if model.geom_rgba[geom_id, 3] > 0
        }
        self._last_key: tuple[str, int] | None = None

    @property
    def available_clips(self) -> tuple[str, ...]:
        return tuple(sorted(self.binding.frame_geom_ids))

    def sample(self, clip: str, phase: float) -> None:
        if clip not in self.binding.frame_geom_ids:
            clip = "idle" if "idle" in self.binding.frame_geom_ids else self.available_clips[0]
        source_clip = self._source_clip(clip)
        frames = self.binding.frame_geom_ids[source_clip]
        frame_numbers = sorted(frames)
        index = min(int(max(0.0, min(phase, 1.0)) * len(frame_numbers)), len(frame_numbers) - 1)
        if source_clip != clip:
            index = len(frame_numbers) - index - 1
        frame_number = frame_numbers[index]
        key = (source_clip, frame_number)
        if key == self._last_key:
            return
        for geom_id in self._active:
            self.model.geom_rgba[geom_id, 3] = 0.0
        self._active = set(frames[frame_number].values())
        for geom_id in self._active:
            self.model.geom_rgba[geom_id, 3] = 1.0
        self._last_key = key

    def _source_clip(self, clip: str) -> str:
        for source_clip in self._REVERSED_CLIP_SOURCES.get(clip, ()):
            if source_clip in self.binding.frame_geom_ids:
                return source_clip
        return clip
