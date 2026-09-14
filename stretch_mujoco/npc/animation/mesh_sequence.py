"""Compatibility backend for pre-baked OBJ frame sequences."""

from __future__ import annotations

import mujoco

from ..binding import NpcBinding


class MeshSequenceBackend:
    capabilities = frozenset({"mesh_sequence", "phase_switch"})
    _REVERSED_CLIP_SOURCES = {"stand_up": ("sit", "sit_down")}
    # This is a logical stable pose, not a separately baked motion.  Keeping
    # the final sit mesh visible prevents a missing clip from falling back to
    # the standing idle sequence between work and stand-up.
    _STATIC_POSE_SOURCES = {"seated_idle": ("sit", "sit_down")}

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
        clips = set(self.binding.frame_geom_ids)
        for logical_clip, sources in self._STATIC_POSE_SOURCES.items():
            if any(source in clips for source in sources):
                clips.add(logical_clip)
        return tuple(sorted(clips))

    def frame_count(self, clip: str) -> int:
        """Return one frame for logical frozen poses and source frames otherwise."""
        if clip in self._STATIC_POSE_SOURCES and any(
            source in self.binding.frame_geom_ids for source in self._STATIC_POSE_SOURCES[clip]
        ):
            return 1
        source_clip = self._source_clip(clip)
        return len(self.binding.frame_geom_ids[source_clip])

    def sample(self, clip: str, phase: float) -> None:
        if clip not in self.available_clips:
            clip = "idle" if "idle" in self.binding.frame_geom_ids else self.available_clips[0]
        source_clip = self._source_clip(clip)
        frames = self.binding.frame_geom_ids[source_clip]
        frame_numbers = sorted(frames)
        if clip in self._STATIC_POSE_SOURCES and source_clip != clip:
            index = len(frame_numbers) - 1
        else:
            index = min(
                int(max(0.0, min(phase, 1.0)) * len(frame_numbers)), len(frame_numbers) - 1
            )
        if source_clip != clip and clip not in self._STATIC_POSE_SOURCES:
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
        self._sync_attachment_anchors(source_clip, frame_number)
        self._last_key = key

    def _sync_attachment_anchors(self, clip: str, frame: int) -> None:
        """Move stable interaction sites onto the hand position of the visible mesh frame."""
        for role, clips in self.binding.frame_anchor_site_ids.items():
            frame_sites = clips.get(clip)
            if frame_sites is None or frame not in frame_sites:
                continue
            target_name = f"npc__{self.binding.npc_id}__{role}"
            target_id = self.binding.interaction_site_ids.get(target_name)
            if target_id is not None:
                self.model.site_pos[target_id] = self.model.site_pos[frame_sites[frame]]

    def _source_clip(self, clip: str) -> str:
        for source_clip in self._STATIC_POSE_SOURCES.get(clip, ()):
            if source_clip in self.binding.frame_geom_ids:
                return source_clip
        for source_clip in self._REVERSED_CLIP_SOURCES.get(clip, ()):
            if source_clip in self.binding.frame_geom_ids:
                return source_clip
        return clip
