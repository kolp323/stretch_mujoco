"""Typed runtime animation state; mesh-sequence backends do not blend skeletons."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AnimationLifecycle(str, Enum):
    REQUESTED = "requested"
    NAVIGATING = "navigating"
    ALIGNING = "aligning"
    PLAYING = "playing"
    COMPLETED = "completed"
    FAILED = "failed"


class InterruptPolicy(str, Enum):
    IMMEDIATE = "immediate"
    SAFE_MARKER = "safe_marker"
    UNINTERRUPTIBLE = "uninterruptible"


@dataclass(frozen=True)
class AnimationState:
    clip: str
    phase: float
    speed: float
    loop: bool
    blend: float = 0.0
    upper_body_overlay: str | None = None
    lifecycle: AnimationLifecycle = AnimationLifecycle.REQUESTED

    def __post_init__(self) -> None:
        if not 0.0 <= self.phase <= 1.0 or self.speed <= 0.0 or not 0.0 <= self.blend <= 1.0:
            raise ValueError("Invalid animation state")
