"""Backend contract for animation representations."""

from __future__ import annotations

from typing import Protocol


class AnimationBackend(Protocol):
    capabilities: frozenset[str]

    @property
    def available_clips(self) -> tuple[str, ...]:
        ...

    def sample(self, clip: str, phase: float) -> None:
        ...
