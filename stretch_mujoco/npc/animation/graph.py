"""Built-in animation graph definitions shared by baking and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..assets import AssetBundle


@dataclass(frozen=True)
class ClipDefinition:
    fps: float = 8.0
    loop: bool = True
    root_motion: str = "in_place"
    markers: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class AnimationGraph:
    """Validated runtime view of one bundle's animation contract.

    Frame ownership remains in the asset manifest.  This graph deliberately
    owns only runtime selection policy, so a mesh-sequence backend can later
    be replaced without changing command semantics.
    """

    graph_id: str
    initial_clip: str
    fallback_clip: str
    clips: dict[str, ClipDefinition]

    def __post_init__(self) -> None:
        if self.initial_clip not in self.clips:
            raise ValueError(f"Animation graph '{self.graph_id}' has no initial clip")
        if self.fallback_clip not in self.clips:
            raise ValueError(f"Animation graph '{self.graph_id}' has no fallback clip")
        for clip, definition in self.clips.items():
            if definition.fps <= 0:
                raise ValueError(f"Animation graph '{self.graph_id}' clip '{clip}' has invalid fps")
            if definition.root_motion not in {"in_place", "authored"}:
                raise ValueError(
                    f"Animation graph '{self.graph_id}' clip '{clip}' has invalid root motion"
                )
            seen: set[str] = set()
            for name, phase in definition.markers:
                if not name or name in seen or not 0.0 <= phase <= 1.0:
                    raise ValueError(
                        f"Animation graph '{self.graph_id}' clip '{clip}' has invalid marker"
                    )
                seen.add(name)

    @classmethod
    def from_bundle(cls, graph_id: str, bundle: "AssetBundle") -> "AnimationGraph":
        return cls(
            graph_id=graph_id,
            initial_clip="idle",
            fallback_clip="idle",
            clips={
                clip_id: ClipDefinition(
                    fps=clip.fps,
                    loop=clip.loop,
                    root_motion=clip.root_motion,
                    markers=tuple(
                        (str(marker["name"]), float(cast(float, marker["phase"])))
                        for marker in clip.markers
                    ),
                )
                for clip_id, clip in bundle.clips.items()
            },
        )


OFFICE_CLIPS: dict[str, ClipDefinition] = {
    "idle": ClipDefinition(),
    "walk": ClipDefinition(markers=(("left_foot", 0.25), ("right_foot", 0.75))),
    "sit": ClipDefinition(loop=False, markers=(("seated", 0.875),)),
    "work": ClipDefinition(markers=(("work_cycle", 0.75),)),
    "eat": ClipDefinition(markers=(("consume", 0.625),)),
}

OFFICE_ANIMATION_GRAPH = AnimationGraph(
    graph_id="office_humanoid_v1",
    initial_clip="idle",
    fallback_clip="idle",
    clips=OFFICE_CLIPS,
)
