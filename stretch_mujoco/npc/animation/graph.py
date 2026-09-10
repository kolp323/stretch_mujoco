"""Built-in animation graph definitions shared by baking and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from .state import InterruptPolicy

if TYPE_CHECKING:
    from ..assets import AssetBundle


@dataclass(frozen=True)
class ClipDefinition:
    fps: float = 8.0
    loop: bool = True
    root_motion: str = "in_place"
    markers: tuple[tuple[str, float], ...] = ()
    completion_clip: str | None = None
    speed: float = 1.0
    interrupt_policy: InterruptPolicy = InterruptPolicy.SAFE_MARKER
    safe_marker: str | None = None
    upper_body_overlay: bool = False


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
            if definition.speed <= 0:
                raise ValueError(
                    f"Animation graph '{self.graph_id}' clip '{clip}' has invalid speed"
                )
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
            if (
                definition.completion_clip is not None
                and definition.completion_clip not in self.clips
            ):
                raise ValueError(
                    f"Animation graph '{self.graph_id}' clip '{clip}' has an unknown completion clip"
                )

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
                    completion_clip=OFFICE_COMPLETION_CLIPS.get(clip_id),
                    speed=OFFICE_CLIPS.get(clip_id, ClipDefinition()).speed,
                    interrupt_policy=OFFICE_CLIPS.get(clip_id, ClipDefinition()).interrupt_policy,
                    safe_marker=OFFICE_CLIPS.get(clip_id, ClipDefinition()).safe_marker,
                    upper_body_overlay=OFFICE_CLIPS.get(
                        clip_id, ClipDefinition()
                    ).upper_body_overlay,
                )
                for clip_id, clip in bundle.clips.items()
            },
        )


OFFICE_COMPLETION_CLIPS = {
    "sit_down": "seated_idle",
    "stand_up": "idle",
}

OFFICE_CLIPS: dict[str, ClipDefinition] = {
    "idle": ClipDefinition(),
    "walk": ClipDefinition(
        markers=(("left_foot", 0.25), ("right_foot", 0.75)),
        safe_marker="right_foot",
    ),
    "sit_down": ClipDefinition(
        loop=False,
        markers=(("seated", 0.875),),
        completion_clip=OFFICE_COMPLETION_CLIPS["sit_down"],
    ),
    "seated_idle": ClipDefinition(),
    "stand_up": ClipDefinition(
        loop=False,
        markers=(("standing", 0.875),),
        completion_clip=OFFICE_COMPLETION_CLIPS["stand_up"],
    ),
    "work": ClipDefinition(markers=(("work_cycle", 0.75),)),
    "use_computer": ClipDefinition(markers=(("computer_cycle", 0.75),)),
    "eat": ClipDefinition(markers=(("consume", 0.625),)),
    "pick_up": ClipDefinition(loop=False, markers=(("grasp", 0.75),)),
    "place": ClipDefinition(loop=False, markers=(("release", 0.75),)),
    "give": ClipDefinition(loop=False, markers=(("handover_ready", 0.6),)),
    "receive": ClipDefinition(loop=False, markers=(("handover_ready", 0.6),)),
    # The approved BEAT talk asset is a foot-locked full-body mesh sequence.
    # MeshSequenceBackend cannot combine an upper-body overlay with an idle
    # sequence, so this clip must remain directly playable until an articulated
    # backend is introduced.
    "talk": ClipDefinition(markers=(("talk_cycle", 0.75),)),
    "gesture_wave": ClipDefinition(
        loop=False,
        markers=(("gesture_wave_complete", 0.932),),
        upper_body_overlay=True,
    ),
    "gesture_point": ClipDefinition(
        loop=False,
        markers=(("gesture_point_complete", 0.75),),
        upper_body_overlay=True,
    ),
}

OFFICE_ANIMATION_GRAPH = AnimationGraph(
    graph_id="office_humanoid_v1",
    initial_clip="idle",
    fallback_clip="idle",
    clips=OFFICE_CLIPS,
)
