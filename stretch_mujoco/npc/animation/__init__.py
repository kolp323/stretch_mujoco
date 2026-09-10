"""NPC animation controllers and backends."""

from .controller import AnimationController, AnimationEvent
from .graph import OFFICE_ANIMATION_GRAPH, OFFICE_CLIPS, AnimationGraph, ClipDefinition
from .mesh_sequence import MeshSequenceBackend
from .state import AnimationLifecycle, AnimationState, InterruptPolicy

__all__ = [
    "AnimationController",
    "AnimationEvent",
    "AnimationGraph",
    "AnimationLifecycle",
    "AnimationState",
    "InterruptPolicy",
    "ClipDefinition",
    "MeshSequenceBackend",
    "OFFICE_CLIPS",
    "OFFICE_ANIMATION_GRAPH",
]
