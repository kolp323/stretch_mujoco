"""NPC animation controllers and backends."""

from .controller import AnimationController, AnimationEvent
from .graph import OFFICE_ANIMATION_GRAPH, OFFICE_CLIPS, AnimationGraph, ClipDefinition
from .mesh_sequence import MeshSequenceBackend

__all__ = [
    "AnimationController",
    "AnimationEvent",
    "AnimationGraph",
    "ClipDefinition",
    "MeshSequenceBackend",
    "OFFICE_CLIPS",
    "OFFICE_ANIMATION_GRAPH",
]
