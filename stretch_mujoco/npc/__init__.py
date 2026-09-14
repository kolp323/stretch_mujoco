"""Versioned NPC definitions and per-instance simulation control."""

from .protocol import CommandStatus, NpcCommand, NpcCommandKind, NpcCommandReceipt, NpcRuntimeState
from .schema import NpcDefinition, NpcEmbodiment, NpcPopulation, NpcSpawn
from .composition import (
    ComposedNpcRuntime,
    NpcSceneComposition,
    compose_npc_scene,
    load_composed_npc_runtime,
)

__all__ = [
    "CommandStatus",
    "NpcCommand",
    "NpcCommandKind",
    "NpcCommandReceipt",
    "NpcDefinition",
    "NpcEmbodiment",
    "NpcPopulation",
    "NpcRuntimeState",
    "NpcSpawn",
    "ComposedNpcRuntime",
    "NpcSceneComposition",
    "compose_npc_scene",
    "load_composed_npc_runtime",
]
