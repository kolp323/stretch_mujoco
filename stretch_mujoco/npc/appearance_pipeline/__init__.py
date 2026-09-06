"""Reproducible texture baking for NPC appearance manifests.

This package owns source-layer composition only.  It deliberately does not
change mesh topology or create MuJoCo geoms; that future responsibility is
reserved for :mod:`stretch_mujoco.npc.appearance_pipeline.slots`.
"""

from .bake import BakedAppearance, bake_appearance, register_baked_appearance
from .flat_layers import generate_flat_layers

__all__ = [
    "BakedAppearance",
    "bake_appearance",
    "generate_flat_layers",
    "register_baked_appearance",
]
