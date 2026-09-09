"""Reproducible texture baking for NPC appearance manifests.

This package owns source-layer composition only.  It deliberately does not
change mesh topology or create MuJoCo geoms; that future responsibility is
reserved for :mod:`stretch_mujoco.npc.appearance_pipeline.slots`.
"""

from .bake import (
    BakedAppearance,
    bake_appearance,
    bake_appearance_definition,
    register_baked_appearance,
)
from .accessory_recipe import FusedAccessoryRecipe, FusedAccessoryRuntimeConfig, recipe_sha256
from .catalog import AppearanceCatalog, VisualIdentity
from .face_details import generate_face_detail_layers
from .flat_layers import generate_flat_layers
from .hair_layers import generate_short_hair_layers
from .semantic_masks import SemanticMaskSet, generate_semantic_masks
from .textile_layers import generate_textile_layers

__all__ = [
    "BakedAppearance",
    "FusedAccessoryRecipe",
    "FusedAccessoryRuntimeConfig",
    "AppearanceCatalog",
    "VisualIdentity",
    "SemanticMaskSet",
    "bake_appearance",
    "bake_appearance_definition",
    "generate_flat_layers",
    "generate_face_detail_layers",
    "generate_short_hair_layers",
    "generate_semantic_masks",
    "generate_textile_layers",
    "register_baked_appearance",
    "recipe_sha256",
]
