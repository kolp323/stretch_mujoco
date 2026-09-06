"""Reserved boundary for future mesh-based NPC appearance slots.

The current pipeline only bakes 2D textures.  A single body mesh cannot be
made into independent hair/top/bottom/shoe geoms by assigning multiple
materials to it: the overlapping copies would render and collide incorrectly.
When production meshes are authoritatively split, this module will own their
slot geometry contract and MuJoCo assembly.  Until then callers receive an
explicit error rather than an apparently working but invalid scene.
"""

from __future__ import annotations

from enum import Enum


class GeometrySlot(str, Enum):
    """Planned named geometry partitions for a production NPC bundle."""

    BODY = "body"
    HAIR = "hair"
    TOP = "top"
    BOTTOM = "bottom"
    SHOES = "shoes"


def require_split_geometry() -> None:
    """Explain that mesh slot assembly is intentionally not implemented yet."""
    raise NotImplementedError(
        "NPC geometry slots require authoritatively split meshes and are not implemented; "
        "use the texture baking pipeline for single-mesh appearances."
    )
