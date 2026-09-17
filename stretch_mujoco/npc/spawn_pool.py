"""Exclusive, geometry-backed NPC spawn anchor allocation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import mujoco


class SpawnAnchorError(ValueError):
    """A spawn pool contract or allocation request is invalid."""


@dataclass(frozen=True)
class SpawnAnchor:
    anchor_id: str
    site: str
    position: tuple[float, float, float]


class SpawnAnchorPool:
    """Allocate each declared spawn anchor to at most one NPC at a time."""

    def __init__(self, model: mujoco.MjModel, policy: Mapping[str, object]) -> None:
        if policy.get("allocation") != "exclusive":
            raise SpawnAnchorError("spawn_policy_allocation_must_be_exclusive")
        raw_ids = policy.get("anchors")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise SpawnAnchorError("spawn_policy_anchors_required")
        capacity = policy.get("capacity")
        if capacity != len(raw_ids):
            raise SpawnAnchorError("spawn_policy_capacity_mismatch")
        separation = float(policy.get("minimum_separation_m", 0.0))
        if not math.isfinite(separation) or separation <= 0:
            raise SpawnAnchorError("spawn_policy_minimum_separation_invalid")
        self.minimum_separation_m = separation
        self.anchors: dict[str, SpawnAnchor] = {}
        self._owners: dict[str, str] = {}
        for anchor_id in raw_ids:
            if not isinstance(anchor_id, str) or anchor_id in self.anchors:
                raise SpawnAnchorError("spawn_policy_anchor_id_invalid")
            site = ""
            # The compiler emits point.spawn.anchor.NN as the semantic ID and
            # npc__<scene>__point__spawn__anchor__NN as the MuJoCo site. Find
            # it by suffix so the pool remains independent of scene ID.
            suffix = "__" + anchor_id.replace(".", "__")
            matches = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
                for i in range(model.nsite)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or "").endswith(suffix)
            ]
            if len(matches) != 1:
                raise SpawnAnchorError(f"spawn_anchor_site_unbound:{anchor_id}")
            site = matches[0]
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
            position = tuple(float(x) for x in model.site_pos[site_id])
            self.anchors[anchor_id] = SpawnAnchor(anchor_id, site, position)
        if len(self.anchors) != capacity:
            raise SpawnAnchorError("spawn_policy_capacity_mismatch")
        anchors = list(self.anchors.values())
        for index, left in enumerate(anchors):
            for right in anchors[index + 1 :]:
                if math.dist(left.position[:2], right.position[:2]) < separation:
                    raise SpawnAnchorError(f"spawn_anchor_spacing_invalid:{left.anchor_id}:{right.anchor_id}")

    def acquire(self, npc_id: str, anchor_id: str | None = None) -> SpawnAnchor:
        if not npc_id:
            raise SpawnAnchorError("spawn_npc_id_required")
        if anchor_id is not None:
            if anchor_id not in self.anchors:
                raise SpawnAnchorError(f"spawn_anchor_unknown:{anchor_id}")
            owner = self._owners.get(anchor_id)
            if owner not in (None, npc_id):
                raise SpawnAnchorError(f"spawn_anchor_occupied:{anchor_id}")
            self._owners[anchor_id] = npc_id
            return self.anchors[anchor_id]
        for key, anchor in self.anchors.items():
            if key not in self._owners:
                self._owners[key] = npc_id
                return anchor
        raise SpawnAnchorError("spawn_anchor_pool_exhausted")

    def release(self, npc_id: str, anchor_id: str | None = None) -> bool:
        for key, owner in list(self._owners.items()):
            if owner == npc_id and (anchor_id is None or key == anchor_id):
                self._owners.pop(key)
                return True
        return False

    def snapshot(self) -> dict[str, str]:
        return dict(self._owners)
