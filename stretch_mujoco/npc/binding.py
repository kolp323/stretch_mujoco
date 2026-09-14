"""Resolve stable NPC names to cached MuJoCo identifiers."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from .naming import (
    candidate_body_names,
    parse_attachment_anchor_site_name,
    parse_frame_geom_name,
)


@dataclass(frozen=True)
class NpcBinding:
    npc_id: str
    body_id: int
    mocap_id: int
    frame_geom_ids: dict[str, dict[int, dict[str, int]]]
    interaction_site_ids: dict[str, int]
    frame_anchor_site_ids: dict[str, dict[str, dict[int, int]]]
    collision_geom_ids: tuple[int, ...]

    @classmethod
    def from_model(cls, model: mujoco.MjModel, npc_id: str) -> "NpcBinding":
        body_id = -1
        for name in candidate_body_names(npc_id):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                break
        if body_id < 0:
            raise ValueError(f"NPC '{npc_id}' has no bound MuJoCo body")
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"NPC '{npc_id}' body is not a mocap body")

        frames: dict[str, dict[int, dict[str, int]]] = {}
        collision_ids: list[int] = []
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not name:
                continue
            parsed = parse_frame_geom_name(name)
            if parsed is not None and parsed[0] == npc_id:
                _, clip, frame, material = parsed
                frames.setdefault(clip, {}).setdefault(frame, {})[material] = geom_id
            if int(model.geom_bodyid[geom_id]) == body_id and (
                "collision" in name or "_collision_" in name
            ):
                collision_ids.append(geom_id)
        if not frames:
            raise ValueError(f"NPC '{npc_id}' has no animation frame geoms")

        sites: dict[str, int] = {}
        frame_anchor_sites: dict[str, dict[str, dict[int, int]]] = {}
        for site_id in range(model.nsite):
            if int(model.site_bodyid[site_id]) != body_id:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
            if name:
                sites[name] = site_id
                parsed_anchor = parse_attachment_anchor_site_name(name)
                if parsed_anchor is not None and parsed_anchor[0] == npc_id:
                    _, role, clip, frame = parsed_anchor
                    frame_anchor_sites.setdefault(role, {}).setdefault(clip, {})[frame] = site_id
        return cls(
            npc_id,
            body_id,
            mocap_id,
            frames,
            sites,
            frame_anchor_sites,
            tuple(collision_ids),
        )


def discover_npc_ids(model: mujoco.MjModel) -> tuple[str, ...]:
    npc_ids: set[str] = set()
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        parsed = parse_frame_geom_name(name or "")
        if parsed is not None:
            npc_ids.add(parsed[0])
    return tuple(sorted(npc_ids))
