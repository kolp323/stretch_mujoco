"""Single source of truth for generated NPC MuJoCo names."""

from __future__ import annotations

import re

CANONICAL_FRAME_PATTERN = re.compile(
    r"^npc__(?P<npc>[a-zA-Z0-9_]+)__clip__(?P<clip>[a-z0-9_]+)"
    r"__frame__(?P<frame>\d+)__slot__(?P<material>[a-z0-9_]+)$"
)
LEGACY_MULTI_FRAME_PATTERN = re.compile(
    r"^em(?P<number>\d+)_frame_(?P<clip>[a-z0-9_]+)_(?P<frame>\d+)_(?P<material>[a-z]+)$"
)
LEGACY_SINGLE_FRAME_PATTERN = re.compile(
    r"^humanoid_preview_frame_(?P<clip>[a-z0-9_]+)_(?P<frame>\d+)_(?P<material>[a-z]+)$"
)
ATTACHMENT_ANCHOR_SITE_PATTERN = re.compile(
    r"^npc__(?P<npc>[a-zA-Z0-9_]+)__anchor__(?P<role>[a-z0-9_]+)"
    r"__clip__(?P<clip>[a-z0-9_]+)__frame__(?P<frame>\d+)$"
)


def body_name(npc_id: str) -> str:
    return f"npc__{npc_id}"


def frame_geom_name(npc_id: str, clip: str, frame: int, material: str) -> str:
    return f"npc__{npc_id}__clip__{clip}__frame__{frame:03d}__slot__{material}"


def accessory_frame_geom_name(npc_id: str, clip: str, frame: int, accessory_id: str) -> str:
    return frame_geom_name(npc_id, clip, frame, f"accessory_{accessory_id}")


def interaction_site_name(npc_id: str, role: str) -> str:
    return f"npc__{npc_id}__{role}"


def attachment_anchor_site_name(npc_id: str, role: str, clip: str, frame: int) -> str:
    return f"npc__{npc_id}__anchor__{role}__clip__{clip}__frame__{frame:03d}"


def parse_attachment_anchor_site_name(name: str) -> tuple[str, str, str, int] | None:
    match = ATTACHMENT_ANCHOR_SITE_PATTERN.match(name)
    if match is None:
        return None
    return (
        match.group("npc"),
        match.group("role"),
        match.group("clip"),
        int(match.group("frame")),
    )


def collision_geom_name(npc_id: str, part: str) -> str:
    return f"npc__{npc_id}__collision__{part}"


def parse_frame_geom_name(name: str) -> tuple[str, str, int, str] | None:
    match = CANONICAL_FRAME_PATTERN.match(name)
    if match:
        return (
            match.group("npc"),
            match.group("clip"),
            int(match.group("frame")),
            match.group("material"),
        )
    match = LEGACY_MULTI_FRAME_PATTERN.match(name)
    if match:
        return (
            f"employee_{int(match.group('number')):02d}",
            match.group("clip"),
            int(match.group("frame")),
            match.group("material"),
        )
    match = LEGACY_SINGLE_FRAME_PATTERN.match(name)
    if match:
        return (
            "employee_01",
            match.group("clip"),
            int(match.group("frame")),
            match.group("material"),
        )
    return None


def candidate_body_names(npc_id: str) -> tuple[str, ...]:
    names = [body_name(npc_id), f"{npc_id}_body"]
    if npc_id == "employee_01":
        names.append("humanoid_preview")
    return tuple(names)
