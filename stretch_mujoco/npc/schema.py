"""Strict, versioned population configuration for embodied office NPCs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from stretch_mujoco.agents.models import (
    EmployeeNeeds,
    EmployeeProfile,
    EmployeeSchedule,
    ScheduleItem,
)

POPULATION_SCHEMA_VERSION = 2
KNOWN_CAPABILITIES = frozenset(
    {
        "locomotion",
        "sit",
        "object_handover",
        "conversation",
        "use_computer",
    }
)


def _require_mapping(payload: object, context: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be an object")
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key '{key}' in NPC population")
        result[key] = value
    return result


def _require_fields(payload: Mapping[str, Any], fields: set[str], context: str) -> None:
    missing = fields - payload.keys()
    if missing:
        raise ValueError(f"{context} missing fields: {', '.join(sorted(missing))}")


@dataclass(frozen=True)
class NpcEmbodiment:
    bundle: str
    appearance: str
    animation_graph: str
    collision_profile: str
    scale: float = 1.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NpcEmbodiment":
        _require_fields(
            payload,
            {"bundle", "appearance", "animation_graph", "collision_profile"},
            "NPC embodiment",
        )
        scale = float(payload.get("scale", 1.0))
        if not 0.1 <= scale <= 10.0:
            raise ValueError("NPC embodiment scale must be between 0.1 and 10.0")
        return cls(
            bundle=str(payload["bundle"]),
            appearance=str(payload["appearance"]),
            animation_graph=str(payload["animation_graph"]),
            collision_profile=str(payload["collision_profile"]),
            scale=scale,
        )


@dataclass(frozen=True)
class NpcSpawn:
    location: str
    site: str
    yaw: float = 0.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NpcSpawn":
        _require_fields(payload, {"location", "site"}, "NPC spawn")
        return cls(
            location=str(payload["location"]),
            site=str(payload["site"]),
            yaw=float(payload.get("yaw", 0.0)),
        )


@dataclass(frozen=True)
class NpcDefinition:
    npc_id: str
    profile: EmployeeProfile
    embodiment: NpcEmbodiment
    spawn: NpcSpawn
    capabilities: frozenset[str]
    needs: EmployeeNeeds
    schedule: EmployeeSchedule

    @classmethod
    def from_dict(
        cls,
        npc_id: str,
        payload: Mapping[str, Any],
        *,
        locations: set[str] | frozenset[str] | None = None,
        sites: set[str] | frozenset[str] | None = None,
    ) -> "NpcDefinition":
        _require_fields(
            payload,
            {"profile", "embodiment", "spawn", "capabilities", "needs", "schedule"},
            f"NPC '{npc_id}'",
        )
        profile_data = _require_mapping(payload["profile"], f"NPC '{npc_id}' profile")
        _require_fields(profile_data, {"role", "department"}, f"NPC '{npc_id}' profile")
        spawn = NpcSpawn.from_dict(_require_mapping(payload["spawn"], f"NPC '{npc_id}' spawn"))
        if locations is not None and spawn.location not in locations:
            raise ValueError(f"NPC '{npc_id}' has unknown spawn location '{spawn.location}'")
        if sites is not None and spawn.site not in sites:
            raise ValueError(f"NPC '{npc_id}' has unknown spawn site '{spawn.site}'")

        raw_capabilities = payload["capabilities"]
        if not isinstance(raw_capabilities, list) or not all(
            isinstance(item, str) for item in raw_capabilities
        ):
            raise ValueError(f"NPC '{npc_id}' capabilities must be a list of strings")
        capabilities = frozenset(raw_capabilities)
        unknown = capabilities - KNOWN_CAPABILITIES
        if unknown:
            raise ValueError(
                f"NPC '{npc_id}' has unknown capabilities: {', '.join(sorted(unknown))}"
            )
        needs_data = _require_mapping(payload["needs"], f"NPC '{npc_id}' needs")
        schedule_data = payload["schedule"]
        if not isinstance(schedule_data, list):
            raise ValueError(f"NPC '{npc_id}' schedule must be a list")
        for item in schedule_data:
            _require_mapping(item, f"NPC '{npc_id}' schedule item")
        personality = _require_mapping(
            profile_data.get("personality", {}), f"NPC '{npc_id}' personality"
        )
        preferences = _require_mapping(
            profile_data.get("preferences", {}), f"NPC '{npc_id}' preferences"
        )
        schedule = EmployeeSchedule(
            tuple(ScheduleItem.from_dict(dict(item)) for item in schedule_data)
        )
        if locations is not None:
            for item in schedule.items:
                if item.location not in locations:
                    raise ValueError(
                        f"NPC '{npc_id}' schedule item '{item.item_id}' has unknown location "
                        f"'{item.location}'"
                    )
        return cls(
            npc_id=npc_id,
            profile=EmployeeProfile(
                role=str(profile_data["role"]),
                department=str(profile_data["department"]),
                personality={str(key): float(value) for key, value in personality.items()},
                preferences=dict(preferences),
            ),
            embodiment=NpcEmbodiment.from_dict(
                _require_mapping(payload["embodiment"], f"NPC '{npc_id}' embodiment")
            ),
            spawn=spawn,
            capabilities=capabilities,
            needs=EmployeeNeeds(
                hunger=float(needs_data.get("hunger", 0.0)),
                thirst=float(needs_data.get("thirst", 0.0)),
                fatigue=float(needs_data.get("fatigue", 0.0)),
            ),
            schedule=schedule,
        )


@dataclass(frozen=True)
class NpcPopulation:
    scene: str
    asset_manifest: str
    npcs: dict[str, NpcDefinition]
    clock: dict[str, Any]
    source_path: Path | None = None

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        source_path: Path | None = None,
        locations: set[str] | frozenset[str] | None = None,
        sites: set[str] | frozenset[str] | None = None,
    ) -> "NpcPopulation":
        version = payload.get("schema_version")
        if version != POPULATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported NPC population schema_version {version!r}; "
                f"expected {POPULATION_SCHEMA_VERSION}"
            )
        _require_fields(payload, {"scene", "asset_manifest", "clock", "npcs"}, "Population")
        npc_payloads = _require_mapping(payload["npcs"], "Population npcs")
        if not npc_payloads:
            raise ValueError("Population must define at least one NPC")
        npcs = {
            str(npc_id): NpcDefinition.from_dict(
                str(npc_id),
                _require_mapping(definition, f"NPC '{npc_id}'"),
                locations=locations,
                sites=sites,
            )
            for npc_id, definition in npc_payloads.items()
        }
        return cls(
            scene=str(payload["scene"]),
            asset_manifest=str(payload["asset_manifest"]),
            npcs=npcs,
            clock=dict(_require_mapping(payload["clock"], "Population clock")),
            source_path=source_path,
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        locations: set[str] | frozenset[str] | None = None,
        sites: set[str] | frozenset[str] | None = None,
    ) -> "NpcPopulation":
        source_path = Path(path).resolve()
        payload = json.loads(
            source_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
        return cls.from_dict(
            _require_mapping(payload, "Population"),
            source_path=source_path,
            locations=locations,
            sites=sites,
        )

    def resolve_path(self, value: str) -> Path:
        base = self.source_path.parent if self.source_path is not None else Path.cwd()
        return (base / value).resolve()
