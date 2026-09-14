"""Strict, versioned population configuration for embodied office NPCs."""

from __future__ import annotations

import json
import math
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
KNOWN_SCHEDULE_ACTIVITIES = frozenset({"work", "rest", "meeting"})
INTERACTION_TEMPLATE_ROLES = {
    "conversation": frozenset({"speaker", "listener"}),
    "handover": frozenset({"giver", "receiver"}),
}


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
class NpcAppearance:
    """JSON-selectable appearance slots bound to one NPC/agent definition.

    The five texture slots are stable semantic IDs.  They are composed into the
    legacy single-body atlas today, but keeping the selection explicit permits
    future split-mesh materials without changing population JSON again.
    """

    skin: str
    hair: str
    top: str
    bottom: str
    shoes: str
    accessories: tuple[str, ...]
    scale: float

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NpcAppearance":
        _require_fields(
            payload,
            {"skin", "hair", "top", "bottom", "shoes", "accessories", "scale"},
            "NPC appearance",
        )
        slots = {name: payload[name] for name in ("skin", "hair", "top", "bottom", "shoes")}
        if not all(isinstance(value, str) and value for value in slots.values()):
            raise ValueError("NPC appearance texture slots must be non-empty string IDs")
        raw_accessories = payload["accessories"]
        if (
            not isinstance(raw_accessories, list)
            or not all(isinstance(item, str) and item for item in raw_accessories)
            or len(set(raw_accessories)) != len(raw_accessories)
        ):
            raise ValueError("NPC appearance accessories must be unique non-empty string IDs")
        scale = float(payload["scale"])
        if not math.isfinite(scale) or not 0.1 <= scale <= 10.0:
            raise ValueError("NPC appearance scale must be between 0.1 and 10.0")
        return cls(**slots, accessories=tuple(raw_accessories), scale=scale)


@dataclass(frozen=True)
class NpcEmbodiment:
    bundle: str
    appearance: str
    animation_graph: str
    collision_profile: str
    scale: float = 1.0
    visual_identity: str | None = None
    accessories: tuple[str, ...] = ()
    appearance_config: NpcAppearance | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NpcEmbodiment":
        _require_fields(
            payload,
            {"bundle", "appearance", "animation_graph", "collision_profile"},
            "NPC embodiment",
        )
        appearance_config = (
            NpcAppearance.from_dict(
                _require_mapping(payload["appearance_config"], "NPC appearance")
            )
            if payload.get("appearance_config") is not None
            else None
        )
        scale = float(payload.get("scale", appearance_config.scale if appearance_config else 1.0))
        if not math.isfinite(scale) or not 0.1 <= scale <= 10.0:
            raise ValueError("NPC embodiment scale must be between 0.1 and 10.0")
        raw_accessories = payload.get(
            "accessories", list(appearance_config.accessories) if appearance_config else []
        )
        if (
            not isinstance(raw_accessories, list)
            or not all(isinstance(item, str) and item for item in raw_accessories)
            or len(set(raw_accessories)) != len(raw_accessories)
        ):
            raise ValueError("NPC embodiment accessories must be unique non-empty string IDs")
        if appearance_config is not None and (
            scale != appearance_config.scale
            or tuple(raw_accessories) != appearance_config.accessories
        ):
            raise ValueError("NPC embodiment scale/accessories must match appearance_config")
        return cls(
            bundle=str(payload["bundle"]),
            appearance=str(payload["appearance"]),
            animation_graph=str(payload["animation_graph"]),
            collision_profile=str(payload["collision_profile"]),
            scale=scale,
            visual_identity=(
                str(payload["visual_identity"])
                if payload.get("visual_identity") is not None
                else None
            ),
            accessories=tuple(raw_accessories),
            appearance_config=appearance_config,
        )


@dataclass(frozen=True)
class NpcSpawn:
    location: str
    site: str
    yaw: float = 0.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NpcSpawn":
        _require_fields(payload, {"location", "site"}, "NPC spawn")
        location, site, yaw = (
            str(payload["location"]),
            str(payload["site"]),
            float(payload.get("yaw", 0.0)),
        )
        if not location or not site or not math.isfinite(yaw):
            raise ValueError("NPC spawn requires non-empty location/site and finite yaw")
        return cls(location=location, site=site, yaw=yaw)


@dataclass(frozen=True)
class NpcInteractionStation:
    """A roster-wildcard station for one interaction role."""

    site: str
    yaw: float | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, context: str) -> "NpcInteractionStation":
        _require_fields(payload, {"site"}, context)
        unknown = set(payload) - {"site", "yaw"}
        if unknown:
            raise ValueError(f"{context} has unknown fields: {', '.join(sorted(unknown))}")
        site = payload["site"]
        if not isinstance(site, str) or not site.strip():
            raise ValueError(f"{context} requires a non-empty site")
        yaw = payload.get("yaw")
        if yaw is not None:
            if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
                raise ValueError(f"{context} yaw must be finite")
            yaw = float(yaw)
        return cls(site=site, yaw=yaw)


@dataclass(frozen=True)
class NpcInteractionTemplate:
    """Role stations expanded for every ordered pair in a population roster."""

    kind: str
    roles: dict[str, NpcInteractionStation]

    @classmethod
    def from_dict(cls, kind: str, payload: Mapping[str, Any]) -> "NpcInteractionTemplate":
        expected_roles = INTERACTION_TEMPLATE_ROLES.get(kind)
        if expected_roles is None:
            raise ValueError(f"Unknown interaction template kind '{kind}'")
        role_names = set(payload)
        if role_names != expected_roles:
            missing = expected_roles - role_names
            unknown = role_names - expected_roles
            detail = []
            if missing:
                detail.append(f"missing roles: {', '.join(sorted(missing))}")
            if unknown:
                detail.append(f"unknown roles: {', '.join(sorted(unknown))}")
            raise ValueError(f"Interaction template '{kind}' {'; '.join(detail)}")
        return cls(
            kind=kind,
            roles={
                role: NpcInteractionStation.from_dict(
                    _require_mapping(value, f"Interaction template '{kind}' role '{role}'"),
                    context=f"Interaction template '{kind}' role '{role}'",
                )
                for role, value in payload.items()
            },
        )


@dataclass(frozen=True)
class NpcDefinition:
    npc_id: str
    agent_id: str
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
        # ``agent_id`` was independently configurable in early schema-v2 files.
        # Keep it readable for compatibility; embodied runtime identity is the
        # canonical NPC mapping key (``npc_id``), not this legacy alias.
        personality = _require_mapping(
            profile_data.get("personality", {}), f"NPC '{npc_id}' personality"
        )
        preferences = _require_mapping(
            profile_data.get("preferences", {}), f"NPC '{npc_id}' preferences"
        )
        for name, value in {**needs_data, **personality}.items():
            valid_value = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                and 0 <= value <= 1
            )
            if not valid_value:
                raise ValueError(f"NPC '{npc_id}' {name} must be a finite value in [0, 1]")
        for item in schedule_data:
            _require_fields(
                item,
                {"id", "start", "end", "activity", "location"},
                f"NPC '{npc_id}' schedule item",
            )
            parsed_item = ScheduleItem.from_dict(dict(item))
            start, end = parsed_item.start_minute, parsed_item.end_minute
            if end <= start:
                raise ValueError(
                    f"NPC '{npc_id}' schedule item '{item['id']}' must end after start"
                )
            if item["activity"] not in KNOWN_SCHEDULE_ACTIVITIES:
                raise ValueError(
                    f"NPC '{npc_id}' schedule item '{item['id']}' has invalid activity"
                )
            variation = item.get("variation_minutes", 0)
            if isinstance(variation, bool) or not isinstance(variation, int) or variation < 0:
                raise ValueError(
                    f"NPC '{npc_id}' schedule item '{item['id']}' "
                    "variation_minutes must be non-negative"
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
            agent_id=str(payload.get("agent_id", npc_id)),
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
    appearance_catalog: str | None = None
    trajectory_profile: str | None = None
    dialogue_policy: dict[str, Any] | None = None
    interaction_templates: dict[str, NpcInteractionTemplate] | None = None
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
        appearance_catalog = (
            str(payload["appearance_catalog"])
            if payload.get("appearance_catalog") is not None
            else None
        )
        trajectory_profile = (
            str(payload["trajectory_profile"])
            if payload.get("trajectory_profile") is not None
            else None
        )
        dialogue_policy = payload.get("dialogue_policy")
        if dialogue_policy is not None:
            dialogue_policy = dict(_require_mapping(dialogue_policy, "Population dialogue_policy"))
        raw_templates = payload.get("interaction_templates")
        interaction_templates: dict[str, NpcInteractionTemplate] = {}
        if raw_templates is not None:
            template_payloads = _require_mapping(raw_templates, "Population interaction_templates")
            interaction_templates = {
                str(kind): NpcInteractionTemplate.from_dict(
                    str(kind), _require_mapping(value, f"Interaction template '{kind}'")
                )
                for kind, value in template_payloads.items()
            }
            if len(npcs) < 2:
                raise ValueError("Population interaction_templates require at least two NPCs")
            if sites is not None:
                for template in interaction_templates.values():
                    for station in template.roles.values():
                        if station.site not in sites:
                            raise ValueError(
                                f"Interaction template '{template.kind}' has unknown site "
                                f"'{station.site}'"
                            )
        if appearance_catalog is not None:
            if source_path is None:
                raise ValueError("Population appearance_catalog requires a source path")
            from .appearance_pipeline.catalog import AppearanceCatalog

            catalog_path = source_path.parent / appearance_catalog
            catalog = AppearanceCatalog.from_json(catalog_path)
            for definition in npcs.values():
                identity_id = definition.embodiment.visual_identity
                if identity_id is None:
                    raise ValueError(
                        f"NPC '{definition.npc_id}' must define visual_identity when "
                        "appearance_catalog is configured"
                    )
                catalog.identity_for_appearance(identity_id, definition.embodiment.appearance)
                if definition.embodiment.appearance_config is not None:
                    catalog.validate_appearance_slots(
                        identity_id, definition.embodiment.appearance_config
                    )
        return cls(
            scene=str(payload["scene"]),
            asset_manifest=str(payload["asset_manifest"]),
            npcs=npcs,
            clock=dict(_require_mapping(payload["clock"], "Population clock")),
            appearance_catalog=appearance_catalog,
            trajectory_profile=trajectory_profile,
            dialogue_policy=dialogue_policy,
            interaction_templates=interaction_templates,
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
