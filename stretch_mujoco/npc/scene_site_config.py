"""Validated, named site plans shared by scene preprocessors and demos.

The plan is deliberately independent of MuJoCo XML: it owns authored site
coordinates and roster-to-spawn bindings, while XML/population/semantic files
are generated projections.  A scene may expose any number of named sites and
select any subset of NPC definitions supplied by its population template.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SceneSite:
    """One named, worldbody-attached MuJoCo site."""

    name: str
    position: tuple[float, float, float]
    yaw: float
    tags: frozenset[str]


@dataclass(frozen=True)
class SceneSpawn:
    """One population-template NPC selected for this scene."""

    npc_id: str
    site: str
    location: str
    yaw: float


@dataclass(frozen=True)
class SceneDemo:
    """Optional named inputs for the home NPC demonstration renderer."""

    activity_site: str
    walker_npc_id: str
    conversation_partner_npc_id: str


@dataclass(frozen=True)
class SceneSitePlan:
    """Authoritative named sites and the roster that initially occupies them."""

    scene_id: str
    sites: dict[str, SceneSite]
    roster: tuple[SceneSpawn, ...]
    demo: SceneDemo | None


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


def _site(name: str, payload: Mapping[str, Any], context: str) -> SceneSite:
    unknown = set(payload) - {"position", "yaw", "tags"}
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(sorted(unknown))}")
    raw_position = payload.get("position")
    if not isinstance(raw_position, list) or len(raw_position) not in {2, 3}:
        raise ValueError(f"{context}.position must contain two or three coordinates")
    coordinates = tuple(_finite_number(value, f"{context}.position") for value in raw_position)
    position = (*coordinates, 0.025) if len(coordinates) == 2 else coordinates
    raw_tags = payload.get("tags", [])
    if not isinstance(raw_tags, list) or not all(isinstance(tag, str) and tag for tag in raw_tags):
        raise ValueError(f"{context}.tags must be a list of non-empty strings")
    return SceneSite(
        name=name,
        position=(float(position[0]), float(position[1]), float(position[2])),
        yaw=_finite_number(payload.get("yaw", 0.0), f"{context}.yaw"),
        tags=frozenset(raw_tags),
    )


def load_scene_site_plans(path: Path) -> dict[str, SceneSitePlan]:
    """Read schema-v2 scene sites; fail before generated artifacts can drift."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError(f"{path}: expected scene site config schema_version 2")
    raw_scenes = _mapping(payload.get("scenes"), f"{path}.scenes")
    plans: dict[str, SceneSitePlan] = {}
    for scene_id, raw_plan in raw_scenes.items():
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"{path}: scene keys must be non-empty strings")
        plan = _mapping(raw_plan, f"{path}.scenes.{scene_id}")
        unknown = set(plan) - {"sites", "roster", "demo"}
        if unknown:
            raise ValueError(f"{scene_id}: unknown fields: {', '.join(sorted(unknown))}")
        raw_sites = _mapping(plan.get("sites"), f"{scene_id}.sites")
        if not raw_sites:
            raise ValueError(f"{scene_id}.sites must not be empty")
        sites = {
            name: _site(name, _mapping(value, f"{scene_id}.sites.{name}"), f"{scene_id}.sites.{name}")
            for name, value in raw_sites.items()
            if isinstance(name, str) and name
        }
        if len(sites) != len(raw_sites):
            raise ValueError(f"{scene_id}.sites must use non-empty string names")
        raw_roster = plan.get("roster")
        if not isinstance(raw_roster, list) or not raw_roster:
            raise ValueError(f"{scene_id}.roster must be a non-empty list")
        roster: list[SceneSpawn] = []
        for index, raw_spawn in enumerate(raw_roster):
            spawn = _mapping(raw_spawn, f"{scene_id}.roster[{index}]")
            unknown_spawn = set(spawn) - {"npc_id", "site", "location", "yaw"}
            if unknown_spawn:
                raise ValueError(
                    f"{scene_id}.roster[{index}] has unknown fields: "
                    f"{', '.join(sorted(unknown_spawn))}"
                )
            npc_id, site = spawn.get("npc_id"), spawn.get("site")
            if not isinstance(npc_id, str) or not npc_id or not isinstance(site, str) or site not in sites:
                raise ValueError(f"{scene_id}.roster[{index}] requires a known npc_id/site")
            location = spawn.get("location", "home")
            if not isinstance(location, str) or not location:
                raise ValueError(f"{scene_id}.roster[{index}].location must be non-empty")
            roster.append(
                SceneSpawn(
                    npc_id=npc_id,
                    site=site,
                    location=location,
                    yaw=_finite_number(spawn.get("yaw", sites[site].yaw), f"{scene_id}.roster[{index}].yaw"),
                )
            )
        if len({entry.npc_id for entry in roster}) != len(roster):
            raise ValueError(f"{scene_id}.roster contains duplicate NPC ids")
        if len({entry.site for entry in roster}) != len(roster):
            raise ValueError(f"{scene_id}.roster assigns more than one NPC to one spawn site")
        demo = None
        if "demo" in plan:
            raw_demo = _mapping(plan["demo"], f"{scene_id}.demo")
            if set(raw_demo) != {"activity_site", "walker_npc_id", "conversation_partner_npc_id"}:
                raise ValueError(f"{scene_id}.demo requires activity_site, walker_npc_id, and conversation_partner_npc_id")
            activity_site = raw_demo["activity_site"]
            walker = raw_demo["walker_npc_id"]
            partner = raw_demo["conversation_partner_npc_id"]
            roster_ids = {entry.npc_id for entry in roster}
            if (
                not isinstance(activity_site, str)
                or activity_site not in sites
                or not isinstance(walker, str)
                or not isinstance(partner, str)
                or walker == partner
                or walker not in roster_ids
                or partner not in roster_ids
            ):
                raise ValueError(f"{scene_id}.demo references an unknown site or roster NPC")
            demo = SceneDemo(activity_site, walker, partner)
        plans[scene_id] = SceneSitePlan(scene_id, sites, tuple(roster), demo)
    return plans
