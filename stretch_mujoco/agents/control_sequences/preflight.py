"""Read-only capability discovery used by the validation and run entry points."""

from __future__ import annotations

import json
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any

from .compiler import ControlCapabilities
from .models import BehaviorSequence, ControlSequenceError


def discover_capabilities(sequence: BehaviorSequence) -> ControlCapabilities:
    """Build a closed compiler allow-list from the declared scene inputs.

    This is intentionally a static preflight: it verifies semantic/site names before a
    simulator is opened.  Dynamic navmesh reachability remains the runner's responsibility.
    """
    scene_path = sequence.scene.get("xml")
    if scene_path is None:
        raise ControlSequenceError("scene.xml is required for control-sequence preflight")
    try:
        root = element_tree.parse(scene_path).getroot()
    except (OSError, element_tree.ParseError) as error:
        raise ControlSequenceError(f"Could not parse scene XML '{scene_path}': {error}") from error
    site_ids = frozenset(
        node.attrib["name"] for node in root.iter("site") if node.attrib.get("name")
    )
    semantics_path = scene_path.with_name("office_semantics.json")
    object_ids: set[str] = set()
    robot_ids: set[str] = set()
    if semantics_path.is_file():
        try:
            semantics: dict[str, Any] = json.loads(semantics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ControlSequenceError(
                f"Could not parse semantics '{semantics_path}': {error}"
            ) from error
        objects = semantics.get("objects", {})
        if not isinstance(objects, dict):
            raise ControlSequenceError("office semantics objects must be a mapping")
        object_ids.update(key for key in objects if isinstance(key, str))
        robot_ids.update(
            key
            for key, value in objects.items()
            if isinstance(value, dict) and value.get("type") == "StretchRobot"
        )
    population_path = sequence.scene.get("population")
    npc_ids: set[str] = set()
    if population_path is not None:
        try:
            population = json.loads(population_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ControlSequenceError(
                f"Could not parse population '{population_path}': {error}"
            ) from error
        npcs = population.get("npcs", {})
        if not isinstance(npcs, dict):
            raise ControlSequenceError("population.npcs must be a mapping")
        npc_ids.update(key for key in npcs if isinstance(key, str))
    # The population is optional for small synthetic tests.  In that case the declared
    # participants still form a closed local allow-list; a real run performs population binding.
    if not npc_ids:
        npc_ids.update(value for value in sequence.participants.values() if value not in robot_ids)
    return ControlCapabilities(
        npc_ids=frozenset(npc_ids),
        robot_ids=frozenset(robot_ids),
        object_ids=frozenset(object_ids),
        site_ids=site_ids,
        reachable_locations=frozenset(object_ids | site_ids),
    )
