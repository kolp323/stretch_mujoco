"""Bounded read-only observations and affordances for LLM plan generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .compiler import ControlCapabilities


@dataclass(frozen=True)
class ControlObservation:
    minute_of_day: float
    agents: Mapping[str, Mapping[str, str | None]]
    active_robot_tasks: tuple[str, ...]


@dataclass(frozen=True)
class AffordanceCatalog:
    actions: frozenset[str]
    locations: frozenset[str]


def build_control_observation(runtime: Any) -> ControlObservation:
    agents = {
        agent_id: {
            "location": agent.state.location,
            "availability": agent.state.availability,
            "action": agent.state.current_action,
        }
        for agent_id, agent in runtime.agents.items()
    }
    tasks = tuple(sorted(task.task_id for task in runtime.pending_robot_tasks()))
    return ControlObservation(runtime.minute_of_day, agents, tasks)


def build_affordances(
    runtime: Any,
    capabilities: ControlCapabilities,
    allowed_actions: frozenset[str],
    allowed_locations: frozenset[str],
) -> AffordanceCatalog:
    """Expose only statically allowed, currently unreserved locations/actions."""
    available_actions = frozenset(allowed_actions)
    locations = frozenset(
        location
        for location in allowed_locations
        if location in capabilities.reachable_locations
        and runtime.reservations.owner(location) is None
    )
    return AffordanceCatalog(available_actions, locations)
