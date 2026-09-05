"""Assembly helpers connecting the agent runtime to the NPC protocol."""

from __future__ import annotations

from .action_recipes import OFFICE_LOCATION_SITES, OFFICE_PLACEMENT_SITES, OFFICE_SEAT_YAWS
from .drivers import MujocoNpcActionDriver, NpcSimulatorClient


def create_mujoco_action_driver(simulator: NpcSimulatorClient) -> MujocoNpcActionDriver:
    return MujocoNpcActionDriver(
        simulator,
        OFFICE_LOCATION_SITES,
        OFFICE_PLACEMENT_SITES,
        seat_yaws=OFFICE_SEAT_YAWS,
    )
