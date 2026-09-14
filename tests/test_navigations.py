from __future__ import annotations

import numpy as np
import pytest

from stretch_mujoco.navigations import (
    AStarPlanner,
    FMMPlanner,
    NavigationController,
    NavigationPathError,
    ObstacleFootprint,
    OccupancyGrid,
)


def make_grid(
    occupancy: np.ndarray,
    obstacles: tuple[ObstacleFootprint, ...] = (),
) -> OccupancyGrid:
    rows, cols = occupancy.shape
    return OccupancyGrid(
        (0.0, float(cols - 1), 0.0, float(rows - 1)),
        occupancy,
        resolution=1.0,
        agent_radius=0.0,
        obstacles=obstacles,
    )


@pytest.mark.parametrize("planner", [AStarPlanner(), FMMPlanner()])
def test_smoothing_does_not_cross_occupied_cells(planner) -> None:
    occupancy = np.zeros((3, 5), dtype=bool)
    occupancy[1, 2] = True
    grid = make_grid(occupancy)

    path = planner.plan(grid, np.array([0.0, 1.0]), np.array([4.0, 1.0]))

    assert len(path) > 2
    for start, end in zip(path, path[1:]):
        assert grid.line_of_sight(grid.world_to_cell(start), grid.world_to_cell(end))


def test_fmm_does_not_cut_an_occupied_corner() -> None:
    occupancy = np.zeros((2, 2), dtype=bool)
    occupancy[0, 0] = True
    obstacle = ObstacleFootprint("corner", (-0.1, -0.1), (0.6, 0.6))
    grid = make_grid(occupancy, (obstacle,))

    path = FMMPlanner().plan(grid, np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    assert len(path) == 3
    for start, end in zip(path, path[1:]):
        assert grid.world_line_of_sight(start, end)


@pytest.mark.parametrize("planner", [AStarPlanner(), FMMPlanner()])
def test_endpoint_connector_cannot_cross_an_obstacle(planner) -> None:
    occupancy = np.array([[False, True, False]], dtype=bool)
    obstacle = ObstacleFootprint("wall", (0.6, -0.4), (1.4, 0.4))
    grid = make_grid(occupancy, (obstacle,))

    with pytest.raises(NavigationPathError):
        planner.plan(grid, np.array([0.0, 0.0]), np.array([1.45, 0.0]))


def test_polygon_footprint_does_not_use_its_aabb_for_collision_queries() -> None:
    # The point is inside this diamond's AABB but outside the actual convex
    # footprint. This guards the mesh-projection navigation path against a
    # regression back to black AABB rectangles.
    diamond = ObstacleFootprint(
        "rotated_mesh",
        (-1.0, -1.0),
        (1.0, 1.0),
        ((0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
    )
    grid = make_grid(np.zeros((3, 3), dtype=bool), (diamond,))

    assert grid.point_inside_obstacle(np.array([0.0, 0.0]))
    assert not grid.point_inside_obstacle(np.array([0.9, 0.9]))


@pytest.mark.parametrize("planner", [AStarPlanner(), FMMPlanner()])
def test_occupied_endpoint_is_returned_as_resolved_free_cell(planner) -> None:
    occupancy = np.zeros((3, 5), dtype=bool)
    occupancy[1, 2] = True
    grid = make_grid(occupancy)

    requested_goal = np.array([2.0, 1.0])
    path = planner.plan(grid, np.array([0.0, 1.0]), requested_goal)

    assert not np.array_equal(path[-1], requested_goal)
    assert grid.is_world_free(path[-1])


def test_refresh_grid_preserves_original_configuration(monkeypatch) -> None:
    controller = NavigationController.__new__(NavigationController)
    controller.model = object()
    controller.data = object()
    controller._grid_kwargs = {
        "resolution": 0.2,
        "agent_radius": 0.3,
        "bounds": (-1.0, 2.0, -3.0, 4.0),
        "floor_geom_name": "custom_floor",
        "minimum_obstacle_height": 0.1,
        "maximum_obstacle_height": 1.5,
    }
    expected_grid = object()
    calls = []

    def fake_from_model(model, data, **kwargs):
        calls.append((model, data, kwargs))
        return expected_grid

    monkeypatch.setattr(OccupancyGrid, "from_model", fake_from_model)

    controller.refresh_grid()

    assert controller.grid is expected_grid
    assert calls == [(controller.model, controller.data, controller._grid_kwargs)]
