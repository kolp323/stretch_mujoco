from pathlib import Path

import mujoco
import numpy as np

from stretch_mujoco.humanoid import OfficeNavigationMesh


OFFICE_SCENE_PATH = (
    Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "office_scene.xml"
)


def load_navigation():
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, OfficeNavigationMesh.from_model(model, data)


def assert_path_is_collision_free(nav, path) -> None:
    assert len(path) >= 2
    for start, end in zip(path, path[1:]):
        distance = float(np.linalg.norm(end - start))
        samples = max(2, int(distance / 0.03))
        for index in range(1, samples + 1):
            point = start + (end - start) * (index / samples)
            assert not nav.point_inside_obstacle(point), point


def test_navmesh_detects_furniture_and_plans_all_human_targets() -> None:
    _, data, nav = load_navigation()
    obstacle_names = {obstacle.geom_name for obstacle in nav.obstacles}
    assert "meeting_table_top" in obstacle_names
    assert "snack_counter_top" in obstacle_names
    assert "desk_right_top" in obstacle_names

    start = data.body("humanoid_preview").xpos[:2]
    for site_name in (
        "meeting_human_stand_site",
        "snack_human_stand_site",
        "cabinet_human_stand_site",
        "coffee_human_stand_site",
    ):
        goal = data.site(site_name).xpos[:2]
        path = nav.plan(start, goal)
        np.testing.assert_allclose(path[-1], goal)
        assert_path_is_collision_free(nav, path)


def test_navmesh_rebuild_changes_path_after_furniture_moves() -> None:
    model, data, nav = load_navigation()
    start = data.body("humanoid_preview").xpos[:2]
    original_goal = data.site("snack_human_stand_site").xpos[:2].copy()
    original_path = nav.plan(start, original_goal)

    meeting_body_id = model.body("meeting_table").id
    model.body_pos[meeting_body_id, 0] -= 1.1
    moved_data = mujoco.MjData(model)
    mujoco.mj_forward(model, moved_data)
    rebuilt = OfficeNavigationMesh.from_model(model, moved_data)
    moved_path = rebuilt.plan(start, original_goal)

    assert_path_is_collision_free(rebuilt, moved_path)
    assert not np.array_equal(np.vstack(original_path), np.vstack(moved_path))


def test_robot_route_reuses_mesh_but_excludes_its_own_base_tree() -> None:
    model, data, _ = load_navigation()
    base_xy = data.body("base_link").xpos[:2]
    robot_nav = OfficeNavigationMesh.from_model(
        model,
        data,
        agent_radius=0.32,
        exclude_body_roots=("base_link",),
    )

    assert robot_nav.is_world_free(base_xy)
    path = robot_nav.plan(base_xy, data.site("snack_human_stand_site").xpos[:2])
    assert_path_is_collision_free(robot_nav, path)
