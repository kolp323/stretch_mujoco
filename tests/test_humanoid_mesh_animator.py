from pathlib import Path

import mujoco
import numpy as np
import pytest

from stretch_mujoco.humanoid.mesh_animator import (
    SIT_ROOT_TO_SEAT_HEIGHT,
    HumanoidMeshAnimator,
)


OFFICE_SCENE_PATH = (
    Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "office_scene.xml"
)


def test_office_humanoid_exposes_all_baked_clips() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    animator = HumanoidMeshAnimator(model)

    assert animator.available_clips == (
        "eat", "give", "idle", "pick_up", "receive", "sit", "talk", "walk", "work"
    )
    assert animator.is_available
    assert tuple(animator.geom_ids) == ("body",)


def test_animator_swaps_each_material_mesh() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    animator = HumanoidMeshAnimator(model)

    animator.update(sim_time=0.2, clip="walk")

    for material in ("body",):
        target_geom = model.geom(f"humanoid_preview_frame_walk_01_{material}")
        initial_geom = model.geom(f"humanoid_preview_frame_idle_00_{material}")
        assert model.geom_rgba[target_geom.id, 3] == 1.0
        assert model.geom_rgba[initial_geom.id, 3] == 0.0


def test_playback_speed_scales_root_motion() -> None:
    def moved_distance(speed_scale: float) -> float:
        model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        animator = HumanoidMeshAnimator(model)
        mocap_id = model.body("humanoid_preview").mocapid[0]
        animator.update(
            0.0,
            "walk",
            data=data,
            navigation_target_site="meeting_human_stand_site",
            speed_scale=speed_scale,
        )
        start = data.mocap_pos[mocap_id, :2].copy()
        animator.update(
            0.1,
            "walk",
            data=data,
            navigation_target_site="meeting_human_stand_site",
            speed_scale=speed_scale,
        )
        return float(np.linalg.norm(data.mocap_pos[mocap_id, :2] - start))

    assert moved_distance(4.5) == pytest.approx(moved_distance(1.0) * 4.5)


def test_playback_speed_does_not_accelerate_work_gestures() -> None:
    def active_frame(clip: str, speed_scale: float) -> tuple[str, int] | None:
        model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
        animator = HumanoidMeshAnimator(model)
        animator.update(0.15, clip, speed_scale=speed_scale)
        return animator._last_key

    assert active_frame("work", 4.5) == active_frame("work", 1.0)
    assert active_frame("eat", 4.5) == active_frame("eat", 1.0)
    assert active_frame("walk", 4.5) != active_frame("walk", 1.0)


def test_sit_walks_to_chair_then_idle_restores_standing_pose() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    animator = HumanoidMeshAnimator(model)
    mocap_id = model.body("humanoid_preview").mocapid[0]
    standing_position = data.mocap_pos[mocap_id].copy()
    standing_orientation = data.mocap_quat[mocap_id].copy()
    sit_site = data.site("chair_left_sit")

    animator.update(0.0, "sit", data=data, sit_target_site="chair_left_sit")

    assert animator.sit_stage == "walk"
    np.testing.assert_allclose(data.mocap_pos[mocap_id], standing_position)

    for step in range(1, 401):
        animator.update(
            step * 0.05,
            "sit",
            data=data,
            sit_target_site="chair_left_sit",
        )

    expected_position = sit_site.xpos.copy()
    expected_position[2] -= SIT_ROOT_TO_SEAT_HEIGHT
    expected_orientation = np.empty(4)
    mujoco.mju_mat2Quat(expected_orientation, sit_site.xmat)
    assert animator.sit_stage == "seated"
    np.testing.assert_allclose(data.mocap_pos[mocap_id], expected_position)
    np.testing.assert_allclose(data.mocap_quat[mocap_id], expected_orientation)

    animator.update(0.0, "idle", data=data, sit_target_site="chair_left_sit")

    np.testing.assert_allclose(data.mocap_pos[mocap_id], standing_position)
    np.testing.assert_allclose(data.mocap_quat[mocap_id], standing_orientation)


def test_walk_navigation_moves_root_to_office_interaction_site() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    animator = HumanoidMeshAnimator(model)
    mocap_id = model.body("humanoid_preview").mocapid[0]
    standing_height = data.mocap_pos[mocap_id, 2]

    for step in range(241):
        animator.update(
            step * 0.05,
            "walk",
            data=data,
            navigation_target_site="meeting_human_stand_site",
        )

    expected = data.site("meeting_human_stand_site").xpos.copy()
    expected[2] = standing_height
    np.testing.assert_allclose(data.mocap_pos[mocap_id], expected, atol=1e-6)


def test_seated_humanoid_can_walk_to_a_new_target_without_resetting_home() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    animator = HumanoidMeshAnimator(model)
    mocap_id = model.body("humanoid_preview").mocapid[0]

    for step in range(241):
        animator.update(step * 0.05, "sit", data=data, sit_target_site="chair_right_sit")
    seated_xy = data.mocap_pos[mocap_id, :2].copy()

    animator.update(
        12.1,
        "walk",
        data=data,
        navigation_target_site="meeting_human_stand_site",
    )

    assert np.linalg.norm(data.mocap_pos[mocap_id, :2] - seated_xy) < 0.1

    for step in range(243, 343):
        animator.update(
            step * 0.05,
            "walk",
            data=data,
            navigation_target_site="meeting_human_stand_site",
        )

    expected = data.site("meeting_human_stand_site").xpos.copy()
    expected[2] = 0.0
    np.testing.assert_allclose(data.mocap_pos[mocap_id], expected, atol=1e-6)


def test_routed_navigation_avoids_meeting_table_and_snack_counter() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    animator = HumanoidMeshAnimator(model)
    mocap_id = model.body("humanoid_preview").mocapid[0]

    def inside_inflated_box(point, center, half_size, radius=0.25):
        return all(abs(point[axis] - center[axis]) < half_size[axis] + radius for axis in range(2))

    meeting_center = data.body("meeting_table").xpos[:2]
    snack_center = data.body("snack_counter").xpos[:2]
    meeting_route = "meeting_human_stand_site"
    snack_route = "snack_human_stand_site"

    for step in range(401):
        animator.update(
            step * 0.05,
            "walk",
            data=data,
            navigation_target_site=meeting_route,
        )
        assert not inside_inflated_box(data.mocap_pos[mocap_id, :2], meeting_center, (0.80, 0.38))

    for step in range(401, 801):
        animator.update(
            step * 0.05,
            "walk",
            data=data,
            navigation_target_site=snack_route,
        )
        root = data.mocap_pos[mocap_id, :2]
        assert not inside_inflated_box(root, meeting_center, (0.80, 0.38))
        assert not inside_inflated_box(root, snack_center, (0.62, 0.38))

    expected = data.site("snack_human_stand_site").xpos.copy()
    expected[2] = 0.0
    np.testing.assert_allclose(data.mocap_pos[mocap_id], expected, atol=1e-6)


def test_sit_route_retraces_snack_exit_before_crossing_office() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    animator = HumanoidMeshAnimator(model)
    mocap_id = model.body("humanoid_preview").mocapid[0]
    route = "snack_human_stand_site"

    for step in range(501):
        animator.update(
            step * 0.05,
            "walk",
            data=data,
            navigation_target_site=route,
        )

    snack_center = data.body("snack_counter").xpos[:2]
    for step in range(501, 1001):
        animator.update(step * 0.05, "sit", data=data, sit_target_site="chair_right_sit")
        root = data.mocap_pos[mocap_id, :2]
        inside_snack = all(
            abs(root[axis] - snack_center[axis]) < (0.87, 0.63)[axis] for axis in range(2)
        )
        assert not inside_snack

    assert animator.sit_stage == "seated"
