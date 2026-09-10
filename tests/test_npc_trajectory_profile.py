import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from stretch_mujoco.npc.trajectory_profile import (
    NpcTrajectoryProfile,
    TrajectoryAnchor,
    TrajectoryProfileError,
    TrajectoryRoute,
)
from stretch_mujoco.npc import CommandStatus, NpcCommand, NpcCommandKind
from stretch_mujoco.npc.system import NpcSystem


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OFFICE_SCENE = REPOSITORY_ROOT / "stretch_mujoco/models/office_scene.xml"
OFFICE_PROFILE = REPOSITORY_ROOT / "stretch_mujoco/npc/trajectory_profiles/office_v1.json"


def test_office_profile_preflights_all_declared_routes() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    profile = NpcTrajectoryProfile.from_json(OFFICE_PROFILE)

    compiled = profile.preflight(model, data)
    clearance = profile.audit_npc_clearance(model, data, "employee_01", sample_period=0.1)

    assert tuple(route.route_id for route in compiled) == (
        "workstation_left_to_meeting",
        "workstation_right_to_meeting",
        "meeting_to_snacks",
        "snacks_to_storage",
    )
    for route, definition in zip(compiled, profile.routes, strict=True):
        source_site = profile.anchors[definition.source].site
        destination_site = profile.anchors[definition.destination].site
        source_id = model.site(source_site).id
        destination_id = model.site(destination_site).id
        assert len(route.waypoints) >= 2
        np.testing.assert_allclose(route.waypoints[0], data.site_xpos[source_id, :2])
        np.testing.assert_allclose(route.waypoints[-1], data.site_xpos[destination_id, :2])
    assert all(audit.sampled_poses > 1 for audit in clearance)


def test_profile_rejects_unknown_route_anchor(tmp_path: Path) -> None:
    profile_path = tmp_path / "invalid_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "invalid",
                "scene": "scene.xml",
                "scene_sha256": "0" * 64,
                "anchors": {"desk": {"site": "desk_site", "role": "workstation"}},
                "routes": [
                    {
                        "route_id": "unknown_destination",
                        "from": "desk",
                        "to": "missing",
                        "actions": ["move_to"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrajectoryProfileError, match="trajectory_route_invalid"):
        NpcTrajectoryProfile.from_json(profile_path)


def test_profile_rejects_application_to_different_scene() -> None:
    profile = NpcTrajectoryProfile.from_json(OFFICE_PROFILE)

    with pytest.raises(TrajectoryProfileError, match="trajectory_profile_scene_mismatch"):
        profile.validate_scene("other_scene.xml")


def test_profile_rejects_scene_bytes_that_differ_from_preflighted_asset(
    tmp_path: Path,
) -> None:
    changed_scene = tmp_path / "office_scene.xml"
    changed_scene.write_text("<mujoco/>", encoding="utf-8")
    profile = NpcTrajectoryProfile.from_json(OFFICE_PROFILE)

    with pytest.raises(TrajectoryProfileError, match="trajectory_profile_scene_digest_mismatch"):
        profile.validate_scene(changed_scene)


def test_profile_preflight_rejects_missing_anchor_site() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="office_floor" type="box" pos="0 0 -.05" size="3 3 .05"/>
            <site name="desk_site" pos="0 0 0"/>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    profile = NpcTrajectoryProfile(
        profile_id="missing_site",
        scene="minimal.xml",
        scene_sha256="0" * 64,
        anchors={
            "desk": TrajectoryAnchor("desk", "missing_site", "workstation"),
            "meeting": TrajectoryAnchor("meeting", "desk_site", "meeting"),
        },
        routes=(TrajectoryRoute("desk_to_meeting", "desk", "meeting", ("move_to",)),),
    )

    with pytest.raises(TrajectoryProfileError, match="trajectory_anchor_site_missing"):
        profile.preflight(model, data)


def test_scene_profile_is_preflighted_and_enforced_by_the_runtime_controller() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE))
    system = NpcSystem.from_model(model, scene_path=OFFICE_SCENE)

    controller = system.controllers["employee_01"]
    assert set(controller.trajectory_routes) == {
        "workstation_left_to_meeting",
        "workstation_right_to_meeting",
        "meeting_to_snacks",
        "snacks_to_storage",
    }
    accepted = system.submit(
        NpcCommand(
            "profile_route_ok",
            0,
            "employee_01",
            NpcCommandKind.MOVE_TO,
            {
                "site": "meeting_human_stand_site",
                "trajectory_route": "workstation_left_to_meeting",
            },
            0.0,
        )
    )
    assert accepted.status == CommandStatus.ACCEPTED
    system.controllers["employee_01"].active_command = None

    rejected = system.submit(
        NpcCommand(
            "profile_route_wrong_target",
            1,
            "employee_01",
            NpcCommandKind.MOVE_TO,
            {
                "site": "snack_human_stand_site",
                "trajectory_route": "workstation_left_to_meeting",
            },
            0.0,
        )
    )
    assert rejected.status == CommandStatus.FAILED
    assert rejected.reason == "trajectory_route_contract_mismatch:workstation_left_to_meeting"
