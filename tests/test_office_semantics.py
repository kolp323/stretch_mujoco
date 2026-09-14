from pathlib import Path

import mujoco
import numpy as np

from stretch_mujoco.semantics import (
    InteractionRole,
    ObjectType,
    RelationType,
    SemanticWorld,
)


MODELS_PATH = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"
OFFICE_SCENE_PATH = MODELS_PATH / "office_scene.xml"
OFFICE_SEMANTICS_PATH = MODELS_PATH / "office_semantics.json"


def load_world() -> tuple[SemanticWorld, mujoco.MjModel, mujoco.MjData]:
    world = SemanticWorld.from_json(OFFICE_SEMANTICS_PATH)
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    world.validate_model(model)
    return world, model, data


def test_office_semantics_registers_required_types_and_relations() -> None:
    world, _, _ = load_world()

    required_types = {
        ObjectType.EMPLOYEE,
        ObjectType.STRETCH_ROBOT,
        ObjectType.WORKSTATION,
        ObjectType.CHAIR,
        ObjectType.COMPUTER,
        ObjectType.DOCUMENT,
        ObjectType.STORAGE_CABINET,
        ObjectType.SNACK,
        ObjectType.MEETING_TABLE,
        ObjectType.COFFEE_MACHINE,
        ObjectType.DOOR,
    }
    assert required_types <= {obj.object_type for obj in world.objects.values()}
    assert world.related_objects("document_report", RelationType.ON)[0].object_id == (
        "storage_cabinet"
    )
    assert not world.related_objects("document_report", RelationType.REQUESTED_BY)
    assert world.object("document_report").get("confidential") is True
    assert not world.pending_requests("employee_01")
    assert world.can_access("employee_01", "document_report")
    assert not world.can_access("stretch_3", "document_report")
    assert world.can_access("stretch_3", "document_invoice")


def test_interaction_points_resolve_to_mujoco_poses() -> None:
    world, model, data = load_world()

    grasp_pose = world.interaction_pose("document_report_grasp", model, data)
    cabinet_pose = world.interaction_pose("cabinet_open", model, data)
    request_pose = world.interaction_pose("stretch_request", model, data)
    nearest_work_point = world.nearest_interaction_point(
        InteractionRole.DESK_WORK,
        np.array([2.3, 0.4, 0.0]),
        model,
        data,
    )

    assert grasp_pose.shape == (4, 4)
    assert cabinet_pose[2, 3] == 0.88
    assert request_pose.shape == (4, 4)
    assert nearest_work_point.point_id == "desk_right_work"


def test_relation_graph_supports_task_state_changes() -> None:
    world, _, _ = load_world()

    world.add_relation("employee_01", RelationType.HOLDS, "document_report")
    world.set_location("document_report", RelationType.ON, "meeting_table")

    assert world.find_relations(
        subject="employee_01", relation=RelationType.HOLDS, object_id="document_report"
    )
    assert not world.find_relations(subject="document_report", relation=RelationType.INSIDE)
    assert world.related_objects("document_report", RelationType.ON)[0].object_id == (
        "meeting_table"
    )
    assert world.object("document_report").get("location") == "meeting_table"

    world.replace_relation("document_report", RelationType.BELONGS_TO, "stretch_3")
    assert world.object("document_report").get("owner") == "stretch_3"


def test_semantic_snapshot_contains_dynamic_objects_and_sites() -> None:
    world, model, data = load_world()

    snapshot = world.pose_snapshot(model, data)

    assert snapshot["time"] == 0.0
    assert "stretch_3" in snapshot["objects"]
    assert "employee_01_handover" in snapshot["interaction_points"]
    assert len(snapshot["objects"]["document_report"]["quaternion"]) == 4
