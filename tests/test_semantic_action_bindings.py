from pathlib import Path

from stretch_mujoco.agents.simulation_bridge import create_mujoco_action_driver
from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.semantics import InteractionPoint, InteractionRole, SemanticWorld


MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


class _Simulator:
    pass


def test_semantic_binding_attributes_override_office_name_constants():
    world = SemanticWorld.from_json(MODELS / "office_semantics.json")
    world.interaction_points["portable_location"] = InteractionPoint(
        "portable_location",
        InteractionRole.HUMAN_STAND,
        "meeting_table",
        "new_scene_meeting_approach",
        {"binding": "location", "target": "meeting_table", "yaw": 0.75},
    )
    world.interaction_points["portable_robot_request"] = InteractionPoint(
        "portable_robot_request",
        InteractionRole.ROBOT_REQUEST,
        "stretch_3",
        "new_scene_robot_request",
        {"binding": "robot_request", "target": "stretch_3"},
    )
    for participant, site, yaw in (
        ("employee_01", "new_scene_conversation_a", 0.0),
        ("employee_02", "new_scene_conversation_b", 3.14),
    ):
        world.interaction_points[site] = InteractionPoint(
            site,
            InteractionRole.CONVERSATION,
            participant,
            site,
            {
                "binding": "conversation_role",
                "participants": ["employee_01", "employee_02"],
                "participant": participant,
                "yaw": yaw,
            },
        )
    driver = create_mujoco_action_driver(_Simulator(), world=world)
    assert driver.location_sites["meeting_table"] == "new_scene_meeting_approach"
    assert driver.interaction_yaws["meeting_table"] == 0.75
    assert driver.robot_request_sites["stretch_3"] == "new_scene_robot_request"
    assert driver.conversation_role_sites[("employee_01", "employee_02")] == (
        "new_scene_conversation_a",
        "new_scene_conversation_b",
    )


def test_population_templates_expand_ordered_roster_pairs_and_semantics_override_them():
    population = NpcPopulation.from_json(MODELS / "office_population.json")
    roster = ("npc_a", "npc_b", "npc_c")
    world = SemanticWorld.from_json(MODELS / "office_semantics.json")
    for participant, site in (("npc_a", "explicit_a"), ("npc_b", "explicit_b")):
        world.interaction_points[site] = InteractionPoint(
            site,
            InteractionRole.CONVERSATION,
            "meeting_table",
            site,
            {
                "binding": "conversation_role",
                "participants": ["npc_a", "npc_b"],
                "participant": participant,
                "yaw": 0.25 if participant == "npc_a" else -2.75,
            },
        )
    for participant, site, yaw in (
        ("npc_a", "explicit_giver", 0.5),
        ("npc_b", "explicit_receiver", None),
    ):
        attributes = {
            "binding": "handover_role",
            "participants": ["npc_a", "npc_b"],
            "participant": participant,
        }
        if yaw is not None:
            attributes["yaw"] = yaw
        world.interaction_points[site] = InteractionPoint(
            site,
            InteractionRole.HANDOVER,
            "meeting_table",
            site,
            attributes,
        )
    driver = create_mujoco_action_driver(
        _Simulator(),
        npc_ids=roster,
        interaction_templates=population.interaction_templates,
        world=world,
    )

    roster_pairs = {(first, second) for first in roster for second in roster if first != second}
    assert {
        pair for pair in driver.conversation_role_sites if set(pair) <= set(roster)
    } == roster_pairs
    assert {pair for pair in driver.handover_role_sites if set(pair) <= set(roster)} == roster_pairs
    assert driver.conversation_role_sites[("npc_b", "npc_c")] == (
        "meeting_conversation_alex_site",
        "meeting_conversation_morgan_site",
    )
    assert driver.conversation_role_sites[("npc_c", "npc_b")] == (
        "meeting_conversation_alex_site",
        "meeting_conversation_morgan_site",
    )
    assert driver.handover_role_sites[("npc_b", "npc_c")] == (
        "npc_handover_giver_stand_site",
        "npc_handover_receiver_stand_site",
    )
    assert driver.handover_role_yaws[("npc_c", "npc_b")] == (1.5708, -1.5708)
    assert driver.conversation_site_yaws["meeting_conversation_alex_site"] == 1.5708
    assert driver.conversation_role_sites[("npc_a", "npc_b")] == ("explicit_a", "explicit_b")
    assert driver.conversation_site_yaws["explicit_a"] == 0.25
    assert driver.handover_role_sites[("npc_a", "npc_b")] == (
        "explicit_giver",
        "explicit_receiver",
    )
    assert driver.handover_role_yaws[("npc_a", "npc_b")] == (0.5, -1.5708)
