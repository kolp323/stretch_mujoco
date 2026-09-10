from stretch_mujoco.npc.naming import parse_frame_geom_name


def test_legacy_npc_frame_names_support_underscore_action_ids() -> None:
    assert parse_frame_geom_name("humanoid_preview_frame_pick_up_00_body") == (
        "employee_01",
        "pick_up",
        0,
        "body",
    )
    assert parse_frame_geom_name("em02_frame_pick_up_11_body") == (
        "employee_02",
        "pick_up",
        11,
        "body",
    )
