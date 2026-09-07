from tools.render_npc_personas import ReviewShot, frame_visibility, review_shots

from stretch_mujoco.npc.naming import accessory_frame_geom_name, frame_geom_name


def test_review_plan_covers_every_npc_front_side_and_sit() -> None:
    assert review_shots(("alex", "morgan")) == (
        ReviewShot("alex", "front", "idle"),
        ReviewShot("alex", "side", "idle"),
        ReviewShot("alex", "sit", "sit"),
        ReviewShot("morgan", "front", "idle"),
        ReviewShot("morgan", "side", "idle"),
        ReviewShot("morgan", "sit", "sit"),
    )


def test_alpha_frame_selection_shows_body_and_accessory_once() -> None:
    names = (
        frame_geom_name("alex", "idle", 0, "body"),
        accessory_frame_geom_name("alex", "idle", 0, "accessory_demo_v1"),
        frame_geom_name("alex", "idle", 1, "body"),
        accessory_frame_geom_name("alex", "idle", 1, "accessory_demo_v1"),
    )

    visible = frame_visibility(names, "alex", "idle", 1)

    assert sum(visible.values()) == 2
    assert visible[frame_geom_name("alex", "idle", 1, "body")]
    assert visible[accessory_frame_geom_name("alex", "idle", 1, "accessory_demo_v1")]
