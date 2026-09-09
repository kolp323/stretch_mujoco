from types import SimpleNamespace

import numpy as np
import pytest

from tools import render_npc_acceptance_video as acceptance


def test_acceptance_shots_cover_required_views_for_walk_and_sit() -> None:
    assert [(shot.name, shot.clip) for shot in acceptance.ACCEPTANCE_SHOTS] == [
        ("front", "walk"),
        ("rear", "walk"),
        ("left", "walk"),
        ("right", "walk"),
        ("top", "walk"),
        ("seated_front", "sit"),
        ("seated_rear", "sit"),
        ("seated_left", "sit"),
        ("seated_right", "sit"),
        ("seated_top", "sit"),
    ]


def test_top_view_uses_a_high_close_camera() -> None:
    top = next(shot for shot in acceptance.ACCEPTANCE_SHOTS if shot.name == "top")

    assert top.elevation == -65.0
    assert top.distance == 2.5
    assert acceptance.ACCEPTANCE_CAMERA_FOV == 65.0


def test_discover_frames_keeps_all_slots_for_a_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    names = [
        "npc__employee_01__clip__walk__frame__000__slot__body",
        "npc__employee_01__clip__walk__frame__000__slot__accessory_cap",
        "npc__employee_01__clip__walk__frame__001__slot__body",
        "npc__employee_02__clip__walk__frame__000__slot__body",
    ]
    monkeypatch.setattr(
        acceptance.mujoco,
        "mj_id2name",
        lambda _model, _object_type, geom_id: names[geom_id],
    )

    assert acceptance._discover_frames(SimpleNamespace(ngeom=len(names)), "employee_01") == {
        "walk": {0: [0, 1], 1: [2]}
    }


def test_show_frame_hides_other_animation_frames() -> None:
    model = SimpleNamespace(geom_rgba=np.ones((3, 4), dtype=float))
    frames = {"idle": {0: [0]}, "walk": {0: [1], 1: [2]}}

    assert acceptance._show_frame(model, frames, "walk", 1) == [2]
    assert model.geom_rgba[:, 3].tolist() == [0.0, 0.0, 1.0]


def test_show_frame_rejects_missing_clip() -> None:
    model = SimpleNamespace(geom_rgba=np.ones((1, 4), dtype=float))

    with pytest.raises(ValueError, match="available: idle"):
        acceptance._show_frame(model, {"idle": {0: [0]}}, "sit", 0)


def test_acceptance_hides_non_target_legacy_and_roster_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = [
        "humanoid_preview_frame_idle_00_body",
        "npc__npc_alex_chen__clip__walk__frame__000__slot__body",
        "npc__npc_morgan_lee__clip__walk__frame__000__slot__body",
        "office_desk_surface",
    ]
    monkeypatch.setattr(
        acceptance.mujoco,
        "mj_id2name",
        lambda _model, _object_type, geom_id: names[geom_id],
    )
    model = SimpleNamespace(geom_rgba=np.ones((len(names), 4), dtype=float), ngeom=len(names))

    hidden = acceptance._hide_non_target_animated_geometries(model, "npc_alex_chen")

    assert hidden == ["employee_01", "npc_morgan_lee"]
    assert model.geom_rgba[:, 3].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_frame_material_checks_rejects_missing_accessory_slot_and_fractional_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = [
        "npc__employee_01__clip__walk__frame__000__slot__body",
        "npc__employee_01__clip__walk__frame__000__slot__glasses",
        "npc__employee_01__clip__walk__frame__001__slot__body",
    ]
    monkeypatch.setattr(
        acceptance.mujoco,
        "mj_id2name",
        lambda _model, _object_type, geom_id: names[geom_id],
    )
    model = SimpleNamespace(
        geom_matid=np.array([1, 2, 1]),
        geom_rgba=np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5]]),
    )

    checks = acceptance._frame_material_checks(model, {"walk": {0: [0, 1], 1: [2]}})

    assert any(
        "source geometry has fractional alpha" in failure
        for failure in checks["transparency_failures"]
    )
    assert any("frame slots differ" in failure for failure in checks["transparency_failures"])
