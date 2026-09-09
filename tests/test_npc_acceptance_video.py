from types import SimpleNamespace

import numpy as np
import pytest

from tools import render_npc_acceptance_video as acceptance


def test_acceptance_shots_cover_front_side_and_seated_review() -> None:
    assert [(shot.name, shot.clip, shot.seated) for shot in acceptance.ACCEPTANCE_SHOTS] == [
        ("front", "walk", False),
        ("side", "walk", False),
        ("seated", "work", True),
    ]


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
