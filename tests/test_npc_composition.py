"""Regression coverage for declarative population-to-office composition."""

import hashlib
import json
from pathlib import Path

import mujoco
import pytest

from stretch_mujoco.agents import ActionCommand, ActionType
from stretch_mujoco.npc import CommandStatus, NpcCommand, NpcCommandKind
from stretch_mujoco.npc.composition import compose_npc_scene, load_composed_npc_runtime

MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


def test_preview_population_composes_portably_and_binds_matching_runtime(tmp_path: Path) -> None:
    base_scene = MODELS / "office_scene.xml"
    base_hash = hashlib.sha256(base_scene.read_bytes()).hexdigest()
    scene_path = tmp_path / "composed" / "preview_office.xml"

    composition = compose_npc_scene(MODELS / "office_population.json", scene_path)
    model = mujoco.MjModel.from_xml_path(str(composition.scene_path))
    receipt = composition.metadata()

    assert base_hash == hashlib.sha256(base_scene.read_bytes()).hexdigest()
    assert receipt["base_scene"] == str(base_scene.resolve())
    assert (
        receipt["sha256"]["generated_scene"] == hashlib.sha256(scene_path.read_bytes()).hexdigest()
    )
    assert composition.receipt_path.is_file()
    for npc_id in ("employee_01", "employee_02"):
        assert model.body(f"npc__{npc_id}").id >= 0
        hand_site = model.site(f"npc__{npc_id}__handover")
        assert hand_site.id >= 0
        assert list(hand_site.pos) == pytest.approx([-0.25, 0.06, 0.90])

    loaded = load_composed_npc_runtime(MODELS / "office_population.json", scene_path)
    assert set(loaded.npc_system.controllers) == {"employee_01", "employee_02"}
    assert loaded.office_runtime.submit_action(
        ActionCommand("employee_02", ActionType.MOVE_TO, "meeting_table")
    ).valid
    assert loaded.semantic_world.object("employee_01").binding.name == "npc__employee_01"
    physical_receipt = loaded.npc_system.submit(
        NpcCommand(
            "composed-employee-01-move",
            0,
            "employee_01",
            NpcCommandKind.MOVE_TO,
            {"site": "meeting_human_stand_site", "speed": 0.5},
            0.0,
        )
    )
    assert physical_receipt.status is CommandStatus.ACCEPTED
    assert physical_receipt.reason != "unknown_npc"

    reused = compose_npc_scene(MODELS / "office_population.json", scene_path)
    assert reused.reused


def test_composition_rejects_population_spawn_site_missing_from_base_scene(tmp_path: Path) -> None:
    payload = json.loads((MODELS / "office_population.json").read_text(encoding="utf-8"))
    payload["scene"] = str((MODELS / "office_scene.xml").resolve())
    payload["asset_manifest"] = str((MODELS / "npc_assets.example.json").resolve())
    payload["npcs"]["employee_01"]["spawn"]["site"] = "missing_spawn_site"
    population_path = tmp_path / "invalid_population.json"
    population_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="spawn site 'missing_spawn_site' is missing"):
        compose_npc_scene(population_path, tmp_path / "invalid.xml", reuse=False)


def test_composition_rejects_interaction_template_site_missing_from_base_scene(tmp_path: Path) -> None:
    payload = json.loads((MODELS / "office_population.json").read_text(encoding="utf-8"))
    payload["scene"] = str((MODELS / "office_scene.xml").resolve())
    payload["asset_manifest"] = str((MODELS / "npc_assets.example.json").resolve())
    payload["interaction_templates"]["conversation"]["speaker"]["site"] = "missing_site"
    population_path = tmp_path / "invalid_template_population.json"
    population_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="interaction_template_site:missing_site"):
        compose_npc_scene(population_path, tmp_path / "invalid-template.xml", reuse=False)
