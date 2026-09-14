from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from stretch_mujoco import launch_sim
from stretch_mujoco import stretch_mujoco_simulator as simulator_module


def test_launcher_stops_simulator_when_start_fails(monkeypatch) -> None:
    class FakeSimulator:
        stopped = False

        def start(self, *, headless: bool) -> None:
            assert headless is True
            raise RuntimeError("startup failed")

        def is_running(self) -> bool:
            return not self.stopped

        def stop(self) -> None:
            self.stopped = True

    simulator = FakeSimulator()
    monkeypatch.setattr(
        launch_sim.stretch_mujoco,
        "StretchMujocoSimulator",
        lambda *args, **kwargs: simulator,
    )

    result = CliRunner().invoke(launch_sim.main, ["--headless"])

    assert isinstance(result.exception, RuntimeError)
    assert simulator.stopped is True


def test_launcher_passes_population_and_semantics(monkeypatch, tmp_path) -> None:
    population, semantics = tmp_path / "population.json", tmp_path / "semantics.json"
    population.write_text("{}")
    semantics.write_text("{}")
    captured = {}

    class FakeSimulator:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def start(self, **kwargs):
            self.stopped = True

        def is_running(self):
            return False

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(launch_sim.stretch_mujoco, "StretchMujocoSimulator", FakeSimulator)
    result = CliRunner().invoke(
        launch_sim.main,
        ["--headless", "--population", str(population), "--semantics", str(semantics)],
    )
    assert result.exit_code == 0
    assert captured["population_path"] == str(population)
    assert captured["semantic_world_path"] == str(semantics)


def test_population_launch_keeps_base_scene_semantics_for_server(monkeypatch, tmp_path) -> None:
    models = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"
    composed = tmp_path / "composed.xml"

    class _Manager:
        @staticmethod
        def Event():
            return object()

    monkeypatch.setattr(simulator_module.multiprocessing, "set_start_method", lambda *a, **k: None)
    monkeypatch.setattr(simulator_module, "Manager", _Manager)
    monkeypatch.setattr(simulator_module.utils, "URDFmodel", lambda: object())
    monkeypatch.setattr(simulator_module.MujocoServerProxies, "default", lambda manager: object())
    monkeypatch.setattr(
        "stretch_mujoco.npc.composition.compose_npc_scene",
        lambda *args, **kwargs: SimpleNamespace(scene_path=composed),
    )

    simulator = simulator_module.StretchMujocoSimulator(
        population_path=str(models / "office_population.json")
    )

    assert simulator.scene_xml_path == str(composed)
    assert simulator._semantic_world_path == str((models / "office_semantics.json").resolve())


def test_embodied_runtime_passes_population_interaction_templates_to_bridge(monkeypatch) -> None:
    templates = {"conversation": object(), "handover": object()}
    runtime = SimpleNamespace(
        trajectory_profile_path=None,
        population_npc_ids=frozenset({"npc_a", "npc_b"}),
        population_interaction_templates=templates,
        agents={},
    )
    captured = {}
    simulator = SimpleNamespace(semantic_world=object(), agent_runtime=None)

    monkeypatch.setattr(
        simulator_module.OfficeAgentRuntime, "from_json", lambda *args, **kwargs: runtime
    )
    monkeypatch.setattr(
        "stretch_mujoco.agents.simulation_bridge.create_mujoco_action_driver",
        lambda *args, **kwargs: captured.update(kwargs) or object(),
    )

    simulator_module.StretchMujocoSimulator.create_office_agent_runtime(
        simulator, "population.json", embodied=True
    )

    assert captured["interaction_templates"] is templates
