import json
from pathlib import Path

import pytest

from stretch_mujoco.agents import ActionCommand, ActionType, DialogueAct, OfficeAgentRuntime
from stretch_mujoco.agents.utility import UtilityGoal, UtilityScore
from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.semantics import SemanticWorld


MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


def test_population_capabilities_reject_social_and_handover(tmp_path):
    payload = json.loads((MODELS / "office_population.json").read_text())
    payload["npcs"]["employee_01"]["capabilities"] = ["locomotion"]
    path = tmp_path / "population.json"
    path.write_text(json.dumps(payload))
    runtime = OfficeAgentRuntime.from_json(
        SemanticWorld.from_json(MODELS / "office_semantics.json"), path
    )
    assert not runtime.validate_action(
        ActionCommand("employee_01", ActionType.TALK, "employee_02")
    ).valid
    assert not runtime.validate_action(
        ActionCommand("employee_01", ActionType.HANDOVER, "employee_02")
    ).valid


def test_population_preserves_legacy_agent_alias_but_uses_npc_runtime_id():
    payload = json.loads((MODELS / "office_population.json").read_text())
    payload["npcs"]["employee_01"]["agent_id"] = "not_employee_01"
    population = NpcPopulation.from_dict(payload)
    definition = population.npcs["employee_01"]
    assert definition.agent_id == "not_employee_01"
    assert (
        OfficeAgentRuntime.from_json(
            SemanticWorld.from_json(MODELS / "office_semantics.json"),
            MODELS / "office_population.json",
        )
        .agents["employee_01"]
        .agent_id
        == "employee_01"
    )


def test_population_rejects_nonfinite_spawn():
    payload = json.loads((MODELS / "office_population.json").read_text())
    payload["npcs"]["employee_01"]["spawn"]["yaw"] = float("nan")
    with pytest.raises(ValueError, match="finite yaw"):
        NpcPopulation.from_dict(payload)


def test_planner_falls_back_before_queuing_an_unsupported_action():
    runtime = OfficeAgentRuntime.from_json(
        SemanticWorld.from_json(MODELS / "office_semantics.json"),
        MODELS / "office_population.json",
        auto_plan=False,
    )
    agent = runtime.agents["employee_01"]
    agent.capabilities = frozenset({"locomotion"})

    class _WorkOnlyUtility:
        @staticmethod
        def evaluate(*args, **kwargs):
            return (UtilityScore(UtilityGoal.WORK, 1.0, {}),)

        @staticmethod
        def choose(scores, **kwargs):
            return scores[0]

    agent.planner.utility = _WorkOnlyUtility()
    plan = agent.planner.choose_plan(agent, runtime.world, 540.0, 0, 0)
    assert plan.variant == "capability_fallback"
    assert tuple(command.action for command in plan.actions) == (ActionType.IDLE,)


def test_population_dialogue_policy_converts_json_enum_keys(tmp_path):
    payload = json.loads((MODELS / "office_population.json").read_text())
    payload["dialogue_policy"] = {
        "max_chars": 80,
        "fallback_templates": {"greeting": ["Hi there."]},
    }
    path = tmp_path / "population.json"
    path.write_text(json.dumps(payload))
    runtime = OfficeAgentRuntime.from_json(
        SemanticWorld.from_json(MODELS / "office_semantics.json"), path
    )
    assert runtime.conversation_policy.config.max_chars == 80
    assert runtime.conversation_policy.config.fallback_templates[DialogueAct.GREETING] == (
        "Hi there.",
    )
    assert DialogueAct.REQUEST in runtime.conversation_policy.config.fallback_templates
