from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from stretch_mujoco.agents.control_sequences.compiler import ControlCapabilities, SequenceCompiler
from stretch_mujoco.agents.control_sequences.executor import SequenceExecutor
from stretch_mujoco.agents.control_sequences.loader import load_control_sequence
from stretch_mujoco.agents.control_sequences.models import (
    FailureHandling,
    CompiledStep,
    SequenceStep,
    StepKind,
    StepStatus,
)
from stretch_mujoco.agents.control_sequences.sources import YamlPlanSource
from stretch_mujoco.agents.actions import (
    ActionCommand,
    ActionExecution,
    ActionType,
    ExecutionStatus,
    RobotTask,
)


def _capabilities() -> ControlCapabilities:
    return ControlCapabilities(
        npc_ids=frozenset({"npc_alex_chen", "npc_morgan_lee", "npc_jordan_patell"}),
        robot_ids=frozenset({"stretch_3"}),
        object_ids=frozenset(
            {"workstation_left", "workstation_right", "bread_snack", "chair_right"}
        ),
        site_ids=frozenset(
            {
                "desk_left_work_site",
                "desk_right_work_site",
                "meeting_conversation_alex_site",
                "meeting_conversation_morgan_site",
                "snack_human_stand_site",
                "chair_right_sit",
                "chair_right_approach_site",
            }
        ),
        reachable_locations=frozenset({"chair_right"}),
    )


def test_llm_mode_consumes_compiler_checked_plan_source_segments() -> None:
    sequence = load_control_sequence(
        "stretch_mujoco/models/control_sequences/npc_full_acceptance_v1.yaml"
    )
    compiler = SequenceCompiler(_capabilities())
    compiled = compiler.compile(sequence, "llm")
    assert not compiled.steps
    wait = SequenceStep(
        "llm_wait",
        StepKind.WAIT,
        2.0,
        FailureHandling(),
        {"id": "llm_wait", "kind": "wait", "duration_s": 1.0},
    )
    executor = SequenceExecutor(
        compiled,
        object(),
        plan_source=YamlPlanSource((wait,)),
        segment_compiler=lambda steps: compiler.compile_segment(sequence, steps),
    )

    assert executor.tick(0.0).status is StepStatus.RUNNING
    assert executor.tick(1.0).status is StepStatus.RUNNING
    assert executor.tick(1.1).status is StepStatus.SUCCEEDED
    assert any(record["event"] == "plan_segment_compiled" for record in executor.audit.records)


def test_composites_advance_without_blocking_or_declaring_missing_receipts_success() -> None:
    sequence = load_control_sequence(
        "stretch_mujoco/models/control_sequences/npc_full_acceptance_v1.yaml"
    )
    # LLM mode gives us the same validated sequence envelope without compiling
    # its unrelated full acceptance action list for this composite-only unit test.
    compiled = SequenceCompiler(_capabilities()).compile(sequence, "llm")

    def wait(step_id: str) -> CompiledStep:
        return CompiledStep(
            SequenceStep(
                step_id,
                StepKind.WAIT,
                2.0,
                FailureHandling(),
                {"id": step_id, "kind": "wait", "duration_s": 1.0},
            )
        )

    parallel = CompiledStep(
        SequenceStep(
            "parallel_waits",
            StepKind.PARALLEL,
            3.0,
            FailureHandling(),
            {"id": "parallel_waits", "kind": "parallel", "join": "all"},
            children=(wait("left").step, wait("right").step),
        ),
        children=(wait("left"), wait("right")),
    )
    choose = CompiledStep(
        SequenceStep(
            "choose_wait",
            StepKind.CHOOSE,
            3.0,
            FailureHandling(),
            {"id": "choose_wait", "kind": "choose"},
            children=(wait("chosen").step,),
        ),
        children=(wait("chosen"),),
    )
    executor = SequenceExecutor(
        replace(compiled, steps=(parallel, choose)), object(), predicate=lambda _: True
    )

    assert executor.tick(0.0).status is StepStatus.RUNNING
    assert executor.tick(1.0).status is StepStatus.RUNNING
    assert executor.tick(1.1).status is StepStatus.RUNNING
    assert executor.tick(2.2).status is StepStatus.RUNNING
    assert executor.tick(2.3).status is StepStatus.SUCCEEDED


def test_request_robot_never_succeeds_without_a_terminal_physical_receipt() -> None:
    sequence = load_control_sequence(
        "stretch_mujoco/models/control_sequences/npc_full_acceptance_v1.yaml"
    )
    compiled = SequenceCompiler(_capabilities()).compile(sequence, "llm")
    request = CompiledStep(
        SequenceStep(
            "request",
            StepKind.ACTION,
            10.0,
            FailureHandling(),
            {
                "id": "request",
                "kind": "action",
                "actor": "alex",
                "action": "request_robot",
                "target": "stretch_3",
                "parameters": {
                    "task": "deliver",
                    "object": "bread_snack",
                    "destination": "workstation_right",
                },
            },
        ),
        actor_id="npc_alex_chen",
        target_id="stretch_3",
    )
    task = RobotTask("npc_alex_chen", "deliver", "bread_snack", "workstation_right")
    agent = SimpleNamespace(executor=ActionExecution(execution_id="request_exec"))

    class Runtime:
        agents = {"npc_alex_chen": agent}
        robot_tasks = {task.task_id: task}

        @staticmethod
        def submit_action(command):
            agent.executor.command = command
            agent.executor.status = StepStatus.SUCCEEDED  # overwritten below with protocol enum
            return SimpleNamespace(valid=True, errors=())

    runtime = Runtime()
    from stretch_mujoco.agents.actions import ExecutionStatus

    original_submit = runtime.submit_action

    def submit(command):
        value = original_submit(command)
        agent.executor.status = ExecutionStatus.SUCCEEDED
        return value

    runtime.submit_action = submit
    executor = SequenceExecutor(replace(compiled, steps=(request,)), runtime)

    assert executor.tick(0.0).status is StepStatus.RUNNING
    assert executor.tick(1.0).status is StepStatus.RUNNING
    task.status = task.status.SUCCEEDED
    assert executor.tick(2.0).status is StepStatus.FAILED
    assert executor.executions[-1].error == "robot_terminal_physical_receipt_missing"


def test_executor_waits_for_the_entire_receipt_gated_desk_work_session() -> None:
    sequence = load_control_sequence(
        "stretch_mujoco/models/control_sequences/npc_full_acceptance_v1.yaml"
    )
    compiled = SequenceCompiler(_capabilities()).compile(sequence, "llm")
    work = CompiledStep(
        SequenceStep(
            "seated_work",
            StepKind.ACTION,
            30.0,
            FailureHandling(),
            {
                "id": "seated_work",
                "kind": "action",
                "actor": "alex",
                "action": "work",
                "target": "workstation_right",
            },
        ),
        actor_id="npc_alex_chen",
        target_id="workstation_right",
    )
    agent = SimpleNamespace(
        executor=ActionExecution(
            execution_id="move_to_chair",
            command=ActionCommand(
                "npc_alex_chen",
                ActionType.MOVE_TO,
                "chair_right",
                {"_desk_work_session_id": "desk_work_000001"},
            ),
            status=ExecutionStatus.RUNNING,
        )
    )

    class Runtime:
        agents = {"npc_alex_chen": agent}

        @staticmethod
        def submit_action(_command):
            return SimpleNamespace(valid=True, errors=())

        @staticmethod
        def desk_work_session_status(_session_id):
            return Runtime.status, Runtime.error

    Runtime.status = ExecutionStatus.RUNNING
    Runtime.error = None
    executor = SequenceExecutor(replace(compiled, steps=(work,)), Runtime())

    assert executor.tick(0.0).status is StepStatus.RUNNING
    agent.executor = ActionExecution(execution_id="stand_up", status=ExecutionStatus.SUCCEEDED)
    assert executor.tick(1.0).status is StepStatus.RUNNING
    Runtime.status = ExecutionStatus.SUCCEEDED
    assert executor.tick(2.0).status is StepStatus.RUNNING
    assert executor.tick(3.0).status is StepStatus.SUCCEEDED
