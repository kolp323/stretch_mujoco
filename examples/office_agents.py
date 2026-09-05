"""Run the deterministic employee action protocol without an LLM."""

import json
from pathlib import Path

import click

from stretch_mujoco.agents import MockRobotExecutor, OfficeAgentRuntime
from stretch_mujoco.semantics import SemanticWorld

MODELS_PATH = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


@click.command()
@click.option(
    "--complete-task",
    is_flag=True,
    help="Let the deterministic mock robot complete the generated task.",
)
def main(complete_task: bool) -> None:
    """Validate and execute the documented request_robot command."""
    world = SemanticWorld.from_json(MODELS_PATH / "office_semantics.json")
    runtime = OfficeAgentRuntime.from_json(
        world,
        MODELS_PATH / "office_agents.json",
        auto_plan=False,
    )
    command = {
        "agent_id": "employee_01",
        "action": "request_robot",
        "target": "stretch",
        "parameters": {
            "task": "deliver",
            "object": "soda_can",
            "destination": "workstation_right",
        },
    }
    validation = runtime.submit_action(command)
    events = runtime.tick(1.0)
    tasks = runtime.pending_robot_tasks()
    if complete_task and tasks:
        MockRobotExecutor().tick(runtime, 1.0)
        events += runtime.drain_events()

    output = {
        "validation": {"valid": validation.valid, "errors": validation.errors},
        "employee_state": vars(runtime.agents["employee_01"].state),
        "reservations": runtime.reservations.snapshot(),
        "robot_tasks": [
            {
                "task_id": task.task_id,
                "task": task.task,
                "object": task.object_id,
                "destination": task.destination,
                "status": task.status.value,
            }
            for task in runtime.robot_tasks.values()
        ],
        "events": [
            {"event": event.event, "agent": event.agent_id, "details": event.details}
            for event in events
        ],
    }
    click.echo(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
