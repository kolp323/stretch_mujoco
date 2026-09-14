from stretch_mujoco.agents.actions import RobotTask, RobotTaskStatus, RobotTaskType
from stretch_mujoco.agents.robot_task_driver import (
    RobotTaskExecutor,
    UnsupportedRobotTaskExecutor,
)
from stretch_mujoco.agents.mock_robot import MockRobotExecutor


class _Runtime:
    def __init__(self, task):
        self.task = task
        self.calls = []

    def pending_robot_tasks(self):
        return () if self.task.status is not RobotTaskStatus.PENDING else (self.task,)

    def complete_robot_task(self, task_id, success, error=None):
        self.calls.append((task_id, success, error))
        self.task.status = RobotTaskStatus.SUCCEEDED if success else RobotTaskStatus.FAILED


def test_external_executor_protocol_declares_task_capability_and_lifecycle():
    class Executor:
        supported_task_types = frozenset({RobotTaskType.PLACE_DELIVERY})

        def tick(self, _runtime, _elapsed_seconds):
            return ()

        def cancel(self, _task_id, _reason="robot_task_cancelled"):
            return None

    assert isinstance(Executor(), RobotTaskExecutor)
    assert isinstance(MockRobotExecutor(), RobotTaskExecutor)
    assert MockRobotExecutor().supported_task_types == frozenset(RobotTaskType)


def test_compatibility_executor_fails_without_running_robot_algorithms():
    task = RobotTask("npc", "deliver", "snack", "desk")
    runtime = _Runtime(task)
    executor = UnsupportedRobotTaskExecutor(object(), waypoints={"desk": (1.0, 0.0)})

    assert executor.tick(runtime, 0.1) == (task.task_id,)
    assert runtime.calls == [(task.task_id, False, "robot_task_executor_not_configured")]
    assert task.receipt_ids == {f"robot:{task.task_id}:unsupported"}
