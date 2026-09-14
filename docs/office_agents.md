# Office Employee Agents

`OfficeAgentRuntime` is the deterministic, LLM-free execution layer for employee
agents. An optional LLM may produce `ActionCommand` JSON outside the simulation,
but it cannot introduce new action names or bypass runtime validation.

## Agent Components

Each configured employee owns independent profile, needs, schedule, memory,
planner, perception, executor, and `EmployeeState` instances. Add employees under
`employees` in `stretch_mujoco/models/office_agents.json`; each ID must also be a
semantic `Employee` object.

Schedule start times receive deterministic per-day jitter. The same seed reproduces
an experiment, while different days do not repeat the exact same timeline.

## Validation Order

Before an action starts, the runtime checks:

1. Agent identity and executor availability.
2. Target and parameter existence.
3. Required location and object type.
4. Action-specific preconditions and perception.
5. Resource reservations and conflicts.
6. `ALLOWED_FOR` permissions for confidential objects.

Rejected actions produce `action_rejected` events and do not mutate the world.

## Simulator Loop

```python
runtime = sim.create_office_agent_runtime(auto_plan=True)
sim.start(headless=True)

while sim.is_running():
    snapshot = sim.pull_semantic_state()
    events = runtime.tick(0.1, snapshot)

    for task in runtime.pending_robot_tasks():
        robot_controller.submit(task)
```

The runtime executes logical actions and verifies semantic effects. Navigation,
grasping, and handover controllers remain responsible for physical execution.

## Robot Task Contract

`request_robot` is only an NPC request action; it is not a robot control
command or a delivery classification. The request is compiled once by
`compile_robot_task_type` into a `RobotTaskType`, and the selected robot control
driver implements that workflow:

```text
request_robot → compile_robot_task_type → RobotTaskType → robot/NPC control driver
```

`place_delivery` releases an object at `destination`. `robot_to_npc_handover`
prepares the recipient NPC, waits for its receive marker, confirms robot release,
then attaches the object to the NPC. `destination` is always a scene location;
`recipient` is a separate NPC ID. For v1 compatibility, `task: deliver` without
`recipient` compiles to `place_delivery`; with `recipient`, it compiles to
`robot_to_npc_handover`.

Controllers must report terminal evidence, not merely a logical delay. A handover
cannot succeed unless the runtime has confirmed all three facts:
`robot_release_confirmed`, `npc_attachment_confirmed`, and
`interaction_confirmed`. `MockRobotExecutor` supplies namespaced receipts for
demo/test integration after the NPC movement/alignment/receive-marker/attachment
flow; it is not a production robot executor. Production integrations implement
the minimal `RobotTaskExecutor` lifecycle and supply their own navigation, IK,
grasp, release, and terminal receipt behavior. This repository does not provide
a physical `robot_to_npc_handover` executor.

For a place delivery, after a controller reaches a terminal state, report its
verified result explicitly:

```python
runtime.complete_robot_task(
    task.task_id,
    success=True,
    semantic_snapshot=sim.pull_semantic_state(),
)
```

Successful place delivery updates `ON` and `REQUESTED_BY`, releases the
reservation, records memory, and emits `robot_task_completed`. Successful
handover instead retains the NPC `HOLDS` attachment and also requires the three
handover evidence flags above. A failed controller result does not claim that the
object moved. When a place snapshot is supplied, success is downgraded to failure
unless the object is physically near the destination.

## Closed Action Set

The allowed actions are `idle`, `move_to`, `sit`, `work`, `rest`, `eat`, `drink`,
`pick_up`, `put_down`, `request_robot`, `use_computer`, `open_cabinet`, `handover`,
and `attend_meeting`.
