from stretch_mujoco.npc.animation import AnimationController, AnimationGraph, ClipDefinition
from stretch_mujoco.npc.animation.state import InterruptPolicy


class Backend:
    capabilities = frozenset({"mesh_sequence"})
    available_clips = ("idle", "walk", "wave")

    def sample(self, clip: str, phase: float) -> None:
        self.last_sample = (clip, phase)


def test_safe_marker_defers_interrupt_and_seeded_phase_is_reproducible() -> None:
    graph = AnimationGraph(
        "test",
        "idle",
        "idle",
        {
            "idle": ClipDefinition(),
            "walk": ClipDefinition(
                fps=1.0,
                markers=(("right_foot", 0.5),),
                safe_marker="right_foot",
                interrupt_policy=InterruptPolicy.SAFE_MARKER,
            ),
            "wave": ClipDefinition(),
        },
    )
    first = AnimationController(Backend(), graph, phase_seed=7)
    second = AnimationController(Backend(), graph, phase_seed=7)
    first.request("walk")
    second.request("walk")
    first.step(0.0)
    second.step(0.0)
    assert first.phase == second.phase

    first.request("wave")
    events = first.step(0.1)
    assert events[0].name == "interrupt_deferred"
    assert first.resolved_clip == "walk"
    first.step(0.3)
    assert first.resolved_clip == "walk"
    first.step(0.4)
    assert first.resolved_clip == "wave"


def test_speed_metadata_controls_phase_progression() -> None:
    graph = AnimationGraph(
        "speed",
        "idle",
        "idle",
        {
            "idle": ClipDefinition(),
            "wave": ClipDefinition(fps=1.0, loop=False, speed=2.0),
        },
    )
    controller = AnimationController(Backend(), graph)
    controller.request("wave")
    controller.step(0.0)
    controller.step(0.25)
    assert controller.phase == 0.5


def test_uninterruptible_clip_defers_until_its_terminal_phase() -> None:
    graph = AnimationGraph(
        "uninterruptible",
        "idle",
        "idle",
        {
            "idle": ClipDefinition(),
            "wave": ClipDefinition(
                fps=1.0, loop=False, interrupt_policy=InterruptPolicy.UNINTERRUPTIBLE
            ),
        },
    )
    controller = AnimationController(Backend(), graph)
    controller.request("wave")
    controller.step(0.0)
    controller.request("idle")
    controller.step(0.5)
    assert controller.resolved_clip == "wave"
    controller.step(1.0)
    assert controller.requested_clip == "idle"
    controller.step(1.1)
    assert controller.resolved_clip == "idle"


def test_phase_offset_is_stable_per_execution_and_distinct_per_npc() -> None:
    graph = AnimationGraph(
        "replay", "idle", "idle", {"idle": ClipDefinition(), "walk": ClipDefinition()}
    )
    first = AnimationController(Backend(), graph, phase_seed=17, npc_id="employee_01")
    replay = AnimationController(Backend(), graph, phase_seed=17, npc_id="employee_01")
    other = AnimationController(Backend(), graph, phase_seed=17, npc_id="employee_02")
    for controller in (first, replay, other):
        controller.set_execution("move_1")
        controller.request("walk")
        controller.step(0.0)
    assert first.phase_offset == replay.phase_offset
    assert first.phase_offset != other.phase_offset
    assert first.pending_clip is None
