"""Lifecycle and state ownership for one embodied NPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import mujoco

from .animation import AnimationController, AnimationGraph, MeshSequenceBackend
from .animation.state import AnimationLifecycle
from .attachment import AttachmentController
from .binding import NpcBinding
from .locomotion import LocomotionController
from .protocol import CommandStatus, NpcCommand, NpcCommandKind, NpcCommandReceipt, NpcRuntimeState


def _payload_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"NPC command payload '{field}' must be numeric")
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"NPC command payload '{field}' must be numeric") from error


def _payload_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"NPC command payload '{field}' must be an integer")
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"NPC command payload '{field}' must be an integer") from error


@dataclass
class ActiveNpcCommand:
    command: NpcCommand
    started_at: float | None = None


@dataclass(frozen=True)
class TrajectoryRouteContract:
    """The controller-facing projection of one preflighted profile route."""

    route_id: str
    destination_site: str
    actions: frozenset[str]


class NpcController:
    """Compose independent locomotion and animation state for a single NPC."""

    def __init__(
        self,
        model: mujoco.MjModel,
        binding: NpcBinding,
        animation_graph: AnimationGraph | None = None,
        simulation_seed: int = 0,
        trajectory_routes: Mapping[str, TrajectoryRouteContract] | None = None,
    ) -> None:
        self.model = model
        self.binding = binding
        self.locomotion = LocomotionController(model, binding)
        self.trajectory_routes = dict(trajectory_routes or {})
        self.animation = AnimationController(
            MeshSequenceBackend(model, binding),
            graph=animation_graph,
            phase_seed=simulation_seed,
            npc_id=binding.npc_id,
        )
        self.attachments = AttachmentController(model, binding)
        self.active_command: ActiveNpcCommand | None = None
        self.last_receipt: NpcCommandReceipt | None = None
        self.revision = 0
        self._last_step_time: float | None = None
        self._animation_events: tuple[str, ...] = ()
        self._walk_motion_started = False
        self._move_completion_pending = False
        self._pending_move_cancel: str | None = None
        self._stop_marker: str | None = None

    def accept(self, command: NpcCommand) -> NpcCommandReceipt:
        if self.active_command is not None:
            return NpcCommandReceipt(
                command.command_id,
                command.npc_id,
                CommandStatus.FAILED,
                reason=f"npc_busy:{self.active_command.command.command_id}",
                finished_at=command.issued_at,
            )
        try:
            if command.kind == NpcCommandKind.MOVE_TO:
                self.animation.set_execution(command.command_id)
                site = str(command.payload["site"])
                speed = _payload_float(command.payload.get("speed", 1.0), "speed")
                progress_timeout = _payload_float(
                    command.payload.get("progress_timeout", 2.0), "progress_timeout"
                )
                max_replans = _payload_int(command.payload.get("max_replans", 0), "max_replans")
                self._validate_trajectory_route(command, site)
                self.locomotion.move_to(
                    site,
                    speed,
                    progress_timeout=progress_timeout,
                    max_replans=max_replans,
                )
                self._walk_motion_started = False
                self._move_completion_pending = False
                self._pending_move_cancel = None
                self._stop_marker = None
                self.animation.lifecycle = AnimationLifecycle.NAVIGATING
            elif command.kind == NpcCommandKind.PLAY_ANIMATION:
                target_site = command.payload.get("target_site")
                if (
                    target_site is not None
                    and mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, str(target_site))
                    < 0
                ):
                    raise ValueError(f"unknown_target_site:{target_site}")
                self.animation.set_execution(command.command_id)
                self.animation.request(str(command.payload["clip"]))
            elif command.kind == NpcCommandKind.INTERACTION_CUE:
                self.animation.set_execution(command.command_id)
                self.animation.request(str(command.payload.get("clip", "idle")))
            elif command.kind == NpcCommandKind.ATTACH_OBJECT:
                object_name = str(command.payload["object"])
                self.attachments.validate_object(object_name)
                self.attachments.handover_site_id()
            elif command.kind == NpcCommandKind.DETACH_OBJECT:
                object_name = str(command.payload["object"])
                if not self.attachments.is_attached(object_name):
                    raise ValueError(f"object_not_attached:{object_name}")
                detach_site = command.payload.get("site")
                if (
                    detach_site is not None
                    and mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, str(detach_site))
                    < 0
                ):
                    raise ValueError(f"unknown_detach_site:{detach_site}")
            elif command.kind == NpcCommandKind.ALIGN_TO:
                _payload_float(command.payload["yaw"], "yaw")
                self.animation.lifecycle = AnimationLifecycle.ALIGNING
            else:
                return NpcCommandReceipt(
                    command.command_id,
                    command.npc_id,
                    CommandStatus.FAILED,
                    reason=f"unsupported_command:{command.kind.value}",
                    finished_at=command.issued_at,
                )
        except (KeyError, TypeError, ValueError) as error:
            return NpcCommandReceipt(
                command.command_id,
                command.npc_id,
                CommandStatus.FAILED,
                reason=str(error),
                finished_at=command.issued_at,
            )
        self.active_command = ActiveNpcCommand(command)
        return NpcCommandReceipt(
            command.command_id,
            command.npc_id,
            CommandStatus.ACCEPTED,
        )

    def _validate_trajectory_route(self, command: NpcCommand, site: str) -> None:
        """Reject a declared profile route whose target/action contract drifted."""
        route_id = command.payload.get("trajectory_route")
        if route_id is None:
            return
        route = self.trajectory_routes.get(str(route_id))
        if route is None:
            raise ValueError(f"trajectory_route_unknown:{route_id}")
        if route.destination_site != site or "move_to" not in route.actions:
            raise ValueError(f"trajectory_route_contract_mismatch:{route_id}")

    def cancel(
        self, command_id: str, sim_time: float, reason: str = "cancelled"
    ) -> NpcCommandReceipt:
        if self.active_command is None or self.active_command.command.command_id != command_id:
            return NpcCommandReceipt(
                command_id,
                self.binding.npc_id,
                CommandStatus.FAILED,
                reason="command_not_active",
                finished_at=sim_time,
            )
        started_at = self.active_command.started_at
        if self.active_command.command.kind == NpcCommandKind.MOVE_TO and self._walk_motion_started:
            self._pending_move_cancel = reason
            self.animation.request("idle")
            return NpcCommandReceipt(
                command_id,
                self.binding.npc_id,
                CommandStatus.RUNNING,
                reason="waiting_for_foot_marker",
                started_at=started_at,
            )
        self.locomotion.cancel()
        self.animation.recover_to_idle()
        self.animation.lifecycle = AnimationLifecycle.FAILED
        self.active_command = None
        receipt = NpcCommandReceipt(
            command_id,
            self.binding.npc_id,
            CommandStatus.CANCELLED,
            reason=reason,
            started_at=started_at,
            finished_at=sim_time,
        )
        self.last_receipt = receipt
        return receipt

    def step(self, data: mujoco.MjData, sim_time: float) -> NpcCommandReceipt | None:
        self.revision += 1
        active = self.active_command
        running_receipt: NpcCommandReceipt | None = None
        if active is not None and active.started_at is None:
            active.started_at = sim_time
            running_receipt = NpcCommandReceipt(
                active.command.command_id,
                active.command.npc_id,
                CommandStatus.RUNNING,
                started_at=sim_time,
            )

        if (
            active is not None
            and active.command.deadline is not None
            and sim_time > active.command.deadline
        ):
            self.locomotion.cancel()
            self.animation.recover_to_idle()
            return self._finish(CommandStatus.TIMED_OUT, sim_time, "deadline_exceeded")

        complete = False
        if active is not None:
            if active.command.kind == NpcCommandKind.MOVE_TO:
                return self._step_move(data, sim_time, active, running_receipt)
            elif active.command.kind == NpcCommandKind.ALIGN_TO:
                dt = (
                    0.0
                    if self._last_step_time is None
                    else min(sim_time - self._last_step_time, 0.1)
                )
                complete = self.locomotion.align_to(
                    data, _payload_float(active.command.payload["yaw"], "yaw"), dt
                )
            elif active.command.kind in {
                NpcCommandKind.PLAY_ANIMATION,
                NpcCommandKind.INTERACTION_CUE,
            }:
                duration = _payload_float(active.command.payload.get("duration", 0.0), "duration")
                complete = duration > 0 and (
                    active.started_at is not None and sim_time - active.started_at >= duration
                )
            elif active.command.kind == NpcCommandKind.ATTACH_OBJECT:
                object_name = str(active.command.payload["object"])
                self.attachments.attach(object_name)
                self.attachments.step(data)
                complete = self.attachments.is_attached(object_name)
            elif active.command.kind == NpcCommandKind.DETACH_OBJECT:
                self.attachments.detach(
                    data,
                    str(active.command.payload["object"]),
                    (
                        None
                        if active.command.payload.get("site") is None
                        else str(active.command.payload["site"])
                    ),
                )
                complete = True
        else:
            self.locomotion.step(data, sim_time)

        locomotion_state = "walk" if self.locomotion.target_site is not None else "stationary"
        events = self.animation.step(
            sim_time,
            locomotion=locomotion_state,
            speed_scale=(self.locomotion.speed if locomotion_state == "walk" else 1.0),
        )
        if active is not None and active.command.kind == NpcCommandKind.MOVE_TO:
            self.animation.lifecycle = AnimationLifecycle.NAVIGATING
        elif active is not None and active.command.kind == NpcCommandKind.ALIGN_TO:
            self.animation.lifecycle = AnimationLifecycle.ALIGNING
        self._animation_events = tuple(event.name for event in events)
        if (
            active is not None
            and active.command.kind
            in {NpcCommandKind.PLAY_ANIMATION, NpcCommandKind.INTERACTION_CUE}
            and self.animation.fallback_event is not None
        ):
            requested_clip = str(active.command.payload.get("clip", "idle"))
            self.animation.recover_to_idle()
            return self._finish(
                CommandStatus.FAILED,
                sim_time,
                self.animation.failure_reason or f"clip_unavailable:{requested_clip}",
            )
        if active is not None and active.command.kind in {
            NpcCommandKind.PLAY_ANIMATION,
            NpcCommandKind.INTERACTION_CUE,
        }:
            completion_marker = active.command.payload.get("completion_marker")
            if completion_marker is not None:
                complete = str(completion_marker) in self._animation_events
            elif _payload_float(active.command.payload.get("duration", 0.0), "duration") <= 0:
                complete = self.animation.phase >= 1.0
        self.attachments.step(data)
        self._last_step_time = sim_time
        if complete:
            if active is not None:
                arrival_clip = active.command.payload.get("arrival_clip")
                if arrival_clip is not None:
                    self.animation.request(str(arrival_clip))
                elif (
                    active.command.kind
                    in {NpcCommandKind.PLAY_ANIMATION, NpcCommandKind.INTERACTION_CUE}
                    and active.command.payload.get("completion_marker") is not None
                ):
                    self.animation.settle_completed_clip()
            return self._finish(CommandStatus.SUCCEEDED, sim_time)
        return running_receipt

    def _step_move(
        self,
        data: mujoco.MjData,
        sim_time: float,
        active: ActiveNpcCommand,
        running_receipt: NpcCommandReceipt | None,
    ) -> NpcCommandReceipt | None:
        """Use walk foot markers as the safe root-motion start and stop boundaries."""
        # A deterministic random walk phase can otherwise delay the first foot
        # marker for hundreds of seconds at a deliberately slow navigation speed.
        # Start on the next normal-cadence footfall, then couple every subsequent
        # walk phase increment to the actual root speed.
        speed_scale = (
            self.locomotion.speed if self._walk_motion_started else max(self.locomotion.speed, 1.0)
        )
        events = self.animation.step(sim_time, locomotion="walk", speed_scale=speed_scale)
        self._animation_events = tuple(event.name for event in events)
        foot_marker = next(
            (event.name for event in events if event.name in {"left_foot", "right_foot"}), None
        )
        self.animation.lifecycle = AnimationLifecycle.NAVIGATING
        if foot_marker is not None and not self._walk_motion_started:
            self._walk_motion_started = True
        if foot_marker is not None and self._pending_move_cancel is not None:
            self._stop_marker = foot_marker
            reason = self._pending_move_cancel
            self._pending_move_cancel = None
            self.locomotion.cancel()
            self.animation.request("idle", force=True)
            return self._finish(CommandStatus.CANCELLED, sim_time, reason)
        if self._walk_motion_started and not self._move_completion_pending:
            self.locomotion.step(data, sim_time)
            if self.locomotion.failure_reason is not None:
                self.animation.recover_to_idle()
                return self._finish(CommandStatus.FAILED, sim_time, self.locomotion.failure_reason)
            if self.locomotion.target_site is None:
                self._move_completion_pending = True
        if self._move_completion_pending and foot_marker is not None:
            self._stop_marker = foot_marker
            self.animation.request(
                str(active.command.payload.get("arrival_clip", "idle")), force=True
            )
            return self._finish(CommandStatus.SUCCEEDED, sim_time)
        self._last_step_time = sim_time
        return running_receipt

    def _finish(
        self, status: CommandStatus, sim_time: float, reason: str | None = None
    ) -> NpcCommandReceipt:
        assert self.active_command is not None
        self.animation.lifecycle = (
            AnimationLifecycle.COMPLETED
            if status == CommandStatus.SUCCEEDED
            else AnimationLifecycle.FAILED
        )
        receipt = NpcCommandReceipt(
            self.active_command.command.command_id,
            self.binding.npc_id,
            status,
            reason=reason,
            started_at=self.active_command.started_at,
            finished_at=sim_time,
        )
        self.active_command = None
        self._walk_motion_started = False
        self._move_completion_pending = False
        self._pending_move_cancel = None
        self.last_receipt = receipt
        return receipt

    def state(self, data: mujoco.MjData, sim_time: float) -> NpcRuntimeState:
        active_id = None if self.active_command is None else self.active_command.command.command_id
        interaction_id = None
        if self.active_command is not None:
            raw_interaction_id = self.active_command.command.payload.get("interaction_id")
            if raw_interaction_id is not None:
                interaction_id = str(raw_interaction_id)
        position = tuple(float(value) for value in data.mocap_pos[self.binding.mocap_id])
        quaternion = tuple(float(value) for value in data.mocap_quat[self.binding.mocap_id])
        return NpcRuntimeState(
            npc_id=self.binding.npc_id,
            revision=self.revision,
            sim_time=sim_time,
            position=position,  # type: ignore[arg-type]
            quaternion=quaternion,  # type: ignore[arg-type]
            locomotion="walk" if self.locomotion.target_site is not None else "stationary",
            requested_animation=self.animation.requested_clip,
            resolved_clip=self.animation.resolved_clip,
            clip_phase=self.animation.phase,
            phase_seed=self.animation.phase_seed,
            phase_offset=self.animation.phase_offset,
            last_marker=(self._animation_events[-1] if self._animation_events else None),
            pending_clip=self.animation.pending_clip,
            deferred_interrupt=self.animation.pending_clip is not None,
            route_revision=self.locomotion.route_revision,
            replan_attempt=self.locomotion.replan_attempt,
            stop_marker=self._stop_marker,
            locomotion_failure=self.locomotion.failure_reason,
            animation_lifecycle=self.animation.lifecycle.value,
            transition=self.animation.transition,
            animation_events=self._animation_events,
            held_objects=self.attachments.held_objects,
            interaction_id=interaction_id,
            active_command_id=active_id,
            last_receipt=self.last_receipt,
        )
