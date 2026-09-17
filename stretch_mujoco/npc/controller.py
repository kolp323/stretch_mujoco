"""Lifecycle and state ownership for one embodied NPC."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import mujoco
import numpy as np

from .animation import AnimationController, AnimationGraph, MeshSequenceBackend
from .animation.state import AnimationLifecycle
from .attachment import AttachmentController
from .binding import NpcBinding
from .locomotion import LocomotionController, angle_delta, yaw_from_quaternion
from .naming import candidate_body_names
from .protocol import CommandStatus, NpcCommand, NpcCommandKind, NpcCommandReceipt, NpcRuntimeState


def _payload_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"NPC command payload '{field}' must be numeric")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"NPC command payload '{field}' must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"NPC command payload '{field}' must be finite")
    return result


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
    requested_clip: str | None = None
    clip_activated: bool = False
    clip_sampled: bool = False
    clip_activated_at: float | None = None
    clip_sampled_at: float | None = None

    def __post_init__(self) -> None:
        if self.command.kind in {
            NpcCommandKind.PLAY_ANIMATION,
            NpcCommandKind.INTERACTION_CUE,
        }:
            self.requested_clip = str(self.command.payload.get("clip", "idle"))


@dataclass(frozen=True)
class TrajectoryRouteContract:
    """The controller-facing projection of one preflighted profile route."""

    route_id: str
    source_anchor: str
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
                allow_final_ingress = command.payload.get("allow_final_ingress", False)
                if not isinstance(allow_final_ingress, bool):
                    raise ValueError("allow_final_ingress_must_be_boolean")
                self._validate_trajectory_route(command, site)
                self.locomotion.move_to(
                    site,
                    speed,
                    progress_timeout=progress_timeout,
                    max_replans=max_replans,
                    navigation_site=(
                        None
                        if command.payload.get("navigation_site") is None
                        else str(command.payload["navigation_site"])
                    ),
                    allow_final_ingress=allow_final_ingress,
                )
                self._walk_motion_started = False
                self._move_completion_pending = False
                self._pending_move_cancel = None
                self._stop_marker = None
                self.animation.lifecycle = AnimationLifecycle.NAVIGATING
            elif command.kind == NpcCommandKind.PLAY_ANIMATION:
                self._validate_target_site(command.payload)
                reveal_object = command.payload.get("reveal_object")
                if reveal_object is not None:
                    if str(command.payload["clip"]) != "give":
                        raise ValueError("reveal_object_requires_give_clip")
                    if not isinstance(reveal_object, str) or not reveal_object:
                        raise ValueError("reveal_object_must_be_non_empty_string")
                    if not self.attachments.is_attached(reveal_object):
                        raise ValueError(f"object_not_attached:{reveal_object}")
                self.animation.set_execution(command.command_id)
                if command.payload.get("interaction_target") is None:
                    # A direct play command is an explicit action boundary.
                    # The preceding MOVE_TO may have just requested its idle
                    # arrival pose while the resolved clip is still walk;
                    # allowing the normal safe-marker deferral here can lose
                    # the requested action when the walk clip transitions to
                    # idle on the next step.
                    self.animation.request(str(command.payload["clip"]), force=True)
                else:
                    self._validate_interaction_gate(command.payload)
                    # Do not expose a social clip until the other participant is
                    # physically close and both actors face one another.
                    self.animation.request("idle", force=True)
            elif command.kind == NpcCommandKind.INTERACTION_CUE:
                self.animation.set_execution(command.command_id)
                requested_clip = str(command.payload.get("clip", "idle"))
                if command.payload.get("interaction_target") is not None:
                    self._validate_interaction_gate(command.payload)
                    self.animation.request("idle", force=True)
                else:
                    # Interaction cues without a spatial gate are explicit
                    # action boundaries just like direct PLAY_ANIMATION
                    # commands; do not strand them behind a completed walk.
                    self.animation.request(requested_clip, force=True)
            elif command.kind == NpcCommandKind.ATTACH_OBJECT:
                object_name = str(command.payload["object"])
                visible = command.payload.get("visible", True)
                if not isinstance(visible, bool):
                    raise ValueError("attach_visible_must_be_boolean")
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
                self._validate_target_site(command.payload)
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
        source = command.payload.get("trajectory_source")
        if source is None:
            raise ValueError(f"trajectory_route_missing_source:{route_id}")
        if (
            route.source_anchor != str(source)
            or route.destination_site != site
            or "move_to" not in route.actions
        ):
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
        interaction_ready = True
        duration: float | None = None
        if active is not None:
            if active.command.kind == NpcCommandKind.MOVE_TO:
                return self._step_move(data, sim_time, active, running_receipt)
            elif active.command.kind == NpcCommandKind.ALIGN_TO:
                try:
                    self._require_target_site_proximity(data, active.command.payload)
                except ValueError as error:
                    return self._finish(CommandStatus.FAILED, sim_time, str(error))
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
                try:
                    self._require_target_site_proximity(data, active.command.payload)
                except ValueError as error:
                    return self._finish(CommandStatus.FAILED, sim_time, str(error))
                if active.command.payload.get("interaction_target") is not None:
                    assert active.requested_clip is not None
                    interaction_ready = self._interaction_ready(data, active.command.payload)
                    self.animation.request(
                        active.requested_clip if interaction_ready else "idle",
                        force=True,
                    )
                reveal_object = active.command.payload.get("reveal_object")
                if interaction_ready and reveal_object is not None:
                    self.attachments.set_visible(str(reveal_object), True)
                duration = _payload_float(active.command.payload.get("duration", 0.0), "duration")
            elif active.command.kind == NpcCommandKind.ATTACH_OBJECT:
                object_name = str(active.command.payload["object"])
                self.attachments.attach(
                    object_name,
                    visible=bool(active.command.payload.get("visible", True)),
                )
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
            assert active.requested_clip is not None
            if (
                interaction_ready
                and self.animation.resolved_clip == active.requested_clip
                and self.animation.last_sampled_clip == active.requested_clip
            ):
                active.clip_activated = True
                active.clip_sampled = True
                if active.clip_activated_at is None:
                    active.clip_activated_at = sim_time
                if active.clip_sampled_at is None:
                    active.clip_sampled_at = sim_time
            completion_marker = active.command.payload.get("completion_marker")
            if not interaction_ready or not active.clip_activated or not active.clip_sampled:
                complete = False
            elif completion_marker is not None:
                complete = str(completion_marker) in self._animation_events
            else:
                assert duration is not None
                if duration > 0:
                    complete = (
                        active.clip_activated_at is not None
                        and sim_time - active.clip_activated_at >= duration
                    )
                else:
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

    def _validate_interaction_gate(self, payload: Mapping[str, object]) -> None:
        target = str(payload.get("interaction_target", ""))
        if not target or self._body_id_for(target) < 0:
            raise ValueError(f"unknown_interaction_target:{target}")
        minimum = _payload_float(
            payload.get("interaction_distance_min", 0.45), "interaction_distance_min"
        )
        maximum = _payload_float(
            payload.get("interaction_distance_max", 0.95), "interaction_distance_max"
        )
        tolerance = _payload_float(
            payload.get("interaction_yaw_tolerance", 0.30), "interaction_yaw_tolerance"
        )
        if minimum < 0 or maximum < minimum or not 0 < tolerance <= math.pi:
            raise ValueError("invalid_interaction_gate")

    def _validate_target_site(self, payload: Mapping[str, object]) -> None:
        target_site = payload.get("target_site")
        if (
            target_site is not None
            and mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, str(target_site)) < 0
        ):
            raise ValueError(f"unknown_target_site:{target_site}")
        tolerance = payload.get("position_tolerance")
        if tolerance is not None and _payload_float(tolerance, "position_tolerance") <= 0:
            raise ValueError("invalid_position_tolerance")

    def _require_target_site_proximity(
        self, data: mujoco.MjData, payload: Mapping[str, object]
    ) -> None:
        """Keep staged actions honest if a participant is moved after approach."""
        self._validate_target_site(payload)
        target_site = payload.get("target_site")
        if target_site is None:
            return
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, str(target_site))
        tolerance = _payload_float(payload.get("position_tolerance", 0.12), "position_tolerance")
        distance = float(
            np.linalg.norm(data.mocap_pos[self.binding.mocap_id, :2] - data.site_xpos[site_id, :2])
        )
        if distance > tolerance:
            raise ValueError(f"target_site_not_reached:{target_site}")

    def _body_id_for(self, npc_id: str) -> int:
        names = candidate_body_names(npc_id)
        if npc_id in {"stretch", "stretch_3"}:
            # Stretch is a semantic participant but not an NPC-generated body.
            # Its root is the authoritative live pose for request interaction.
            names = (*names, "base_link")
        for name in names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return body_id
        return -1

    def _interaction_ready(self, data: mujoco.MjData, payload: Mapping[str, object]) -> bool:
        self._validate_interaction_gate(payload)
        target_body = self._body_id_for(str(payload["interaction_target"]))
        target_mocap = int(self.model.body_mocapid[target_body])
        target_position = (
            data.mocap_pos[target_mocap] if target_mocap >= 0 else data.xpos[target_body]
        )
        own_position = data.mocap_pos[self.binding.mocap_id]
        delta = target_position[:2] - own_position[:2]
        distance = float(np.linalg.norm(delta))
        minimum = _payload_float(
            payload.get("interaction_distance_min", 0.45), "interaction_distance_min"
        )
        maximum = _payload_float(
            payload.get("interaction_distance_max", 0.95), "interaction_distance_max"
        )
        if not minimum <= distance <= maximum:
            return False
        bearing = math.atan2(float(delta[0]), float(-delta[1]))
        own_yaw = yaw_from_quaternion(data.mocap_quat[self.binding.mocap_id])
        target_yaw = yaw_from_quaternion(
            data.mocap_quat[target_mocap] if target_mocap >= 0 else data.xquat[target_body]
        )
        tolerance = _payload_float(
            payload.get("interaction_yaw_tolerance", 0.30), "interaction_yaw_tolerance"
        )
        return (
            abs(angle_delta(own_yaw, bearing)) <= tolerance
            and abs(angle_delta(target_yaw, math.remainder(bearing + math.pi, 2 * math.pi)))
            <= tolerance
        )

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
