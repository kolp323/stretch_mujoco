import atexit
import hashlib
import multiprocessing
import platform
import signal
import sys
import tempfile
import threading
import time
import uuid
import warnings
from multiprocessing import Lock, Manager, Process
from pathlib import Path

import click
import numpy as np
from mujoco._structs import MjModel

import stretch_mujoco.utils as utils
from stretch_mujoco.agents import OfficeAgentRuntime
from stretch_mujoco.datamodels.status_command import (
    CommandBaseVelocity,
    CommandCoordinateFrameArrowsViz,
    CommandKeyframe,
    CommandMove,
    StatusCommand,
)
from stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch_mujoco.datamodels.status_stretch_joints import StatusStretchJoints
from stretch_mujoco.datamodels.status_stretch_sensors import StatusStretchSensors
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.mujoco_server import MujocoServer, MujocoServerProxies
from stretch_mujoco.mujoco_server_managed import MujocoServerManaged
from stretch_mujoco.mujoco_server_passive import MujocoServerPassive
from stretch_mujoco.npc import NpcCommand, NpcCommandKind, NpcCommandReceipt, NpcRuntimeState
from stretch_mujoco.semantics import SemanticWorld
from stretch_mujoco.utils import block_until_check_succeeds, require_connection


class StretchMujocoSimulator:
    """
    Stretch Mujoco Simulator class for interfacing with the Mujoco Server.

    Calling `start()` will spawn a new process that runs `MujocoServer` and the simulator.

    You can specify `start(headless=True)` to run the simulation without a GUI.

    Data from the MujocoServer is sent to StretchMujocoSimulator using proxies.

    Use `pull_status()` and `pull_camera_data()` to access simulation data.
    """

    def __init__(
        self,
        scene_xml_path: str | None = None,
        model: MjModel | None = None,
        camera_hz: float = 30,
        cameras_to_use: list[StretchCameras] | None = None,
        start_translation: list | None = None,
        start_rotation_quat: list | None = None,
        semantic_world_path: str | None = None,
        population_path: str | None = None,
    ) -> None:
        self.population_path = (
            None if population_path is None else str(Path(population_path).resolve())
        )
        if self.population_path is not None:
            if model is not None:
                raise ValueError("population_path cannot be combined with a precompiled model")
            from stretch_mujoco.npc.composition import compose_npc_scene
            from stretch_mujoco.npc.schema import NpcPopulation

            population = NpcPopulation.from_json(self.population_path)
            if (
                scene_xml_path is not None
                and Path(scene_xml_path).resolve() != population.resolve_path(population.scene)
            ):
                raise ValueError("scene_xml_path conflicts with authoritative population.scene")
            composed_dir = Path(tempfile.gettempdir()) / "stretch_mujoco_npc_compositions"
            population_key = hashlib.sha256(self.population_path.encode()).hexdigest()[:12]
            scene_xml_path = str(
                compose_npc_scene(
                    self.population_path,
                    composed_dir
                    / f"{Path(self.population_path).stem}-{population_key}.xml",
                ).scene_path
            )
            semantic_scene_path = population.resolve_path(population.scene)
        else:
            semantic_scene_path = scene_xml_path
        self.scene_xml_path = scene_xml_path
        self.model = model
        self.camera_hz = camera_hz
        self.urdf_model = utils.URDFmodel()
        self._server_process: Process | None = None
        self._cameras_to_use = list(cameras_to_use or ())
        self._start_translation = start_translation
        self._start_rotation_quat = start_rotation_quat
        self.semantic_world = (
            SemanticWorld.from_json(semantic_world_path)
            if semantic_world_path is not None
            else SemanticWorld.for_scene(semantic_scene_path)
        )
        self._semantic_world_path = (
            str(self.semantic_world.source_path)
            if self.semantic_world is not None and self.semantic_world.source_path is not None
            else semantic_world_path
        )
        self.agent_runtime: OfficeAgentRuntime | None = None

        self.is_stop_called = False

        # Manager must use spawn when this simulator is constructed from an MCP worker thread.
        multiprocessing.set_start_method("spawn", force=True)
        self._manager = Manager()
        self._stop_mujoco_process_event = self._manager.Event()

        self.data_proxies = MujocoServerProxies.default(self._manager)

        self._command_lock = Lock()
        self._npc_sequences: dict[str, int] = {}
        self._npc_command_ids: set[str] = set()
        self._legacy_humanoid_sit_target = "chair_right_sit"
        self._legacy_humanoid_navigation_target = ""
        self._legacy_humanoid_playback_speed = 1.0

    def start(
        self,
        show_viewer_ui: bool = False,
        headless: bool = False,
        use_passive_viewer: bool = True,
        home_on_start: bool = True,
    ) -> None:
        """
        Start the simulator

        Args:
            show_viewer_ui: bool, whether to show the Mujoco viewer UI
            headless: bool, whether to run the simulation in headless mode
            use_passive_viewer: bool, to use the passive or managed mujoco UI viewer.
        """
        self.is_stop_called = False

        mujoco_server = MujocoServer  # Headless

        if not headless:
            mujoco_server = MujocoServerPassive if use_passive_viewer else MujocoServerManaged

        if platform.system() == "Darwin" and mujoco_server is MujocoServerPassive:
            # On a mac, the process for MujocoServerPassive needs to be started with mjpython
            mjpython_path = sys.executable.replace("bin/python3", "bin/mjpython").replace(
                "bin/python", "bin/mjpython"
            )
            print(f"{mjpython_path=}")
            multiprocessing.set_executable(mjpython_path)

        self._server_process = Process(
            target=mujoco_server.launch_server,
            name="MujocoProcess",
            args=(
                self.scene_xml_path,
                self.model,
                self.camera_hz,
                show_viewer_ui,
                self._stop_mujoco_process_event,
                self.data_proxies,
                self._cameras_to_use,
                self._start_translation,
                self._start_rotation_quat,
                self._semantic_world_path,
                self.population_path,
            ),
            daemon=False,  # We're gonna handle terminating this in stop_mujoco_process()
        )
        self._server_process.start()

        # Handle stopping, in all its various ways:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, lambda num, sig: self.stop())
            signal.signal(signal.SIGINT, lambda num, sig: self.stop())
        atexit.register(self.stop)

        click.secho("Starting Stretch Mujoco Simulator...", fg="green")
        while self.pull_status().time == 0 or self.pull_camera_data().time == 0:
            time.sleep(1)
            click.secho("Still waiting to connect to the MuJoCo simulator.", fg="yellow")

            if not self.is_running():
                click.secho("The simulator is not running anymore, quitting..", fg="yellow")
                return

        click.secho("The MuJoCo simulator is connected.", fg="green")

        if home_on_start:
            self.home()

    def stop(self) -> None:
        """
        This is called at exit to gracefully terminate the simulation and the Mujoco Process, and their many threads.

        Fingers-crossed we get a SIGTERM, and not a SIGKILL..
        """
        if self.is_stop_called:
            return

        self.is_stop_called = True

        try:
            simulation_time = self.data_proxies.get_status().time
            simulation_time_message = f" simulated runtime= {simulation_time:.1f}s"
        except (AttributeError, BrokenPipeError, EOFError, OSError):
            simulation_time_message = ""

        click.secho(
            f"Stopping Stretch Mujoco Simulator...{simulation_time_message}",
            fg="red",
        )

        self.stop_mujoco_process()

        click.secho(
            "The Stretch MuJoCo Simulator has ended. Good-bye!",
            fg="red",
        )

    def stop_mujoco_process(self):

        if self._server_process and not self._server_process.is_alive():
            click.secho(
                "The MuJoCo process has already terminated.",
                fg="red",
            )
            return

        click.secho(
            "Sending signal to stop the MuJoCo process...",
            fg="red",
        )

        # Wait until the main control loop ends before sending this stop event.
        self._stop_mujoco_process_event.set()
        if self._server_process:
            # self._server_process.terminate() # ask it nicely.
            self._server_process.join()

        click.secho(
            "The MuJoCo process has ended.",
            fg="red",
        )

    @require_connection
    def home(self) -> None:
        """
        Move the robot to home position
        """
        with self._command_lock:
            self.data_proxies.set_command(
                StatusCommand(keyframe=CommandKeyframe(name="home", trigger=True))
            )
        self.wait_while_is_moving(Actuators.lift)

    @require_connection
    def stow(self) -> None:
        """
        Move the robot to stow position
        """
        with self._command_lock:
            self.data_proxies.set_command(
                StatusCommand(keyframe=CommandKeyframe(name="stow", trigger=True))
            )

        self.wait_while_is_moving(Actuators.wrist_pitch)

    def set_humanoid_animation(self, animation: str) -> None:
        """Select a clip for the default NPC (deprecated compatibility API)."""
        available_animations = {"eat", "idle", "sit", "walk", "work"}
        if animation not in available_animations:
            available = ", ".join(sorted(available_animations))
            raise ValueError(f"Unknown humanoid animation '{animation}'. Available: {available}")
        warnings.warn(
            "set_humanoid_animation() is deprecated; use submit_npc_command()",
            DeprecationWarning,
            stacklevel=2,
        )
        if animation in {"sit", "work"}:
            self._submit_legacy_move(self._legacy_humanoid_sit_target, arrival_clip=animation)
        elif animation == "walk" and self._legacy_humanoid_navigation_target:
            self._submit_legacy_move(self._legacy_humanoid_navigation_target)
        else:
            self.submit_npc_command(
                self._new_npc_command(
                    "employee_01",
                    NpcCommandKind.PLAY_ANIMATION,
                    {"clip": animation},
                )
            )

    def set_humanoid_sit_target(self, chair: str) -> None:
        """Choose which office chair the NPC uses for the sit animation."""
        site_name = chair if chair.endswith("_sit") else f"{chair}_sit"
        available_targets = {"chair_left_sit", "chair_right_sit"}
        if site_name not in available_targets:
            available = ", ".join(
                sorted(target.removesuffix("_sit") for target in available_targets)
            )
            raise ValueError(f"Unknown office chair '{chair}'. Available: {available}")
        self._legacy_humanoid_sit_target = site_name

    def set_humanoid_navigation_target(self, target: str) -> None:
        """Choose an office interaction site for the NPC walk root motion."""
        target_sites = {
            "workstation_left": "desk_left_work_site",
            "workstation_right": "desk_right_work_site",
            "chair_left": "chair_left_sit",
            "chair_right": "chair_right_sit",
            "meeting_table": "meeting_human_stand_site",
            "storage_cabinet": "cabinet_human_stand_site",
            "snack_counter": "snack_human_stand_site",
            "coffee_bar": "coffee_human_stand_site",
            "coffee_machine": "coffee_human_stand_site",
        }
        site_name = target_sites.get(target, target)
        if site_name not in set(target_sites.values()):
            available = ", ".join(sorted(target_sites))
            raise ValueError(
                f"Unknown humanoid navigation target '{target}'. Available: {available}"
            )
        self._legacy_humanoid_navigation_target = site_name

    def submit_npc_command(self, command: NpcCommand) -> str:
        """Submit an ordered NPC command and return its idempotency key."""
        with self._command_lock:
            if command.command_id in self._npc_command_ids:
                return command.command_id
            last_sequence = self._npc_sequences.get(command.npc_id, -1)
            if command.sequence <= last_sequence:
                raise ValueError(
                    f"NPC command sequence {command.sequence} is not newer than {last_sequence}"
                )
            self._npc_sequences[command.npc_id] = command.sequence
            self._npc_command_ids.add(command.command_id)
            self.data_proxies.submit_npc_command(command)
        return command.command_id

    def cancel_npc_command(self, npc_id: str, command_id: str) -> None:
        """Cancel an active command through the same ordered transport."""
        command = self._new_npc_command(
            npc_id,
            NpcCommandKind.CANCEL,
            {"command_id": command_id},
        )
        self.submit_npc_command(command)

    def pull_npc_states(self) -> dict[str, NpcRuntimeState]:
        """Return the latest observed per-NPC simulation states."""
        return {
            npc_id: NpcRuntimeState.from_dict(payload)
            for npc_id, payload in self.data_proxies.get_npc_states().items()
        }

    def pull_npc_receipts(self) -> tuple[NpcCommandReceipt, ...]:
        """Drain command lifecycle receipts emitted by the simulator process."""
        return tuple(
            NpcCommandReceipt.from_dict(payload)
            for payload in self.data_proxies.drain_npc_receipts()
        )

    def _new_npc_command(
        self,
        npc_id: str,
        kind: NpcCommandKind,
        payload: dict[str, object],
        *,
        deadline: float | None = None,
    ) -> NpcCommand:
        sequence = self._npc_sequences.get(npc_id, -1) + 1
        issued_at = float(self.pull_status().time)
        return NpcCommand(
            command_id=f"npc_{uuid.uuid4().hex}",
            sequence=sequence,
            npc_id=npc_id,
            kind=kind,
            payload=payload,
            issued_at=issued_at,
            deadline=deadline,
        )

    def _submit_legacy_move(self, site: str, *, arrival_clip: str = "idle") -> str:
        return self.submit_npc_command(
            self._new_npc_command(
                "employee_01",
                NpcCommandKind.MOVE_TO,
                {
                    "site": site,
                    "speed": self._legacy_humanoid_playback_speed,
                    "arrival_clip": arrival_clip,
                },
            )
        )

    def set_humanoid_playback_speed(self, speed: float) -> None:
        """Scale NPC root motion and baked animation playback together."""
        if speed <= 0:
            raise ValueError("Humanoid playback speed must be positive")
        self._legacy_humanoid_playback_speed = float(speed)

    def set_object_visibility(self, object_id: str, visible: bool) -> None:
        """Show or hide an object body and enable or disable its collisions."""
        self.data_proxies.set_object_visibility(object_id, visible)

    def consume_office_object(self, object_id: str) -> None:
        """Mark an office object consumed and remove it from physics/rendering."""
        if self.semantic_world is not None and object_id in self.semantic_world.objects:
            semantic_object = self.semantic_world.object(object_id)
            semantic_object.attributes["consumed"] = True
            semantic_object.attributes["available"] = False
        self.set_object_visibility(object_id, False)

    def restore_office_object(self, object_id: str) -> None:
        """Restore a consumed object for a fresh replay or simulation day."""
        if self.semantic_world is not None and object_id in self.semantic_world.objects:
            semantic_object = self.semantic_world.object(object_id)
            semantic_object.attributes["consumed"] = False
            semantic_object.attributes["available"] = True
        self.set_object_visibility(object_id, True)

    def attach_object_to_gripper(self, object_id: str) -> None:
        """Keep a free object fixed to the current gripper-relative pose."""
        self.data_proxies.set_grasped_object(object_id)

    def release_grasped_object(self) -> None:
        """Release an object previously attached to the gripper."""
        self.data_proxies.set_grasped_object("")

    def request_grasp_metrics(self, object_id: str) -> None:
        """Ask the physics process to report finger contacts for an object."""
        self.data_proxies.set_grasp_validation_target(object_id)

    def pull_grasp_metrics(self) -> dict:
        """Return the latest object-to-gripper distance and finger contacts."""
        return self.data_proxies.get_grasp_metrics()

    def set_robot_motion_speed(self, speed: float) -> None:
        """Scale lift and arm controller response for accelerated demonstrations."""
        if speed <= 0:
            raise ValueError("Robot motion speed must be positive")
        self.data_proxies.set_robot_motion_speed(speed)

    def create_office_agent_runtime(
        self,
        config_path: str | None = None,
        *,
        seed: int | None = None,
        auto_plan: bool = True,
        embodied: bool = False,
    ) -> OfficeAgentRuntime:
        """Create the deterministic employee runtime for this semantic scene."""
        if self.semantic_world is None:
            raise ValueError("The current scene does not define a semantic world")
        if config_path is None:
            if self.semantic_world.source_path is None:
                raise ValueError("Cannot infer the office agent configuration path")
            config_path = str(self.semantic_world.source_path.with_name("office_agents.json"))
        # Load the roster before assembling the bridge: schema-v2 populations
        # must use their generated NPC IDs, whereas schema-v1 keeps legacy sites.
        self.agent_runtime = OfficeAgentRuntime.from_json(
            self.semantic_world,
            config_path,
            seed=seed,
            auto_plan=auto_plan,
        )
        if embodied:
            from stretch_mujoco.agents.simulation_bridge import create_mujoco_action_driver
            from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile

            trajectory_profile = (
                None
                if self.agent_runtime.trajectory_profile_path is None
                else NpcTrajectoryProfile.from_json(self.agent_runtime.trajectory_profile_path)
            )
            self.agent_runtime.action_driver = create_mujoco_action_driver(
                self,
                npc_ids=self.agent_runtime.population_npc_ids,
                trajectory_profile=trajectory_profile,
                agent_locations={
                    agent_id: agent.state.location
                    for agent_id, agent in self.agent_runtime.agents.items()
                },
                world=self.semantic_world,
                interaction_templates=self.agent_runtime.population_interaction_templates,
                interaction_station_allocator=getattr(
                    self.agent_runtime, "interaction_station_allocator", None
                ),
            )
            self.agent_runtime.interaction_driver = self.agent_runtime.action_driver
        return self.agent_runtime

    def is_reached_set_position(self, actuator: str | Actuators, position_tolerance: float = 0.05):
        """
        Checks if the joint has reached a previously commanded location.

        Only listens to the `move_to` command.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [
            Actuators.base_rotate,
            Actuators.base_translate,
            Actuators.left_wheel_vel,
            Actuators.right_wheel_vel,
        ]:
            raise NotImplementedError(f"Check joint reached is not supported for {actuator}.")

        move_command = self.data_proxies.get_command().move_to.get(actuator.name)

        if not move_command:
            click.secho(
                "Warning: Position check requested, but the joint was not commanded to move.",
                fg="yellow",
            )
            return True

        set_position = move_command.pos

        current_position = actuator.get_position(self.pull_status())

        return bool(np.isclose(current_position, set_position, atol=position_tolerance))

    def wait_until_at_setpoint(
        self, actuator: str | Actuators, timeout: float = 5.0, position_tolerance: float = 0.05
    ):
        """Blocks until the actuator reaches its previously set point."""
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        move_command = self.data_proxies.get_command().move_to.get(actuator.name)

        if not move_command:
            return True

        if not block_until_check_succeeds(
            wait_timeout=timeout,
            check=lambda: self.is_reached_set_position(
                actuator=actuator, position_tolerance=position_tolerance
            ),
            is_alive=self.is_running,
        ):
            pos = move_command.pos
            actual = actuator.get_position(self.pull_status())
            error = pos - actual
            click.secho(
                f"Timeout: Joint {actuator.name} did not reach {pos}. "
                f"Actual: {actual:.4f} Diff: {error * 100:.4f}cm",
                fg="red",
            )
            return False
        return True

    _last_movement_positions: dict[Actuators, float | tuple[float, float, float]] = {}

    def wait_while_is_moving(
        self,
        actuator: str | Actuators,
        timeout: float | None = 5.0,
        check_interval: float = 0.1,
        position_tolerance: float = 0.0005,
    ):
        """
        Checks position after a delay, and blocks if position has changed.
        If `timeout` is None, will block indefinitely.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        def check_if_moved():
            """Checks movement, returns True if movement is detected."""
            time.sleep(check_interval)

            if actuator in [
                Actuators.left_wheel_vel,
                Actuators.right_wheel_vel,
                Actuators.base_rotate,
                Actuators.base_translate,
            ]:
                current_position = actuator.get_position_relative(self.pull_status())
                if actuator == Actuators.left_wheel_vel or actuator == Actuators.base_translate:
                    current_position = current_position[0]
                elif actuator == Actuators.right_wheel_vel:
                    current_position = current_position[1]
                elif actuator == Actuators.base_rotate:
                    current_position = current_position[2]
            else:
                current_position = actuator.get_position(self.pull_status())

            if actuator not in self._last_movement_positions:
                self._last_movement_positions[actuator] = current_position
                return True

            last_position = self._last_movement_positions[actuator]

            is_moved = not np.isclose(current_position, last_position, atol=position_tolerance)

            self._last_movement_positions[actuator] = current_position

            return is_moved

        if not block_until_check_succeeds(
            wait_timeout=timeout,
            check=lambda: not check_if_moved(),
            is_alive=self.is_running,
        ):
            if timeout is not None:
                click.secho(
                    f"Timeout: Joint {actuator.name} is still moving after {timeout}.",
                    fg="red",
                )
            return False
        return True

    @require_connection
    def move_to(self, actuator: str | Actuators, pos: float) -> None:
        """
        Move the actuator to an absolute position.
        Args:
            actuator: string name of the actuator or Actuator enum instance
            pos: float, absolute position goal

        Use `wait_until_at_setpoint()` or `wait_while_is_moving()` to block until the joint reaches its location.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [
            Actuators.left_wheel_vel,
            Actuators.right_wheel_vel,
            Actuators.base_rotate,
            Actuators.base_translate,
        ]:
            raise Exception(
                f"Cannot set an absolute position for a continuous joint {actuator.name}"
            )

        with self._command_lock:
            command = self.data_proxies.get_command()
            command.set_move_to(CommandMove(actuator_name=actuator.name, pos=pos, trigger=True))

            self.data_proxies.set_command(command)

    @require_connection
    def move_by(self, actuator: str | Actuators, pos: float):
        """
        Move the actuator by a relative amount.
        Args:
            actuator: string name of the actuator or Actuator enum instance
            pos: float, position to increment by

        Use `wait_until_at_setpoint()` or `wait_while_is_moving()` to block until the joint reaches its location.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [Actuators.left_wheel_vel, Actuators.right_wheel_vel]:
            click.secho(
                f"Cannot set a position for a velocity joint {actuator.name}",
                fg="red",
            )
            raise Exception(
                f"Cannot set an absolute position for a continuous joint {actuator.name}"
            )

        with self._command_lock:
            command = self.data_proxies.get_command()

            command.set_move_by(
                # The server handles the relative-motion math, so pass pos directly.
                CommandMove(actuator_name=actuator.name, pos=pos, trigger=True)
            )

            self.data_proxies.set_command(command)

    @require_connection
    def set_base_velocity(self, v_linear: float, omega: float) -> None:
        """
        Set the base velocity of the robot
        Args:
            v_linear: float, linear velocity
            omega: float, angular velocity
        """

        with self._command_lock:
            command = self.data_proxies.get_command()
            command.set_base_velocity(
                CommandBaseVelocity(v_linear=v_linear, omega=omega, trigger=True)
            )

            self.data_proxies.set_command(command)

    @require_connection
    def add_world_frame(
        self,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Add a world frame to the simulator for visualization.
        Args:
            position: tuple of (x, y, z) coordinates in the world frame
            rotation: tuple of (x, y, z) angle in radians for the rotation around each axis
        """
        with self._command_lock:
            command = self.data_proxies.get_command()
            command.coordinate_frame_arrows_viz.append(
                CommandCoordinateFrameArrowsViz(position=position, rotation=rotation, trigger=True)
            )
            self.data_proxies.set_command(command)

    @require_connection
    def get_base_pose(self):
        """Get the se(2) base pose: x, y, and theta"""
        status = self.pull_status()
        return (status.base.x, status.base.y, status.base.theta)

    @require_connection
    def get_ee_pose(self) -> np.ndarray:
        return self.get_link_pose("link_grasp_center")

    @require_connection
    def get_link_pose(self, link_name: str) -> np.ndarray:
        """Pose of link in world frame"""
        status = self.pull_status()
        cfg = {
            "wrist_yaw": status.wrist_yaw.pos,
            "wrist_pitch": status.wrist_pitch.pos,
            "wrist_roll": status.wrist_roll.pos,
            "lift": status.lift.pos,
            "arm": status.arm.pos,
            "head_pan": status.head_pan.pos,
            "head_tilt": status.head_tilt.pos,
        }
        transform = self.urdf_model.get_transform(cfg, link_name)
        base_xyt = self.get_base_pose()
        base_4x4 = np.eye(4)
        base_4x4[:3, :3] = utils.Rz(base_xyt[2])
        base_4x4[:2, 3] = base_xyt[:2]
        world_coord = np.matmul(base_4x4, transform)
        return world_coord

    @require_connection
    def pull_camera_data(self) -> StatusStretchCameras:
        """
        Pull camera data from the simulator and return as a StatusStretchCameras
        """
        return self.data_proxies.get_cameras()

    @require_connection
    def pull_sensor_data(self) -> StatusStretchSensors:
        """
        Pull sensor data from the simulator and return as a StatusStretchSensors
        """
        return self.data_proxies.get_sensors()

    @require_connection
    def pull_semantic_state(self) -> dict:
        """Return the latest low-frequency semantic object and interaction poses."""
        return self.data_proxies.get_semantic_state()

    @require_connection
    def pull_status(self) -> StatusStretchJoints:
        """
        Pull robot joint states from the simulator and return as a StatusStretchJoints
        """
        return self.data_proxies.get_status()

    @require_connection
    def pull_joint_limits(self) -> dict[Actuators, tuple[float, float]]:
        """
        Pull robot joint limuts from the simulator and return as a dict
        """
        return self.data_proxies.get_joint_limits()

    def is_mujoco_process_dead_or_stopevent_triggered(self):
        return (
            self._server_process is None
            or not self._server_process.is_alive()
            or self._stop_mujoco_process_event.is_set()
        )

    def is_running(self) -> bool:
        """
        Check whether the simulator is running and its stop event has not fired.

        If the MuJoCo process has terminated or the stop event is set, this
        method also finalizes the client through :meth:`stop`.
        """
        if self.is_mujoco_process_dead_or_stopevent_triggered():
            # Send the signal to stop the program:
            self.stop()
            return False

        return not self.is_stop_called
