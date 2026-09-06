import contextlib
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass
from multiprocessing.managers import DictProxy, SyncManager
from typing import Callable

import click
import mujoco
import mujoco._enums
import mujoco._functions
import numpy as np
from mujoco._structs import MjData, MjModel

import stretch_mujoco.config as config
import stretch_mujoco.utils as utils
from stretch_mujoco.datamodels.status_command import CommandBaseVelocity, CommandMove, StatusCommand
from stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch_mujoco.datamodels.status_stretch_joints import StatusStretchJoints
from stretch_mujoco.datamodels.status_stretch_sensors import StatusStretchSensors
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.enums.stretch_sensors import StretchSensors
from stretch_mujoco.mujoco_server_camera_manager import (
    MujocoServerCameraManagerSync,
    MujocoServerCameraManagerThreaded,
)
from stretch_mujoco.mujoco_server_sensor_manager import MujocoServerSensorManagerThreaded
from stretch_mujoco.npc import NpcCommand
from stretch_mujoco.npc.system import NpcSystem
from stretch_mujoco.semantics import SemanticWorld
from stretch_mujoco.utils import FpsCounter


@dataclass
class MujocoServerProxies:
    _command: "DictProxy[str, StatusCommand]"
    _status: "DictProxy[str, StatusStretchJoints]"
    _cameras: "DictProxy[str, StatusStretchCameras]"
    _sensors: "DictProxy[str, StatusStretchSensors]"
    _joint_limits: "DictProxy[str, dict[Actuators, tuple[float, float]]]"
    _object_visibility: "DictProxy[str, dict[str, bool]]"
    _grasped_object: "DictProxy[str, str]"
    _grasp_validation_target: "DictProxy[str, str]"
    _grasp_metrics: "DictProxy[str, dict]"
    _robot_motion_speed: "DictProxy[str, float]"
    _semantic_state: "DictProxy[str, dict]"
    _npc_command_queue: object
    _npc_states: "DictProxy[str, dict]"
    _npc_receipt_queue: object

    def __setattr__(self, name: str, value) -> None:
        try:
            super().__setattr__(name, value)
        except BrokenPipeError:
            ...

    def get_status(self) -> StatusStretchJoints:
        return self._status["val"]

    def set_status(self, value: StatusStretchJoints):
        self._status["val"] = value

    def get_command(self) -> StatusCommand:
        return self._command["val"]

    def set_command(self, value: StatusCommand):
        self._command["val"] = value

    def get_cameras(self) -> StatusStretchCameras:
        return self._cameras["val"]

    def set_cameras(self, value: StatusStretchCameras):
        self._cameras["val"] = value

    def get_sensors(self) -> StatusStretchSensors:
        return self._sensors["val"]

    def set_sensors(self, value: StatusStretchSensors):
        self._sensors["val"] = value

    def get_joint_limits(self) -> dict[Actuators, tuple[float, float]]:
        return self._joint_limits["val"]

    def set_joint_limit(self, actuator: Actuators, min_max: tuple[float, float]):
        limits = self._joint_limits["val"]
        limits[actuator] = min_max

        self._joint_limits["val"] = limits

    def get_object_visibility(self) -> dict[str, bool]:
        return dict(self._object_visibility["val"])

    def set_object_visibility(self, object_id: str, visible: bool) -> None:
        visibility = dict(self._object_visibility["val"])
        visibility[object_id] = bool(visible)
        self._object_visibility["val"] = visibility

    def get_grasped_object(self) -> str:
        return str(self._grasped_object["val"])

    def set_grasped_object(self, object_id: str) -> None:
        self._grasped_object["val"] = object_id

    def get_grasp_validation_target(self) -> str:
        return str(self._grasp_validation_target["val"])

    def set_grasp_validation_target(self, object_id: str) -> None:
        self._grasp_validation_target["val"] = object_id

    def get_grasp_metrics(self) -> dict:
        return dict(self._grasp_metrics["val"])

    def set_grasp_metrics(self, metrics: dict) -> None:
        self._grasp_metrics["val"] = metrics

    def get_robot_motion_speed(self) -> float:
        return float(self._robot_motion_speed["val"])

    def set_robot_motion_speed(self, speed: float) -> None:
        self._robot_motion_speed["val"] = float(speed)

    def get_semantic_state(self) -> dict:
        return self._semantic_state["val"]

    def set_semantic_state(self, state: dict) -> None:
        self._semantic_state["val"] = state

    def submit_npc_command(self, command: NpcCommand) -> None:
        self._npc_command_queue.put(command.to_dict())

    def get_pending_npc_command(self) -> dict | None:
        try:
            return self._npc_command_queue.get_nowait()
        except queue.Empty:
            return None

    def set_npc_states(self, states: dict[str, dict]) -> None:
        self._npc_states["val"] = states

    def get_npc_states(self) -> dict[str, dict]:
        return dict(self._npc_states["val"])

    def push_npc_receipt(self, receipt: dict) -> None:
        self._npc_receipt_queue.put(receipt)

    def drain_npc_receipts(self) -> tuple[dict, ...]:
        receipts = []
        while True:
            try:
                receipts.append(self._npc_receipt_queue.get_nowait())
            except queue.Empty:
                return tuple(receipts)

    @staticmethod
    def default(manager: SyncManager) -> "MujocoServerProxies":
        return MujocoServerProxies(
            _command=manager.dict({"val": StatusCommand.default()}),
            _status=manager.dict({"val": StatusStretchJoints.default()}),
            _cameras=manager.dict({"val": StatusStretchCameras.default()}),
            _sensors=manager.dict({"val": StatusStretchSensors.default()}),
            _joint_limits=manager.dict({"val": {}}),
            _object_visibility=manager.dict({"val": {}}),
            _grasped_object=manager.dict({"val": ""}),
            _grasp_validation_target=manager.dict({"val": ""}),
            _grasp_metrics=manager.dict({"val": {}}),
            _robot_motion_speed=manager.dict({"val": 1.0}),
            _semantic_state=manager.dict(
                {"val": {"time": 0.0, "objects": {}, "interaction_points": {}}}
            ),
            _npc_command_queue=manager.Queue(),
            _npc_states=manager.dict({"val": {}}),
            _npc_receipt_queue=manager.Queue(),
        )


class BaseController:
    def __init__(self, mujoco_server: "MujocoServer") -> None:
        self.mujoco_server = mujoco_server
        self.last_command: CommandMove | CommandBaseVelocity | None = None
        self.start_pose = np.array([0, 0, 0])

    def push_command(self, command: CommandMove | CommandBaseVelocity):
        """Push a command to the base. Call `update()` to set the next trajectory."""
        self.last_command = command
        self.start_pose = self.get_base_pose()

    def _clear_command(self, is_stop_motion: bool):
        self.last_command = None

        if is_stop_motion:
            self._set_base_velocity(0.0, 0.0)

    def update(self):
        """
        The update method to set mujoco ctrl's for the base while in motion.
        """
        if self.last_command is None:
            return

        if isinstance(self.last_command, CommandMove):
            return self.handle_move_by(self.last_command)

        if isinstance(self.last_command, CommandBaseVelocity):
            return self._set_base_velocity(self.last_command.v_linear, self.last_command.omega)

    def get_base_pose(self) -> np.ndarray:
        """Get the se(2) base pose: x, y, and theta"""
        xyz = self.mujoco_server.mjdata.body("base_link").xpos
        rotation = self.mujoco_server.mjdata.body("base_link").xmat.reshape(3, 3)
        theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.array([xyz[0], xyz[1], theta])

    def handle_move_by(self, command: CommandMove):
        if command.actuator_name == Actuators.base_translate.name:
            return self._base_translate_by(
                command.pos,
            )

        if command.actuator_name == Actuators.base_rotate.name:
            return self._base_rotate_by(
                command.pos,
            )

        raise NotImplementedError(f"Actuator {command.actuator_name} is not supported.")

    def _base_translate_by(self, x_inc: float) -> None:
        """
        Translate the base by a certain w.r.t base global pose
        """
        start_pose = self.start_pose[:2]

        sign = 1 if x_inc > 0 else -1
        if not np.linalg.norm(self.get_base_pose()[:2] - start_pose) <= abs(x_inc):
            return self._clear_command(is_stop_motion=True)

        self._set_base_velocity(config.base_motion["default_x_vel"] * sign, 0)

    def _base_rotate_by(self, theta_inc: float) -> None:
        """
        Rotate the base by a certain w.r.t base global pose
        """
        start_pose = self.start_pose[-1]
        sign = 1 if theta_inc > 0 else -1
        if not abs(start_pose - self.get_base_pose()[-1]) <= abs(theta_inc):
            return self._clear_command(is_stop_motion=True)

        self._set_base_velocity(0, config.base_motion["default_r_vel"] * sign)

    def _set_base_velocity(self, v_linear: float, omega: float) -> None:
        """
        Set the base velocity of the robot
        Args:
            v_linear: float, linear velocity
            omega: float, angular velocity
        """
        w_left, w_right = utils.diff_drive_inv_kinematics(v_linear, omega)
        model = self.mujoco_server.mjmodel
        data = self.mujoco_server.mjdata
        ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in (Actuators.left_wheel_vel.name, Actuators.right_wheel_vel.name)
        ]
        targets = np.array([w_left, w_right]) * model.actuator_gear[ids, 0]
        limits = np.max(np.abs(model.actuator_ctrlrange[ids]), axis=1)
        scale = max(1.0, float(np.max(np.abs(targets) / limits)))
        data.ctrl[ids] = targets / scale


class MujocoServer:
    """
    Use `MucocoServer.launch_server()` to start the headless simulator.

    This uses the mujoco simulator in headless mode.
    """

    @classmethod
    def launch_server(
        cls,
        scene_xml_path: str | None,
        model: MjModel | None,
        camera_hz: float,
        show_viewer_ui: bool,
        stop_mujoco_process_event: threading.Event,
        data_proxies: MujocoServerProxies,
        cameras_to_use: list[StretchCameras],
        start_translation: list | None,
        start_rotation_quat: list | None,
    ):
        server = cls(
            scene_xml_path,
            model,
            stop_mujoco_process_event,
            data_proxies,
            start_translation,
            start_rotation_quat,
        )
        server.run(
            show_viewer_ui=show_viewer_ui,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

    def change_start_pose(
        self, model: MjModel, translation: list | None, rotation_quat: list | None
    ):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        if body_id == -1:
            raise ValueError("Body 'base_link' not found in the MjModel.")

        joint_id = -1
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == body_id:
                joint_id = j
                break

        # Since the model has a Free Joint, we must change the default QPOS (qpos0).
        qadr = model.jnt_qposadr[joint_id]

        if translation is not None:
            model.qpos0[qadr : qadr + 3] = translation

        if rotation_quat is not None:
            model.qpos0[qadr + 3 : qadr + 7] = rotation_quat

        print(f"Start pose: {model.qpos0[qadr:qadr+3]}, {model.qpos0[qadr+3:qadr+7]}")

        return model

    def __init__(
        self,
        scene_xml_path: str | None,
        model: MjModel | None,
        stop_mujoco_process_event: threading.Event,
        data_proxies: MujocoServerProxies,
        start_translation: list | None,
        start_rotation_quat: list | None,
    ):
        """
        Initialize the Simulator handle with a scene
        Args:
            scene_xml_path: str, path to the scene xml file
            model: MjModel, Mujoco model object
        """
        if scene_xml_path is None:
            scene_xml_path = utils.default_scene_xml_path

        if model is None:
            model = MjModel.from_xml_path(scene_xml_path)

        model = self.change_start_pose(model, start_translation, start_rotation_quat)

        self.mjmodel = model

        # Wheel velocity controls target gear-scaled actuator velocity.  The bundled
        # [-6, 6] range only permits about 0.1 m/s with gear=3, despite the 0.3 m/s
        # configured base speed.
        for actuator_name in (Actuators.left_wheel_vel.name, Actuators.right_wheel_vel.name):
            actuator_id = mujoco.mj_name2id(
                self.mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            self.mjmodel.actuator_ctrlrange[actuator_id] = (-40.0, 40.0)

        self.mjdata = MjData(self.mjmodel)
        self.npc_system = NpcSystem.from_model(self.mjmodel)
        self._object_visibility_state: dict[str, bool] = {}
        self._object_geom_defaults: dict[int, tuple[float, int, int]] = {}
        self._grasp_attachment_object = ""
        self._grasp_attachment_transform: np.ndarray | None = None
        self._grasp_attachment_gravcomp: float | None = None
        self._robot_motion_speed = 1.0
        self._robot_actuator_defaults: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for actuator_name in ("lift", "arm", "gripper"):
            actuator_id = mujoco.mj_name2id(
                self.mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            self._robot_actuator_defaults[actuator_id] = (
                self.mjmodel.actuator_gainprm[actuator_id].copy(),
                self.mjmodel.actuator_biasprm[actuator_id].copy(),
                self.mjmodel.actuator_forcerange[actuator_id].copy(),
            )
        self.semantic_world = SemanticWorld.for_scene(scene_xml_path)
        self._next_semantic_update_time = 0.0
        if self.semantic_world is not None:
            self.semantic_world.validate_model(self.mjmodel)

        self._base_in_pos_motion = False

        self._stop_mujoco_process_event = stop_mujoco_process_event

        self.data_proxies = data_proxies

        self.base_controller = BaseController(self)

        self.physics_fps_counter = FpsCounter()

        self.sensor_manager = MujocoServerSensorManagerThreaded(
            sensor_hz=15,
            sensors_to_use=StretchSensors.from_mjmodel(self.mjmodel),
            mujoco_server=self,
        )

        self.update_joint_limits()

        signal.signal(signal.SIGTERM, lambda num, h: self.request_to_stop())
        signal.signal(signal.SIGINT, lambda num, h: self.request_to_stop())

    def update_joint_limits(self):
        for i in range(self.mjmodel.njnt):
            name = mujoco._functions.mj_id2name(self.mjmodel, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            joint_range = self.mjmodel.jnt_range[i]  # This gives [lower_limit, upper_limit]
            try:
                actuator = Actuators.get_actuator_by_joint_names_in_mjcf(name)
                self.data_proxies.set_joint_limit(
                    actuator=actuator, min_max=(joint_range[0], joint_range[1])
                )
            except:
                ...

    def set_camera_manager(
        self,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
        *,
        use_camera_thread: bool,
        use_threadpool_executor: bool,
    ):
        """
        This should be called before trying to render offscreen cameras.

        If `use_camera_thread` is false, `self.camera_manager.pull_camera_data_at_camera_rate()` should be called on a UI thread.
        This is the recommended usage.

        If `use_camera_thread` is true, a thread will be spawned to call Renderer.render().
        This may not work on all platforms since rendering should happen on the main thread.
        This mode is mainly used with the Mujoco Managed Viewer, to avoid rendering on the physics thread.
        """
        if use_camera_thread or use_threadpool_executor:
            self.camera_manager = MujocoServerCameraManagerThreaded(
                use_camera_thread=use_camera_thread,
                use_threadpool_executor=use_threadpool_executor,
                camera_hz=camera_hz,
                cameras_to_use=cameras_to_use,
                mujoco_server=self,
            )
        else:
            self.camera_manager = MujocoServerCameraManagerSync(
                camera_hz=camera_hz, cameras_to_use=cameras_to_use, mujoco_server=self
            )

    def run(
        self,
        show_viewer_ui: bool,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
    ):
        # self.__run_headless_simulation(camera_hz=camera_hz, cameras_to_use=cameras_to_use)
        self.__run_headless_simulation_with_physics_thread(
            camera_hz=camera_hz, cameras_to_use=cameras_to_use
        )

    def _is_requested_to_stop(self):
        try:
            return self._stop_mujoco_process_event.is_set()
        except (EOFError, BrokenPipeError):
            # We likely lost connection to the main process if we've hit this.
            return True

    def request_to_stop(self):
        try:
            self._stop_mujoco_process_event.set()
        except (EOFError, BrokenPipeError):
            # We likely lost connection to the main process if we've hit this.
            ...

    def close(self):
        """
        Clean up C++ resources
        """
        self.npc_system.fail_active_commands(float(self.mjdata.time), "simulator_restarted")
        for receipt in self.npc_system.drain_receipts():
            self.data_proxies.push_npc_receipt(receipt.to_dict())
        self.request_to_stop()

        if isinstance(self.camera_manager, MujocoServerCameraManagerThreaded):
            self.camera_manager.cameras_thread.join()

        if isinstance(self.sensor_manager, MujocoServerSensorManagerThreaded):
            self.sensor_manager.sensors_thread.join()

        self.camera_manager.close()

    def _run_ui_simulation(self, show_viewer_ui: bool) -> None:
        """
        Run the simulation with the viewer
        """
        raise NotImplementedError(
            "This is headless mode. Use MujocoServerPassive or MujocoServerManaged to run the UI simulator."
        )

    def _physics_step(self, lock: contextlib.AbstractContextManager):
        """
        Calls mj_step and _ctrl_callback, and sleeps until the next timestep.
        """
        start_time = time.perf_counter()

        with lock:
            mujoco._functions.mj_step(self.mjmodel, self.mjdata)
            self._ctrl_callback(self.mjmodel, self.mjdata)

        time_until_next_step = self.mjmodel.opt.timestep - (time.perf_counter() - start_time)
        if time_until_next_step > 0:
            # Sleep to match the timestep.
            time.sleep(time_until_next_step)

    def _physics_loop(
        self, lock: contextlib.AbstractContextManager, termination_check: Callable[[], bool]
    ):
        """
        A loop to use when starting physics in a thread.
        """
        while termination_check():
            self._physics_step(lock=lock)

        click.secho("Physics Loop has terminated.", fg="red")

    def __run_headless_simulation(
        self, camera_hz: float, cameras_to_use: list[StretchCameras]
    ) -> None:
        """
        Run the simulation without the viewer headless.

        Headless mode manages its own `set_camera_manager()` call.
        """
        print("Running headless simulation...")

        self.set_camera_manager(
            use_camera_thread=False,
            use_threadpool_executor=False,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

        while not self._is_requested_to_stop():
            self._physics_step(contextlib.nullcontext())
            self.camera_manager.pull_camera_data_at_camera_rate(is_sleep_until_ready=False)

        self.close()

    def __run_headless_simulation_with_physics_thread(
        self, camera_hz: float, cameras_to_use: list[StretchCameras]
    ) -> None:
        """
        Run the simulation without the viewer headless.

        Headless mode manages its own `set_camera_manager()` call.
        """
        print("Running headless simulation...")

        self.set_camera_manager(
            use_camera_thread=False,
            use_threadpool_executor=False,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

        physics_thread = threading.Thread(
            target=self._physics_loop,
            args=(self.camera_manager.camera_lock, lambda: not self._is_requested_to_stop()),
            daemon=True,
        )
        physics_thread.start()

        while not self._is_requested_to_stop():
            self.camera_manager.pull_camera_data_at_camera_rate(is_sleep_until_ready=True)

        physics_thread.join()
        self.close()

    def _ctrl_callback(self, model: MjModel, data: MjData) -> None:
        """
        Callback function that gets executed with mj_step
        """
        self.mjdata = data
        self.mjmodel = model

        if not self.mjdata or not self.mjdata.time:
            print("WARNING: no mujoco data to report")
            return

        self.physics_fps_counter.tick(sim_time=data.time)
        self._apply_robot_motion_speed()
        while True:
            command_payload = self.data_proxies.get_pending_npc_command()
            if command_payload is None:
                break
            self.npc_system.submit(NpcCommand.from_dict(command_payload))
        self.npc_system.step(model, data, float(data.time))
        self.data_proxies.set_npc_states(
            {npc_id: state.to_dict() for npc_id, state in self.npc_system.states(data).items()}
        )
        for receipt in self.npc_system.drain_receipts():
            self.data_proxies.push_npc_receipt(receipt.to_dict())
        self._apply_grasp_attachment(data)
        self._update_grasp_metrics(data)
        self._apply_object_visibility()
        if self.semantic_world is not None and data.time >= self._next_semantic_update_time:
            self.data_proxies.set_semantic_state(self.semantic_world.pose_snapshot(model, data))
            self._next_semantic_update_time = data.time + 0.2
        self.pull_status()
        self.push_command(self.data_proxies.get_command())

    def _apply_object_visibility(self) -> None:
        for object_id, visible in self.data_proxies.get_object_visibility().items():
            if self._object_visibility_state.get(object_id) == visible:
                continue
            body_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, object_id)
            if body_id < 0:
                continue
            for geom_id in range(self.mjmodel.ngeom):
                if int(self.mjmodel.geom_bodyid[geom_id]) != body_id:
                    continue
                defaults = self._object_geom_defaults.setdefault(
                    geom_id,
                    (
                        float(self.mjmodel.geom_rgba[geom_id, 3]),
                        int(self.mjmodel.geom_contype[geom_id]),
                        int(self.mjmodel.geom_conaffinity[geom_id]),
                    ),
                )
                self.mjmodel.geom_rgba[geom_id, 3] = defaults[0] if visible else 0.0
                self.mjmodel.geom_contype[geom_id] = defaults[1] if visible else 0
                self.mjmodel.geom_conaffinity[geom_id] = defaults[2] if visible else 0
            self._object_visibility_state[object_id] = visible

    def _apply_robot_motion_speed(self) -> None:
        speed = self.data_proxies.get_robot_motion_speed()
        if np.isclose(speed, self._robot_motion_speed):
            return
        self._robot_motion_speed = speed
        for actuator_id, defaults in self._robot_actuator_defaults.items():
            gain, bias, force_range = defaults
            self.mjmodel.actuator_gainprm[actuator_id] = gain * speed
            self.mjmodel.actuator_biasprm[actuator_id] = bias * speed
            self.mjmodel.actuator_forcerange[actuator_id] = force_range * speed

    def _apply_grasp_attachment(self, data: MjData) -> None:
        requested_object = self.data_proxies.get_grasped_object()
        if requested_object != self._grasp_attachment_object:
            if self._grasp_attachment_object:
                previous_id = mujoco.mj_name2id(
                    self.mjmodel,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self._grasp_attachment_object,
                )
                if previous_id >= 0:
                    self._set_attachment_physics(previous_id, enabled=True)
            self._grasp_attachment_object = requested_object
            self._grasp_attachment_transform = None
            self._grasp_attachment_gravcomp = None
            if requested_object:
                gripper_id = mujoco.mj_name2id(
                    self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, "link_grasp_center"
                )
                object_id = mujoco.mj_name2id(
                    self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, requested_object
                )
                if gripper_id < 0 or object_id < 0:
                    self._grasp_attachment_object = ""
                    self.data_proxies.set_grasped_object("")
                    return
                self._grasp_attachment_transform = np.linalg.inv(
                    self._body_pose(data, gripper_id)
                ) @ self._body_pose(data, object_id)
                self._set_attachment_physics(object_id, enabled=False)

        if not self._grasp_attachment_object or self._grasp_attachment_transform is None:
            return
        gripper_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, "link_grasp_center")
        object_id = mujoco.mj_name2id(
            self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, self._grasp_attachment_object
        )
        joint_id = int(self.mjmodel.body_jntadr[object_id])
        if joint_id < 0 or self.mjmodel.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            return
        target = self._body_pose(data, gripper_id) @ self._grasp_attachment_transform
        qpos_address = int(self.mjmodel.jnt_qposadr[joint_id])
        dof_address = int(self.mjmodel.jnt_dofadr[joint_id])
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, target[:3, :3].reshape(9))
        data.qpos[qpos_address : qpos_address + 3] = target[:3, 3]
        data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion
        data.qvel[dof_address : dof_address + 6] = 0.0

    def _set_attachment_physics(self, body_id: int, *, enabled: bool) -> None:
        if enabled:
            if self._grasp_attachment_gravcomp is not None:
                self.mjmodel.body_gravcomp[body_id] = self._grasp_attachment_gravcomp
        else:
            self._grasp_attachment_gravcomp = float(self.mjmodel.body_gravcomp[body_id])
            self.mjmodel.body_gravcomp[body_id] = 1.0
        for geom_id in range(self.mjmodel.ngeom):
            if int(self.mjmodel.geom_bodyid[geom_id]) != body_id:
                continue
            defaults = self._object_geom_defaults.setdefault(
                geom_id,
                (
                    float(self.mjmodel.geom_rgba[geom_id, 3]),
                    int(self.mjmodel.geom_contype[geom_id]),
                    int(self.mjmodel.geom_conaffinity[geom_id]),
                ),
            )
            self.mjmodel.geom_contype[geom_id] = defaults[1] if enabled else 0
            self.mjmodel.geom_conaffinity[geom_id] = defaults[2] if enabled else 0

    def _update_grasp_metrics(self, data: MjData) -> None:
        object_name = self.data_proxies.get_grasp_validation_target()
        if not object_name:
            return
        object_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, object_name)
        grasp_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, "link_grasp_center")
        left_id = mujoco.mj_name2id(
            self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, "link_gripper_finger_left"
        )
        right_id = mujoco.mj_name2id(
            self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, "link_gripper_finger_right"
        )
        if min(object_id, grasp_id, left_id, right_id) < 0:
            self.data_proxies.set_grasp_metrics(
                {"object_id": object_name, "error": "Missing grasp validation body"}
            )
            return

        left_contacts = 0
        right_contacts = 0
        minimum_contact_distance = None
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = int(self.mjmodel.geom_bodyid[contact.geom1])
            body2 = int(self.mjmodel.geom_bodyid[contact.geom2])
            if body1 == object_id:
                other_body = body2
            elif body2 == object_id:
                other_body = body1
            else:
                continue
            if self._body_is_descendant(other_body, left_id):
                left_contacts += 1
            if self._body_is_descendant(other_body, right_id):
                right_contacts += 1
            if left_contacts or right_contacts:
                distance = float(contact.dist)
                minimum_contact_distance = (
                    distance
                    if minimum_contact_distance is None
                    else min(minimum_contact_distance, distance)
                )

        center_distance = float(np.linalg.norm(data.xpos[object_id] - data.xpos[grasp_id]))
        self.data_proxies.set_grasp_metrics(
            {
                "time": float(data.time),
                "object_id": object_name,
                "center_distance_m": center_distance,
                "left_finger_contacts": left_contacts,
                "right_finger_contacts": right_contacts,
                "bilateral_contact": left_contacts > 0 and right_contacts > 0,
                "minimum_contact_distance_m": minimum_contact_distance,
                "object_position": data.xpos[object_id].tolist(),
                "grasp_center_position": data.xpos[grasp_id].tolist(),
            }
        )

    def _body_is_descendant(self, body_id: int, ancestor_id: int) -> bool:
        while body_id > 0:
            if body_id == ancestor_id:
                return True
            body_id = int(self.mjmodel.body_parentid[body_id])
        return False

    @staticmethod
    def _body_pose(data: MjData, body_id: int) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
        pose[:3, 3] = data.xpos[body_id]
        return pose

    def pull_status(self):
        """
        Pull joints status of the robot from the simulator
        """

        new_status = StatusStretchJoints.default()
        new_status.fps = self.physics_fps_counter.fps

        new_status.time = self.mjdata.time
        new_status.sim_to_real_time_ratio_msg = self.physics_fps_counter.sim_to_real_time_ratio_msg
        new_status.lift.pos = self.mjdata.actuator("lift").length[0]
        new_status.lift.vel = self.mjdata.actuator("lift").velocity[0]

        new_status.arm.pos = self.mjdata.actuator("arm").length[0]
        new_status.arm.vel = self.mjdata.actuator("arm").velocity[0]

        new_status.head_pan.pos = self.mjdata.actuator("head_pan").length[0]
        new_status.head_pan.vel = self.mjdata.actuator("head_pan").velocity[0]

        new_status.head_tilt.pos = self.mjdata.actuator("head_tilt").length[0]
        new_status.head_tilt.vel = self.mjdata.actuator("head_tilt").velocity[0]

        new_status.wrist_yaw.pos = self.mjdata.actuator("wrist_yaw").length[0]
        new_status.wrist_yaw.vel = self.mjdata.actuator("wrist_yaw").velocity[0]

        new_status.wrist_pitch.pos = self.mjdata.actuator("wrist_pitch").length[0]
        new_status.wrist_pitch.vel = self.mjdata.actuator("wrist_pitch").velocity[0]

        new_status.wrist_roll.pos = self.mjdata.actuator("wrist_roll").length[0]
        new_status.wrist_roll.vel = self.mjdata.actuator("wrist_roll").velocity[0]

        new_status.gripper.pos = self._to_real_gripper_range(
            self.mjdata.actuator("gripper").length[0]
        )
        new_status.gripper.vel = self.mjdata.actuator("gripper").velocity[
            0
        ]  # This is still in sim gripper range

        left_wheel_vel = self.mjdata.actuator("left_wheel_vel").velocity[0]
        right_wheel_vel = self.mjdata.actuator("right_wheel_vel").velocity[0]
        (
            new_status.base.x_vel,
            new_status.base.theta_vel,
        ) = utils.diff_drive_fwd_kinematics(left_wheel_vel, right_wheel_vel)
        (
            new_status.base.x,
            new_status.base.y,
            new_status.base.theta,
        ) = self.base_controller.get_base_pose()

        self.data_proxies.set_status(new_status)

    def _to_real_gripper_range(self, pos: float) -> float:
        """
        Map the gripper position to real gripper range
        """
        return utils.map_between_ranges(
            pos,
            config.robot_settings["sim_gripper_min_max"],
            config.robot_settings["gripper_min_max"],
        )

    def push_command(self, command_status: StatusCommand):
        """
        Handles setting mujoco ctrl properties to move joints.
        """
        # move_by
        for _, command in command_status.move_by.items():
            if command.trigger:
                command.trigger = False
                actuator_name = command.actuator_name
                pos = command.pos
                if actuator_name in (Actuators.base_translate.name, Actuators.base_rotate.name):
                    self.base_controller.push_command(command)
                else:
                    if actuator_name == Actuators.gripper.name:
                        current_value = self._to_real_gripper_range(
                            self.mjdata.actuator("gripper").length[0]
                        )
                        self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(
                            current_value + pos
                        )
                    else:
                        current_value = self.mjdata.actuator(actuator_name).length[0]
                        self.mjdata.actuator(actuator_name).ctrl = current_value + pos

        # move_to
        for _, command in command_status.move_to.items():
            if command.trigger:
                command.trigger = False
                actuator_name = command.actuator_name
                pos = command.pos
                if actuator_name == Actuators.gripper.name:
                    self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(pos)
                elif actuator_name in (Actuators.base_translate.name, Actuators.base_rotate.name):
                    raise NotImplementedError(
                        f"Cannot set move_to for {actuator_name}, which is a relative joint."
                    )
                else:
                    self.mjdata.actuator(actuator_name).ctrl = pos

        # set_base_velocity
        if command_status.base_velocity is not None and command_status.base_velocity.trigger:
            command_status.base_velocity.trigger = False
            self.base_controller.push_command(command_status.base_velocity)

        # keyframe
        if command_status.keyframe is not None and command_status.keyframe.trigger:
            command_status.keyframe.trigger = False
            self.mjdata.ctrl = self.mjmodel.keyframe(command_status.keyframe.name).ctrl

        self.base_controller.update()

        self.data_proxies.set_command(command_status)

    def _to_sim_gripper_range(self, pos: float) -> float:
        """
        Map the gripper position to sim gripper range
        """
        return utils.map_between_ranges(
            pos,
            config.robot_settings["gripper_min_max"],
            config.robot_settings["sim_gripper_min_max"],
        )
