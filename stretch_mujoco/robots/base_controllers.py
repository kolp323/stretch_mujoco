"""Concrete base locomotion controllers.

- ``DiffDriveBaseController`` — differential-drive (Stretch 3)
- ``OmniPositionBaseController`` — omnidirectional position-controlled (Google Robot, TidyBot)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from stretch_mujoco import config as stretch_config
from stretch_mujoco.enums.actuators import Actuators as StretchActuators
from stretch_mujoco.robots.base import RobotBaseController
import stretch_mujoco.utils as utils

if TYPE_CHECKING:
    from stretch_mujoco.mujoco_server import MujocoServer
    from stretch_mujoco.datamodels.status_command import CommandBaseVelocity, CommandMove


class DiffDriveBaseController(RobotBaseController):
    """Differential-drive base using velocity-controlled left/right wheels.

    Extracted from the existing ``mujoco_server.BaseController`` and
    made reusable.
    """

    def __init__(self, mujoco_server: "MujocoServer") -> None:
        self._server = mujoco_server
        self._last_command: "CommandMove | CommandBaseVelocity | None" = None
        self._start_pose = np.array([0.0, 0.0, 0.0])

    # -- RobotBaseController interface -----------------------------------

    def push_command(self, command: "CommandMove | CommandBaseVelocity") -> None:
        self._last_command = command
        self._start_pose = self.get_base_pose()

    def update(self) -> None:
        if self._last_command is None:
            return
        from stretch_mujoco.datamodels.status_command import (
            CommandBaseVelocity,
            CommandMove,
        )

        if isinstance(self._last_command, CommandMove):
            self.handle_move_by(self._last_command)
        elif isinstance(self._last_command, CommandBaseVelocity):
            self._set_base_velocity(self._last_command.v_linear, self._last_command.omega)

    def get_base_pose(self) -> np.ndarray:
        xyz = self._server.mjdata.body("base_link").xpos
        rotation = self._server.mjdata.body("base_link").xmat.reshape(3, 3)
        theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.array([xyz[0], xyz[1], theta])

    def handle_move_by(self, command: "CommandMove") -> None:
        name = command.actuator_name
        if name == StretchActuators.base_translate.name:
            self._base_translate_by(command.pos)
        elif name == StretchActuators.base_rotate.name:
            self._base_rotate_by(command.pos)
        else:
            raise NotImplementedError(f"Diff-drive base does not support '{name}'")

    def stop(self) -> None:
        self._last_command = None
        self._set_base_velocity(0.0, 0.0)

    # -- internals -------------------------------------------------------

    def _set_base_velocity(self, v_linear: float, omega: float) -> None:
        w_left, w_right = utils.diff_drive_inv_kinematics(v_linear, omega)
        self._server.mjdata.actuator(StretchActuators.left_wheel_vel.name).ctrl = w_left
        self._server.mjdata.actuator(StretchActuators.right_wheel_vel.name).ctrl = w_right

    def _base_translate_by(self, x_inc: float) -> None:
        start_pose = self._start_pose[:2]
        sign = 1 if x_inc > 0 else -1
        if not np.linalg.norm(self.get_base_pose()[:2] - start_pose) <= abs(x_inc):
            self.stop()
            return
        self._set_base_velocity(stretch_config.base_motion["default_x_vel"] * sign, 0.0)

    def _base_rotate_by(self, theta_inc: float) -> None:
        start_pose = self._start_pose[-1]
        sign = 1 if theta_inc > 0 else -1
        if not abs(start_pose - self.get_base_pose()[-1]) <= abs(theta_inc):
            self.stop()
            return
        self._set_base_velocity(0.0, stretch_config.base_motion["default_r_vel"] * sign)

    def _clear_command(self, is_stop_motion: bool) -> None:
        self._last_command = None
        if is_stop_motion:
            self._set_base_velocity(0.0, 0.0)


class OmniPositionBaseController(RobotBaseController):
    """Omnidirectional base with three position-controlled planar joints.

    ``set_base_velocity(v, ω)`` is emulated by incrementing the x/y/θ
    position targets at each physics step.

    Parameters
    ----------
    mujoco_server:
        The owning ``MujocoServer`` instance.
    x_actuator_name:
        MuJoCo actuator name for the X slide joint (e.g. ``"base_x"``).
    y_actuator_name:
        MuJoCo actuator name for the Y slide joint.
    heading_actuator_name:
        MuJoCo actuator name for the heading hinge joint.
    max_linear_vel:
        Safety clamp for forward/backward speed [m/s].
    max_angular_vel:
        Safety clamp for rotational speed [rad/s].
    """

    def __init__(
        self,
        mujoco_server: "MujocoServer",
        *,
        x_actuator_name: str = "base_x",
        y_actuator_name: str = "base_y",
        heading_actuator_name: str = "base_theta",
        max_linear_vel: float = 2.0,
        max_angular_vel: float = 3.0,
    ) -> None:
        self._server = mujoco_server
        self._x_name = x_actuator_name
        self._y_name = y_actuator_name
        self._heading_name = heading_actuator_name
        self._max_linear = float(max_linear_vel)
        self._max_angular = float(max_angular_vel)
        self._last_v: float = 0.0
        self._last_omega: float = 0.0

    # -- RobotBaseController interface -----------------------------------

    def push_command(self, command: Any) -> None:
        from stretch_mujoco.datamodels.status_command import CommandBaseVelocity

        if isinstance(command, CommandBaseVelocity):
            self._last_v = float(np.clip(command.v_linear, -self._max_linear, self._max_linear))
            self._last_omega = float(np.clip(command.omega, -self._max_angular, self._max_angular))

    def update(self) -> None:
        if self._last_v == 0.0 and self._last_omega == 0.0:
            return
        dt = self._server.mjmodel.opt.timestep
        heading = float(self._server.mjdata.actuator(self._heading_name).length[0])
        dx = self._last_v * np.cos(heading) * dt
        dy = self._last_v * np.sin(heading) * dt
        dth = self._last_omega * dt

        self._server.mjdata.actuator(self._x_name).ctrl = (
            float(self._server.mjdata.actuator(self._x_name).length[0]) + dx
        )
        self._server.mjdata.actuator(self._y_name).ctrl = (
            float(self._server.mjdata.actuator(self._y_name).length[0]) + dy
        )
        self._server.mjdata.actuator(self._heading_name).ctrl = heading + dth

    def get_base_pose(self) -> np.ndarray:
        x = float(self._server.mjdata.actuator(self._x_name).length[0])
        y = float(self._server.mjdata.actuator(self._y_name).length[0])
        th = float(self._server.mjdata.actuator(self._heading_name).length[0])
        return np.array([x, y, th])

    def handle_move_by(self, command: Any) -> None:
        name = command.actuator_name
        if name == self._x_name:
            current = float(self._server.mjdata.actuator(self._x_name).length[0])
            self._server.mjdata.actuator(self._x_name).ctrl = current + command.pos
        elif name == self._y_name:
            current = float(self._server.mjdata.actuator(self._y_name).length[0])
            self._server.mjdata.actuator(self._y_name).ctrl = current + command.pos
        elif name == self._heading_name:
            current = float(self._server.mjdata.actuator(self._heading_name).length[0])
            self._server.mjdata.actuator(self._heading_name).ctrl = current + command.pos
        else:
            raise NotImplementedError(f"Omni base does not support move_by for '{name}'")

    def _set_base_velocity(self, v_linear: float, omega: float) -> None:
        self._last_v = float(v_linear)
        self._last_omega = float(omega)

    def stop(self) -> None:
        self._last_v = 0.0
        self._last_omega = 0.0
