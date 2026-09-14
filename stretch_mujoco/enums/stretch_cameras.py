from typing import Callable

import mujoco
import numpy as np

from stretch_mujoco import config, utils
from stretch_mujoco.datamodels.camera import CameraCrop, CameraSettings  # noqa: F401 — re-export
from stretch_mujoco.robots.base import RobotCameras


class StretchCameras(RobotCameras):
    """
    An enum of the camera's available to the simulation.
    """

    cam_d405_rgb = 0
    cam_d405_depth = 1

    cam_d435i_rgb = 2
    cam_d435i_depth = 3

    cam_nav_rgb = 4
    office_overview_rgb = 5

    def get_render_params(self):
        return (self.camera_name_in_mjcf, self.name, self.post_processing_callback)

    @staticmethod
    def all() -> list["StretchCameras"]:
        """
        Returns all the available cameras
        """
        return [camera for camera in StretchCameras]

    @staticmethod
    def from_mjmodel(mjmodel: mujoco.MjModel) -> list["StretchCameras"]:
        """Return only cameras whose MJCF camera exists in ``mjmodel``."""
        return [
            camera
            for camera in StretchCameras
            if mujoco.mj_name2id(
                mjmodel,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera.camera_name_in_mjcf,
            )
            >= 0
        ]

    @staticmethod
    def none() -> list["StretchCameras"]:
        """
        Short-hand for not using any cameras.
        """
        return []

    @staticmethod
    def rgb() -> list["StretchCameras"]:
        """
        Returns the RGB camera's only
        """
        return [
            StretchCameras.cam_d405_rgb,
            StretchCameras.cam_d435i_rgb,
            StretchCameras.cam_nav_rgb,
            StretchCameras.office_overview_rgb,
        ]

    @staticmethod
    def depth() -> list["StretchCameras"]:
        """
        Returns the Depth camera's only
        """
        return [StretchCameras.cam_d405_depth, StretchCameras.cam_d435i_depth]

    @property
    def camera_name_in_mjcf(self) -> str:
        if self == StretchCameras.cam_d405_rgb:
            return "d405_rgb"
        if self == StretchCameras.cam_d405_depth:
            return "d405_depth"
        if self == StretchCameras.cam_d435i_rgb:
            return "d435i_camera_rgb"
        if self == StretchCameras.cam_d435i_depth:
            return "d435i_camera_depth"
        if self == StretchCameras.cam_nav_rgb:
            return "nav_camera_rgb"
        if self == StretchCameras.office_overview_rgb:
            return "office_overview"

        raise NotImplementedError(f"Camera {self} camera_name_in_mjcf is not implemented")

    @property
    def is_rgb(self) -> bool:
        return not self.is_depth

    @property
    def is_depth(self) -> bool:
        if self == StretchCameras.cam_d405_depth or self == StretchCameras.cam_d435i_depth:
            return True
        if (
            self == StretchCameras.cam_d405_rgb
            or self == StretchCameras.cam_d435i_rgb
            or self == StretchCameras.cam_nav_rgb
            or self == StretchCameras.office_overview_rgb
        ):
            return False

        raise NotImplementedError(f"Camera {self} is_depth is not implemented")

    @property
    def post_processing_callback(self) -> Callable[[np.ndarray], np.ndarray] | None:

        if self == StretchCameras.cam_d405_depth:
            return lambda render: utils.limit_depth_distance(render, config.depth_limits["d405"])

        if self == StretchCameras.cam_d435i_depth:
            return lambda render: utils.limit_depth_distance(render, config.depth_limits["d435i"])

        if (
            self == StretchCameras.cam_d405_rgb
            or self == StretchCameras.cam_d435i_rgb
            or self == StretchCameras.cam_nav_rgb
            or self == StretchCameras.office_overview_rgb
        ):
            return None

        raise NotImplementedError(f"Camera {self} post_processing_callback is not implemented")

    @property
    def initial_camera_settings(self):

        if self == StretchCameras.cam_d405_rgb:
            return CameraSettings(
                field_of_view_vertical_in_degrees=58,  # from spec
                focal=(242.56, 242.34),  # from calibration on SE3-3044
                width=480,  # from webteleop
                height=270,  # from webteleop
                crop=CameraCrop(y_min=0, y_max=270, x_min=125, x_max=395),  # from webteleop
                sensor_resolution=(1280, 720),  # from ov9782 spec
                # sensor_pixel_size_micrometers=3.0 # from ov9782 spec
            )

        if self == StretchCameras.cam_d405_depth:
            # Stereo camera, we just use a depth camera camera in mujoco:
            return StretchCameras.cam_d405_rgb.initial_camera_settings

        if self == StretchCameras.cam_d435i_rgb:
            return CameraSettings(
                field_of_view_vertical_in_degrees=42,  # from spec
                focal=(304.24, 304.07),  # from calibration on SE3-3044
                width=424,  # from webteleop
                height=240,  # from webteleop
                sensor_resolution=(1920, 1080),  # from ov2740 spec
                # sensor_pixel_size_micrometers=1.4 # from ov2740 spec
            )

        if self == StretchCameras.cam_d435i_depth:
            return StretchCameras.cam_d435i_rgb.initial_camera_settings
            # TODO: To use these values, depth disparity must be corrected:
        #     return CameraSettings(
        #         field_of_view_vertical_in_degrees=58,  # 58 from spec
        #         focal=(212.31, 212.31),  # from calibration on SE3-3044
        #         width=424,  # from webteleop
        #         height=240,  # from webteleop
        #     )

        if self == StretchCameras.cam_nav_rgb:
            # Arducam B0385 - 70 degrees FOV-X from spec, converted to FOV-Y by field_of_view_vertical_from_horizontal()
            field_of_view_vertical_in_degrees = (
                CameraSettings.field_of_view_vertical_from_horizontal(70, 1280, 720)
            )
            return CameraSettings(
                field_of_view_vertical_in_degrees=field_of_view_vertical_in_degrees,
                focal=(0.0, 0.0),  # TODO We don't have calibrated values.
                width=800,  # from webteleop
                height=600,  # from webteleop
                sensor_resolution=(1280, 720),  # from ov9782 spec
                # sensor_pixel_size_micrometers=3.0 # from ov9782 spec, note: enabling this will not work with 0 `focal`
            )

        if self == StretchCameras.office_overview_rgb:
            return CameraSettings(
                field_of_view_vertical_in_degrees=55,
                focal=(0.0, 0.0),
                width=960,
                height=540,
                sensor_resolution=(960, 540),
            )

        raise NotImplementedError(f"Camera {self} initial settings are not implemented")
