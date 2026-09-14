"""Constrained numerical IK for GraspGen candidates, including base translation.

Replicates the real Stretch robot's manipulation IK (``stretch.motion.kinematics``
``manip_ik``): the six-DOF configuration ``[base_x, lift, arm, wrist_yaw,
wrist_pitch, wrist_roll]`` is solved, so the base-forward translation participates
in the solve and the robot can position the gripper laterally by driving the base
instead of contorting the wrist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from stretch_mujoco.graspgen.calibration import world_to_base_pose
from stretch_mujoco.utils import URDFmodel

# Base-forward travel bounds (meters), sized for the small table workspace.
BASE_X_LIMITS = (-0.15, 0.15)
MAX_POSITION_ERROR = 0.01

# Joint bounds for [base_x, lift, arm, wrist_yaw, wrist_pitch, wrist_roll].
JOINT_LOWER = np.array([BASE_X_LIMITS[0], 0.0, 0.0, -1.39, -1.57, -3.14])
JOINT_UPPER = np.array([BASE_X_LIMITS[1], 1.1, 0.52, 4.42, 0.56, 3.14])

DEFAULT_INITIAL = np.array([0.0, 0.7, 0.4, 0.0, -0.5, 0.0])


@dataclass(frozen=True)
class GraspIKCandidate:
    index: int
    joints: np.ndarray
    target_pose: np.ndarray
    position_error: float
    orientation_error: float
    confidence: float

    @property
    def score(self) -> float:
        task_error = (
            self.position_error * 100.0 + self.orientation_error * 0.2 - self.confidence * 0.05
        )
        # A parallel-jaw gripper is symmetric under a 180 degree rotation about its
        # own approach axis, so callers typically also IK-solve each candidate's
        # ``pose @ diag([-1, -1, 1, 1])`` twin (same grasp, wrist rolled by pi) to
        # give the solver more reachable options. Both twins land on nearly
        # identical position/orientation error, so without a roll term here the
        # twin that wins is essentially numerical noise -- which silently picks
        # the "upside down" wrist_roll ~= +-pi solution as often as the natural
        # wrist_roll ~= 0 one. Penalizing |wrist_roll| breaks that tie toward the
        # smaller, more natural rotation from the (wrist_roll = 0) rest posture,
        # while staying far too small to override a genuine task_error difference.
        wrist_posture = (
            abs(float(self.joints[3])) * 0.02
            + abs(float(self.joints[4])) * 0.01
            + abs(float(self.joints[5])) * 0.01
        )
        return task_error + wrist_posture


class StretchGraspIK:
    def __init__(self, urdf_model: URDFmodel | None = None) -> None:
        self.urdf_model = urdf_model or URDFmodel()

    @staticmethod
    def _configuration(joints: np.ndarray) -> dict[str, float]:
        return {
            "base_x": float(joints[0]),
            "lift": float(joints[1]),
            "arm": float(joints[2]),
            "wrist_yaw": float(joints[3]),
            "wrist_pitch": float(joints[4]),
            "wrist_roll": float(joints[5]),
            "head_pan": 0.0,
            "head_tilt": 0.0,
        }

    def forward(self, joints: np.ndarray) -> np.ndarray:
        return self.urdf_model.get_transform(
            self._configuration(np.asarray(joints, dtype=float)), "link_grasp_center"
        )

    def solve(
        self,
        target_pose: np.ndarray,
        *,
        initial: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Solve for ``[base_x, lift, arm, wrist_yaw, wrist_pitch, wrist_roll]`` so the
        gripper reaches ``target_pose`` (in the base frame). ``base_x`` is the forward
        displacement of the base relative to its current pose."""
        target = np.asarray(target_pose, dtype=float)
        if target.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 target pose, got {target.shape}")
        initial = DEFAULT_INITIAL if initial is None else np.asarray(initial, dtype=float)

        def residual(joints: np.ndarray) -> np.ndarray:
            actual = self.forward(joints)
            position = (actual[:3, 3] - target[:3, 3]) * 20.0
            orientation = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
            return np.r_[position, orientation * 0.3]

        result = least_squares(
            residual,
            x0=np.clip(initial, JOINT_LOWER, JOINT_UPPER),
            bounds=(JOINT_LOWER, JOINT_UPPER),
            max_nfev=500,
        )
        actual = self.forward(result.x)
        position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        orientation_error = float(
            Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()
        )
        return result.x, position_error, orientation_error

    def solve_ik(
        self,
        pos: np.ndarray,
        quat: np.ndarray | None = None,
        initial: np.ndarray | None = None,
        base_pose: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    ) -> np.ndarray | None:
        """Solve for ``[base_x, lift, arm, wrist_yaw, wrist_pitch, wrist_roll]`` to
        place the gripper at the world-frame pose ``(pos, quat)``, allowing the base
        to translate forward/backward. Returns None if the pose is not reachable.

        Mirrors ``HomeRobotZmqClient.solve_ik``: the desired world pose is mapped into
        the current base frame before solving, and the solved ``base_x`` is the base
        displacement the robot must drive.
        """
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (3,):
            raise ValueError(f"Expected a 3D position, got {pos.shape}")
        initial = DEFAULT_INITIAL if initial is None else np.asarray(initial, dtype=float)
        if quat is None:
            # Keep the orientation of the initial configuration's end effector.
            current = self.forward(initial)
            quat = Rotation.from_matrix(current[:3, :3]).as_quat()
        target = np.eye(4)
        target[:3, :3] = Rotation.from_quat(np.asarray(quat, dtype=float)).as_matrix()
        target[:3, 3] = pos
        target_in_base = world_to_base_pose(base_pose) @ target
        joints, position_error, orientation_error = self.solve(target_in_base, initial=initial)
        if position_error > MAX_POSITION_ERROR or orientation_error > 0.05:
            return None
        return joints

    def rank_candidates(
        self,
        target_poses: np.ndarray,
        confidences: np.ndarray,
        *,
        max_position_error: float = 0.015,
    ) -> list[GraspIKCandidate]:
        ranked: list[GraspIKCandidate] = []
        for index, (pose, confidence) in enumerate(zip(target_poses, confidences)):
            joints, position_error, orientation_error = self.solve(pose)
            if position_error <= max_position_error:
                ranked.append(
                    GraspIKCandidate(
                        index=index,
                        joints=joints,
                        target_pose=np.asarray(pose),
                        position_error=position_error,
                        orientation_error=orientation_error,
                        confidence=float(confidence),
                    )
                )
        return sorted(ranked, key=lambda candidate: candidate.score)
