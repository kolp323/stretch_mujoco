"""Physical object attachment owned by one NPC controller."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .binding import NpcBinding


@dataclass
class HeldObject:
    name: str
    body_id: int
    joint_id: int
    qpos_address: int
    dof_address: int
    gravcomp: float
    geom_collisions: dict[int, tuple[int, int]]


class AttachmentController:
    """Kinematically follows an NPC site while preserving reversible physics defaults."""

    def __init__(self, model: mujoco.MjModel, binding: NpcBinding) -> None:
        self.model = model
        self.binding = binding
        self.held: dict[str, HeldObject] = {}

    def validate_object(self, object_name: str) -> tuple[int, int]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id < 0:
            raise ValueError(f"unknown_object_body:{object_name}")
        joint_id = int(self.model.body_jntadr[body_id])
        if joint_id < 0 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"object_not_free_body:{object_name}")
        return body_id, joint_id

    def attach(self, object_name: str) -> None:
        if object_name in self.held:
            return
        body_id, joint_id = self.validate_object(object_name)
        collisions = {
            geom_id: (
                int(self.model.geom_contype[geom_id]),
                int(self.model.geom_conaffinity[geom_id]),
            )
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) == body_id
        }
        held = HeldObject(
            object_name,
            body_id,
            joint_id,
            int(self.model.jnt_qposadr[joint_id]),
            int(self.model.jnt_dofadr[joint_id]),
            float(self.model.body_gravcomp[body_id]),
            collisions,
        )
        self.model.body_gravcomp[body_id] = 1.0
        for geom_id in collisions:
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
        self.held[object_name] = held

    def detach(self, data: mujoco.MjData, object_name: str, site_name: str | None = None) -> None:
        held = self.held.pop(object_name, None)
        if held is None:
            raise ValueError(f"object_not_attached:{object_name}")
        if site_name:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                self.held[object_name] = held
                raise ValueError(f"unknown_detach_site:{site_name}")
            self._write_pose(data, held, data.site_xpos[site_id], data.site_xmat[site_id])
        self.model.body_gravcomp[held.body_id] = held.gravcomp
        for geom_id, (contype, conaffinity) in held.geom_collisions.items():
            self.model.geom_contype[geom_id] = contype
            self.model.geom_conaffinity[geom_id] = conaffinity

    def step(self, data: mujoco.MjData) -> None:
        if not self.held:
            return
        site_id = self.handover_site_id()
        position, rotation = self._site_pose(data, site_id)
        for held in self.held.values():
            self._write_pose(data, held, position, rotation)

    def is_attached(self, object_name: str) -> bool:
        return object_name in self.held

    @property
    def held_objects(self) -> tuple[str, ...]:
        return tuple(sorted(self.held))

    def handover_site_id(self) -> int:
        suffixes = ("__handover", "_handover_site")
        for name, site_id in self.binding.interaction_site_ids.items():
            if name.endswith(suffixes):
                return site_id
        raise ValueError(f"npc_missing_handover_site:{self.binding.npc_id}")

    def _site_pose(self, data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray]:
        if int(self.model.site_bodyid[site_id]) != self.binding.body_id:
            return data.site_xpos[site_id], data.site_xmat[site_id]
        root_rotation = np.empty(9)
        mujoco.mju_quat2Mat(root_rotation, data.mocap_quat[self.binding.mocap_id])
        root_matrix = root_rotation.reshape(3, 3)
        site_rotation = np.empty(9)
        mujoco.mju_quat2Mat(site_rotation, self.model.site_quat[site_id])
        position = (
            data.mocap_pos[self.binding.mocap_id] + root_matrix @ self.model.site_pos[site_id]
        )
        return position, root_matrix @ site_rotation.reshape(3, 3)

    @staticmethod
    def _write_pose(
        data: mujoco.MjData, held: HeldObject, position: np.ndarray, rotation: np.ndarray
    ) -> None:
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, np.asarray(rotation).reshape(9))
        data.qpos[held.qpos_address : held.qpos_address + 3] = position
        data.qpos[held.qpos_address + 3 : held.qpos_address + 7] = quaternion
        data.qvel[held.dof_address : held.dof_address + 6] = 0.0
