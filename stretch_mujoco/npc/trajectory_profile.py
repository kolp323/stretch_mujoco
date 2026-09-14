"""Portable, preflightable NPC trajectory contracts for authored scenes.

Profiles own stable semantic anchors and permitted route pairs.  They never
persist world-coordinate waypoints: those are a projection of a particular
scene's current collision geometry and must be rebuilt during preflight and
runtime replanning.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco

from stretch_mujoco.humanoid.navigation import NavigationPathError, OfficeNavigationMesh

from .binding import NpcBinding
from .locomotion import LocomotionController


class TrajectoryProfileError(ValueError):
    """Raised when a trajectory profile is malformed or cannot be preflighted."""


@dataclass(frozen=True)
class TrajectoryAnchor:
    anchor_id: str
    site: str
    role: str


@dataclass(frozen=True)
class TrajectoryRoute:
    route_id: str
    source: str
    destination: str
    actions: tuple[str, ...]


@dataclass(frozen=True)
class PreflightRoute:
    route_id: str
    waypoints: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RouteClearanceAudit:
    """Continuous controller samples proving one route has no proxy contact."""

    route_id: str
    sampled_poses: int


@dataclass(frozen=True)
class NpcTrajectoryProfile:
    """One scene's declarative NPC navigation contract (schema version 1)."""

    profile_id: str
    scene: str
    scene_sha256: str
    anchors: dict[str, TrajectoryAnchor]
    routes: tuple[TrajectoryRoute, ...]
    source_path: Path | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "NpcTrajectoryProfile":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise TrajectoryProfileError("trajectory_profile_schema_version_must_be_1")
        profile_id = str(payload.get("profile_id", ""))
        scene = str(payload.get("scene", ""))
        if not profile_id or not scene:
            raise TrajectoryProfileError("trajectory_profile_requires_profile_id_and_scene")
        scene_sha256 = str(payload.get("scene_sha256", ""))
        if len(scene_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in scene_sha256
        ):
            raise TrajectoryProfileError("trajectory_profile_requires_scene_sha256")
        raw_anchors = payload.get("anchors")
        if not isinstance(raw_anchors, dict) or not raw_anchors:
            raise TrajectoryProfileError("trajectory_profile_requires_anchors")
        anchors: dict[str, TrajectoryAnchor] = {}
        for anchor_id, definition in raw_anchors.items():
            if not isinstance(definition, dict):
                raise TrajectoryProfileError(f"trajectory_anchor_invalid:{anchor_id}")
            site = str(definition.get("site", ""))
            role = str(definition.get("role", ""))
            if not anchor_id or not site or not role:
                raise TrajectoryProfileError(
                    f"trajectory_anchor_requires_site_and_role:{anchor_id}"
                )
            anchors[str(anchor_id)] = TrajectoryAnchor(str(anchor_id), site, role)
        raw_routes = payload.get("routes")
        if not isinstance(raw_routes, list) or not raw_routes:
            raise TrajectoryProfileError("trajectory_profile_requires_routes")
        routes: list[TrajectoryRoute] = []
        route_ids: set[str] = set()
        for definition in raw_routes:
            if not isinstance(definition, dict):
                raise TrajectoryProfileError("trajectory_route_invalid")
            route_id = str(definition.get("route_id", ""))
            source_anchor = str(definition.get("from", ""))
            destination = str(definition.get("to", ""))
            actions = definition.get("actions", [])
            if (
                not route_id
                or route_id in route_ids
                or source_anchor not in anchors
                or destination not in anchors
                or source_anchor == destination
                or not isinstance(actions, list)
                or not actions
                or not all(isinstance(action, str) and action for action in actions)
            ):
                raise TrajectoryProfileError(f"trajectory_route_invalid:{route_id or '<unnamed>'}")
            route_ids.add(route_id)
            routes.append(TrajectoryRoute(route_id, source_anchor, destination, tuple(actions)))
        return cls(profile_id, scene, scene_sha256, anchors, tuple(routes), source.resolve())

    def route_for(
        self, source_anchor: str | None, destination_site: str, action: str
    ) -> TrajectoryRoute | None:
        """Return the one declared route that authorizes this driver transition."""
        if source_anchor is None:
            return None
        matches = [
            route
            for route in self.routes
            if route.source == source_anchor
            and self.anchors[route.destination].site == destination_site
            and action in route.actions
        ]
        if len(matches) > 1:
            raise TrajectoryProfileError(
                f"trajectory_route_ambiguous:{source_anchor}:{destination_site}:{action}"
            )
        return matches[0] if matches else None

    def validate_scene(self, scene: str | Path) -> None:
        """Reject applying a profile to a different named scene asset."""
        expected = Path(self.scene).name
        actual = Path(scene).name
        if actual != expected:
            raise TrajectoryProfileError(f"trajectory_profile_scene_mismatch:{expected}!={actual}")
        actual_sha256 = hashlib.sha256(Path(scene).read_bytes()).hexdigest()
        if actual_sha256 != self.scene_sha256:
            raise TrajectoryProfileError("trajectory_profile_scene_digest_mismatch")

    def preflight(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[PreflightRoute, ...]:
        """Resolve every declared route against the current collision geometry.

        A profile is valid only when all anchor sites and an ``office_floor``
        navigation surface are present and every route has a collision-free path.
        Runtime is intentionally free to recompute these paths after furniture
        moves; the profile remains the stable source of route intent.
        """
        missing_sites = [
            anchor.site
            for anchor in self.anchors.values()
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, anchor.site) < 0
        ]
        if missing_sites:
            raise TrajectoryProfileError(
                "trajectory_anchor_site_missing:" + ",".join(sorted(missing_sites))
            )
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "office_floor") < 0:
            raise TrajectoryProfileError("trajectory_navigation_geometry_missing:office_floor")
        try:
            navigation = OfficeNavigationMesh.from_model(model, data)
        except NavigationPathError as error:
            raise TrajectoryProfileError(
                "trajectory_navigation_geometry_invalid:office_floor"
            ) from error
        compiled: list[PreflightRoute] = []
        for route in self.routes:
            source = self.anchors[route.source]
            destination = self.anchors[route.destination]
            source_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, source.site)
            destination_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, destination.site)
            try:
                path = navigation.plan(data.site_xpos[source_id], data.site_xpos[destination_id])
            except NavigationPathError as error:
                raise TrajectoryProfileError(
                    f"trajectory_route_unavailable:{route.route_id}"
                ) from error
            waypoints = tuple((float(point[0]), float(point[1])) for point in path)
            if len(waypoints) < 2:
                raise TrajectoryProfileError(f"trajectory_route_empty:{route.route_id}")
            compiled.append(PreflightRoute(route.route_id, waypoints))
        return tuple(compiled)

    def audit_npc_clearance(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        npc_id: str,
        *,
        sample_period: float = 0.01,
    ) -> tuple[RouteClearanceAudit, ...]:
        """Reject routes whose real move-and-turn poses touch a scene model.

        Navigation's inflated 2-D geometry prevents path crossings.  This
        complementary check drives the same ``LocomotionController`` used at
        runtime, including its gradual yaw updates, and asks MuJoCo for contacts
        between the NPC's collision proxies and all non-floor scene geometry.
        """
        if sample_period <= 0 or sample_period > 0.1:
            raise ValueError("trajectory clearance sample_period must be in (0, 0.1]")
        binding = NpcBinding.from_model(model, npc_id)
        if not binding.collision_geom_ids:
            raise TrajectoryProfileError(f"trajectory_npc_collision_proxies_missing:{npc_id}")
        original_position = data.mocap_pos[binding.mocap_id].copy()
        original_quaternion = data.mocap_quat[binding.mocap_id].copy()
        collision_ids = set(binding.collision_geom_ids)
        results: list[RouteClearanceAudit] = []
        try:
            for route in self.routes:
                source = self.anchors[route.source]
                source_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, source.site)
                root_position = original_position.copy()
                root_position[:2] = data.site_xpos[source_id, :2]
                data.mocap_pos[binding.mocap_id] = root_position
                mujoco.mju_mat2Quat(data.mocap_quat[binding.mocap_id], data.site_xmat[source_id])
                mujoco.mj_forward(model, data)
                controller = LocomotionController(model, binding, dynamic_obstacles=False)
                controller.move_to(self.anchors[route.destination].site, speed=1.0)
                for sample_index in range(10000):
                    completed = controller.step(data, sample_index * sample_period)
                    mujoco.mj_forward(model, data)
                    contacts = self._npc_scene_contacts(model, data, collision_ids)
                    if contacts:
                        raise TrajectoryProfileError(
                            f"trajectory_route_collision:{route.route_id}:{contacts[0]}"
                        )
                    if completed:
                        results.append(RouteClearanceAudit(route.route_id, sample_index + 1))
                        break
                    if controller.failure_reason is not None:
                        raise TrajectoryProfileError(
                            f"trajectory_route_unavailable:{route.route_id}"
                        )
                else:
                    raise TrajectoryProfileError(f"trajectory_route_timeout:{route.route_id}")
        finally:
            data.mocap_pos[binding.mocap_id] = original_position
            data.mocap_quat[binding.mocap_id] = original_quaternion
            mujoco.mj_forward(model, data)
        return tuple(results)

    @staticmethod
    def _npc_scene_contacts(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        collision_ids: set[int],
    ) -> tuple[str, ...]:
        contacts: set[str] = set()
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first = int(contact.geom1)
            second = int(contact.geom2)
            if first not in collision_ids and second not in collision_ids:
                continue
            other = second if first in collision_ids else first
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or str(other)
            if other_name == "office_floor" or other in collision_ids:
                continue
            npc = first if first in collision_ids else second
            npc_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, npc) or str(npc)
            contacts.add(f"{npc_name}->{other_name}")
        return tuple(sorted(contacts))
