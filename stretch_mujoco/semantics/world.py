"""Typed semantic world graph bound to MuJoCo entities and sites."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np


class ObjectType(str, Enum):
    EMPLOYEE = "Employee"
    STRETCH_ROBOT = "StretchRobot"
    WORKSTATION = "Workstation"
    CHAIR = "Chair"
    COMPUTER = "Computer"
    DOCUMENT = "Document"
    STORAGE_CABINET = "StorageCabinet"
    SNACK = "Snack"
    MEETING_TABLE = "MeetingTable"
    COFFEE_MACHINE = "CoffeeMachine"
    DOOR = "Door"
    COUNTER = "Counter"


class RelationType(str, Enum):
    INSIDE = "INSIDE"
    ON = "ON"
    NEAR = "NEAR"
    HOLDS = "HOLDS"
    BELONGS_TO = "BELONGS_TO"
    OCCUPIED_BY = "OCCUPIED_BY"
    REQUESTED_BY = "REQUESTED_BY"
    RESERVED_BY = "RESERVED_BY"
    ALLOWED_FOR = "ALLOWED_FOR"


class InteractionRole(str, Enum):
    CHAIR_SIT = "chair_sit_site"
    DESK_WORK = "desk_work_site"
    CABINET_OPEN = "cabinet_open_site"
    DOCUMENT_GRASP = "document_grasp_site"
    SNACK_PLACE = "snack_place_site"
    HANDOVER = "handover_site"
    MEETING_PLACE = "meeting_place_site"
    COFFEE_USE = "coffee_use_site"
    DOOR_HANDLE = "door_handle_site"
    HUMAN_STAND = "human_stand_site"


class BindingKind(str, Enum):
    BODY = "body"
    GEOM = "geom"
    JOINT = "joint"
    SITE = "site"


@dataclass(frozen=True)
class SemanticBinding:
    kind: BindingKind
    name: str


@dataclass
class SemanticObject:
    object_id: str
    object_type: ObjectType
    binding: SemanticBinding
    attributes: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        return self.attributes.get(name, default)


@dataclass(frozen=True)
class SemanticRelation:
    subject: str
    relation: RelationType
    object: str


@dataclass(frozen=True)
class InteractionPoint:
    point_id: str
    role: InteractionRole
    owner: str
    site: str
    attributes: dict[str, Any] = field(default_factory=dict)


class SemanticValidationError(ValueError):
    """Raised when semantic data does not match its graph or MuJoCo model."""


_MJ_OBJECT_TYPES = {
    BindingKind.BODY: mujoco.mjtObj.mjOBJ_BODY,
    BindingKind.GEOM: mujoco.mjtObj.mjOBJ_GEOM,
    BindingKind.JOINT: mujoco.mjtObj.mjOBJ_JOINT,
    BindingKind.SITE: mujoco.mjtObj.mjOBJ_SITE,
}


def _enum_value(enum_type: type[Enum], value: str) -> Enum:
    try:
        return enum_type(value)
    except ValueError as error:
        valid = ", ".join(item.value for item in enum_type)
        raise SemanticValidationError(
            f"Unknown {enum_type.__name__} value '{value}'. Available: {valid}"
        ) from error


class SemanticWorld:
    """Mutable relation graph with immutable object and interaction definitions."""

    def __init__(
        self,
        objects: dict[str, SemanticObject],
        relations: Iterable[SemanticRelation],
        interaction_points: dict[str, InteractionPoint],
        *,
        scene: str | None = None,
        source_path: Path | None = None,
    ) -> None:
        self.objects = objects
        self.relations = set(relations)
        self.interaction_points = interaction_points
        self.scene = scene
        self.source_path = source_path
        self._validate_graph()

    @classmethod
    def from_json(cls, path: str | Path) -> "SemanticWorld":
        source_path = Path(path).resolve()
        payload = json.loads(source_path.read_text(encoding="utf-8"))

        objects: dict[str, SemanticObject] = {}
        for object_id, definition in payload.get("objects", {}).items():
            binding = definition.get("binding", {})
            objects[object_id] = SemanticObject(
                object_id=object_id,
                object_type=_enum_value(ObjectType, definition["type"]),
                binding=SemanticBinding(
                    kind=_enum_value(BindingKind, binding["kind"]),
                    name=binding["name"],
                ),
                attributes=dict(definition.get("attributes", {})),
            )

        relations = [
            SemanticRelation(
                subject=item["subject"],
                relation=_enum_value(RelationType, item["relation"]),
                object=item["object"],
            )
            for item in payload.get("relations", [])
        ]
        interaction_points = {
            point_id: InteractionPoint(
                point_id=point_id,
                role=_enum_value(InteractionRole, definition["role"]),
                owner=definition["owner"],
                site=definition["site"],
                attributes=dict(definition.get("attributes", {})),
            )
            for point_id, definition in payload.get("interaction_points", {}).items()
        }
        return cls(
            objects,
            relations,
            interaction_points,
            scene=payload.get("scene"),
            source_path=source_path,
        )

    @classmethod
    def for_scene(cls, scene_xml_path: str | Path | None) -> "SemanticWorld | None":
        if scene_xml_path is None:
            return None
        scene_path = Path(scene_xml_path).resolve()
        candidates = (
            scene_path.with_suffix(".semantic.json"),
            scene_path.with_name(f"{scene_path.stem.removesuffix('_scene')}_semantics.json"),
        )
        for candidate in candidates:
            if candidate.exists():
                return cls.from_json(candidate)
        return None

    def _validate_graph(self) -> None:
        errors: list[str] = []
        for object_id, semantic_object in self.objects.items():
            if object_id != semantic_object.object_id:
                errors.append(f"Object key '{object_id}' does not match its object_id")
            for attribute_name in ("location", "owner"):
                reference = semantic_object.attributes.get(attribute_name)
                if reference is not None and reference not in self.objects:
                    errors.append(
                        f"Object '{object_id}' has unknown {attribute_name} '{reference}'"
                    )
        for relation in self.relations:
            if relation.subject not in self.objects:
                errors.append(f"Relation subject '{relation.subject}' is not registered")
            if relation.object not in self.objects:
                errors.append(f"Relation object '{relation.object}' is not registered")
        for point in self.interaction_points.values():
            if point.owner not in self.objects:
                errors.append(
                    f"Interaction point '{point.point_id}' has unknown owner '{point.owner}'"
                )
        for object_id, semantic_object in self.objects.items():
            location = semantic_object.attributes.get("location")
            if location is not None and not any(
                item.subject == object_id
                and item.object == location
                and item.relation in {RelationType.INSIDE, RelationType.ON}
                for item in self.relations
            ):
                errors.append(f"Object '{object_id}' location attribute has no matching relation")
            owner = semantic_object.attributes.get("owner")
            if (
                owner is not None
                and SemanticRelation(object_id, RelationType.BELONGS_TO, owner)
                not in self.relations
            ):
                errors.append(f"Object '{object_id}' owner attribute has no matching relation")
        if errors:
            raise SemanticValidationError("; ".join(errors))

    def register_population_npcs(self, npc_ids: Iterable[str]) -> None:
        """Bind schema-v2 NPC identities to the generated MuJoCo body contract.

        Population bodies are generated per scene and therefore must not be
        permanently declared in the legacy two-employee office semantics file.
        Registering them while loading a schema-v2 population makes the semantic
        graph use the same canonical IDs and handover sites as the scene builder.
        """
        for npc_id in npc_ids:
            body = f"npc__{npc_id}"
            handover_site = f"npc__{npc_id}__handover"
            existing = self.objects.get(npc_id)
            if existing is not None:
                if (
                    existing.object_type != ObjectType.EMPLOYEE
                    or existing.binding != SemanticBinding(BindingKind.BODY, body)
                ):
                    raise SemanticValidationError(
                        f"NPC '{npc_id}' conflicts with an existing semantic binding"
                    )
            else:
                self.objects[npc_id] = SemanticObject(
                    object_id=npc_id,
                    object_type=ObjectType.EMPLOYEE,
                    binding=SemanticBinding(BindingKind.BODY, body),
                )
            point_id = f"{npc_id}_handover"
            existing_point = self.interaction_points.get(point_id)
            point = InteractionPoint(
                point_id=point_id,
                role=InteractionRole.HANDOVER,
                owner=npc_id,
                site=handover_site,
            )
            if existing_point is not None and existing_point != point:
                raise SemanticValidationError(
                    f"NPC '{npc_id}' conflicts with an existing handover interaction point"
                )
            self.interaction_points[point_id] = point
        self._validate_graph()

    def validate_model(self, model: mujoco.MjModel) -> None:
        errors: list[str] = []
        for semantic_object in self.objects.values():
            binding = semantic_object.binding
            if mujoco.mj_name2id(model, _MJ_OBJECT_TYPES[binding.kind], binding.name) < 0:
                errors.append(
                    f"{semantic_object.object_id} binds missing {binding.kind.value} "
                    f"'{binding.name}'"
                )
        for point in self.interaction_points.values():
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, point.site) < 0:
                errors.append(f"{point.point_id} binds missing site '{point.site}'")
        if errors:
            raise SemanticValidationError("; ".join(errors))

    def object(self, object_id: str) -> SemanticObject:
        try:
            return self.objects[object_id]
        except KeyError as error:
            raise KeyError(f"Unknown semantic object '{object_id}'") from error

    def objects_of_type(self, object_type: ObjectType | str) -> tuple[SemanticObject, ...]:
        resolved_type = (
            object_type
            if isinstance(object_type, ObjectType)
            else _enum_value(ObjectType, object_type)
        )
        return tuple(obj for obj in self.objects.values() if obj.object_type == resolved_type)

    def find_relations(
        self,
        *,
        subject: str | None = None,
        relation: RelationType | str | None = None,
        object_id: str | None = None,
    ) -> tuple[SemanticRelation, ...]:
        resolved_relation = relation
        if isinstance(relation, str):
            resolved_relation = _enum_value(RelationType, relation)
        matches = (
            item
            for item in self.relations
            if (subject is None or item.subject == subject)
            and (resolved_relation is None or item.relation == resolved_relation)
            and (object_id is None or item.object == object_id)
        )
        return tuple(
            sorted(
                matches,
                key=lambda item: (item.subject, item.relation.value, item.object),
            )
        )

    def related_objects(
        self, subject: str, relation: RelationType | str
    ) -> tuple[SemanticObject, ...]:
        return tuple(
            self.objects[item.object]
            for item in self.find_relations(subject=subject, relation=relation)
        )

    def location_of(self, object_id: str) -> SemanticObject | None:
        self.object(object_id)
        locations = tuple(
            item
            for item in self.relations
            if item.subject == object_id and item.relation in {RelationType.INSIDE, RelationType.ON}
        )
        if len(locations) > 1:
            raise SemanticValidationError(
                f"Object '{object_id}' has more than one semantic location"
            )
        return self.objects[locations[0].object] if locations else None

    def pending_requests(self, requester: str | None = None) -> tuple[SemanticObject, ...]:
        requests = self.find_relations(
            relation=RelationType.REQUESTED_BY,
            object_id=requester,
        )
        return tuple(self.objects[item.subject] for item in requests)

    def can_access(self, actor_id: str, object_id: str) -> bool:
        self.object(actor_id)
        semantic_object = self.object(object_id)
        if not semantic_object.get("confidential", False):
            return True
        return bool(
            self.find_relations(
                subject=object_id,
                relation=RelationType.ALLOWED_FOR,
                object_id=actor_id,
            )
        )

    def add_relation(
        self, subject: str, relation: RelationType | str, object_id: str
    ) -> SemanticRelation:
        self.object(subject)
        self.object(object_id)
        resolved_relation = (
            relation if isinstance(relation, RelationType) else _enum_value(RelationType, relation)
        )
        item = SemanticRelation(subject, resolved_relation, object_id)
        self.relations.add(item)
        return item

    def remove_relation(self, subject: str, relation: RelationType | str, object_id: str) -> bool:
        resolved_relation = (
            relation if isinstance(relation, RelationType) else _enum_value(RelationType, relation)
        )
        item = SemanticRelation(subject, resolved_relation, object_id)
        if item not in self.relations:
            return False
        self.relations.remove(item)
        return True

    def replace_relation(
        self, subject: str, relation: RelationType | str, object_id: str
    ) -> SemanticRelation:
        resolved_relation = (
            relation if isinstance(relation, RelationType) else _enum_value(RelationType, relation)
        )
        self.relations = {
            item
            for item in self.relations
            if not (item.subject == subject and item.relation == resolved_relation)
        }
        item = self.add_relation(subject, resolved_relation, object_id)
        if resolved_relation == RelationType.BELONGS_TO:
            self.objects[subject].attributes["owner"] = object_id
        return item

    def set_location(
        self, object_id: str, relation: RelationType | str, container_id: str
    ) -> SemanticRelation:
        resolved_relation = (
            relation if isinstance(relation, RelationType) else _enum_value(RelationType, relation)
        )
        if resolved_relation not in {RelationType.INSIDE, RelationType.ON}:
            raise ValueError("Location relation must be INSIDE or ON")
        self.relations = {
            item
            for item in self.relations
            if not (
                item.subject == object_id
                and item.relation in {RelationType.INSIDE, RelationType.ON}
            )
        }
        item = self.add_relation(object_id, resolved_relation, container_id)
        self.objects[object_id].attributes["location"] = container_id
        return item

    def interaction_points_for(
        self,
        *,
        role: InteractionRole | str | None = None,
        owner: str | None = None,
    ) -> tuple[InteractionPoint, ...]:
        resolved_role = role
        if isinstance(role, str):
            resolved_role = _enum_value(InteractionRole, role)
        return tuple(
            point
            for point in self.interaction_points.values()
            if (resolved_role is None or point.role == resolved_role)
            and (owner is None or point.owner == owner)
        )

    def interaction_pose(
        self, point_id: str, model: mujoco.MjModel, data: mujoco.MjData
    ) -> np.ndarray:
        try:
            point = self.interaction_points[point_id]
        except KeyError as error:
            raise KeyError(f"Unknown interaction point '{point_id}'") from error
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, point.site)
        if site_id < 0:
            raise SemanticValidationError(f"Interaction site '{point.site}' is missing")
        return self._pose_matrix(data.site_xpos[site_id], data.site_xmat[site_id])

    def object_pose(self, object_id: str, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        binding = self.object(object_id).binding
        if binding.kind == BindingKind.JOINT:
            raise ValueError("Joint bindings do not expose a Cartesian pose")
        mj_type = _MJ_OBJECT_TYPES[binding.kind]
        entity_id = mujoco.mj_name2id(model, mj_type, binding.name)
        if entity_id < 0:
            raise SemanticValidationError(
                f"Semantic object '{object_id}' binding '{binding.name}' is missing"
            )
        if binding.kind == BindingKind.BODY:
            return self._pose_matrix(data.xpos[entity_id], data.xmat[entity_id])
        if binding.kind == BindingKind.GEOM:
            return self._pose_matrix(data.geom_xpos[entity_id], data.geom_xmat[entity_id])
        return self._pose_matrix(data.site_xpos[entity_id], data.site_xmat[entity_id])

    def nearest_interaction_point(
        self,
        role: InteractionRole | str,
        position: np.ndarray,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> InteractionPoint:
        candidates = self.interaction_points_for(role=role)
        if not candidates:
            raise KeyError(f"No interaction points registered for role '{role}'")
        origin = np.asarray(position, dtype=float)[:3]
        return min(
            candidates,
            key=lambda point: float(
                np.linalg.norm(self.interaction_pose(point.point_id, model, data)[:3, 3] - origin)
            ),
        )

    def pose_snapshot(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        objects: dict[str, Any] = {}
        for object_id, semantic_object in self.objects.items():
            if semantic_object.binding.kind == BindingKind.JOINT:
                continue
            pose = self.object_pose(object_id, model, data)
            objects[object_id] = self._serialize_pose(pose)
        points = {
            point_id: self._serialize_pose(self.interaction_pose(point_id, model, data))
            for point_id in self.interaction_points
        }
        return {
            "time": float(data.time),
            "objects": objects,
            "interaction_points": points,
        }

    @staticmethod
    def _pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(rotation).reshape(3, 3)
        pose[:3, 3] = position
        return pose

    @staticmethod
    def _serialize_pose(pose: np.ndarray) -> dict[str, list[float]]:
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, pose[:3, :3].reshape(9))
        return {
            "position": pose[:3, 3].tolist(),
            "quaternion": quaternion.tolist(),
        }
