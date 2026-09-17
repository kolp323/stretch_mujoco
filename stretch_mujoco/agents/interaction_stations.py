"""Immutable interaction-station catalog and atomic runtime leases."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Protocol

from stretch_mujoco.npc.schema import (
    NpcInteractionCatalogEntry,
    NpcInteractionStation,
    NpcPopulation,
)


class InteractionStationAllocationError(ValueError):
    """A station request or journal transition violates the allocator contract."""


class InteractionLeaseOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class InteractionStationRole:
    role: str
    site: str
    yaw: float | None


@dataclass(frozen=True)
class InteractionStationDefinition:
    station_id: str
    kind: str
    roles: tuple[InteractionStationRole, ...]
    actor_modes: tuple[str, ...] = ()
    object_ids: tuple[str, ...] = ()
    transfer_site: str | None = None
    distance_min_m: float | None = None
    distance_max_m: float | None = None
    yaw_tolerance_rad: float | None = None
    owner_entity: str | None = None
    seat_type: str | None = None
    slot_index: int | None = None
    clearance_radius_m: float | None = None
    workstation_entity: str | None = None
    computer_entity: str | None = None
    seat_slot: str | None = None

    def role(self, name: str) -> InteractionStationRole:
        for role in self.roles:
            if role.role == name:
                return role
        raise KeyError(name)


class InteractionStationCatalog:
    """Deeply immutable runtime projection of a population-v3 station catalog."""

    _ROLE_CONTRACTS = {
        "conversation": frozenset({"speaker", "listener"}),
        "handover": frozenset({"giver", "receiver", "robot"}),
        "seat": frozenset({"ingress", "sit"}),
        "workstation": frozenset({"work"}),
    }
    _CONVERSATION_MODES = frozenset({"npc_npc", "robot_npc"})
    _HANDOVER_MODES = frozenset(
        {"npc_to_npc", "robot_to_npc", "npc_to_robot"}
    )

    def __init__(self, definitions: Iterable[InteractionStationDefinition]) -> None:
        stations = tuple(sorted(definitions, key=lambda item: item.station_id))
        if len({station.station_id for station in stations}) != len(stations):
            raise InteractionStationAllocationError("interaction_station_catalog_invalid")
        self._stations = stations
        self._by_id = MappingProxyType(
            {station.station_id: station for station in stations}
        )

    @classmethod
    def from_population(cls, population: NpcPopulation) -> "InteractionStationCatalog":
        if (
            population.schema_version != 3
            or population.legacy_single_station
            or population.interaction_stations is None
        ):
            raise InteractionStationAllocationError(
                "interaction_station_catalog_legacy_rejected"
            )
        raw_catalog = population.interaction_stations
        if set(raw_catalog) != set(cls._ROLE_CONTRACTS):
            raise InteractionStationAllocationError("interaction_station_catalog_invalid")
        definitions: list[InteractionStationDefinition] = []
        seen_ids: set[str] = set()
        for kind in sorted(cls._ROLE_CONTRACTS):
            entries = raw_catalog[kind]
            if not isinstance(entries, Mapping):
                raise InteractionStationAllocationError(
                    "interaction_station_catalog_invalid"
                )
            for station_id in sorted(entries):
                entry = entries[station_id]
                definitions.append(
                    cls._adapt_entry(kind, station_id, entry, seen_ids)
                )
                seen_ids.add(station_id)
        return cls(definitions)

    @classmethod
    def _adapt_entry(
        cls,
        kind: str,
        station_id: str,
        entry: NpcInteractionCatalogEntry,
        seen_ids: set[str],
    ) -> InteractionStationDefinition:
        if (
            not isinstance(entry, NpcInteractionCatalogEntry)
            or not isinstance(station_id, str)
            or not station_id
            or station_id in seen_ids
            or entry.station_id != station_id
            or entry.kind != kind
            or not isinstance(entry.roles, Mapping)
            or set(entry.roles) != cls._ROLE_CONTRACTS[kind]
            or not isinstance(entry.attributes, Mapping)
        ):
            raise InteractionStationAllocationError("interaction_station_catalog_invalid")
        if (
            entry.attributes.get("legacy_single_station") is True
            or entry.attributes.get("production_evidence") is False
        ):
            raise InteractionStationAllocationError(
                "interaction_station_catalog_legacy_rejected"
            )
        roles = []
        for role_name in sorted(entry.roles):
            role = entry.roles[role_name]
            if (
                not isinstance(role, NpcInteractionStation)
                or not isinstance(role.site, str)
                or not role.site
                or (
                    role.yaw is not None
                    and (
                        isinstance(role.yaw, bool)
                        or not isinstance(role.yaw, (int, float))
                        or not math.isfinite(role.yaw)
                    )
                )
            ):
                raise InteractionStationAllocationError(
                    "interaction_station_catalog_invalid"
                )
            roles.append(
                InteractionStationRole(
                    role_name,
                    role.site,
                    None if role.yaw is None else float(role.yaw),
                )
            )

        modes: tuple[str, ...] = ()
        object_ids: tuple[str, ...] = ()
        transfer_site = None
        distance_min_m = None
        distance_max_m = None
        yaw_tolerance_rad = None
        owner_entity = None
        seat_type = None
        slot_index = None
        clearance_radius_m = None
        workstation_entity = None
        computer_entity = None
        seat_slot = None
        if kind == "conversation":
            modes = cls._strict_strings(
                entry.attributes.get("allowed_actor_pairs"),
                allowed=cls._CONVERSATION_MODES,
            )
        elif kind == "handover":
            modes = cls._strict_strings(
                entry.attributes.get("modes"),
                allowed=cls._HANDOVER_MODES,
            )
            object_ids = cls._strict_strings(entry.attributes.get("object_ids"))
            transfer_site = entry.attributes.get("transfer_site")
            if not isinstance(transfer_site, str) or not transfer_site:
                raise InteractionStationAllocationError(
                    "interaction_station_catalog_invalid"
                )
        elif kind == "seat":
            owner_entity = entry.attributes.get("owner_entity")
            seat_type = entry.attributes.get("seat_type")
            slot_index = entry.attributes.get("slot_index")
            clearance_radius_m = entry.attributes.get("clearance_radius_m")
            if (
                not isinstance(owner_entity, str)
                or not owner_entity
                or not isinstance(seat_type, str)
                or not seat_type
                or isinstance(slot_index, bool)
                or not isinstance(slot_index, int)
                or slot_index < 0
            ):
                raise InteractionStationAllocationError(
                    "interaction_station_catalog_invalid"
                )
            clearance_radius_m = cls._strict_positive_number(clearance_radius_m)
        elif kind == "workstation":
            workstation_entity = entry.attributes.get("workstation_entity")
            computer_entity = entry.attributes.get("computer_entity")
            seat_slot = entry.attributes.get("seat_slot")
            if any(
                not isinstance(value, str) or not value
                for value in (workstation_entity, computer_entity, seat_slot)
            ):
                raise InteractionStationAllocationError(
                    "interaction_station_catalog_invalid"
                )
        if kind in {"conversation", "handover"}:
            distance_min_m, distance_max_m = cls._strict_distance(
                entry.attributes.get("distance_m")
            )
            yaw_tolerance_rad = cls._strict_positive_number(
                entry.attributes.get("yaw_tolerance_rad"), maximum=math.pi
            )
        return InteractionStationDefinition(
            station_id,
            kind,
            tuple(roles),
            modes,
            object_ids,
            transfer_site,
            distance_min_m,
            distance_max_m,
            yaw_tolerance_rad,
            owner_entity,
            seat_type,
            slot_index,
            clearance_radius_m,
            workstation_entity,
            computer_entity,
            seat_slot,
        )

    @staticmethod
    def _strict_strings(
        value: object, *, allowed: frozenset[str] | None = None
    ) -> tuple[str, ...]:
        if (
            not isinstance(value, (list, tuple))
            or not value
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)
            or (allowed is not None and not set(value) <= allowed)
        ):
            raise InteractionStationAllocationError(
                "interaction_station_catalog_invalid"
            )
        return tuple(value)

    @classmethod
    def _strict_distance(cls, value: object) -> tuple[float, float]:
        if not isinstance(value, Mapping) or set(value) != {"min", "max"}:
            raise InteractionStationAllocationError(
                "interaction_station_catalog_invalid"
            )
        minimum = cls._strict_positive_number(value["min"])
        maximum = cls._strict_positive_number(value["max"])
        if maximum < minimum:
            raise InteractionStationAllocationError(
                "interaction_station_catalog_invalid"
            )
        return minimum, maximum

    @staticmethod
    def _strict_positive_number(value: object, *, maximum: float | None = None) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            or (maximum is not None and value > maximum)
        ):
            raise InteractionStationAllocationError(
                "interaction_station_catalog_invalid"
            )
        return float(value)

    def station(self, station_id: str) -> InteractionStationDefinition:
        try:
            return self._by_id[station_id]
        except KeyError as error:
            raise KeyError(f"Unknown interaction station '{station_id}'") from error

    def stations(self, kind: str | None = None) -> tuple[InteractionStationDefinition, ...]:
        if kind is None:
            return self._stations
        return tuple(station for station in self._stations if station.kind == kind)


@dataclass(frozen=True)
class SeatSlotLease:
    slot_id: str
    owner_id: str
    session_id: str
    occupied: bool = False
    verification_receipt_id: str | None = None


class SeatSlotAllocator:
    """Atomically reserve and occupy authored seat slots."""

    def __init__(self, catalog: InteractionStationCatalog) -> None:
        self.catalog = catalog
        self._lock = RLock()
        self._definitions = MappingProxyType(
            {station.station_id: station for station in catalog.stations("seat")}
        )
        self._leases: dict[str, SeatSlotLease] = {}

    def definition(self, slot_id: str) -> InteractionStationDefinition:
        try:
            return self._definitions[slot_id]
        except KeyError as error:
            raise InteractionStationAllocationError(
                f"seat_slot_unknown:{slot_id}"
            ) from error

    def reserve(self, slot_id: str, owner_id: str, session_id: str) -> SeatSlotLease:
        if not owner_id or not session_id:
            raise InteractionStationAllocationError("seat_slot_request_invalid")
        self.definition(slot_id)
        with self._lock:
            current = self._leases.get(slot_id)
            if current is not None:
                if current.owner_id == owner_id and current.session_id == session_id:
                    return current
                raise InteractionStationAllocationError(f"seat_slot_busy:{slot_id}")
            lease = SeatSlotLease(slot_id, owner_id, session_id)
            self._leases[slot_id] = lease
            return lease

    def occupy(
        self, slot_id: str, owner_id: str, session_id: str, verification_receipt_id: str
    ) -> SeatSlotLease:
        if not verification_receipt_id:
            raise InteractionStationAllocationError(
                "seat_slot_verification_receipt_required"
            )
        with self._lock:
            current = self._matching(slot_id, owner_id, session_id)
            if current.occupied:
                if current.verification_receipt_id == verification_receipt_id:
                    return current
                raise InteractionStationAllocationError("seat_slot_occupancy_conflict")
            occupied = replace(
                current,
                occupied=True,
                verification_receipt_id=verification_receipt_id,
            )
            self._leases[slot_id] = occupied
            return occupied

    def release_reservation(self, slot_id: str, owner_id: str, session_id: str) -> bool:
        with self._lock:
            current = self._matching(slot_id, owner_id, session_id)
            if current.occupied:
                raise InteractionStationAllocationError(
                    "seat_slot_stand_receipt_required"
                )
            del self._leases[slot_id]
            return True

    def release_occupied(
        self, slot_id: str, owner_id: str, session_id: str, stand_receipt_id: str
    ) -> bool:
        if not stand_receipt_id:
            raise InteractionStationAllocationError(
                "seat_slot_stand_receipt_required"
            )
        with self._lock:
            current = self._matching(slot_id, owner_id, session_id)
            if not current.occupied:
                raise InteractionStationAllocationError("seat_slot_not_occupied")
            del self._leases[slot_id]
            return True

    def lease(self, slot_id: str) -> SeatSlotLease | None:
        with self._lock:
            return self._leases.get(slot_id)

    def snapshot(self) -> tuple[SeatSlotLease, ...]:
        with self._lock:
            return tuple(self._leases[key] for key in sorted(self._leases))

    def _matching(self, slot_id: str, owner_id: str, session_id: str) -> SeatSlotLease:
        current = self._leases.get(slot_id)
        if current is None:
            raise InteractionStationAllocationError(f"seat_slot_not_reserved:{slot_id}")
        if current.owner_id != owner_id or current.session_id != session_id:
            raise InteractionStationAllocationError(f"seat_slot_busy:{slot_id}")
        return current


@dataclass(frozen=True)
class InteractionRoleAssignment:
    participant_id: str
    actor_kind: str
    role: str
    site: str


@dataclass(frozen=True)
class InteractionStationLease:
    lease_id: str
    session_id: str
    station_id: str
    kind: str
    mode: str
    assignments: tuple[InteractionRoleAssignment, ...]
    object_id: str | None
    preferred_station_id: str | None
    resources: tuple[str, ...]
    terminal_receipt_id: str | None = None
    outcome: InteractionLeaseOutcome | None = None
    cleanup_evidence_id: str | None = None

    @property
    def active(self) -> bool:
        return self.outcome is None


@dataclass(frozen=True)
class InteractionStationLeaseEvent:
    sequence: int
    event_type: str
    lease: InteractionStationLease

    def to_json(self) -> str:
        return json.dumps(
            _event_to_dict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> "InteractionStationLeaseEvent":
        try:
            raw = json.loads(payload, object_pairs_hook=_strict_json_object)
            return _event_from_dict(raw)
        except InteractionStationAllocationError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise InteractionStationAllocationError(
                "interaction_station_journal_invalid"
            ) from error


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InteractionStationAllocationError(
                "interaction_station_journal_invalid"
            )
        result[key] = value
    return result


def _event_to_dict(event: InteractionStationLeaseEvent) -> dict[str, object]:
    lease = event.lease
    return {
        "schema_version": 1,
        "sequence": event.sequence,
        "event_type": event.event_type,
        "lease": {
            "lease_id": lease.lease_id,
            "session_id": lease.session_id,
            "station_id": lease.station_id,
            "kind": lease.kind,
            "mode": lease.mode,
            "assignments": [
                {
                    "participant_id": assignment.participant_id,
                    "actor_kind": assignment.actor_kind,
                    "role": assignment.role,
                    "site": assignment.site,
                }
                for assignment in lease.assignments
            ],
            "object_id": lease.object_id,
            "preferred_station_id": lease.preferred_station_id,
            "resources": list(lease.resources),
            "terminal_receipt_id": lease.terminal_receipt_id,
            "outcome": None if lease.outcome is None else lease.outcome.value,
            "cleanup_evidence_id": lease.cleanup_evidence_id,
        },
    }


def _event_from_dict(payload: object) -> InteractionStationLeaseEvent:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "sequence",
        "event_type",
        "lease",
    }:
        raise InteractionStationAllocationError("interaction_station_journal_invalid")
    sequence = payload["sequence"]
    if (
        isinstance(payload["schema_version"], bool)
        or payload["schema_version"] != 1
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 1
        or payload["event_type"] not in {"acquired", "released"}
    ):
        raise InteractionStationAllocationError("interaction_station_journal_invalid")
    raw_lease = payload["lease"]
    lease_fields = {
        "lease_id",
        "session_id",
        "station_id",
        "kind",
        "mode",
        "assignments",
        "object_id",
        "preferred_station_id",
        "resources",
        "terminal_receipt_id",
        "outcome",
        "cleanup_evidence_id",
    }
    if not isinstance(raw_lease, dict) or set(raw_lease) != lease_fields:
        raise InteractionStationAllocationError("interaction_station_journal_invalid")
    string_fields = ("lease_id", "session_id", "station_id", "kind", "mode")
    if any(
        not isinstance(raw_lease[field], str) or not raw_lease[field]
        for field in string_fields
    ):
        raise InteractionStationAllocationError("interaction_station_journal_invalid")
    assignments_payload = raw_lease["assignments"]
    if not isinstance(assignments_payload, list):
        raise InteractionStationAllocationError("interaction_station_journal_invalid")
    assignments = []
    assignment_fields = {"participant_id", "actor_kind", "role", "site"}
    for raw_assignment in assignments_payload:
        if (
            not isinstance(raw_assignment, dict)
            or set(raw_assignment) != assignment_fields
            or any(
                not isinstance(raw_assignment[field], str)
                or not raw_assignment[field]
                for field in assignment_fields
            )
        ):
            raise InteractionStationAllocationError(
                "interaction_station_journal_invalid"
            )
        assignments.append(
            InteractionRoleAssignment(
                raw_assignment["participant_id"],
                raw_assignment["actor_kind"],
                raw_assignment["role"],
                raw_assignment["site"],
            )
        )
    resources = raw_lease["resources"]
    if (
        not isinstance(resources, list)
        or any(not isinstance(item, str) or not item for item in resources)
        or len(set(resources)) != len(resources)
    ):
        raise InteractionStationAllocationError("interaction_station_journal_invalid")
    for optional in (
        "object_id",
        "preferred_station_id",
        "terminal_receipt_id",
        "cleanup_evidence_id",
    ):
        value = raw_lease[optional]
        if value is not None and (not isinstance(value, str) or not value):
            raise InteractionStationAllocationError(
                "interaction_station_journal_invalid"
            )
    outcome = raw_lease["outcome"]
    try:
        parsed_outcome = None if outcome is None else InteractionLeaseOutcome(outcome)
    except (TypeError, ValueError) as error:
        raise InteractionStationAllocationError(
            "interaction_station_journal_invalid"
        ) from error
    lease = InteractionStationLease(
        raw_lease["lease_id"],
        raw_lease["session_id"],
        raw_lease["station_id"],
        raw_lease["kind"],
        raw_lease["mode"],
        tuple(assignments),
        raw_lease["object_id"],
        raw_lease["preferred_station_id"],
        tuple(resources),
        raw_lease["terminal_receipt_id"],
        parsed_outcome,
        raw_lease["cleanup_evidence_id"],
    )
    return InteractionStationLeaseEvent(sequence, payload["event_type"], lease)


class InteractionLeaseJournal(Protocol):
    def append(self, event: InteractionStationLeaseEvent) -> None: ...

    def events(self) -> tuple[InteractionStationLeaseEvent, ...]: ...


class InMemoryInteractionLeaseJournal:
    def __init__(self, events: Iterable[InteractionStationLeaseEvent] = ()) -> None:
        self._lock = RLock()
        self._events = list(events)

    def append(self, event: InteractionStationLeaseEvent) -> None:
        with self._lock:
            if (
                not isinstance(event, InteractionStationLeaseEvent)
                or event.sequence != len(self._events) + 1
            ):
                raise InteractionStationAllocationError(
                    "interaction_station_journal_invalid"
                )
            self._events.append(event)

    def events(self) -> tuple[InteractionStationLeaseEvent, ...]:
        with self._lock:
            return tuple(self._events)


class JsonlInteractionLeaseJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()
        self._events: list[InteractionStationLeaseEvent] = []
        self._size = 0
        if self.path.exists():
            raw = self.path.read_bytes()
            self._size = len(raw)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise InteractionStationAllocationError(
                    "interaction_station_journal_invalid"
                ) from error
            if text and not text.endswith("\n"):
                raise InteractionStationAllocationError(
                    "interaction_station_journal_invalid"
                )
            for line in text.splitlines():
                if not line:
                    raise InteractionStationAllocationError(
                        "interaction_station_journal_invalid"
                    )
                self._events.append(InteractionStationLeaseEvent.from_json(line))

    def append(self, event: InteractionStationLeaseEvent) -> None:
        with self._lock:
            if (
                not isinstance(event, InteractionStationLeaseEvent)
                or event.sequence != len(self._events) + 1
            ):
                raise InteractionStationAllocationError(
                    "interaction_station_journal_invalid"
                )
            if self.path.exists() and self.path.stat().st_size != self._size:
                raise InteractionStationAllocationError(
                    "interaction_station_journal_invalid"
                )
            payload = (event.to_json() + "\n").encode("utf-8")
            existed = self.path.exists()
            mode = "r+b" if existed else "w+b"
            try:
                with self.path.open(mode) as stream:
                    stream.seek(0, os.SEEK_END)
                    if stream.tell() != self._size:
                        raise InteractionStationAllocationError(
                            "interaction_station_journal_invalid"
                        )
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                try:
                    if self.path.exists():
                        with self.path.open("r+b") as stream:
                            stream.truncate(self._size)
                            stream.flush()
                            os.fsync(stream.fileno())
                        if not existed and self._size == 0:
                            self.path.unlink(missing_ok=True)
                finally:
                    raise
            self._events.append(event)
            self._size += len(payload)

    def events(self) -> tuple[InteractionStationLeaseEvent, ...]:
        with self._lock:
            return tuple(self._events)


@dataclass(frozen=True)
class InteractionStationAllocatorSnapshot:
    active_leases: tuple[InteractionStationLease, ...]
    resource_owners: tuple[tuple[str, str], ...]


RouteCostCallback = Callable[[str, str, str], float | None]


class InteractionStationAllocator:
    """Select and atomically lease one compatible station for a session."""

    _ACTOR_KINDS = frozenset({"npc", "robot"})

    def __init__(
        self,
        catalog: InteractionStationCatalog,
        *,
        route_cost: RouteCostCallback | None = None,
        journal: InteractionLeaseJournal | None = None,
    ) -> None:
        self.catalog = catalog
        self._route_cost = route_cost or (
            lambda _participant, _actor_kind, _site: 0.0
        )
        self._lock = RLock()
        self._journal = journal or InMemoryInteractionLeaseJournal()
        if self._journal.events():
            raise InteractionStationAllocationError(
                "interaction_station_journal_requires_reconstruction"
            )
        self._leases: dict[str, InteractionStationLease] = {}
        self._sessions: dict[str, str] = {}
        self._session_requests: dict[str, tuple[object, ...]] = {}
        self._resource_owners: dict[str, str] = {}
        self._events: list[InteractionStationLeaseEvent] = []

    def acquire(
        self,
        *,
        session_id: str,
        kind: str,
        participants: tuple[str, str],
        actor_kinds: tuple[str, str],
        object_id: str | None = None,
        preferred_station_id: str | None = None,
    ) -> InteractionStationLease:
        request = self._request_signature(
            session_id,
            kind,
            participants,
            actor_kinds,
            object_id,
            preferred_station_id,
        )
        with self._lock:
            previous_lease_id = self._sessions.get(session_id)
            if previous_lease_id is not None:
                if self._session_requests[session_id] != request:
                    raise InteractionStationAllocationError(
                        "interaction_station_session_conflict"
                    )
                return self._leases[previous_lease_id]

            stations = self.catalog.stations(kind)
            if preferred_station_id is not None:
                stations = tuple(
                    station
                    for station in stations
                    if station.station_id == preferred_station_id
                )
            compatible: list[
                tuple[
                    InteractionStationDefinition,
                    str,
                    tuple[InteractionRoleAssignment, ...],
                ]
            ] = []
            for station in stations:
                matched = self._assign(station, participants, actor_kinds, object_id)
                if matched is not None:
                    mode, assignments = matched
                    compatible.append((station, mode, assignments))
            if not compatible:
                raise InteractionStationAllocationError(
                    "interaction_station_no_compatible"
                )

            available = []
            for station, mode, assignments in compatible:
                resources = self._resources(
                    station.station_id, assignments, object_id
                )
                if all(resource not in self._resource_owners for resource in resources):
                    available.append((station, mode, assignments, resources))
            if not available:
                raise InteractionStationAllocationError("interaction_station_busy")

            routed = []
            for station, mode, assignments, resources in available:
                total = self._total_route_cost(assignments)
                if total is not None:
                    routed.append(
                        (total, station.station_id, station, mode, assignments, resources)
                    )
            if not routed:
                raise InteractionStationAllocationError(
                    "interaction_station_route_unavailable"
                )
            _, _, station, mode, assignments, resources = min(routed)
            lease_id = f"interaction_lease:{session_id}"
            lease = InteractionStationLease(
                lease_id,
                session_id,
                station.station_id,
                kind,
                mode,
                assignments,
                object_id,
                preferred_station_id,
                resources,
            )
            event = self._event_for(lease, "acquired")
            self._journal.append(event)
            self._leases[lease_id] = lease
            self._sessions[session_id] = lease_id
            self._session_requests[session_id] = request
            for resource in resources:
                self._resource_owners[resource] = lease_id
            self._events.append(event)
            return lease

    def release(
        self,
        lease_id: str,
        *,
        terminal_receipt_id: str,
        outcome: InteractionLeaseOutcome | str,
    ) -> InteractionStationLease:
        if not isinstance(terminal_receipt_id, str) or not terminal_receipt_id:
            raise InteractionStationAllocationError(
                "interaction_station_terminal_receipt_required"
            )
        try:
            terminal_outcome = InteractionLeaseOutcome(outcome)
        except (TypeError, ValueError) as error:
            raise InteractionStationAllocationError(
                "interaction_station_terminal_outcome_required"
            ) from error
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise InteractionStationAllocationError(
                    "interaction_station_unknown_lease"
                )
            if not lease.active:
                if (
                    lease.terminal_receipt_id == terminal_receipt_id
                    and lease.outcome == terminal_outcome
                ):
                    return lease
                raise InteractionStationAllocationError(
                    "interaction_station_release_conflict"
                )
            terminal = replace(
                lease,
                terminal_receipt_id=terminal_receipt_id,
                outcome=terminal_outcome,
            )
            event = self._event_for(terminal, "released")
            self._journal.append(event)
            for resource in lease.resources:
                if self._resource_owners.get(resource) == lease_id:
                    del self._resource_owners[resource]
            self._leases[lease_id] = terminal
            self._events.append(event)
            return terminal

    def release_unconfirmed(
        self,
        lease_id: str,
        *,
        cleanup_evidence_id: str,
        outcome: InteractionLeaseOutcome | str,
    ) -> InteractionStationLease:
        if not isinstance(cleanup_evidence_id, str) or not cleanup_evidence_id:
            raise InteractionStationAllocationError(
                "interaction_station_cleanup_evidence_required"
            )
        try:
            terminal_outcome = InteractionLeaseOutcome(outcome)
        except (TypeError, ValueError) as error:
            raise InteractionStationAllocationError(
                "interaction_station_terminal_outcome_required"
            ) from error
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise InteractionStationAllocationError(
                    "interaction_station_unknown_lease"
                )
            if not lease.active:
                if (
                    lease.cleanup_evidence_id == cleanup_evidence_id
                    and lease.outcome == terminal_outcome
                ):
                    return lease
                raise InteractionStationAllocationError(
                    "interaction_station_release_conflict"
                )
            terminal = replace(
                lease,
                cleanup_evidence_id=cleanup_evidence_id,
                outcome=terminal_outcome,
            )
            event = self._event_for(terminal, "released")
            self._journal.append(event)
            for resource in lease.resources:
                if self._resource_owners.get(resource) == lease_id:
                    del self._resource_owners[resource]
            self._leases[lease_id] = terminal
            self._events.append(event)
            return terminal

    def owner(self, resource: str) -> str | None:
        with self._lock:
            return self._resource_owners.get(resource)

    def lease(self, lease_id: str) -> InteractionStationLease | None:
        with self._lock:
            return self._leases.get(lease_id)

    def lease_for_session(self, session_id: str) -> InteractionStationLease | None:
        with self._lock:
            lease_id = self._sessions.get(session_id)
            return None if lease_id is None else self._leases[lease_id]

    def snapshot(self) -> InteractionStationAllocatorSnapshot:
        with self._lock:
            active = tuple(
                sorted(
                    (lease for lease in self._leases.values() if lease.active),
                    key=lambda lease: lease.lease_id,
                )
            )
            return InteractionStationAllocatorSnapshot(
                active,
                tuple(sorted(self._resource_owners.items())),
            )

    def events(self) -> tuple[InteractionStationLeaseEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @classmethod
    def reconstruct(
        cls,
        catalog: InteractionStationCatalog,
        journal: InteractionLeaseJournal,
        *,
        route_cost: RouteCostCallback | None = None,
    ) -> "InteractionStationAllocator":
        records = journal.events()
        allocator = cls(catalog, route_cost=route_cost)
        allocator._journal = journal
        for expected_sequence, event in enumerate(records, start=1):
            allocator._replay_event(event, expected_sequence)
        return allocator

    def _replay_event(
        self,
        event: InteractionStationLeaseEvent,
        expected_sequence: int,
    ) -> None:
        if (
            not isinstance(event, InteractionStationLeaseEvent)
            or event.sequence != expected_sequence
        ):
            raise InteractionStationAllocationError(
                "interaction_station_journal_invalid"
            )
        lease = event.lease
        if event.event_type == "acquired":
            if (
                not lease.active
                or lease.terminal_receipt_id is not None
                or lease.cleanup_evidence_id is not None
                or lease.lease_id != f"interaction_lease:{lease.session_id}"
                or lease.lease_id in self._leases
                or lease.session_id in self._sessions
                or (
                    lease.preferred_station_id is not None
                    and lease.preferred_station_id != lease.station_id
                )
            ):
                raise InteractionStationAllocationError(
                    "interaction_station_journal_invalid"
                )
            participants = tuple(
                assignment.participant_id for assignment in lease.assignments
            )
            actor_kinds = tuple(
                assignment.actor_kind for assignment in lease.assignments
            )
            request = self._request_signature(
                lease.session_id,
                lease.kind,
                participants,
                actor_kinds,
                lease.object_id,
                lease.preferred_station_id,
            )
            try:
                station = self.catalog.station(lease.station_id)
            except KeyError as error:
                raise InteractionStationAllocationError(
                    "interaction_station_journal_invalid"
                ) from error
            matched = self._assign(
                station,
                participants,
                actor_kinds,
                lease.object_id,
            )
            if (
                matched != (lease.mode, lease.assignments)
                or lease.kind != station.kind
                or lease.resources
                != self._resources(
                    lease.station_id, lease.assignments, lease.object_id
                )
                or any(
                    resource in self._resource_owners
                    for resource in lease.resources
                )
            ):
                raise InteractionStationAllocationError(
                    "interaction_station_journal_invalid"
                )
            self._leases[lease.lease_id] = lease
            self._sessions[lease.session_id] = lease.lease_id
            self._session_requests[lease.session_id] = request
            for resource in lease.resources:
                self._resource_owners[resource] = lease.lease_id
        elif event.event_type == "released":
            active = self._leases.get(lease.lease_id)
            if (
                active is None
                or not active.active
                or (lease.terminal_receipt_id is None)
                == (lease.cleanup_evidence_id is None)
                or lease.outcome is None
                or replace(
                    lease,
                    terminal_receipt_id=None,
                    outcome=None,
                    cleanup_evidence_id=None,
                )
                != active
            ):
                raise InteractionStationAllocationError(
                    "interaction_station_journal_invalid"
                )
            for resource in active.resources:
                if self._resource_owners.get(resource) != active.lease_id:
                    raise InteractionStationAllocationError(
                        "interaction_station_journal_invalid"
                    )
            for resource in active.resources:
                del self._resource_owners[resource]
            self._leases[lease.lease_id] = lease
        else:
            raise InteractionStationAllocationError(
                "interaction_station_journal_invalid"
            )
        self._events.append(event)

    def _event_for(
        self,
        lease: InteractionStationLease,
        event_type: str,
    ) -> InteractionStationLeaseEvent:
        return InteractionStationLeaseEvent(
            len(self._events) + 1,
            event_type,
            lease,
        )

    @classmethod
    def _assign(
        cls,
        station: InteractionStationDefinition,
        participants: tuple[str, str],
        actor_kinds: tuple[str, str],
        object_id: str | None,
    ) -> tuple[str, tuple[InteractionRoleAssignment, ...]] | None:
        if station.kind == "conversation":
            if object_id is not None:
                return None
            if actor_kinds == ("npc", "npc") and "npc_npc" in station.actor_modes:
                mode, roles = "npc_npc", ("speaker", "listener")
            elif (
                actor_kinds in {("robot", "npc"), ("npc", "robot")}
                and "robot_npc" in station.actor_modes
            ):
                mode, roles = "robot_npc", ("speaker", "listener")
            else:
                return None
        elif station.kind == "handover":
            if object_id is None or object_id not in station.object_ids:
                return None
            contracts = {
                ("npc", "npc"): ("npc_to_npc", ("giver", "receiver")),
                ("robot", "npc"): ("robot_to_npc", ("robot", "receiver")),
                ("npc", "robot"): ("npc_to_robot", ("giver", "robot")),
            }
            contract = contracts.get(actor_kinds)
            if contract is None or contract[0] not in station.actor_modes:
                return None
            mode, roles = contract
        else:
            return None
        assignments = tuple(
            InteractionRoleAssignment(
                participant,
                actor_kind,
                role_name,
                station.role(role_name).site,
            )
            for participant, actor_kind, role_name in zip(
                participants, actor_kinds, roles
            )
        )
        return mode, assignments

    @classmethod
    def _request_signature(
        cls,
        session_id: str,
        kind: str,
        participants: tuple[str, str],
        actor_kinds: tuple[str, str],
        object_id: str | None,
        preferred_station_id: str | None,
    ) -> tuple[object, ...]:
        if (
            not isinstance(session_id, str)
            or not session_id
            or kind not in {"conversation", "handover"}
            or not isinstance(participants, tuple)
            or len(participants) != 2
            or any(not isinstance(item, str) or not item for item in participants)
            or len(set(participants)) != 2
            or not isinstance(actor_kinds, tuple)
            or len(actor_kinds) != 2
            or any(item not in cls._ACTOR_KINDS for item in actor_kinds)
            or (object_id is not None and (not isinstance(object_id, str) or not object_id))
            or (
                preferred_station_id is not None
                and (
                    not isinstance(preferred_station_id, str)
                    or not preferred_station_id
                )
            )
        ):
            raise InteractionStationAllocationError(
                "interaction_station_request_invalid"
            )
        return (
            kind,
            participants,
            actor_kinds,
            object_id,
            preferred_station_id,
        )

    @staticmethod
    def _resources(
        station_id: str,
        assignments: tuple[InteractionRoleAssignment, ...],
        object_id: str | None,
    ) -> tuple[str, ...]:
        ordered = [f"station:{station_id}"]
        ordered.extend(f"site:{assignment.site}" for assignment in assignments)
        ordered.extend(
            f"participant:{assignment.participant_id}" for assignment in assignments
        )
        if object_id is not None:
            ordered.append(f"object:{object_id}")
        return tuple(dict.fromkeys(ordered))

    def _total_route_cost(
        self, assignments: tuple[InteractionRoleAssignment, ...]
    ) -> float | None:
        total = 0.0
        for assignment in assignments:
            value = self._route_cost(
                assignment.participant_id,
                assignment.actor_kind,
                assignment.site,
            )
            if (
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                return None
            total += float(value)
        return total
