from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from stretch_mujoco.agents.interaction_stations import (
    InteractionLeaseOutcome,
    InMemoryInteractionLeaseJournal,
    InteractionStationAllocationError,
    InteractionStationAllocator,
    InteractionStationCatalog,
    InteractionStationLeaseEvent,
    JsonlInteractionLeaseJournal,
)
from stretch_mujoco.npc.schema import (
    NpcInteractionCatalogEntry,
    NpcInteractionStation,
    NpcPopulation,
)


def _role(site: str, yaw: float = 0.0) -> NpcInteractionStation:
    return NpcInteractionStation(site, yaw)


def _conversation(station_id: str, *, modes=None) -> NpcInteractionCatalogEntry:
    prefix = station_id.replace("conversation.", "")
    return NpcInteractionCatalogEntry(
        station_id,
        "conversation",
        {
            "speaker": _role(f"{prefix}.speaker"),
            "listener": _role(f"{prefix}.listener"),
        },
        {
            "allowed_actor_pairs": modes or ["npc_npc", "robot_npc"],
            "distance_m": {"min": 0.55, "max": 1.05},
            "yaw_tolerance_rad": 0.35,
        },
    )


def _handover(station_id: str, *, objects=None, modes=None) -> NpcInteractionCatalogEntry:
    prefix = station_id.replace("handover.", "")
    return NpcInteractionCatalogEntry(
        station_id,
        "handover",
        {
            "giver": _role(f"{prefix}.giver"),
            "receiver": _role(f"{prefix}.receiver"),
            "robot": _role(f"{prefix}.robot"),
        },
        {
            "modes": modes
            or ["npc_to_npc", "robot_to_npc", "npc_to_robot"],
            "object_ids": objects or ["cup"],
            "transfer_site": f"{prefix}.transfer",
            "distance_m": {"min": 0.6, "max": 1.0},
            "yaw_tolerance_rad": 0.4,
        },
    )


def _population(
    *,
    conversations=None,
    handovers=None,
    schema_version: int = 3,
    legacy_single_station: bool = False,
) -> NpcPopulation:
    return NpcPopulation(
        scene="fixture.xml",
        asset_manifest="fixture.json",
        npcs={},
        clock={},
        interaction_stations={
            "conversation": conversations
            or {
                "conversation.a": _conversation("conversation.a"),
                "conversation.b": _conversation("conversation.b"),
            },
            "handover": handovers
            or {
                "handover.a": _handover("handover.a"),
                "handover.b": _handover("handover.b"),
            },
            "seat": {},
            "workstation": {},
        },
        schema_version=schema_version,
        legacy_single_station=legacy_single_station,
    )


def _catalog(**kwargs) -> InteractionStationCatalog:
    return InteractionStationCatalog.from_population(_population(**kwargs))


def _acquire_conversation(
    allocator: InteractionStationAllocator,
    session_id: str,
    participants=("npc.1", "npc.2"),
    *,
    actor_kinds=("npc", "npc"),
    preferred_station_id=None,
):
    return allocator.acquire(
        session_id=session_id,
        kind="conversation",
        participants=participants,
        actor_kinds=actor_kinds,
        preferred_station_id=preferred_station_id,
    )


def test_catalog_adapter_is_immutable_and_rejects_legacy_or_malformed() -> None:
    catalog = _catalog()
    station = catalog.station("conversation.a")
    assert isinstance(station.roles, tuple)
    assert station.distance_min_m == 0.55
    assert station.distance_max_m == 1.05
    assert station.yaw_tolerance_rad == 0.35
    handover = catalog.station("handover.a")
    assert handover.distance_min_m == 0.6
    assert handover.distance_max_m == 1.0
    assert handover.yaw_tolerance_rad == 0.4
    assert handover.transfer_site == "a.transfer"
    assert handover.object_ids == ("cup",)
    with pytest.raises(FrozenInstanceError):
        station.station_id = "changed"  # type: ignore[misc]

    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_catalog_legacy_rejected$",
    ):
        InteractionStationCatalog.from_population(
            _population(schema_version=2, legacy_single_station=True)
        )

    legacy_entry = _conversation("conversation.legacy")
    legacy_entry.attributes["production_evidence"] = False
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_catalog_legacy_rejected$",
    ):
        _catalog(conversations={"conversation.legacy": legacy_entry})

    malformed = _conversation("conversation.bad")
    malformed.roles.pop("listener")
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_catalog_invalid$",
    ):
        _catalog(conversations={"conversation.bad": malformed})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("distance_m", {"min": 0.55}),
        ("distance_m", {"min": 1.05, "max": 0.55}),
        ("distance_m", {"min": True, "max": 1.05}),
        ("yaw_tolerance_rad", 3.15),
    ],
)
def test_catalog_rejects_malformed_spatial_contract(field: str, value: object) -> None:
    malformed = _conversation("conversation.bad")
    malformed.attributes[field] = value
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_catalog_invalid$",
    ):
        _catalog(conversations={"conversation.bad": malformed})


def test_route_failure_is_atomic_and_leaves_no_partial_resources() -> None:
    allocator = InteractionStationAllocator(
        _catalog(),
        route_cost=lambda participant, _actor_kind, _site: (
            None if participant == "npc.2" else 1.0
        ),
    )
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_route_unavailable$",
    ):
        _acquire_conversation(allocator, "session.route")
    assert allocator.snapshot().active_leases == ()
    assert allocator.snapshot().resource_owners == ()


def test_two_stations_can_be_acquired_concurrently_without_partial_overlap() -> None:
    allocator = InteractionStationAllocator(_catalog())

    def acquire(index: int):
        return _acquire_conversation(
            allocator,
            f"session.{index}",
            (f"npc.{index}.a", f"npc.{index}.b"),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        leases = tuple(pool.map(acquire, (1, 2)))
    assert {lease.station_id for lease in leases} == {
        "conversation.a",
        "conversation.b",
    }
    assert len(allocator.snapshot().resource_owners) == 10


def test_same_handover_object_conflicts_across_distinct_stations() -> None:
    allocator = InteractionStationAllocator(_catalog())
    first = allocator.acquire(
        session_id="handover.first",
        kind="handover",
        participants=("npc.1", "npc.2"),
        actor_kinds=("npc", "npc"),
        object_id="cup",
    )
    assert allocator.owner("object:cup") == first.lease_id
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_busy$",
    ):
        allocator.acquire(
            session_id="handover.second",
            kind="handover",
            participants=("npc.3", "npc.4"),
            actor_kinds=("npc", "npc"),
            object_id="cup",
        )


@pytest.mark.parametrize(
    ("actor_kinds", "mode", "roles"),
    [
        (("npc", "npc"), "npc_to_npc", ("giver", "receiver")),
        (("robot", "npc"), "robot_to_npc", ("robot", "receiver")),
        (("npc", "robot"), "npc_to_robot", ("giver", "robot")),
    ],
)
def test_handover_actor_modes_assign_correct_role_sites(
    actor_kinds, mode: str, roles: tuple[str, str]
) -> None:
    allocator = InteractionStationAllocator(_catalog())
    lease = allocator.acquire(
        session_id=f"session.{mode}",
        kind="handover",
        participants=("actor.1", "actor.2"),
        actor_kinds=actor_kinds,
        object_id="cup",
        preferred_station_id="handover.a",
    )
    assert lease.mode == mode
    assert tuple(assignment.role for assignment in lease.assignments) == roles
    assert tuple(assignment.participant_id for assignment in lease.assignments) == (
        "actor.1",
        "actor.2",
    )
    assert all(f"site:{assignment.site}" in lease.resources for assignment in lease.assignments)


def test_conversation_actor_filtering_and_object_whitelist() -> None:
    route_calls: list[tuple[str, str, str]] = []

    def route_cost(participant: str, actor_kind: str, site: str) -> float:
        route_calls.append((participant, actor_kind, site))
        return 1.0

    allocator = InteractionStationAllocator(_catalog(), route_cost=route_cost)
    robot_npc = _acquire_conversation(
        allocator,
        "conversation.robot",
        ("robot.1", "npc.1"),
        actor_kinds=("robot", "npc"),
    )
    assert robot_npc.mode == "robot_npc"
    assert tuple(assignment.role for assignment in robot_npc.assignments) == (
        "speaker",
        "listener",
    )
    assert any(
        participant == "robot.1" and actor_kind == "robot"
        for participant, actor_kind, _site in route_calls
    )
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_no_compatible$",
    ):
        _acquire_conversation(
            allocator,
            "conversation.invalid",
            ("robot.2", "robot.3"),
            actor_kinds=("robot", "robot"),
        )
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_no_compatible$",
    ):
        allocator.acquire(
            session_id="handover.object",
            kind="handover",
            participants=("npc.2", "npc.3"),
            actor_kinds=("npc", "npc"),
            object_id="not-whitelisted",
        )


def test_route_cost_selection_and_station_id_tie_break_are_deterministic() -> None:
    lower_b = InteractionStationAllocator(
        _catalog(),
        route_cost=lambda _participant, _actor_kind, site: (
            0.25 if site.startswith("b.") else 2.0
        ),
    )
    assert _acquire_conversation(lower_b, "session.cost").station_id == "conversation.b"

    tied = InteractionStationAllocator(
        _catalog(), route_cost=lambda _participant, _actor_kind, _site: 1.0
    )
    assert _acquire_conversation(tied, "session.tie").station_id == "conversation.a"


def test_preferred_station_busy_does_not_fall_back() -> None:
    allocator = InteractionStationAllocator(_catalog())
    _acquire_conversation(
        allocator,
        "session.first",
        preferred_station_id="conversation.a",
    )
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_busy$",
    ):
        _acquire_conversation(
            allocator,
            "session.second",
            ("npc.3", "npc.4"),
            preferred_station_id="conversation.a",
        )


def test_acquire_and_release_are_idempotent_and_detect_conflicts() -> None:
    allocator = InteractionStationAllocator(_catalog())
    lease = _acquire_conversation(allocator, "session.idempotent")
    assert allocator.lease(lease.lease_id) is lease
    assert allocator.lease_for_session("session.idempotent") is lease
    assert allocator.lease("unknown") is None
    assert allocator.lease_for_session("unknown") is None
    assert _acquire_conversation(allocator, "session.idempotent") is lease
    assert len(allocator.events()) == 1
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_session_conflict$",
    ):
        _acquire_conversation(
            allocator,
            "session.idempotent",
            ("npc.1", "npc.other"),
        )

    terminal = allocator.release(
        lease.lease_id,
        terminal_receipt_id="receipt.1",
        outcome="succeeded",
    )
    assert terminal.outcome == InteractionLeaseOutcome.SUCCEEDED
    assert allocator.release(
        lease.lease_id,
        terminal_receipt_id="receipt.1",
        outcome="succeeded",
    ) is terminal
    assert _acquire_conversation(allocator, "session.idempotent") is terminal
    assert allocator.lease(lease.lease_id) is terminal
    assert allocator.lease_for_session("session.idempotent") is terminal
    assert allocator.snapshot().active_leases == ()
    assert len(allocator.events()) == 2
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_release_conflict$",
    ):
        allocator.release(
            lease.lease_id,
            terminal_receipt_id="receipt.other",
            outcome="failed",
        )


@pytest.mark.parametrize("outcome", ["cancelled", "timed_out"])
def test_cancel_and_timeout_release_all_resources(outcome: str) -> None:
    allocator = InteractionStationAllocator(_catalog())
    lease = _acquire_conversation(allocator, f"session.{outcome}")
    terminal = allocator.release(
        lease.lease_id,
        terminal_receipt_id=f"receipt.{outcome}",
        outcome=outcome,
    )
    assert terminal.outcome is InteractionLeaseOutcome(outcome)
    assert all(allocator.owner(resource) is None for resource in lease.resources)
    assert [event.event_type for event in allocator.events()] == [
        "acquired",
        "released",
    ]


def test_release_requires_terminal_receipt_and_terminal_outcome() -> None:
    allocator = InteractionStationAllocator(_catalog())
    lease = _acquire_conversation(allocator, "session.release")
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_terminal_receipt_required$",
    ):
        allocator.release(lease.lease_id, terminal_receipt_id="", outcome="failed")
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_terminal_outcome_required$",
    ):
        allocator.release(
            lease.lease_id,
            terminal_receipt_id="receipt.invalid",
            outcome="running",
        )


def test_reconstruction_restores_active_and_terminal_sessions() -> None:
    catalog = _catalog()
    journal = InMemoryInteractionLeaseJournal()
    allocator = InteractionStationAllocator(catalog, journal=journal)
    active = _acquire_conversation(
        allocator,
        "session.active",
        preferred_station_id="conversation.a",
    )
    released = _acquire_conversation(
        allocator,
        "session.released",
        ("npc.3", "npc.4"),
        preferred_station_id="conversation.b",
    )
    terminal = allocator.release(
        released.lease_id,
        terminal_receipt_id="receipt.released",
        outcome="failed",
    )

    restored = InteractionStationAllocator.reconstruct(catalog, journal)
    assert restored.events() == journal.events()
    assert restored.snapshot().active_leases == (active,)
    assert all(restored.owner(resource) == active.lease_id for resource in active.resources)
    assert all(restored.owner(resource) is None for resource in released.resources)
    event_count = len(restored.events())
    assert _acquire_conversation(
        restored,
        "session.released",
        ("npc.3", "npc.4"),
        preferred_station_id="conversation.b",
    ) == terminal
    assert len(restored.events()) == event_count

    replacement = _acquire_conversation(
        restored,
        "session.replacement",
        ("npc.5", "npc.6"),
        preferred_station_id="conversation.b",
    )
    assert replacement.station_id == "conversation.b"
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_busy$",
    ):
        _acquire_conversation(
            restored,
            "session.competing",
            ("npc.7", "npc.8"),
            preferred_station_id="conversation.a",
        )


class _FailingJournal(InMemoryInteractionLeaseJournal):
    def __init__(self, fail_sequence: int) -> None:
        super().__init__()
        self.fail_sequence = fail_sequence

    def append(self, event: InteractionStationLeaseEvent) -> None:
        if event.sequence == self.fail_sequence:
            raise OSError("planted journal failure")
        super().append(event)


def test_acquire_append_failure_leaves_allocator_unchanged() -> None:
    journal = _FailingJournal(1)
    allocator = InteractionStationAllocator(_catalog(), journal=journal)
    with pytest.raises(OSError, match="planted journal failure"):
        _acquire_conversation(allocator, "session.failed-acquire")
    assert allocator.snapshot().active_leases == ()
    assert allocator.snapshot().resource_owners == ()
    assert allocator.lease_for_session("session.failed-acquire") is None
    assert allocator.events() == ()
    assert journal.events() == ()


def test_release_append_failure_preserves_active_lease_and_resources() -> None:
    journal = _FailingJournal(2)
    allocator = InteractionStationAllocator(_catalog(), journal=journal)
    lease = _acquire_conversation(allocator, "session.failed-release")
    with pytest.raises(OSError, match="planted journal failure"):
        allocator.release(
            lease.lease_id,
            terminal_receipt_id="receipt.failed",
            outcome="failed",
        )
    assert allocator.lease(lease.lease_id) is lease
    assert allocator.snapshot().active_leases == (lease,)
    assert all(allocator.owner(resource) == lease.lease_id for resource in lease.resources)
    assert len(allocator.events()) == 1
    assert len(journal.events()) == 1


def test_jsonl_journal_is_deterministic_and_reconstructable(tmp_path: Path) -> None:
    path = tmp_path / "leases.jsonl"
    catalog = _catalog()
    journal = JsonlInteractionLeaseJournal(path)
    allocator = InteractionStationAllocator(catalog, journal=journal)
    lease = _acquire_conversation(allocator, "session.jsonl")
    allocator.release(
        lease.lease_id,
        terminal_receipt_id="receipt.jsonl",
        outcome="succeeded",
    )
    expected = "".join(event.to_json() + "\n" for event in allocator.events())
    assert path.read_text(encoding="utf-8") == expected

    reopened = JsonlInteractionLeaseJournal(path)
    restored = InteractionStationAllocator.reconstruct(catalog, reopened)
    assert restored.events() == allocator.events()
    assert restored.snapshot().active_leases == ()
    assert restored.lease_for_session("session.jsonl").outcome is (
        InteractionLeaseOutcome.SUCCEEDED
    )


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":1,"schema_version":1}\n',
        '{"schema_version":true,"sequence":1,"event_type":"acquired","lease":{}}\n',
        '{"schema_version":1}\n',
        "not-json\n",
        '{}',
    ],
)
def test_jsonl_journal_rejects_malformed_or_duplicate_json(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "malformed.jsonl"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_journal_invalid$",
    ):
        JsonlInteractionLeaseJournal(path)


def test_jsonl_reconstruction_rejects_tampered_resources(tmp_path: Path) -> None:
    path = tmp_path / "tampered.jsonl"
    catalog = _catalog()
    journal = JsonlInteractionLeaseJournal(path)
    allocator = InteractionStationAllocator(catalog, journal=journal)
    _acquire_conversation(allocator, "session.tampered")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["lease"]["resources"].append("site:tampered")
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tampered = JsonlInteractionLeaseJournal(path)
    with pytest.raises(
        InteractionStationAllocationError,
        match="^interaction_station_journal_invalid$",
    ):
        InteractionStationAllocator.reconstruct(catalog, tampered)


def test_reconstruction_rejects_release_before_acquire_and_mismatch() -> None:
    catalog = _catalog()
    source = InteractionStationAllocator(catalog)
    lease = _acquire_conversation(source, "session.source")
    source.release(
        lease.lease_id,
        terminal_receipt_id="receipt.source",
        outcome="failed",
    )
    acquired, released = source.events()
    release_first = replace(released, sequence=1)
    mismatched = replace(
        released,
        lease=replace(released.lease, station_id="conversation.b"),
    )
    for records in ((release_first,), (acquired, mismatched)):
        with pytest.raises(
            InteractionStationAllocationError,
            match="^interaction_station_journal_invalid$",
        ):
            InteractionStationAllocator.reconstruct(
                catalog, InMemoryInteractionLeaseJournal(records)
            )


def test_reconstruction_rejects_resource_conflict_and_nonmonotonic_event() -> None:
    catalog = _catalog()
    first_allocator = InteractionStationAllocator(catalog)
    first = _acquire_conversation(
        first_allocator,
        "session.first",
        preferred_station_id="conversation.a",
    )
    second_allocator = InteractionStationAllocator(catalog)
    second = _acquire_conversation(
        second_allocator,
        "session.second",
        ("npc.3", "npc.4"),
        preferred_station_id="conversation.a",
    )
    conflict = (first_allocator.events()[0], replace(second_allocator.events()[0], sequence=2))
    nonmonotonic = (first_allocator.events()[0], first_allocator.events()[0])
    for records in (conflict, nonmonotonic):
        with pytest.raises(
            InteractionStationAllocationError,
            match="^interaction_station_journal_invalid$",
        ):
            InteractionStationAllocator.reconstruct(
                catalog, InMemoryInteractionLeaseJournal(records)
            )
    assert first.station_id == second.station_id == "conversation.a"
