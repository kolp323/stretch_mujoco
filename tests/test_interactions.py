import pytest

from stretch_mujoco.agents.interactions import InteractionCoordinator, InteractionStatus


def test_handover_barriers_require_physical_participants_in_order() -> None:
    coordinator = InteractionCoordinator()
    session = coordinator.start(
        "handover",
        ("giver", "receiver"),
        "parcel",
        (
            ("ready", ("giver", "receiver")),
            ("released", ("giver",)),
            ("received", ("receiver",)),
        ),
        session_id="handover_1",
    )

    coordinator.acknowledge(session.session_id, "giver", "ready")
    assert session.phase == "ready"
    coordinator.acknowledge(session.session_id, "receiver", "ready")
    coordinator.acknowledge(session.session_id, "giver", "released")
    coordinator.acknowledge(session.session_id, "receiver", "received")

    assert session.status == InteractionStatus.SUCCEEDED
    assert (
        coordinator.start(
            "handover",
            ("giver", "receiver"),
            "parcel",
            (
                ("ready", ("giver", "receiver")),
                ("released", ("giver",)),
                ("received", ("receiver",)),
            ),
            session_id="handover_1",
        )
        is session
    )


def test_interaction_session_id_rejects_different_payload() -> None:
    coordinator = InteractionCoordinator()
    coordinator.start(
        "handover", ("giver", "receiver"), "parcel", (("ready", ("giver",)),), session_id="same"
    )

    with pytest.raises(ValueError, match="interaction_session_id_conflict"):
        coordinator.start(
            "handover", ("giver", "other"), "parcel", (("ready", ("giver",)),), session_id="same"
        )


def test_interaction_rejects_out_of_order_acknowledgement() -> None:
    coordinator = InteractionCoordinator()
    session = coordinator.start("handover", ("robot", "npc"), "parcel", (("released", ("robot",)),))

    with pytest.raises(ValueError, match="Expected interaction phase"):
        coordinator.acknowledge(session.session_id, "robot", "received")
