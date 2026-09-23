"""Sessions: the description of a run, and the identifiers it produces."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from simulator.domain.errors import SimulationValidationError
from simulator.domain.scenario import Scenario
from simulator.domain.session import (
    MAX_EVENT_ID_LENGTH,
    MAX_SESSION_ID_LENGTH,
    SimulationSession,
    default_session_id,
)
from simulator.tests.conftest import FIXED_START


def _create(**overrides: Any) -> SimulationSession:  # noqa: ANN401 - test helper
    """Build a session, overriding only what a test cares about."""
    parameters: dict[str, Any] = {
        "machine_ids": ["M001"],
        "scenario": Scenario.BEARING_DEGRADATION,
        "seed": 1234,
        "duration": timedelta(hours=1),
        "started_at": FIXED_START,
    }
    parameters.update(overrides)
    return SimulationSession.create(**parameters)


def test_tick_count_matches_the_masterplan_arithmetic() -> None:
    """Thirty days at one reading per minute is the dataset figure.

    `MASTERPLAN.md` §6 states 43,200 records per machine for that shape. Getting
    this wrong would silently change every downstream dataset size, so it is
    asserted against the documented number rather than against a formula.
    """
    month = _create(duration=timedelta(days=30))

    assert month.tick_count == 43_200


def test_the_final_instant_is_exclusive() -> None:
    """A one-hour run samples at 0..59 minutes, not 0..60."""
    assert _create(duration=timedelta(hours=1)).tick_count == 60


def test_one_profile_is_built_per_machine() -> None:
    """The fleet is whatever the caller asked for, in order."""
    session = _create(machine_ids=["M003", "M001", "M002"])

    assert [profile.machine_id for profile in session.machines] == ["M003", "M001", "M002"]


def test_profiles_are_reproducible_from_the_seed() -> None:
    """Same seed, same fleet — including each machine's susceptibility."""
    first = _create(machine_ids=["M001", "M002"])
    second = _create(machine_ids=["M001", "M002"])

    assert first.machines == second.machines


def test_adding_a_machine_does_not_disturb_the_existing_ones() -> None:
    """Each machine is seeded on its own identity, not on its position.

    Otherwise growing a fleet for a later Phase 3 run would silently rewrite the
    data already generated, and a dataset would not be reproducible.
    """
    two = _create(machine_ids=["M001", "M002"])
    three = _create(machine_ids=["M001", "M002", "M003"])

    assert two.machines[0] == three.machines[0]
    assert two.machines[1] == three.machines[1]


def test_event_ids_identify_session_machine_and_index() -> None:
    """The key is readable and reconstructible, not merely unique."""
    session = _create(machine_ids=["M003"])

    event_id = session.event_id("M003", 7)

    assert session.session_id in event_id
    assert "M003" in event_id
    assert event_id.endswith("00000007")


def test_event_ids_fit_the_column_for_the_worst_case() -> None:
    """A maximal session id, a maximal machine id, and a large index.

    The API's `event_id` column is 64 characters. This asserts the budget holds
    at its limit rather than only for the values a test happens to use.
    """
    session = _create(machine_ids=["PUMP012"], session_id="s" * MAX_SESSION_ID_LENGTH)

    assert len(session.event_id("PUMP012", 99_999_999)) <= MAX_EVENT_ID_LENGTH


def test_default_session_id_is_readable_and_bounded() -> None:
    """A session id appears in every event id and in log output."""
    session_id = default_session_id(Scenario.BEARING_DEGRADATION, 1234)

    assert session_id == "sim-bd-1234"
    assert len(session_id) <= MAX_SESSION_ID_LENGTH


@pytest.mark.parametrize(
    "session_id",
    ["", "   ", "s" * (MAX_SESSION_ID_LENGTH + 1)],
)
def test_an_unusable_session_id_is_rejected(session_id: str) -> None:
    """Rejected here rather than as a truncated column at ingestion."""
    with pytest.raises(SimulationValidationError):
        _create(session_id=session_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_interval", timedelta(0)),
        ("sample_interval", timedelta(seconds=-1)),
        ("duration", timedelta(0)),
        ("duration", timedelta(minutes=-5)),
    ],
)
def test_a_degenerate_run_is_rejected(field: str, value: timedelta) -> None:
    """A run that produces nothing, or divides by zero, fails at construction."""
    with pytest.raises(SimulationValidationError):
        _create(**{field: value})


def test_a_session_needs_machines() -> None:
    """An empty fleet would report success while generating nothing."""
    with pytest.raises(SimulationValidationError):
        _create(machine_ids=[])


def test_duplicate_machines_are_rejected() -> None:
    """Two simulators for one machine would emit colliding event ids."""
    with pytest.raises(SimulationValidationError):
        _create(machine_ids=["M001", "M001"])


def test_a_naive_start_instant_is_rejected() -> None:
    """Every generated timestamp derives from this, so it must carry a zone."""
    with pytest.raises(SimulationValidationError):
        _create(started_at=datetime(2026, 9, 23, 12, 0))
