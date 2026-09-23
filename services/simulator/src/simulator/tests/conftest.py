"""Shared fixtures for the simulator's tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from simulator.domain.engine import MachineSimulator
from simulator.domain.scenario import Scenario
from simulator.domain.session import SimulationSession

#: A fixed instant, so assertions read as constants and nothing depends on the
#: wall clock. The simulation model never reads the clock; this is a value the
#: caller supplies, which is what makes a run reproducible.
FIXED_START = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)

#: Two hours at one reading per minute: long enough for a degradation scenario to
#: reach its steep section and cross into imminent failure.
STANDARD_DURATION = timedelta(minutes=120)


@pytest.fixture
def started_at() -> datetime:
    """Return the fixed instant runs start from."""
    return FIXED_START


@pytest.fixture
def session() -> SimulationSession:
    """A single-machine bearing-degradation run."""
    return SimulationSession.create(
        machine_ids=["M001"],
        scenario=Scenario.BEARING_DEGRADATION,
        seed=1234,
        duration=STANDARD_DURATION,
        started_at=FIXED_START,
    )


@pytest.fixture
def simulator(session: SimulationSession) -> MachineSimulator:
    """A simulator for the session's only machine."""
    return MachineSimulator(session, session.machines[0])
