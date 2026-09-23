"""The engine: determinism, ordering, and what each tick carries."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from simulator.domain.demo import (
    DEMO_DURATION,
    DEMO_MACHINE_ID,
    DEMO_SCENARIO,
    DEMO_SEED,
)
from simulator.domain.engine import MachineSimulator
from simulator.domain.errors import SimulationValidationError
from simulator.domain.readings import SensorReading
from simulator.domain.session import MAX_EVENT_ID_LENGTH, SimulationSession
from simulator.tests.conftest import FIXED_START
from simulator.tests.support import readings_between


def _full_series(session: SimulationSession) -> list[SensorReading]:
    """Every reading in a session, for the session's only machine."""
    simulator = MachineSimulator(session, session.machines[0])
    return readings_between(simulator, 0, simulator.tick_count)


def test_the_same_seed_produces_an_identical_series(session: SimulationSession) -> None:
    """Reproducibility is a product requirement, not a convenience.

    `MASTERPLAN.md` §3.5 requires datasets, experiments, and demonstrations to be
    regenerable. Two simulators built from the same session description must
    therefore agree exactly — not approximately, and not merely in trend.
    """
    assert _full_series(session) == _full_series(session)


def test_a_different_seed_produces_a_different_series(session: SimulationSession) -> None:
    """The seed actually drives the output, rather than being decorative."""
    assert _full_series(session) != _full_series(replace(session, seed=session.seed + 1))


def test_ticks_are_independent_of_the_order_they_are_requested(
    simulator: MachineSimulator,
) -> None:
    """A tick's value depends on its index, not on what came before it.

    This is what lets a realtime run resume at any index and what removes a
    whole class of bug: there is no accumulator to drift, and no reliance on
    ticks having been generated in sequence.
    """
    forward = [simulator.tick(index) for index in range(6)]
    backward = [simulator.tick(index) for index in reversed(range(6))]

    assert forward == list(reversed(backward))


def test_timestamps_advance_by_the_sample_interval(
    simulator: MachineSimulator,
    session: SimulationSession,
) -> None:
    """The simulated clock moves one interval per tick, from the run's start."""
    first = simulator.tick(0).telemetry.recorded_at
    second = simulator.tick(1).telemetry.recorded_at

    assert first == FIXED_START
    assert second - first == session.sample_interval


def test_every_sample_carries_its_session(
    simulator: MachineSimulator,
    session: SimulationSession,
) -> None:
    """Session identity travels with each observation, so provenance is checkable."""
    assert simulator.tick(0).telemetry.session_id == session.session_id


def test_event_ids_are_unique_and_fit_the_column(simulator: MachineSimulator) -> None:
    """Idempotency keys must not repeat, and must fit `String(64)`.

    The API stores `event_id` as the primary key of a 64-character column, so an
    identifier that outgrew it would fail at ingestion rather than here.
    """
    event_ids = {simulator.tick(index).telemetry.event_id for index in range(simulator.tick_count)}

    assert len(event_ids) == simulator.tick_count
    assert max(len(event_id) for event_id in event_ids) <= MAX_EVENT_ID_LENGTH


def test_index_outside_the_run_is_rejected(simulator: MachineSimulator) -> None:
    """Asking for a tick that does not exist fails loudly."""
    with pytest.raises(SimulationValidationError):
        simulator.tick(simulator.tick_count)
    with pytest.raises(SimulationValidationError):
        simulator.tick(-1)


def test_ground_truth_advances_alongside_the_readings(simulator: MachineSimulator) -> None:
    """The hidden state tracks the observable one, step for step.

    Phase 3 labels from these two streams, so they must stay aligned: a label
    that drifted away from its reading would mislabel the whole dataset.
    """
    first = simulator.tick(0)
    last = simulator.tick(simulator.tick_count - 1)

    assert last.ground_truth.health_index < first.ground_truth.health_index
    assert last.ground_truth.failure_imminent
    assert not first.ground_truth.failure_imminent
    assert first.ground_truth.recorded_at == first.telemetry.recorded_at
    assert last.ground_truth.recorded_at == last.telemetry.recorded_at


def test_the_canonical_demonstration_is_reproducible() -> None:
    """PRD section 12: M003, bearing degradation, demo mode, the same sequence.

    A demonstration that cannot be reproduced is not a demonstration, so this
    asserts both halves of the requirement: the sequence is identical every
    time, and it visibly degrades.
    """
    session = SimulationSession.create(
        machine_ids=[DEMO_MACHINE_ID],
        scenario=DEMO_SCENARIO,
        seed=DEMO_SEED,
        duration=DEMO_DURATION,
        started_at=FIXED_START,
    )
    simulator = MachineSimulator(session, session.machines[0])
    readings = _full_series(session)

    # Reproducible.
    assert readings == _full_series(session)

    # And visibly degrading: vibration at least doubles across the run, which is
    # the movement the demonstration exists to show.
    assert readings[-1].vibration > readings[0].vibration * 2
    assert readings[-1].temperature > readings[0].temperature
    assert readings[-1].rpm < readings[0].rpm

    # And the machine actually fails, rather than merely trending.
    assert simulator.tick(simulator.tick_count - 1).ground_truth.failure_imminent


def test_a_run_of_one_tick_is_valid() -> None:
    """The degenerate case works, rather than dividing by zero.

    `elapsed_fraction` divides by the duration; a single-tick run is the
    smallest case where that could go wrong.
    """
    session = SimulationSession.create(
        machine_ids=["M001"],
        scenario=DEMO_SCENARIO,
        seed=DEMO_SEED,
        duration=timedelta(minutes=1),
        started_at=FIXED_START,
    )
    simulator = MachineSimulator(session, session.machines[0])

    assert simulator.tick_count == 1
    assert simulator.tick(0).telemetry.machine_id == "M001"
