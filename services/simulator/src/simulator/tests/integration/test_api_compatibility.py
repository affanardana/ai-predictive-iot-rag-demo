"""The simulator emits data the API will accept.

A test-only import across the service boundary. The two services deliberately do
not share code — see the Phase 2 plan — so this is where their agreement is
enforced. If either side changes the telemetry shape, the build fails here
rather than the pipeline failing at ingestion, where the symptom would be
dropped messages with no obvious cause.

The import is confined to tests. Neither service depends on the other at
runtime, so the simulator can still be deployed on its own.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from api.domain.value_objects.machine_id import MachineId as ApiMachineId
from api.domain.value_objects.sensor_reading import SensorReading as ApiSensorReading
from api.infrastructure.persistence.sql.models.machine import MACHINE_ID_MAX_LENGTH
from api.infrastructure.persistence.sql.models.telemetry import (
    EVENT_ID_MAX_LENGTH,
    SESSION_ID_MAX_LENGTH,
)
from api.presentation.schemas.telemetry_ingest import TelemetryIngestItem
from simulator.domain.engine import MachineSimulator
from simulator.domain.readings import signal_names
from simulator.domain.scenario import Scenario
from simulator.domain.session import MAX_SESSION_ID_LENGTH, SimulationSession
from simulator.infrastructure.sinks.jsonl import json_default
from simulator.tests.conftest import FIXED_START


def _session(scenario: Scenario, machine_id: str = "M001") -> SimulationSession:
    """A run short enough to validate every tick of."""
    return SimulationSession.create(
        machine_ids=[machine_id],
        scenario=scenario,
        seed=2026,
        duration=timedelta(minutes=120),
        started_at=FIXED_START,
    )


@pytest.mark.parametrize("scenario", list(Scenario))
def test_every_generated_reading_is_accepted_by_the_api(scenario: Scenario) -> None:
    """Every tick, not a sample.

    The API rejects a value the moment it violates its invariants, and one bad
    tick in a long run is still a dataset that cannot be ingested — so validating
    the first reading would prove almost nothing.
    """
    session = _session(scenario)
    simulator = MachineSimulator(session, session.machines[0])

    for index in range(simulator.tick_count):
        reading = simulator.tick(index).telemetry.reading
        accepted = ApiSensorReading(**reading.as_row())

        # Accepted, and unchanged: a lossy conversion would be its own bug.
        assert accepted.temperature == reading.temperature
        assert accepted.vibration == reading.vibration
        assert accepted.rpm == reading.rpm


def test_the_signal_order_matches_the_api() -> None:
    """Signal order is part of the contract, not a detail.

    The API's bucketed telemetry query maps result columns back onto signals by
    position. A different order here would silently transpose two signals —
    temperature reported as vibration — with nothing failing.
    """
    assert signal_names() == ApiSensorReading.signal_names()


def test_generated_machine_ids_are_accepted_by_the_api() -> None:
    """A fleet the simulator invents has to be one the API will register."""
    for index in range(1, 40):
        machine_id = f"M{index:03d}"

        assert str(ApiMachineId(machine_id)) == machine_id


def test_identifiers_fit_the_api_column_widths() -> None:
    """Read against the API's own constants, not restated numbers.

    Taking the limits from the API's models means this fails if *it* widens or
    narrows a column, rather than only if the simulator changes.
    """
    session = SimulationSession.create(
        machine_ids=["PUMP012"],
        scenario=Scenario.NORMAL,
        seed=1,
        duration=timedelta(minutes=1),
        started_at=FIXED_START,
        session_id="s" * MAX_SESSION_ID_LENGTH,
    )
    sample = MachineSimulator(session, session.machines[0]).tick(0).telemetry

    assert len(sample.event_id) <= EVENT_ID_MAX_LENGTH
    assert len(sample.session_id) <= SESSION_ID_MAX_LENGTH
    assert len(sample.machine_id) <= MACHINE_ID_MAX_LENGTH


def test_timestamps_carry_a_timezone() -> None:
    """The API stores them in `timestamptz` and rejects a naive value outright."""
    session = _session(Scenario.NORMAL)
    sample = MachineSimulator(session, session.machines[0]).tick(0).telemetry

    assert sample.recorded_at.tzinfo is not None
    assert sample.recorded_at.utcoffset() is not None


def test_the_published_payload_is_accepted_by_the_ingest_schema() -> None:
    """The wire shape, through the schema that will receive it.

    Every other telemetry schema in the API nests the six signals under
    `reading`; the ingest one must not, because the wire format is what the
    simulator publishes. This runs a real payload through the real schema
    instead of comparing hand-written examples, which is the failure mode worth
    catching: two sides that each agree with a docstring and not with each
    other.

    Serialised and reparsed through JSON first, because that round trip is what
    the broker performs -- an ISO string on one side and a `datetime` on the
    other is exactly the sort of difference a direct call would hide.
    """
    session = _session(Scenario.NORMAL)
    sample = MachineSimulator(session, session.machines[0]).tick(0).telemetry

    on_the_wire = json.loads(json.dumps(sample.as_row(), default=json_default))
    item = TelemetryIngestItem(**on_the_wire)

    assert item.event_id == sample.event_id
    assert item.machine_id == sample.machine_id
    assert item.session_id == sample.session_id
    assert item.recorded_at == sample.recorded_at
    assert item.temperature == sample.reading.temperature
    assert item.vibration == sample.reading.vibration


def test_the_payload_is_flat_rather_than_nested() -> None:
    """A guard on the one structural difference between the two vocabularies.

    The API's responses group the signals under `reading`. If the simulator ever
    adopted that grouping, the ingest schema would reject every message -- and
    it would do so at runtime, in the pipeline, rather than here.
    """
    session = _session(Scenario.NORMAL)
    row = MachineSimulator(session, session.machines[0]).tick(0).telemetry.as_row()

    assert "reading" not in row
    assert set(signal_names()) <= set(row)
