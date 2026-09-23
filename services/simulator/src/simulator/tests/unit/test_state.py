"""State value objects: their invariants, and what they serialise to."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from simulator.domain.errors import SimulationValidationError
from simulator.domain.readings import SIGNAL_NAMES, SensorReading
from simulator.domain.scenario import Scenario
from simulator.domain.state import (
    CHANNEL_NAMES,
    DegradationState,
    GroundTruthState,
    TelemetrySample,
)

MOMENT = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)


def test_degradation_channels_accept_the_full_range() -> None:
    """Both endpoints are valid: no damage, and total damage."""
    assert DegradationState().worst() == 0.0
    assert DegradationState(bearing_wear=1.0).worst() == 1.0


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_degradation_channels_outside_the_range_are_rejected(value: float) -> None:
    """A channel is a fraction. Anything else is a physics bug, not data."""
    with pytest.raises(SimulationValidationError):
        DegradationState(bearing_wear=value)


def test_worst_returns_the_most_advanced_mechanism() -> None:
    """The channel health is derived from."""
    state = DegradationState(bearing_wear=0.2, thermal_stress=0.7, load_stress=0.4)

    assert state.worst() == pytest.approx(0.7)


def test_ground_truth_serialises_its_hidden_fields() -> None:
    """The row carries the scenario and the channels, which is what labelling needs."""
    state = GroundTruthState(
        machine_id="M003",
        recorded_at=MOMENT,
        scenario=Scenario.OVERHEATING,
        degradation=DegradationState(thermal_stress=0.6),
        health_index=0.4,
        failure_imminent=False,
    )

    assert set(state.as_row()) == {
        "machine_id",
        "recorded_at",
        "scenario",
        *CHANNEL_NAMES,
        "health_index",
        "failure_imminent",
    }
    assert state.as_row()["scenario"] == "OVERHEATING"


@pytest.mark.parametrize("health", [-0.01, 1.01])
def test_a_health_index_outside_the_range_is_rejected(health: float) -> None:
    """Health is a fraction, like the channels it derives from."""
    with pytest.raises(SimulationValidationError):
        GroundTruthState(
            machine_id="M003",
            recorded_at=MOMENT,
            scenario=Scenario.NORMAL,
            degradation=DegradationState(),
            health_index=health,
            failure_imminent=False,
        )


def _sample(**overrides: object) -> TelemetrySample:
    """Build a sample, overriding only what a test cares about."""
    values: dict[str, object] = {
        "event_id": "evt-1",
        "machine_id": "M003",
        "recorded_at": MOMENT,
        "session_id": "sim-bd-1",
        "reading": SensorReading(
            temperature=65.0,
            vibration=1.4,
            rpm=1480.0,
            current=12.5,
            load=0.72,
            voltage=400.0,
        ),
    }
    values.update(overrides)
    return TelemetrySample(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["event_id", "machine_id", "session_id"])
def test_a_blank_identifier_is_rejected(field: str) -> None:
    """Every identifier is used as a key or a column; a blank one is unusable."""
    with pytest.raises(SimulationValidationError):
        _sample(**{field: "  "})


def test_a_naive_timestamp_is_rejected() -> None:
    """The API stores these in `timestamptz` and would refuse a naive value."""
    with pytest.raises(SimulationValidationError):
        _sample(recorded_at=datetime(2026, 9, 23, 12, 0))


def test_a_sample_serialises_to_identity_and_signals() -> None:
    """The telemetry row's shape is the contract with everything downstream."""
    assert set(_sample().as_row()) == {
        "event_id",
        "machine_id",
        "recorded_at",
        "session_id",
        *SIGNAL_NAMES,
    }
