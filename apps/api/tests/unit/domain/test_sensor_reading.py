"""Sensor reading value object."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from api.domain.errors import DomainValidationError
from api.domain.value_objects.sensor_reading import SensorReading
from tests.support.factories import make_reading


def _with_signal(field: str, value: float) -> SensorReading:
    """Return a reading identical to the default except for one signal.

    `dataclasses.replace` re-runs `__post_init__`, so this exercises the same
    validation path as constructing a reading from scratch.
    """
    return replace(make_reading(), **{field: value})


def test_accepts_a_plausible_reading() -> None:
    """Ordinary values pass validation unchanged."""
    reading = make_reading(temperature=72.5, vibration=2.3)

    assert reading.temperature == 72.5
    assert reading.vibration == 2.3


@pytest.mark.parametrize(
    "field",
    ["temperature", "vibration", "rpm", "current", "load", "voltage"],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_signals(field: str, value: float) -> None:
    """Every signal must be a real number."""
    with pytest.raises(DomainValidationError):
        _with_signal(field, value)


@pytest.mark.parametrize("field", ["vibration", "rpm", "load", "voltage"])
def test_rejects_negative_values_where_impossible(field: str) -> None:
    """Quantities that cannot go negative are rejected when they do."""
    with pytest.raises(DomainValidationError):
        _with_signal(field, -1.0)


@pytest.mark.parametrize("field", ["temperature", "current"])
def test_allows_negative_values_where_physically_possible(field: str) -> None:
    """Temperature may be below zero, and current may be reported signed.

    Neither is bounded below by physics, so validating a range here would reject
    legitimate data. Physically plausible *ranges* belong to the simulator.
    """
    assert getattr(_with_signal(field, -5.0), field) == -5.0


def test_signal_names_defines_the_canonical_order() -> None:
    """The declared signal order matches the dataclass fields.

    This order is what the SQL aggregation relies on to map result columns back
    onto signals, so a mismatch would silently transpose two signals rather
    than fail.
    """
    declared = SensorReading.signal_names()

    assert declared == ("temperature", "vibration", "rpm", "current", "load", "voltage")
    assert set(declared) == set(SensorReading.__dataclass_fields__)


def test_reading_is_frozen() -> None:
    """A measurement cannot be edited after the fact."""
    reading = make_reading()

    with pytest.raises(FrozenInstanceError):
        reading.temperature = 99.0  # type: ignore[misc]
