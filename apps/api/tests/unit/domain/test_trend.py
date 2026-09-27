"""What a signal has been doing, computed from hand-worked series."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from api.domain.read_models import TelemetryPoint, TelemetrySeries
from api.domain.services.trend import Trend, TrendDirection, summarise
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.sensor_reading import SensorReading
from api.domain.value_objects.time_window import (
    Aggregation,
    ResolvedWindow,
    SeriesResolution,
    TimeWindow,
)
from tests.support.factories import make_reading

START = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def a_series(*vibrations: float, bucket_minutes: int | None = 1) -> TelemetrySeries:
    """Build a series whose vibration follows the values given."""
    points = tuple(
        TelemetryPoint(
            timestamp=START + timedelta(minutes=index),
            reading=make_reading(vibration=value),
            sample_count=1,
        )
        for index, value in enumerate(vibrations)
    )
    resolution = (
        SeriesResolution(bucket=None, aggregation=Aggregation.RAW)
        if bucket_minutes is None
        else SeriesResolution(
            bucket=timedelta(minutes=bucket_minutes), aggregation=Aggregation.MEAN
        )
    )
    return TelemetrySeries(
        machine_id=MachineId("M003"),
        window=TimeWindow.FIVE_HOURS,
        resolution=resolution,
        interval=ResolvedWindow(start=START, end=START + timedelta(hours=5)),
        points=points,
    )


def vibration_of(series: TelemetrySeries) -> Trend:
    """Return the vibration trend from a summary."""
    return next(trend for trend in summarise(series) if trend.signal == "vibration")


def test_a_ramp_is_rising() -> None:
    """The demonstration's shape: 1.4 climbing to 2.3."""
    trend = vibration_of(a_series(1.4, 1.6, 1.9, 2.1, 2.3))

    assert trend.direction is TrendDirection.RISING
    assert trend.first == 1.4
    assert trend.last == 2.3
    assert trend.fitted_change > 0


def test_a_descent_is_falling() -> None:
    """The other direction, so the sign is not a coincidence of the input."""
    trend = vibration_of(a_series(2.3, 2.1, 1.9, 1.6, 1.4))

    assert trend.direction is TrendDirection.FALLING
    assert trend.fitted_change < 0


def test_noise_inside_the_deadband_is_flat() -> None:
    """Otherwise every question would be answered "vibration is rising"."""
    trend = vibration_of(a_series(1.40, 1.41, 1.39, 1.40, 1.40))

    assert trend.direction is TrendDirection.FLAT


def test_the_fit_ignores_a_single_noisy_endpoint() -> None:
    """The case that separates a fitted slope from `last - first`.

    The series rises and then one bucket reads low. The endpoints say the
    signal fell; the fit says it rose, which is what the other four points are
    saying. This is why the statistic is a slope rather than a subtraction, and
    a series steep enough that one bad bucket reverses the endpoints is the case
    that separates them.
    """
    trend = vibration_of(a_series(2.0, 2.3, 2.6, 2.9, 1.9))

    assert trend.fitted_change > 0
    assert trend.direction is TrendDirection.RISING
    # The endpoints still report what was read, and they disagree -- which is
    # why both are shown rather than one being chosen for the reader.
    assert trend.change < 0


def test_a_constant_zero_signal_has_no_percentage() -> None:
    """A percentage of zero is not a number, and must not be one."""
    trend = vibration_of(a_series(0.0, 0.0, 0.0))

    assert trend.direction is TrendDirection.FLAT
    assert trend.percent_change is None


def test_a_single_point_has_no_trend() -> None:
    """One reading has no direction, and saying "flat" would be a claim."""
    assert summarise(a_series(1.4)) == ()


def test_an_empty_series_has_no_trend() -> None:
    """Nothing recorded is not the same as steady."""
    assert summarise(a_series()) == ()


def test_every_trend_states_its_resolution() -> None:
    """A mean of five readings and a reading are different claims."""
    raw = vibration_of(a_series(1.0, 1.2, bucket_minutes=None))
    bucketed = vibration_of(a_series(1.0, 1.2, bucket_minutes=5))

    assert raw.resolution == "each stored measurement"
    assert bucketed.resolution == "mean of 5-minute buckets"


def test_the_summary_carries_both_claims() -> None:
    """The reader gets the endpoints and the fit, each labelled."""
    trend = vibration_of(a_series(1.4, 1.9, 2.3))

    assert "1.4" in trend.endpoints and "2.3" in trend.endpoints
    assert "least-squares" in trend.fit
    # The summary is what the model reads, so it carries the numbers and the
    # resolution; the method lives on `fit`, which is the evidence line a reader
    # checks the derivation against.
    assert "1.4" in trend.summary and "1-minute buckets" in trend.summary


def test_every_signal_is_summarised() -> None:
    """A machine's condition is not one number."""
    trends = summarise(a_series(1.0, 2.0))

    assert {trend.signal for trend in trends} == set(SensorReading.signal_names())
