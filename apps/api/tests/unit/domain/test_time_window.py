"""Time windows and series resolution."""

from __future__ import annotations

from datetime import timedelta

import pytest

from api.domain.value_objects.time_window import (
    DEFAULT_MAX_RAW_POINTS,
    DEFAULT_SAMPLE_INTERVAL,
    Aggregation,
    SeriesResolution,
    TimeWindow,
)
from tests.support.factories import DEFAULT_NOW


def test_windows_match_the_prd_allowlist() -> None:
    """Only the five windows the PRD names are selectable."""
    assert {window.value for window in TimeWindow} == {"1h", "5h", "24h", "7d", "30d"}


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (TimeWindow.ONE_HOUR, timedelta(hours=1)),
        (TimeWindow.FIVE_HOURS, timedelta(hours=5)),
        (TimeWindow.ONE_DAY, timedelta(hours=24)),
        (TimeWindow.SEVEN_DAYS, timedelta(days=7)),
        (TimeWindow.THIRTY_DAYS, timedelta(days=30)),
    ],
)
def test_durations(window: TimeWindow, expected: timedelta) -> None:
    """Each window spans the advertised duration."""
    assert window.duration == expected


def test_start_from_subtracts_the_duration() -> None:
    """A window ends at a supplied instant and begins one duration earlier."""
    assert TimeWindow.FIVE_HOURS.start_from(DEFAULT_NOW) == DEFAULT_NOW - timedelta(hours=5)


@pytest.mark.parametrize(
    ("window", "expected_bucket"),
    [
        # Short windows stay raw: at one reading per minute they fit well under
        # the 720-point ceiling.
        (TimeWindow.ONE_HOUR, None),
        (TimeWindow.FIVE_HOURS, None),
        # Longer windows bucket to whole minutes.
        (TimeWindow.ONE_DAY, timedelta(minutes=2)),
        (TimeWindow.SEVEN_DAYS, timedelta(minutes=14)),
        (TimeWindow.THIRTY_DAYS, timedelta(minutes=60)),
    ],
)
def test_resolution_for_each_window(window: TimeWindow, expected_bucket: timedelta | None) -> None:
    """Resolution is chosen from the window's length.

    The exact bucket widths matter: a 14-minute bucket is not a unit
    `date_trunc` accepts, which is why the aggregation floors on the epoch
    instead.
    """
    resolution = SeriesResolution.for_window(window)

    assert resolution.bucket == expected_bucket
    assert resolution.is_raw is (expected_bucket is None)


@pytest.mark.parametrize("window", list(TimeWindow))
def test_bucketing_keeps_series_within_the_point_ceiling(window: TimeWindow) -> None:
    """No window returns more than the configured maximum number of points.

    This is the non-functional requirement that historical queries not ship
    large raw datasets, expressed as an assertion.
    """
    resolution = SeriesResolution.for_window(window)
    point_count = window.duration / DEFAULT_SAMPLE_INTERVAL

    if resolution.is_raw:
        assert point_count <= DEFAULT_MAX_RAW_POINTS
        return

    assert resolution.bucket is not None
    bucketed_points = window.duration / resolution.bucket
    assert bucketed_points <= DEFAULT_MAX_RAW_POINTS


def test_raw_uses_the_raw_aggregation() -> None:
    """A series with no bucket is labelled RAW."""
    assert SeriesResolution.for_window(TimeWindow.ONE_HOUR).aggregation is Aggregation.RAW


def test_rejects_inconsistent_resolutions() -> None:
    """Bucket and aggregation must agree."""
    with pytest.raises(ValueError, match="no bucket"):
        SeriesResolution(bucket=None, aggregation=Aggregation.MEAN)

    with pytest.raises(ValueError, match="non-RAW"):
        SeriesResolution(bucket=timedelta(minutes=5), aggregation=Aggregation.RAW)


@pytest.mark.parametrize(
    ("sample_interval", "max_raw_points"),
    [(timedelta(0), 720), (timedelta(minutes=-1), 720), (timedelta(minutes=1), 0)],
)
def test_rejects_degenerate_configuration(sample_interval: timedelta, max_raw_points: int) -> None:
    """Impossible sampling configurations fail loudly."""
    with pytest.raises(ValueError):
        SeriesResolution.for_window(
            TimeWindow.ONE_HOUR,
            sample_interval=sample_interval,
            max_raw_points=max_raw_points,
        )


def test_honours_a_custom_point_ceiling() -> None:
    """A tighter ceiling forces bucketing on a window that would otherwise be raw."""
    resolution = SeriesResolution.for_window(TimeWindow.ONE_HOUR, max_raw_points=10)

    assert not resolution.is_raw
    assert resolution.bucket is not None
    assert resolution.bucket >= timedelta(minutes=6)
