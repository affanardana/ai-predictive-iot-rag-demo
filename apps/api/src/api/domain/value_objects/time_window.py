"""Time windows and series resolution.

The PRD fixes the windows users may inspect (1h, 5h, 24h, 7d, 30d) and the
non-functional requirements demand that historical queries not transfer large
raw datasets to the frontend. `SeriesResolution` is the value object that
carries how a series was reduced, so the response can always state whether it
contains measurements or aggregates.

That marker also keeps a Phase 10 question open: the PRD calls a statement
like "vibration increased over the last five hours" *Observed*, but a value
derived from bucketed averages is a statistic rather than a measurement.
Recording the resolution on every response means either reading of the
evidence model stays implementable without changing the telemetry endpoints.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from api.domain.timestamps import ensure_aware

#: One reading per minute, per the masterplan's dataset definition.
DEFAULT_SAMPLE_INTERVAL = timedelta(minutes=1)

#: Beyond this many points a series is bucketed before leaving the API.
DEFAULT_MAX_RAW_POINTS = 720


class Aggregation(StrEnum):
    """How points within a bucket were combined."""

    RAW = "RAW"
    MEAN = "MEAN"
    MIN = "MIN"
    MAX = "MAX"


class TimeWindow(StrEnum):
    """A selectable history window."""

    ONE_HOUR = "1h"
    FIVE_HOURS = "5h"
    ONE_DAY = "24h"
    SEVEN_DAYS = "7d"
    THIRTY_DAYS = "30d"

    @property
    def duration(self) -> timedelta:
        """Return the wall-clock length of the window."""
        return _WINDOW_DURATIONS[self]

    def start_from(self, end: datetime) -> datetime:
        """Return the inclusive start of this window ending at `end`."""
        return end - self.duration


_WINDOW_DURATIONS: Mapping[TimeWindow, timedelta] = {
    TimeWindow.ONE_HOUR: timedelta(hours=1),
    TimeWindow.FIVE_HOURS: timedelta(hours=5),
    TimeWindow.ONE_DAY: timedelta(hours=24),
    TimeWindow.SEVEN_DAYS: timedelta(days=7),
    TimeWindow.THIRTY_DAYS: timedelta(days=30),
}


@dataclass(frozen=True, slots=True)
class SeriesResolution:
    """How a returned telemetry series was reduced before transmission."""

    bucket: timedelta | None
    aggregation: Aggregation

    def __post_init__(self) -> None:
        """Keep the two fields consistent: raw series have no bucket."""
        if self.bucket is None and self.aggregation is not Aggregation.RAW:
            raise ValueError("A series with no bucket must use the RAW aggregation.")
        if self.bucket is not None and self.aggregation is Aggregation.RAW:
            raise ValueError("A bucketed series must use a non-RAW aggregation.")

    @property
    def is_raw(self) -> bool:
        """Whether the series contains unmodified measurements."""
        return self.bucket is None

    @classmethod
    def for_window(
        cls,
        window: TimeWindow,
        sample_interval: timedelta = DEFAULT_SAMPLE_INTERVAL,
        max_raw_points: int = DEFAULT_MAX_RAW_POINTS,
        aggregation: Aggregation = Aggregation.MEAN,
    ) -> SeriesResolution:
        """Choose a resolution that keeps a window under `max_raw_points`."""
        if sample_interval <= timedelta(0):
            raise ValueError("Sample interval must be positive.")
        if max_raw_points < 1:
            raise ValueError("Max raw points must be at least 1.")

        point_count = window.duration / sample_interval
        if point_count <= max_raw_points:
            return cls(bucket=None, aggregation=Aggregation.RAW)

        bucket_seconds = (
            math.ceil(
                (window.duration.total_seconds() / max_raw_points) / sample_interval.total_seconds()
            )
            * sample_interval.total_seconds()
        )
        return cls(bucket=timedelta(seconds=bucket_seconds), aggregation=aggregation)


@dataclass(frozen=True, slots=True)
class ResolvedWindow:
    """The instants a telemetry window was actually resolved against.

    `SeriesResolution` says how a series was *reduced*; this says where it was
    *cut*. They are separate because the cut is not always what the caller
    asked for. `TimeWindow` names a duration, and a duration needs an anchor --
    and the anchor is the newest stored reading rather than the wall clock, so
    that a simulator whose timestamps run ahead of real time still produces a
    series (see `GetTelemetryHistory`).

    Carried on the response because a consumer must be able to tell. Without
    it, a client holding a `1h` series has no way to know whether it covers the
    last hour of wall-clock time or an hour of data ending somewhere else --
    and during a fast-forwarded demonstration the answer is the second, which
    looks like a bug in the chart until it is stated.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        """Reject naive timestamps and an inverted interval."""
        ensure_aware(self.start, "start")
        ensure_aware(self.end, "end")
        if self.start > self.end:
            raise ValueError(
                f"Resolved window start ({self.start.isoformat()}) is after its "
                f"end ({self.end.isoformat()})."
            )

    @property
    def duration(self) -> timedelta:
        """Return the wall-clock length of the resolved interval."""
        return self.end - self.start
