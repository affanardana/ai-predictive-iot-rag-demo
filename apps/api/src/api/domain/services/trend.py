"""What a signal has been doing over a window.

`PRD.md` section 15 asks the Copilot for trends, section 8 requires historical
analysis, and `MASTERPLAN.md` names `get_machine_trend()` among the seven tools —
but until now the API had no statistic of any kind. It served raw series and
bucketed aggregates, and a trend was something a reader did in their head.

**The endpoints are `OBSERVED`; the fit is `INFERRED`.** `CHANGELOG.md` recorded
that as an open question for this phase, and the answer is that they are two
different claims. *"Vibration was 1.42 mm/s then and 2.31 mm/s now"* is two
readings and a subtraction — `PRD.md` section 15 lists exactly that under
Observed. *"It fitted a rise of 0.89 mm/s over the window"* is a derivation,
whose answer depends on the method chosen, and section 16's closing line forbids
presenting that as directly observed. A trend therefore returns both, each
carrying what makes it checkable: the endpoints, and the method.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from api.domain.read_models import TelemetrySeries
from api.domain.value_objects.sensor_reading import SensorReading

#: How much a signal must move, relative to its own mean, before its direction
#: is called rather than left flat. Two per cent separates the corpus's rising
#: vibration (1.4 → 2.3 mm/s, 64%) from ordinary sensor noise, which is what
#: stops every question being answered "vibration is rising".
DEADBAND_RATIO = 0.02


class TrendDirection(StrEnum):
    """Which way a signal moved."""

    RISING = "RISING"
    FALLING = "FALLING"
    FLAT = "FLAT"


@dataclass(frozen=True, slots=True)
class Trend:
    """One signal's movement across a series."""

    signal: str
    #: The series' own endpoints, in the series' own order.
    first: float
    last: float
    #: The least-squares slope's total movement across the window. Preferred
    #: over `last - first` because a bucketed series' endpoints are two single
    #: buckets: at 30 days that compares two hours and ignores the other 718.
    fitted_change: float
    direction: TrendDirection
    samples: int
    #: How the points were derived, never omitted. `TelemetrySeries` already
    #: distinguishes a measurement from an aggregate, and a trend that hid
    #: which it used would undo that.
    resolution: str

    @property
    def change(self) -> float:
        """The plain difference between the endpoints."""
        return self.last - self.first

    @property
    def percent_change(self) -> float | None:
        """The fitted change as a percentage of where it started, or None at zero."""
        if self.first == 0:
            return None
        return 100.0 * self.fitted_change / abs(self.first)

    @property
    def summary(self) -> str:
        """Render the endpoints and the fitted change, both, for the model."""
        movement = (
            "flat"
            if self.direction is TrendDirection.FLAT
            else f"{self.direction.value.lower()} by {abs(self.fitted_change):g}"
        )
        return (
            f"{self.signal}: {self.first:g} then {self.last:g} "
            f"(change {self.change:+g}), {movement} across the window "
            f"({self.resolution})"
        )

    @property
    def endpoints(self) -> str:
        """Render only what was read, for the OBSERVED statement."""
        return f"{self.signal} was {self.first:g} and is now {self.last:g}"

    @property
    def fit(self) -> str:
        """Render the derivation and its method, for the INFERRED statement."""
        return (
            f"{self.signal} fitted a change of {self.fitted_change:+g} across the window "
            f"(least-squares slope over {self.samples} points, {self.resolution})"
        )


def summarise(
    series: TelemetrySeries,
    *,
    deadband_ratio: float = DEADBAND_RATIO,
) -> tuple[Trend, ...]:
    """Return one trend per signal in `series`.

    A series with fewer than two points has no direction to state, so it yields
    nothing rather than a trend of zero change — which would read as "steady"
    when the truth is "unknown".
    """
    points = series.points
    if len(points) < 2:
        return ()

    resolution = _resolution(series)
    trends: list[Trend] = []
    for signal in SensorReading.signal_names():
        values = [float(getattr(point.reading, signal)) for point in points]
        fitted = _fitted_change(values)
        trends.append(
            Trend(
                signal=signal,
                first=values[0],
                last=values[-1],
                fitted_change=fitted,
                direction=_direction(values, fitted, deadband_ratio),
                samples=len(values),
                resolution=resolution,
            )
        )
    return tuple(trends)


def _fitted_change(values: list[float]) -> float:
    """Return the least-squares slope's movement across the whole series.

    Fitted over the point index rather than over elapsed time: the points are
    evenly spaced by construction — raw readings at the sample interval, or
    buckets of a fixed width — so the index is the clock, and using it avoids a
    second place where timestamps could be mishandled.
    """
    count = len(values)
    mean_index = (count - 1) / 2
    mean_value = sum(values) / count
    covariance = sum(
        (index - mean_index) * (value - mean_value) for index, value in enumerate(values)
    )
    variance = sum((index - mean_index) ** 2 for index in range(count))
    if variance == 0:
        return 0.0
    slope = covariance / variance
    return slope * (count - 1)


def _direction(values: list[float], fitted_change: float, deadband_ratio: float) -> TrendDirection:
    """Return which way the values moved, or FLAT inside the deadband."""
    mean = sum(abs(value) for value in values) / len(values)
    if mean == 0:
        return TrendDirection.FLAT
    threshold = deadband_ratio * mean
    if fitted_change > threshold:
        return TrendDirection.RISING
    if fitted_change < -threshold:
        return TrendDirection.FALLING
    return TrendDirection.FLAT


def _resolution(series: TelemetrySeries) -> str:
    """Describe how the points were derived."""
    bucket = series.resolution.bucket
    if bucket is None:
        return "each stored measurement"
    minutes = int(bucket.total_seconds() // 60) or 1
    return f"{series.resolution.aggregation.value.lower()} of {minutes}-minute buckets"
