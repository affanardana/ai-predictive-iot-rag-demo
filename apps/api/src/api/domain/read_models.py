"""Read models for telemetry history.

These describe a *view* of stored telemetry rather than stored state, which is
why they are not entities. A series is defined by the window it covers and the
resolution it was reduced to, so a consumer always knows whether it is looking
at measurements or at aggregates -- see `SeriesResolution`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from api.domain.timestamps import ensure_aware
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.sensor_reading import SensorReading
from api.domain.value_objects.time_window import ResolvedWindow, SeriesResolution, TimeWindow


@dataclass(frozen=True, slots=True)
class TelemetryPoint:
    """A single point on a telemetry series.

    One shape serves both raw and aggregated series: `sample_count` is 1 for a
    raw measurement and the number of readings combined for a bucket. Keeping
    a single type means consumers never branch on which kind of series they
    received, and `TelemetrySeries.resolution` carries the interpretation.
    """

    timestamp: datetime
    reading: SensorReading
    sample_count: int

    def __post_init__(self) -> None:
        """Validate the timestamp and sample count."""
        ensure_aware(self.timestamp, "timestamp")
        if self.sample_count < 1:
            raise ValueError("Telemetry point sample_count must be at least 1.")

    @property
    def is_aggregated(self) -> bool:
        """Whether this point combines more than one measurement."""
        return self.sample_count > 1

    @classmethod
    def from_reading(cls, timestamp: datetime, reading: SensorReading) -> TelemetryPoint:
        """Build a raw, single-sample point."""
        return cls(timestamp=timestamp, reading=reading, sample_count=1)


@dataclass(frozen=True, slots=True)
class TelemetrySeries:
    """A window of telemetry, reduced to a stated resolution.

    `interval` is the part of the answer a client cannot infer for itself. The
    window it asked for names a duration; these are the instants that duration
    was applied to, which is what a consumer needs to plot the series against
    the right stretch of time -- and what tells it that the series does not end
    at the wall clock.
    """

    machine_id: MachineId
    window: TimeWindow
    resolution: SeriesResolution
    interval: ResolvedWindow
    points: Sequence[TelemetryPoint]

    def __len__(self) -> int:
        """Return the number of points in the series."""
        return len(self.points)
