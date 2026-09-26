"""Use case: retrieve a window of telemetry at a stated resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from api.domain.errors import MachineNotFoundError
from api.domain.ports.clock import Clock
from api.domain.ports.repositories import TelemetryRepository
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.read_models import TelemetryPoint, TelemetrySeries
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.sensor_reading import SensorReading
from api.domain.value_objects.time_window import (
    DEFAULT_MAX_RAW_POINTS,
    DEFAULT_SAMPLE_INTERVAL,
    ResolvedWindow,
    SeriesResolution,
    TimeWindow,
)

#: Shape returned by the bucketed repository query:
#: `(bucket_start, aggregated_signals, sample_count)`.
BucketRow = tuple[datetime, tuple[float, ...], int]


@dataclass(frozen=True, slots=True)
class GetTelemetryHistory:
    """Return telemetry for a machine over a window, reduced as needed.

    The window boundary is computed from the injected clock rather than from
    wall-clock time, so the 1h/5h/24h/7d/30d windows are deterministic in
    tests.

    Whether the series comes back raw or bucketed is decided here, from the
    window's length, rather than left to the caller. The non-functional
    requirements ask that historical queries not ship large raw datasets, and
    keeping that rule in one place stops each caller from having to remember
    it.
    """

    unit_of_work_factory: UnitOfWorkFactory
    clock: Clock
    sample_interval: timedelta = DEFAULT_SAMPLE_INTERVAL
    max_raw_points: int = DEFAULT_MAX_RAW_POINTS

    async def execute(self, machine_id: MachineId, window: TimeWindow) -> TelemetrySeries:
        """Return the telemetry series for one machine and window.

        Raises:
            MachineNotFoundError: if the machine is not registered. Checked
                explicitly so an unknown machine returns 404 rather than an
                empty series, which would be indistinguishable from a machine
                that is merely quiet.
        """
        resolution = SeriesResolution.for_window(
            window,
            sample_interval=self.sample_interval,
            max_raw_points=self.max_raw_points,
        )
        async with self.unit_of_work_factory() as uow:
            machine = await uow.machines.get(machine_id)
            if machine is None:
                raise MachineNotFoundError(str(machine_id))

            interval = await self._resolve(uow.telemetry, machine_id, window)
            points = await self._collect(
                telemetry=uow.telemetry,
                machine_id=machine_id,
                start=interval.start,
                end=interval.end,
                resolution=resolution,
            )

        return TelemetrySeries(
            machine_id=machine_id,
            window=window,
            resolution=resolution,
            interval=interval,
            points=points,
        )

    async def _resolve(
        self,
        telemetry: TelemetryRepository,
        machine_id: MachineId,
        window: TimeWindow,
    ) -> ResolvedWindow:
        """Return the instants this window covers.

        The end is the **later** of the wall clock and the newest stored
        reading, rather than the wall clock alone.

        The simulator's `recorded_at` advances at `sample_interval x index`
        while its paced loop sleeps `tick_seconds` per tick. Played back at
        300x, the newest stored reading is an hour ahead of real time within
        twelve seconds -- so a window bounded by `clock.now()` selects nothing
        at all, and every chart is empty during the demonstration this project
        exists to give.

        `max` rather than simply taking the newest record, and the difference
        is the whole point. A machine that stopped reporting an hour ago has
        `newest < now`, and this must leave the window where it was: the last
        hour means the last hour. Returning that machine's final hour of data
        under a "1h" label would be a quiet redefinition of the window, and the
        staleness is already carried by `is_reporting` and the reading's own
        timestamp. So in the ordinary case this is a no-op, and it engages only
        when the stored data leads the clock.

        The clock is read here rather than in the repository. `window_raw` and
        `window_bucketed` take explicit bounds, and the port documents never
        consulting a clock as a virtue -- which is what makes the same query
        usable for a model input window and for a chart alike.
        """
        newest = await telemetry.latest_for(machine_id)
        end = self.clock.now()
        if newest is not None and newest.recorded_at > end:
            end = newest.recorded_at
        return ResolvedWindow(start=window.start_from(end), end=end)

    async def _collect(
        self,
        telemetry: TelemetryRepository,
        machine_id: MachineId,
        start: datetime,
        end: datetime,
        resolution: SeriesResolution,
    ) -> list[TelemetryPoint]:
        """Read the series using whichever query matches the resolution."""
        bucket = resolution.bucket
        if bucket is None:
            records = await telemetry.window_raw(machine_id, start, end, limit=self.max_raw_points)
            return [
                TelemetryPoint.from_reading(record.recorded_at, record.reading)
                for record in records
            ]

        rows = await telemetry.window_bucketed(
            machine_id,
            start,
            end,
            bucket_seconds=int(bucket.total_seconds()),
            aggregation=resolution.aggregation,
        )
        return [_bucket_row_to_point(row) for row in rows]


def _bucket_row_to_point(row: BucketRow) -> TelemetryPoint:
    """Turn a raw aggregation row into a domain point."""
    bucket_start, signals, sample_count = row
    reading = SensorReading(**dict(zip(SensorReading.signal_names(), signals, strict=True)))
    return TelemetryPoint(timestamp=bucket_start, reading=reading, sample_count=sample_count)
