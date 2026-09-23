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
        end = self.clock.now()
        start = window.start_from(end)

        async with self.unit_of_work_factory() as uow:
            machine = await uow.machines.get(machine_id)
            if machine is None:
                raise MachineNotFoundError(str(machine_id))

            points = await self._collect(
                telemetry=uow.telemetry,
                machine_id=machine_id,
                start=start,
                end=end,
                resolution=resolution,
            )

        return TelemetrySeries(
            machine_id=machine_id,
            window=window,
            resolution=resolution,
            points=points,
        )

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
