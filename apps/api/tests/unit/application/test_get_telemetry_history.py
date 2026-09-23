"""Telemetry history use case."""

from __future__ import annotations

from datetime import timedelta

import pytest

from api.application.use_cases import GetTelemetryHistory
from api.domain.errors import MachineNotFoundError
from api.domain.ports.clock import Clock
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.time_window import Aggregation, TimeWindow
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.factories import DEFAULT_NOW, make_machine, make_reading, make_telemetry


def _use_case(
    uow_factory: InMemoryUnitOfWorkFactory,
    clock: Clock,
) -> GetTelemetryHistory:
    return GetTelemetryHistory(unit_of_work_factory=uow_factory, clock=clock)


async def test_raises_for_an_unknown_machine(
    uow_factory: InMemoryUnitOfWorkFactory, clock: Clock
) -> None:
    """An unregistered machine returns 404 rather than an empty series.

    An empty series would be indistinguishable from a machine that is simply
    quiet, which is a materially different situation for an operator.
    """
    with pytest.raises(MachineNotFoundError):
        await _use_case(uow_factory, clock).execute(MachineId("M999"), TimeWindow.ONE_HOUR)


async def test_returns_a_raw_series_for_a_short_window(
    uow_factory: InMemoryUnitOfWorkFactory, clock: Clock
) -> None:
    """A one-hour window is returned unbucketed."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.telemetry.add_many_idempotent(
            [
                make_telemetry(
                    event_id=f"evt-{minutes}",
                    machine_id="M003",
                    recorded_at=DEFAULT_NOW - timedelta(minutes=minutes),
                    reading=make_reading(vibration=1.0 + minutes / 10),
                )
                for minutes in (0, 10, 20)
            ]
        )

    series = await _use_case(uow_factory, clock).execute(MachineId("M003"), TimeWindow.ONE_HOUR)

    assert series.resolution.is_raw
    assert series.resolution.aggregation is Aggregation.RAW
    assert len(series.points) == 3
    assert all(point.sample_count == 1 for point in series.points)


async def test_returns_points_oldest_first(
    uow_factory: InMemoryUnitOfWorkFactory, clock: Clock
) -> None:
    """A series is chronological, ready to plot."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.telemetry.add_many_idempotent(
            [
                make_telemetry(
                    event_id=f"evt-{minutes}",
                    machine_id="M003",
                    recorded_at=DEFAULT_NOW - timedelta(minutes=minutes),
                )
                for minutes in (20, 0, 10)
            ]
        )

    series = await _use_case(uow_factory, clock).execute(MachineId("M003"), TimeWindow.ONE_HOUR)

    timestamps = [point.timestamp for point in series.points]
    assert timestamps == sorted(timestamps)


async def test_excludes_points_outside_the_window(
    uow_factory: InMemoryUnitOfWorkFactory, clock: Clock
) -> None:
    """The window boundary is enforced from the injected clock."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.telemetry.add_many_idempotent(
            [
                make_telemetry(
                    event_id="inside",
                    machine_id="M003",
                    recorded_at=DEFAULT_NOW - timedelta(minutes=30),
                ),
                make_telemetry(
                    event_id="outside",
                    machine_id="M003",
                    recorded_at=DEFAULT_NOW - timedelta(hours=2),
                ),
            ]
        )

    series = await _use_case(uow_factory, clock).execute(MachineId("M003"), TimeWindow.ONE_HOUR)

    assert len(series.points) == 1
    assert series.points[0].timestamp == DEFAULT_NOW - timedelta(minutes=30)


async def test_buckets_a_long_window(uow_factory: InMemoryUnitOfWorkFactory, clock: Clock) -> None:
    """A 24-hour window is aggregated, and the response says so."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.telemetry.add_many_idempotent(
            [
                make_telemetry(
                    event_id=f"evt-{minutes}",
                    machine_id="M003",
                    recorded_at=DEFAULT_NOW - timedelta(minutes=minutes),
                    reading=make_reading(temperature=60.0 + minutes),
                )
                for minutes in (0, 1, 2, 3)
            ]
        )

    series = await _use_case(uow_factory, clock).execute(MachineId("M003"), TimeWindow.ONE_DAY)

    assert not series.resolution.is_raw
    assert series.resolution.bucket == timedelta(minutes=2)
    assert series.resolution.aggregation is Aggregation.MEAN
    # Four readings across four minutes straddle three two-minute buckets --
    # 11:56, 11:58, and 12:00 -- because buckets align to the Unix epoch rather
    # than to the start of the requested window.
    assert len(series.points) == 3
    assert sum(point.sample_count for point in series.points) == 4
    # Chronological, so the middle bucket is the one holding two readings.
    assert [point.sample_count for point in series.points] == [1, 2, 1]
    assert series.points[1].is_aggregated


async def test_aggregates_with_the_mean(
    uow_factory: InMemoryUnitOfWorkFactory, clock: Clock
) -> None:
    """Bucket values are averages of their members.

    Both readings are offset so they share a single two-minute bucket; the
    offsets are not arbitrary, since a pair straddling a bucket boundary would
    each form its own bucket regardless of how close together they are.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.telemetry.add_many_idempotent(
            [
                make_telemetry(
                    event_id="a",
                    machine_id="M003",
                    recorded_at=DEFAULT_NOW - timedelta(seconds=10),
                    reading=make_reading(temperature=60.0),
                ),
                make_telemetry(
                    event_id="b",
                    machine_id="M003",
                    recorded_at=DEFAULT_NOW - timedelta(seconds=50),
                    reading=make_reading(temperature=70.0),
                ),
            ]
        )

    series = await _use_case(uow_factory, clock).execute(MachineId("M003"), TimeWindow.ONE_DAY)

    assert len(series.points) == 1
    assert series.points[0].reading.temperature == pytest.approx(65.0)
