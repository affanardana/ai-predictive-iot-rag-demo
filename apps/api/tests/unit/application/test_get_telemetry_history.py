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
from tests.support.fakes import FixedClock


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


async def test_returns_points_that_lead_the_clock(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """Stored data ahead of the wall clock still produces a series.

    This is the regression that made the dashboard impossible. The simulator
    advances `recorded_at` at `sample_interval x index` while its paced loop
    sleeps `tick_seconds` per tick, so played back at 300x the newest stored
    reading is an hour ahead of real time within twelve seconds. Every window
    bounded by `clock.now()` then selects nothing, and every chart is empty
    during the demonstration the project exists to give.
    """
    ahead = DEFAULT_NOW + timedelta(hours=3)
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.telemetry.add_many_idempotent(
            [
                make_telemetry(event_id="newest", machine_id="M003", recorded_at=ahead),
                make_telemetry(
                    event_id="within",
                    machine_id="M003",
                    recorded_at=ahead - timedelta(minutes=30),
                ),
                make_telemetry(
                    event_id="too-old",
                    machine_id="M003",
                    recorded_at=ahead - timedelta(hours=2),
                ),
            ]
        )

    series = await _use_case(uow_factory, FixedClock(DEFAULT_NOW)).execute(
        MachineId("M003"), TimeWindow.ONE_HOUR
    )

    assert [point.timestamp for point in series.points] == [
        ahead - timedelta(minutes=30),
        ahead,
    ]
    # And the response says where the window landed, so a client can place the
    # series on a time axis instead of assuming it ends at the browser's clock.
    assert series.interval.end == ahead
    assert series.interval.start == ahead - timedelta(hours=1)


async def test_window_ends_at_the_clock_when_data_does_not_lead_it(
    uow_factory: InMemoryUnitOfWorkFactory, clock: Clock
) -> None:
    """Ordinary operation is unchanged: the anchor is the clock.

    The anchor follows the data only when the data is ahead of it. A machine
    reporting normally must produce exactly the window it always did, or this
    fix would have redefined every chart in the product rather than repairing
    one case.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.telemetry.add_many_idempotent(
            [make_telemetry(event_id="recent", machine_id="M003", recorded_at=DEFAULT_NOW)]
        )

    series = await _use_case(uow_factory, clock).execute(MachineId("M003"), TimeWindow.ONE_HOUR)

    assert series.interval.end == DEFAULT_NOW
    assert series.interval.start == DEFAULT_NOW - timedelta(hours=1)


async def test_a_silent_machine_reports_an_empty_window_not_stale_data(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """A machine that stopped reporting has an empty window, not an old one.

    The distinction the anchor deliberately preserves. `end` is the *later* of
    the clock and the newest reading, never simply the newest reading: were it
    the latter, this machine's final hour of data would be returned under a
    "1h" label, quietly redefining the window to mean "the last hour of
    whatever data exists". Staleness is already carried by `is_reporting` and
    by the reading's own timestamp, and neither of those should be a chart's
    job to express.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.telemetry.add_many_idempotent(
            [
                make_telemetry(
                    event_id="last-gasp",
                    machine_id="M003",
                    recorded_at=DEFAULT_NOW - timedelta(hours=5),
                )
            ]
        )

    series = await _use_case(uow_factory, FixedClock(DEFAULT_NOW)).execute(
        MachineId("M003"), TimeWindow.ONE_HOUR
    )

    assert series.points == []
    assert series.interval.end == DEFAULT_NOW


async def test_an_empty_store_anchors_to_the_clock(
    uow_factory: InMemoryUnitOfWorkFactory, clock: Clock
) -> None:
    """A registered machine that has never reported still answers.

    There is no newest reading to anchor to, so the clock is the only sensible
    choice -- and a machine in this state is ordinary on a fresh deployment.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))

    series = await _use_case(uow_factory, clock).execute(MachineId("M003"), TimeWindow.ONE_HOUR)

    assert series.points == []
    assert series.interval.end == DEFAULT_NOW
    assert series.interval.start == DEFAULT_NOW - timedelta(hours=1)


async def test_the_resolved_window_is_the_window_that_was_asked_for(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """Whatever the anchor, the interval is exactly one window long.

    The anchor moves; the duration must not. A series that quietly covered a
    different span would make every chart's x-axis a lie.
    """
    ahead = DEFAULT_NOW + timedelta(hours=3)
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.telemetry.add_many_idempotent(
            [make_telemetry(event_id="newest", machine_id="M003", recorded_at=ahead)]
        )

    for window in TimeWindow:
        series = await _use_case(uow_factory, FixedClock(DEFAULT_NOW)).execute(
            MachineId("M003"), window
        )
        assert series.interval.duration == window.duration
        assert series.interval.end == ahead
