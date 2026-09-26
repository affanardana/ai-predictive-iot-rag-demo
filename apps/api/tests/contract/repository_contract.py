"""Shared behavioural contract for repository adapters.

Each `*Contract` class is a mixin: concrete test classes pair it with a fixture
supplying the unit-of-work factory under test. Both the in-memory and SQL
adapters must satisfy every assertion here.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from api.domain.errors import (
    IncidentNotFoundError,
    KnowledgeDocumentNotFoundError,
    PersistenceError,
    SimulationRunNotFoundError,
)
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.value_objects.document_category import DocumentCategory
from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.risk_level import IncidentSeverity
from api.domain.value_objects.run_status import RunStatus
from api.domain.value_objects.simulation_scenario import SimulationScenario
from api.domain.value_objects.time_window import Aggregation
from tests.support.factories import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NOW,
    make_embedding,
    make_incident,
    make_knowledge_chunk,
    make_knowledge_document,
    make_machine,
    make_prediction,
    make_reading,
    make_simulation_run,
    make_telemetry,
)


class MachineRepositoryContract:
    """Behaviour every `MachineRepository` must exhibit."""

    async def test_get_returns_none_for_an_unknown_machine(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """An unknown identifier is absent, not an error.

        Use cases turn this `None` into a 404; raising from the repository
        instead would make a typo indistinguishable from a storage failure.
        """
        async with uow_factory() as uow:
            assert await uow.machines.get(MachineId("M999")) is None

    async def test_round_trips_a_machine(self, uow_factory: UnitOfWorkFactory) -> None:
        """Every persisted field comes back unchanged."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M003", name="Compressor"))

        async with uow_factory() as uow:
            machine = await uow.machines.get(MachineId("M003"))

        assert machine is not None
        assert machine.name == "Compressor"
        assert machine.registered_at == DEFAULT_NOW

    async def test_lists_machines_in_identifier_order(self, uow_factory: UnitOfWorkFactory) -> None:
        """The fleet listing is deterministic rather than storage order.

        PostgreSQL and the in-memory dict would otherwise return rows in
        different orders, so the frontend would reshuffle between environments.
        """
        async with uow_factory() as uow:
            for machine_id in ("M003", "M001", "M002"):
                await uow.machines.add(make_machine(machine_id))

        async with uow_factory() as uow:
            machines = await uow.machines.list_all()

        assert [machine.id.value for machine in machines] == ["M001", "M002", "M003"]

    async def test_lists_nothing_when_empty(self, uow_factory: UnitOfWorkFactory) -> None:
        """An empty store lists nothing rather than failing.

        This is the expected state at the end of Phase 1, so the endpoint must
        treat it as ordinary.
        """
        async with uow_factory() as uow:
            assert await uow.machines.list_all() == []

    async def test_returned_machines_are_detached(self, uow_factory: UnitOfWorkFactory) -> None:
        """Mutating a returned entity must not change stored state.

        A real database returns copies; a naive in-memory store would hand back
        the very object it holds, so a use case that mutated an entity and
        forgot to persist it would pass here and fail against PostgreSQL.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001", name="Original"))

        async with uow_factory() as uow:
            machine = await uow.machines.get(MachineId("M001"))
            assert machine is not None
            machine.rename("Mutated")

        async with uow_factory() as uow:
            stored = await uow.machines.get(MachineId("M001"))

        assert stored is not None
        assert stored.name == "Original"


class TelemetryRepositoryContract:
    """Behaviour every `TelemetryRepository` must exhibit."""

    async def test_inserts_new_records(self, uow_factory: UnitOfWorkFactory) -> None:
        """New records are stored, and the count reports how many."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            inserted = await uow.telemetry.add_many_idempotent(
                [make_telemetry(event_id=f"evt-{index}") for index in range(3)]
            )

        assert inserted == 3

    async def test_inserting_nothing_is_a_no_op(self, uow_factory: UnitOfWorkFactory) -> None:
        """An empty batch succeeds and reports zero, rather than erroring.

        An ingestion cycle that finds nothing to publish is normal.
        """
        async with uow_factory() as uow:
            assert await uow.telemetry.add_many_idempotent([]) == 0

    async def test_redelivering_the_same_event_creates_no_duplicate(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """The central idempotency guarantee, enforced by the schema.

        MQTT redelivery is an expected event rather than an error, so the
        second insert is dropped and reports zero new rows.
        """
        record = make_telemetry(event_id="evt-duplicate")

        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            first = await uow.telemetry.add_many_idempotent([record])
            second = await uow.telemetry.add_many_idempotent([record])

        assert first == 1
        assert second == 0

        async with uow_factory() as uow:
            window = await uow.telemetry.window_raw(
                MachineId("M001"),
                DEFAULT_NOW - timedelta(hours=1),
                DEFAULT_NOW + timedelta(hours=1),
                limit=100,
            )

        assert len(window) == 1

    async def test_deduplicates_within_a_single_batch(self, uow_factory: UnitOfWorkFactory) -> None:
        """A batch containing the same event twice inserts it once."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            inserted = await uow.telemetry.add_many_idempotent(
                [make_telemetry(event_id="evt-a"), make_telemetry(event_id="evt-a")]
            )

        assert inserted == 1

    async def test_returns_the_most_recent_record(self, uow_factory: UnitOfWorkFactory) -> None:
        """`latest_for` selects by timestamp, not by insertion order.

        A late-arriving backfill must not masquerade as the current reading.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(event_id="old", recorded_at=DEFAULT_NOW - timedelta(hours=2)),
                    make_telemetry(
                        event_id="new",
                        recorded_at=DEFAULT_NOW,
                        reading=make_reading(vibration=2.5),
                    ),
                ]
            )

        async with uow_factory() as uow:
            latest = await uow.telemetry.latest_for(MachineId("M001"))

        assert latest is not None
        assert latest.event_id == "new"
        assert latest.reading.vibration == pytest.approx(2.5)

    async def test_latest_is_none_without_records(self, uow_factory: UnitOfWorkFactory) -> None:
        """A machine that has never reported has no latest record."""
        async with uow_factory() as uow:
            assert await uow.telemetry.latest_for(MachineId("M001")) is None

    async def test_latest_for_many_agrees_with_latest_for(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """The bulk read names the same record as the single read, every time.

        This is the assertion the fleet view depends on. `GET /api/v1/machines`
        reads every machine's latest record in one query while
        `GET /api/v1/machines/{id}` reads one -- and the row that links to a
        machine must show the same reading as the page it opens. Two queries
        that disagree about which record is newest would make the dashboard
        contradict itself.

        The two rows sharing a timestamp are the point: without a tie-break the
        answer would depend on the query plan, and this test would pass or fail
        by luck.
        """
        async with uow_factory() as uow:
            for machine_id in ("M001", "M002"):
                await uow.machines.add(make_machine(machine_id))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(
                        event_id="m1-old",
                        machine_id="M001",
                        recorded_at=DEFAULT_NOW - timedelta(hours=1),
                    ),
                    make_telemetry(event_id="m1-new-a", machine_id="M001", recorded_at=DEFAULT_NOW),
                    make_telemetry(event_id="m1-new-b", machine_id="M001", recorded_at=DEFAULT_NOW),
                    make_telemetry(
                        event_id="m2-old",
                        machine_id="M002",
                        recorded_at=DEFAULT_NOW - timedelta(days=3),
                    ),
                    make_telemetry(
                        event_id="m2-new",
                        machine_id="M002",
                        recorded_at=DEFAULT_NOW - timedelta(minutes=5),
                    ),
                ]
            )

        async with uow_factory() as uow:
            bulk = await uow.telemetry.latest_for_many(
                [MachineId("M001"), MachineId("M002"), MachineId("M003")]
            )

        assert set(bulk) == {MachineId("M001"), MachineId("M002")}
        for machine_id in ("M001", "M002"):
            async with uow_factory() as uow:
                single = await uow.telemetry.latest_for(MachineId(machine_id))
            assert single is not None
            assert bulk[MachineId(machine_id)].event_id == single.event_id

    async def test_latest_for_many_is_scoped_to_the_machines_asked_for(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """A machine that was not asked about never appears in the result.

        The fleet read builds one mapping for the whole registry, so an adapter
        that ignored the identifier list would silently attach one machine's
        reading to another -- the worst possible failure for a monitoring view,
        and one that looks like plausible data.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.machines.add(make_machine("M002"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(event_id="seen", machine_id="M001"),
                    make_telemetry(event_id="unasked", machine_id="M002"),
                ]
            )

        async with uow_factory() as uow:
            bulk = await uow.telemetry.latest_for_many([MachineId("M001")])

        assert set(bulk) == {MachineId("M001")}
        assert bulk[MachineId("M001")].event_id == "seen"

    async def test_latest_for_many_is_empty_for_an_empty_request(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Being asked about nothing returns nothing, and does not error.

        A fleet with no registered machines is the ordinary state on a fresh
        deployment, so it must not be a query the adapter cannot run.

        The machine is registered even though nothing is asked about it, and
        that is not tidiness: SQLite does not enforce foreign keys by default
        while PostgreSQL does, so the orphan row this test used to write passed
        the SQLite tier and failed the PostgreSQL one with a translated
        integrity error. The store has to hold *something* for the assertion to
        mean anything, and on one of the two dialects it can only hold a row
        whose machine exists.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.telemetry.add_many_idempotent([make_telemetry(event_id="orphan")])

        async with uow_factory() as uow:
            assert await uow.telemetry.latest_for_many([]) == {}

    async def test_latest_for_many_omits_machines_that_never_reported(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """A silent machine is absent rather than mapped to None.

        The distinction matters at the call site: absent means "no reading
        exists", and the fleet summary renders that as `is_reporting: false`
        rather than as a reading of zero.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.machines.add(make_machine("M002"))
            await uow.telemetry.add_many_idempotent(
                [make_telemetry(event_id="only-m1", machine_id="M001")]
            )

        async with uow_factory() as uow:
            bulk = await uow.telemetry.latest_for_many([MachineId("M001"), MachineId("M002")])

        assert MachineId("M002") not in bulk

    async def test_latest_records_returns_the_most_recent_in_chronological_order(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """The tail of the series, oldest-first.

        Both halves matter and they pull in opposite directions: the LIMIT
        keeps the *newest* rows, and the ordering presents them as a series.
        An adapter that gets one right and the other wrong still returns three
        plausible-looking records.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(
                        event_id=f"evt-{minutes:02d}",
                        recorded_at=DEFAULT_NOW - timedelta(minutes=minutes),
                    )
                    for minutes in (40, 30, 20, 10, 0)
                ]
            )

        async with uow_factory() as uow:
            records = await uow.telemetry.latest_records(MachineId("M001"), limit=3)

        assert [record.event_id for record in records] == ["evt-20", "evt-10", "evt-00"]
        timestamps = [record.recorded_at for record in records]
        assert timestamps == sorted(timestamps)

    async def test_latest_records_is_independent_of_insertion_order(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Newest-inserted-first still selects by timestamp.

        The SQL adapter reverses its result, so an insert-ordered store would
        pass a naive test while returning the oldest rows in production.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(event_id="newest", recorded_at=DEFAULT_NOW),
                    make_telemetry(event_id="middle", recorded_at=DEFAULT_NOW - timedelta(hours=1)),
                    make_telemetry(event_id="oldest", recorded_at=DEFAULT_NOW - timedelta(hours=2)),
                ]
            )

        async with uow_factory() as uow:
            records = await uow.telemetry.latest_records(MachineId("M001"), limit=2)

        assert [record.event_id for record in records] == ["middle", "newest"]

    async def test_latest_records_returns_fewer_when_history_is_short(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """A short history is truncated, not an error.

        The caller decides whether what came back is enough; refusing here
        would leave the use case unable to report *how far short* it fell.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(
                        event_id="only-1", recorded_at=DEFAULT_NOW - timedelta(minutes=1)
                    ),
                    make_telemetry(event_id="only-2", recorded_at=DEFAULT_NOW),
                ]
            )

        async with uow_factory() as uow:
            records = await uow.telemetry.latest_records(MachineId("M001"), limit=60)

        assert len(records) == 2

    async def test_latest_records_is_empty_without_records(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """A machine that has never reported yields an empty series."""
        async with uow_factory() as uow:
            assert await uow.telemetry.latest_records(MachineId("M001"), limit=60) == []

    async def test_latest_records_is_scoped_to_one_machine(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Another machine's readings must never enter the window.

        This is the failure that would matter most: a window silently padded
        from a neighbour's history would still be sixty readings long, and the
        model would score it without complaint.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.machines.add(make_machine("M002"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(event_id="m1-a", machine_id="M001"),
                    make_telemetry(event_id="m2-a", machine_id="M002"),
                    make_telemetry(event_id="m1-b", machine_id="M001"),
                ]
            )

        async with uow_factory() as uow:
            records = await uow.telemetry.latest_records(MachineId("M001"), limit=60)

        assert [record.event_id for record in records] == ["m1-a", "m1-b"]

    async def test_count_for_counts_only_that_machine(self, uow_factory: UnitOfWorkFactory) -> None:
        """Readiness is per machine, so the count is too."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.machines.add(make_machine("M002"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(event_id="m1-a", machine_id="M001"),
                    make_telemetry(event_id="m2-a", machine_id="M002"),
                    make_telemetry(event_id="m2-b", machine_id="M002"),
                ]
            )

        async with uow_factory() as uow:
            assert await uow.telemetry.count_for(MachineId("M001")) == 1
            assert await uow.telemetry.count_for(MachineId("M002")) == 2

    async def test_count_for_is_zero_without_records(self, uow_factory: UnitOfWorkFactory) -> None:
        """An unstarted machine counts zero rather than failing."""
        async with uow_factory() as uow:
            assert await uow.telemetry.count_for(MachineId("M001")) == 0

    async def test_count_for_ignores_redelivered_records(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """A redelivered batch must not look like new history.

        This is what keeps readiness from being reached by a retry: if the
        count moved on a duplicate, a redelivering transport could push a
        machine over the threshold without a single new reading arriving.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            batch = [make_telemetry(event_id=f"evt-{index}") for index in range(3)]
            assert await uow.telemetry.add_many_idempotent(batch) == 3

        async with uow_factory() as uow:
            assert await uow.telemetry.add_many_idempotent(batch) == 0
            assert await uow.telemetry.count_for(MachineId("M001")) == 3

    async def test_window_returns_chronological_order(self, uow_factory: UnitOfWorkFactory) -> None:
        """A window is oldest-first so it can be plotted directly.

        Note this is the opposite of `history_for`, which answers "what
        happened most recently" and is newest-first.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(
                        event_id=f"evt-{minutes}",
                        recorded_at=DEFAULT_NOW - timedelta(minutes=minutes),
                    )
                    for minutes in (30, 0, 15)
                ]
            )

        async with uow_factory() as uow:
            window = await uow.telemetry.window_raw(
                MachineId("M001"),
                DEFAULT_NOW - timedelta(hours=1),
                DEFAULT_NOW + timedelta(hours=1),
                limit=100,
            )

        timestamps = [record.recorded_at for record in window]
        assert timestamps == sorted(timestamps)

    async def test_window_excludes_records_outside_the_bounds(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Both bounds are inclusive, and nothing outside them is returned."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(event_id="inside", recorded_at=DEFAULT_NOW),
                    make_telemetry(event_id="before", recorded_at=DEFAULT_NOW - timedelta(hours=5)),
                ]
            )

        async with uow_factory() as uow:
            window = await uow.telemetry.window_raw(
                MachineId("M001"),
                DEFAULT_NOW - timedelta(hours=1),
                DEFAULT_NOW + timedelta(hours=1),
                limit=100,
            )

        assert [record.event_id for record in window] == ["inside"]

    async def test_window_limit_keeps_the_most_recent_points(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Truncation discards old points, never new ones."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(
                        event_id=f"evt-{minutes}",
                        recorded_at=DEFAULT_NOW - timedelta(minutes=minutes),
                    )
                    for minutes in (0, 10, 20, 30)
                ]
            )

        async with uow_factory() as uow:
            window = await uow.telemetry.window_raw(
                MachineId("M001"),
                DEFAULT_NOW - timedelta(hours=1),
                DEFAULT_NOW + timedelta(hours=1),
                limit=2,
            )

        assert [record.event_id for record in window] == ["evt-10", "evt-0"]

    async def test_window_is_scoped_to_one_machine(self, uow_factory: UnitOfWorkFactory) -> None:
        """One machine's window never contains another's readings."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.machines.add(make_machine("M002"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(event_id="mine", machine_id="M001"),
                    make_telemetry(event_id="theirs", machine_id="M002"),
                ]
            )

        async with uow_factory() as uow:
            window = await uow.telemetry.window_raw(
                MachineId("M001"),
                DEFAULT_NOW - timedelta(hours=1),
                DEFAULT_NOW + timedelta(hours=1),
                limit=100,
            )

        assert [record.event_id for record in window] == ["mine"]

    async def test_buckets_aggregate_by_mean(self, uow_factory: UnitOfWorkFactory) -> None:
        """Bucket values are the mean of their members.

        Both readings sit deliberately inside one 300-second bucket. Buckets are
        aligned to the Unix epoch rather than to the window start, so readings
        placed either side of an epoch boundary land in different buckets no
        matter how close together they are.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(
                        event_id="a",
                        recorded_at=DEFAULT_NOW - timedelta(minutes=1),
                        reading=make_reading(temperature=60.0, vibration=1.0),
                    ),
                    make_telemetry(
                        event_id="b",
                        recorded_at=DEFAULT_NOW - timedelta(minutes=4),
                        reading=make_reading(temperature=70.0, vibration=3.0),
                    ),
                ]
            )

            buckets = await uow.telemetry.window_bucketed(
                MachineId("M001"),
                DEFAULT_NOW - timedelta(minutes=5),
                DEFAULT_NOW + timedelta(minutes=5),
                bucket_seconds=300,
                aggregation=Aggregation.MEAN,
            )

        assert len(buckets) == 1
        _bucket_start, signals, sample_count = buckets[0]
        assert sample_count == 2
        # Signals follow SensorReading.signal_names() order.
        assert signals[0] == pytest.approx(65.0)
        assert signals[1] == pytest.approx(2.0)

    async def test_buckets_separate_distinct_intervals(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Readings in different buckets are not merged."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(
                        event_id="early",
                        recorded_at=DEFAULT_NOW - timedelta(minutes=10),
                        reading=make_reading(temperature=50.0),
                    ),
                    make_telemetry(
                        event_id="late",
                        recorded_at=DEFAULT_NOW,
                        reading=make_reading(temperature=80.0),
                    ),
                ]
            )

            buckets = await uow.telemetry.window_bucketed(
                MachineId("M001"),
                DEFAULT_NOW - timedelta(minutes=30),
                DEFAULT_NOW + timedelta(minutes=1),
                bucket_seconds=300,
                aggregation=Aggregation.MEAN,
            )

        assert len(buckets) == 2
        assert [count for _start, _signals, count in buckets] == [1, 1]
        # Chronological order, matching the raw-window contract.
        assert buckets[0][0] < buckets[1][0]

    @pytest.mark.parametrize(
        ("aggregation", "expected"),
        [(Aggregation.MIN, 60.0), (Aggregation.MAX, 70.0)],
    )
    async def test_buckets_support_min_and_max(
        self,
        uow_factory: UnitOfWorkFactory,
        aggregation: Aggregation,
        expected: float,
    ) -> None:
        """The other aggregations select the extreme values.

        Both readings share one 300-second bucket; see
        `test_buckets_aggregate_by_mean` for why the offsets are chosen that
        way.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(
                        event_id="a",
                        recorded_at=DEFAULT_NOW - timedelta(minutes=1),
                        reading=make_reading(temperature=60.0),
                    ),
                    make_telemetry(
                        event_id="b",
                        recorded_at=DEFAULT_NOW - timedelta(minutes=4),
                        reading=make_reading(temperature=70.0),
                    ),
                ]
            )

            buckets = await uow.telemetry.window_bucketed(
                MachineId("M001"),
                DEFAULT_NOW - timedelta(minutes=5),
                DEFAULT_NOW + timedelta(minutes=5),
                bucket_seconds=300,
                aggregation=aggregation,
            )

        assert buckets[0][1][0] == pytest.approx(expected)

    async def test_buckets_are_empty_outside_the_window(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Aggregation applies the same bounds as the raw query.

        The bucketing expression is dialect-specific, so a bound applied in the
        wrong place would only show up here.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.telemetry.add_many_idempotent([make_telemetry(event_id="a")])

            buckets = await uow.telemetry.window_bucketed(
                MachineId("M001"),
                DEFAULT_NOW + timedelta(hours=1),
                DEFAULT_NOW + timedelta(hours=2),
                bucket_seconds=300,
                aggregation=Aggregation.MEAN,
            )

        assert buckets == []


class PredictionRepositoryContract:
    """Behaviour every `PredictionRepository` must exhibit."""

    async def test_round_trips_a_prediction(self, uow_factory: UnitOfWorkFactory) -> None:
        """Including the horizon, which is stored as whole seconds.

        A round-trip through an integer column is where a `timedelta` would
        silently lose resolution if the unit were chosen carelessly.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.predictions.add(make_prediction(probability=0.81, model_version="lstm-v3"))

        async with uow_factory() as uow:
            latest = await uow.predictions.latest_for(MachineId("M001"))

        assert latest is not None
        assert latest.probability.value == pytest.approx(0.81)
        assert latest.model_version == "lstm-v3"
        assert latest.horizon == timedelta(minutes=60)
        assert latest.predicted_at == DEFAULT_NOW

    async def test_latest_is_none_without_predictions(self, uow_factory: UnitOfWorkFactory) -> None:
        """A machine that has never been scored has no prediction.

        Distinct from a prediction of zero, which the fleet summary must report
        as a genuine NORMAL risk.
        """
        async with uow_factory() as uow:
            assert await uow.predictions.latest_for(MachineId("M001")) is None

    async def test_latest_for_many_agrees_with_latest_for(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """The bulk read names the same prediction as the single read.

        The fleet table shows each machine's failure probability, and the
        machine page shows the same number. They are two different queries, so
        they are asserted to be one answer.

        Two predictions share an instant here deliberately: without a
        tie-break, which one is "latest" would depend on the plan.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.machines.add(make_machine("M002"))
            await uow.predictions.add(
                make_prediction(
                    prediction_id="m1-old",
                    machine_id="M001",
                    predicted_at=DEFAULT_NOW - timedelta(hours=1),
                )
            )
            for suffix in ("a", "b"):
                await uow.predictions.add(
                    make_prediction(
                        prediction_id=f"m1-tie-{suffix}",
                        machine_id="M001",
                        predicted_at=DEFAULT_NOW,
                    )
                )
            await uow.predictions.add(make_prediction(prediction_id="m2-only", machine_id="M002"))

        async with uow_factory() as uow:
            bulk = await uow.predictions.latest_for_many(
                [MachineId("M001"), MachineId("M002"), MachineId("M003")]
            )

        assert set(bulk) == {MachineId("M001"), MachineId("M002")}
        for machine_id in ("M001", "M002"):
            async with uow_factory() as uow:
                single = await uow.predictions.latest_for(MachineId(machine_id))
            assert single is not None
            assert bulk[MachineId(machine_id)].prediction_id == single.prediction_id

    async def test_latest_for_many_is_empty_for_an_empty_request(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Being asked about nothing returns nothing, and does not error.

        The machine is registered first so the store is non-empty on both
        dialects -- see the telemetry case above for why an orphan row cannot
        be used to make that point.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.predictions.add(make_prediction(prediction_id="orphan"))

        async with uow_factory() as uow:
            assert await uow.predictions.latest_for_many([]) == {}

    async def test_returns_most_recent_first(self, uow_factory: UnitOfWorkFactory) -> None:
        """History is newest-first so the newest score is the first element.

        Unlike a plotted telemetry window, this answers "what does the model
        say now", so the ordering is deliberately the reverse.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            for hours in (2, 0, 1):
                await uow.predictions.add(
                    make_prediction(
                        prediction_id=f"pred-{hours}",
                        predicted_at=DEFAULT_NOW - timedelta(hours=hours),
                    )
                )

        async with uow_factory() as uow:
            history = await uow.predictions.history_for(MachineId("M001"), limit=10)

        assert [prediction.prediction_id for prediction in history] == [
            "pred-0",
            "pred-1",
            "pred-2",
        ]

    async def test_history_respects_the_limit(self, uow_factory: UnitOfWorkFactory) -> None:
        """Truncation keeps the newest predictions, discarding old ones."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            for hours in range(5):
                await uow.predictions.add(
                    make_prediction(
                        prediction_id=f"pred-{hours}",
                        predicted_at=DEFAULT_NOW - timedelta(hours=hours),
                    )
                )

        async with uow_factory() as uow:
            history = await uow.predictions.history_for(MachineId("M001"), limit=2)

        assert [prediction.prediction_id for prediction in history] == ["pred-0", "pred-1"]


class IncidentRepositoryContract:
    """Behaviour every `IncidentRepository` must exhibit."""

    async def test_round_trips_an_incident(self, uow_factory: UnitOfWorkFactory) -> None:
        """A new incident round-trips in the OPEN state it was created in."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            incident = make_incident(
                incident_id="inc-1",
                severity=IncidentSeverity.CRITICAL,
                probability=0.9,
            )
            await uow.incidents.add(incident)

        async with uow_factory() as uow:
            stored = await uow.incidents.get("inc-1")

        assert stored is not None
        assert stored.status is IncidentStatus.OPEN
        assert stored.severity is IncidentSeverity.CRITICAL
        assert stored.probability.value == pytest.approx(0.9)
        assert stored.detected_at == DEFAULT_NOW

    async def test_get_returns_none_for_an_unknown_incident(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """An unknown identifier is absent, not an error."""
        async with uow_factory() as uow:
            assert await uow.incidents.get("missing") is None

    async def test_update_persists_a_status_change(self, uow_factory: UnitOfWorkFactory) -> None:
        """A lifecycle transition survives the round trip.

        The entity enforces which transitions are legal; this checks the new
        status actually reaches storage rather than only mutating the object.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.incidents.add(make_incident(incident_id="inc-1"))

        async with uow_factory() as uow:
            incident = await uow.incidents.get("inc-1")
            assert incident is not None
            incident.acknowledge()
            await uow.incidents.update(incident)

        async with uow_factory() as uow:
            stored = await uow.incidents.get("inc-1")

        assert stored is not None
        assert stored.status is IncidentStatus.ACKNOWLEDGED

    async def test_update_raises_for_an_unknown_incident(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Updating a row that does not exist fails loudly.

        Silently doing nothing would let a caller believe it had acknowledged
        an incident that was never there.
        """
        async with uow_factory() as uow:
            with pytest.raises(IncidentNotFoundError):
                await uow.incidents.update(make_incident(incident_id="missing"))

    async def test_lists_a_machines_incidents_most_recent_first(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """A machine's incidents are ordered newest-first."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            for hours in (2, 0, 1):
                await uow.incidents.add(
                    make_incident(
                        incident_id=f"inc-{hours}",
                        detected_at=DEFAULT_NOW - timedelta(hours=hours),
                    )
                )

        async with uow_factory() as uow:
            incidents = await uow.incidents.list_for_machine(MachineId("M001"), limit=10)

        assert [incident.incident_id for incident in incidents] == [
            "inc-0",
            "inc-1",
            "inc-2",
        ]

    async def test_filters_by_status_and_severity(self, uow_factory: UnitOfWorkFactory) -> None:
        """Filters combine conjunctively, and omitted filters match everything.

        Both halves matter: a `None` that filtered everything out would make
        the unfiltered incident page silently empty.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            critical = make_incident(incident_id="inc-critical", severity=IncidentSeverity.CRITICAL)
            low = make_incident(incident_id="inc-low", severity=IncidentSeverity.LOW)
            acknowledged = make_incident(incident_id="inc-ack", severity=IncidentSeverity.CRITICAL)
            acknowledged.acknowledge()
            for incident in (critical, low, acknowledged):
                await uow.incidents.add(incident)

        async with uow_factory() as uow:
            critical_open = await uow.incidents.list_filtered(
                status=IncidentStatus.OPEN, severity=IncidentSeverity.CRITICAL, limit=10
            )
            all_incidents = await uow.incidents.list_filtered(status=None, severity=None, limit=10)

        assert [incident.incident_id for incident in critical_open] == ["inc-critical"]
        assert len(all_incidents) == 3

    async def test_open_counts_agrees_with_is_open(self, uow_factory: UnitOfWorkFactory) -> None:
        """The counted statuses are exactly the ones the domain calls open.

        `is_open` is true for OPEN *and* ACKNOWLEDGED -- an acknowledged
        incident still needs attention until it is resolved or dismissed. The
        count is computed in SQL while `is_open` is computed in Python, so the
        two are asserted against each other rather than trusted to agree.

        A machine with no open incidents is absent from the mapping; the caller
        reads that as zero.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.machines.add(make_machine("M002"))
            await uow.machines.add(make_machine("M003"))

            incidents = {
                "open": make_incident(incident_id="a", machine_id="M001"),
                "acknowledged": make_incident(incident_id="b", machine_id="M001"),
                "resolved": make_incident(incident_id="c", machine_id="M001"),
                "dismissed": make_incident(incident_id="d", machine_id="M002"),
            }
            incidents["acknowledged"].acknowledge()
            incidents["resolved"].resolve()
            incidents["dismissed"].dismiss()
            for incident in incidents.values():
                await uow.incidents.add(incident)

        machine_ids = [MachineId("M001"), MachineId("M002"), MachineId("M003")]
        async with uow_factory() as uow:
            counts = await uow.incidents.open_counts_by_machine(machine_ids)
            expected = {}
            for machine_id in machine_ids:
                listed = await uow.incidents.list_for_machine(machine_id, limit=100)
                open_count = sum(1 for incident in listed if incident.is_open)
                if open_count:
                    expected[machine_id] = open_count

        assert counts == expected
        assert counts[MachineId("M001")] == 2
        assert MachineId("M002") not in counts
        assert MachineId("M003") not in counts

    async def test_open_counts_is_empty_for_an_empty_request(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Being asked about nothing returns nothing, and does not error.

        The machine is registered first so the store is non-empty on both
        dialects -- see the telemetry case earlier in this file for why an
        orphan row cannot be used to make that point.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M001"))
            await uow.incidents.add(make_incident(incident_id="orphan"))

        async with uow_factory() as uow:
            assert await uow.incidents.open_counts_by_machine([]) == {}


class SimulationRunRepositoryContract:
    """Behaviour every `SimulationRunRepository` must exhibit."""

    async def test_round_trips_a_run(self, uow_factory: UnitOfWorkFactory) -> None:
        """Every field a run is recorded with comes back unchanged.

        The configuration is stored rather than referenced so the API can
        re-issue a run without asking anyone -- which only works if the plan
        survives the round trip intact.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M003"))
            await uow.simulations.add(
                make_simulation_run(
                    session_id="sim-bd-1a2b3c4d",
                    machine_id="M003",
                    seed=20_260_923,
                    minutes=240,
                    completed_ticks=37,
                )
            )

        async with uow_factory() as uow:
            run = await uow.simulations.get("sim-bd-1a2b3c4d")

        assert run is not None
        assert run.machine_id == MachineId("M003")
        assert run.scenario is SimulationScenario.BEARING_DEGRADATION
        assert run.seed == 20_260_923
        assert run.tick_count == 240
        assert run.completed_ticks == 37
        assert run.status is RunStatus.PENDING

    async def test_get_returns_none_for_an_unknown_run(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """An unknown identifier is absent, not an error."""
        async with uow_factory() as uow:
            assert await uow.simulations.get("never-existed") is None

    async def test_update_persists_a_status_change(self, uow_factory: UnitOfWorkFactory) -> None:
        """The lifecycle is stored, not merely returned to the caller."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M003"))
            await uow.simulations.add(make_simulation_run(machine_id="M003"))

        async with uow_factory() as uow:
            run = await uow.simulations.get("sim-bd-test")
            assert run is not None
            run.mark_running()
            run.record_progress(12, DEFAULT_NOW)
            await uow.simulations.update(run)

        async with uow_factory() as uow:
            stored = await uow.simulations.get("sim-bd-test")

        assert stored is not None
        assert stored.status is RunStatus.RUNNING
        assert stored.completed_ticks == 12

    async def test_update_raises_for_an_unknown_run(self, uow_factory: UnitOfWorkFactory) -> None:
        """Updating a run that was never recorded is an error, not an insert.

        An upsert would let a caller believe it had updated a run that does not
        exist, and the SQL adapter cannot update a row that is not there -- so
        the two adapters would disagree about what just happened.
        """
        with pytest.raises(SimulationRunNotFoundError):
            async with uow_factory() as uow:
                await uow.simulations.update(make_simulation_run(session_id="absent"))

    async def test_delete_removes_a_run(self, uow_factory: UnitOfWorkFactory) -> None:
        """Deleting is what `reset` does to a finished run."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M003"))
            await uow.simulations.add(make_simulation_run(machine_id="M003"))

        async with uow_factory() as uow:
            await uow.simulations.delete("sim-bd-test")

        async with uow_factory() as uow:
            assert await uow.simulations.get("sim-bd-test") is None

    async def test_delete_raises_for_an_unknown_run(self, uow_factory: UnitOfWorkFactory) -> None:
        """Deleting something that is not there is reported, not ignored."""
        with pytest.raises(SimulationRunNotFoundError):
            async with uow_factory() as uow:
                await uow.simulations.delete("never-existed")

    async def test_a_machine_may_have_only_one_active_run(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Two active runs on one machine are refused by the store itself.

        The check in `StartSimulation` reports a clean conflict, so reaching the
        storage layer means two requests raced. The constraint is nevertheless
        enforced here, in the schema and in this adapter alike, because the
        alternative is two scenarios interleaving readings on one machine with
        nothing to explain the resulting risk band.

        Both adapters must raise the same error: the SQL one translates its
        driver's unique violation, so this asserts the domain error rather than
        anything driver-specific.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M003"))
            await uow.simulations.add(
                make_simulation_run(session_id="sim-first", machine_id="M003")
            )

        with pytest.raises(PersistenceError):
            async with uow_factory() as uow:
                await uow.simulations.add(
                    make_simulation_run(session_id="sim-second", machine_id="M003")
                )

    async def test_a_finished_run_does_not_block_the_next_one(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """The constraint is over *active* runs, not all of them.

        Without this, a machine could be demonstrated exactly once and never
        again -- the first run would hold the slot forever.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M003"))
            finished = make_simulation_run(session_id="sim-first", machine_id="M003")
            finished.mark_running()
            finished.mark_completed(DEFAULT_NOW)
            await uow.simulations.add(finished)

        async with uow_factory() as uow:
            await uow.simulations.add(
                make_simulation_run(session_id="sim-second", machine_id="M003")
            )

        async with uow_factory() as uow:
            active = await uow.simulations.active_for_machine(MachineId("M003"))
            assert active is not None
            assert active.session_id == "sim-second"

    async def test_runs_on_different_machines_do_not_block_each_other(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Concurrency is per machine, which is what the fleet view needs."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M003"))
            await uow.machines.add(make_machine("M004"))
            await uow.simulations.add(make_simulation_run(session_id="sim-m3", machine_id="M003"))
            await uow.simulations.add(make_simulation_run(session_id="sim-m4", machine_id="M004"))

        async with uow_factory() as uow:
            assert await uow.simulations.count_active() == 2

    async def test_active_for_machine_ignores_finished_runs(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """A stopped run is not active, however recently it ran."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M003"))
            stopped = make_simulation_run(session_id="sim-done", machine_id="M003")
            stopped.mark_running()
            stopped.mark_stopped(DEFAULT_NOW)
            await uow.simulations.add(stopped)

        async with uow_factory() as uow:
            assert await uow.simulations.active_for_machine(MachineId("M003")) is None

    async def test_list_recent_is_newest_first_and_bounded(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """The ordering a run list wants, and a bound on how much of it.

        One machine each: they are all `PENDING`, and the store refuses two
        active runs on one machine -- which is itself asserted above.
        """
        async with uow_factory() as uow:
            for index in range(5):
                machine_id = f"M{index + 1:03d}"
                await uow.machines.add(make_machine(machine_id))
                await uow.simulations.add(
                    make_simulation_run(
                        session_id=f"sim-{index}",
                        machine_id=machine_id,
                        created_at=DEFAULT_NOW + timedelta(minutes=index),
                    )
                )

        async with uow_factory() as uow:
            recent = await uow.simulations.list_recent(limit=3)

        assert [run.session_id for run in recent] == ["sim-4", "sim-3", "sim-2"]

    async def test_list_for_machine_is_scoped(self, uow_factory: UnitOfWorkFactory) -> None:
        """One machine's runs never include another's."""
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M003"))
            await uow.machines.add(make_machine("M004"))
            await uow.simulations.add(make_simulation_run(session_id="sim-m3", machine_id="M003"))
            await uow.simulations.add(make_simulation_run(session_id="sim-m4", machine_id="M004"))

        async with uow_factory() as uow:
            runs = await uow.simulations.list_for_machine(MachineId("M003"), limit=10)

        assert [run.session_id for run in runs] == ["sim-m3"]

    async def test_a_rolled_back_run_is_not_stored(self, uow_factory: UnitOfWorkFactory) -> None:
        """The unit of work discards a run when the block raises.

        The in-memory store snapshots on entry, and `InMemoryStore.restore`
        lists every collection by hand -- so a new one added to the store but
        not to `restore` would leave this passing in SQL and failing here, or
        worse, the other way round in production.
        """
        async with uow_factory() as uow:
            await uow.machines.add(make_machine("M003"))

        with pytest.raises(RuntimeError):
            async with uow_factory() as uow:
                await uow.simulations.add(make_simulation_run(machine_id="M003"))
                raise RuntimeError("something went wrong mid-transaction")

        async with uow_factory() as uow:
            assert await uow.simulations.get("sim-bd-test") is None


class KnowledgeRepositoryContract:
    """Behaviour every `KnowledgeRepository` must exhibit."""

    async def test_get_returns_none_for_an_unknown_document(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """An unknown document is absent, not an error."""
        async with uow_factory() as uow:
            assert await uow.knowledge.get_document("bearing-inspection-sop", "9.9") is None

    async def test_round_trips_a_document(self, uow_factory: UnitOfWorkFactory) -> None:
        """Every persisted field comes back unchanged."""
        async with uow_factory() as uow:
            await uow.knowledge.add_document(
                make_knowledge_document(
                    document_key="lubrication-procedure",
                    title="Lubrication Procedure",
                    category=DocumentCategory.LUBRICATION,
                    version="2.1",
                    is_active=True,
                    page_count=3,
                )
            )

        async with uow_factory() as uow:
            document = await uow.knowledge.get_document("lubrication-procedure", "2.1")

        assert document is not None
        assert document.title == "Lubrication Procedure"
        assert document.category is DocumentCategory.LUBRICATION
        assert document.version == "2.1"
        assert document.page_count == 3
        assert document.is_active is True
        assert document.is_synthetic is True
        assert document.ingested_at == DEFAULT_NOW

    async def test_round_trips_a_chunk_and_its_embedding(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """The vector survives storage, dimension for dimension.

        SQLite stores it as JSON and PostgreSQL as a `vector`, so this is the
        round trip that shows the two agree on what a chunk is.
        """
        async with uow_factory() as uow:
            document = make_knowledge_document()
            await uow.knowledge.add_document(document)
            await uow.knowledge.add_chunks(
                [
                    make_knowledge_chunk(
                        document_id=document.document_id,
                        chunk_index=0,
                        section="3. Inspection Steps",
                        page=2,
                        embedding=make_embedding(0.5, 0.5),
                    )
                ]
            )

        async with uow_factory() as uow:
            chunks = await uow.knowledge.chunks_for(document.document_id)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.section == "3. Inspection Steps"
        assert chunk.page == 2
        assert chunk.embedding[:2] == pytest.approx((0.5, 0.5))
        assert len(chunk.embedding) == 384

    async def test_a_second_active_version_is_refused(self, uow_factory: UnitOfWorkFactory) -> None:
        """At most one version of a key is active, enforced by the schema.

        The in-memory adapter mirrors the partial unique index by hand, so an
        adapter that let this through would pass in memory and fail in
        production -- the divergence this suite exists to catch.
        """
        async with uow_factory() as uow:
            await uow.knowledge.add_document(make_knowledge_document(version="1.4", is_active=True))

        with pytest.raises(PersistenceError):
            async with uow_factory() as uow:
                await uow.knowledge.add_document(
                    make_knowledge_document(version="2.0", is_active=True)
                )

    async def test_activating_a_version_supersedes_the_previous_one(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Activation is one statement, so no index is transiently broken."""
        async with uow_factory() as uow:
            await uow.knowledge.add_document(make_knowledge_document(version="1.4", is_active=True))
            await uow.knowledge.add_document(make_knowledge_document(version="2.0"))

        async with uow_factory() as uow:
            await uow.knowledge.activate("bearing-inspection-sop", "2.0")

        async with uow_factory() as uow:
            old = await uow.knowledge.get_document("bearing-inspection-sop", "1.4")
            new = await uow.knowledge.get_document("bearing-inspection-sop", "2.0")

        assert old is not None and old.is_active is False
        assert new is not None and new.is_active is True

    async def test_withdrawing_leaves_no_active_version(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """`activate(key, None)` is how a document leaves the evidence base."""
        async with uow_factory() as uow:
            await uow.knowledge.add_document(make_knowledge_document(is_active=True))

        async with uow_factory() as uow:
            await uow.knowledge.activate("bearing-inspection-sop", None)

        async with uow_factory() as uow:
            document = await uow.knowledge.get_document("bearing-inspection-sop", "1.4")

        assert document is not None
        assert document.is_active is False

    async def test_activating_an_unknown_version_is_not_found(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """A typo in a version must not silently withdraw the document.

        The statement would match every row of the key and deactivate all of
        them, which is why existence is checked before it runs rather than
        inferred from its row count.
        """
        async with uow_factory() as uow:
            await uow.knowledge.add_document(make_knowledge_document(version="1.4", is_active=True))

        with pytest.raises(KnowledgeDocumentNotFoundError):
            async with uow_factory() as uow:
                await uow.knowledge.activate("bearing-inspection-sop", "9.9")

        async with uow_factory() as uow:
            document = await uow.knowledge.get_document("bearing-inspection-sop", "1.4")

        assert document is not None
        assert document.is_active is True

    async def test_activating_an_unknown_document_is_not_found(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Withdrawing a document that was never ingested is an error."""
        with pytest.raises(KnowledgeDocumentNotFoundError):
            async with uow_factory() as uow:
                await uow.knowledge.activate("no-such-document", None)

    async def test_list_documents_is_newest_first_and_bounded(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """The corpus listing is ordered and bounded."""
        entries = [
            ("vibration-diagnosis-guide", "1.0"),
            ("bearing-inspection-sop", "1.4"),
            ("lubrication-procedure", "2.1"),
        ]
        async with uow_factory() as uow:
            for index, (key, version) in enumerate(entries):
                await uow.knowledge.add_document(
                    make_knowledge_document(
                        document_key=key,
                        version=version,
                        ingested_at=DEFAULT_NOW + timedelta(minutes=index),
                    )
                )

        async with uow_factory() as uow:
            documents = await uow.knowledge.list_documents(limit=2)

        assert [document.document_key for document in documents] == [
            "lubrication-procedure",
            "bearing-inspection-sop",
        ]

    async def test_a_rolled_back_document_and_its_chunks_are_not_stored(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """The unit of work discards both tables when the block raises.

        `InMemoryStore.restore` lists every collection by hand, so a new one
        added to the store but not to `restore` would leave its write behind
        after a rollback -- passing against PostgreSQL and failing here.
        """
        stored_id: str = ""
        with pytest.raises(RuntimeError):
            async with uow_factory() as uow:
                document = make_knowledge_document()
                stored_id = document.document_id
                await uow.knowledge.add_document(document)
                await uow.knowledge.add_chunks(
                    [make_knowledge_chunk(document_id=document.document_id)]
                )
                raise RuntimeError("something went wrong mid-transaction")

        async with uow_factory() as uow:
            assert await uow.knowledge.get_document("bearing-inspection-sop", "1.4") is None
            assert await uow.knowledge.chunks_for(stored_id) == []


class ChunkSearchContract:
    """Behaviour every adapter that can rank vectors must exhibit.

    Applied to the in-memory and PostgreSQL adapters. SQLite cannot rank at all
    -- its chunks are stored as JSON -- and asserts that it raises instead.
    """

    async def test_returns_the_nearest_chunk_first(self, uow_factory: UnitOfWorkFactory) -> None:
        """Ranking is by cosine distance, best first."""
        near = make_knowledge_document(document_key="bearing-inspection-sop", is_active=True)
        far = make_knowledge_document(
            document_key="electrical-motor-safety",
            category=DocumentCategory.ELECTRICAL_SAFETY,
            is_active=True,
        )
        async with uow_factory() as uow:
            await uow.knowledge.add_document(near)
            await uow.knowledge.add_document(far)
            await uow.knowledge.add_chunks(
                [
                    make_knowledge_chunk(
                        document_id=far.document_id,
                        content="Isolate the supply.",
                        embedding=make_embedding(0.0, 1.0),
                    ),
                    make_knowledge_chunk(
                        document_id=near.document_id,
                        content="Inspect the bearing.",
                        embedding=make_embedding(1.0),
                    ),
                ]
            )

        async with uow_factory() as uow:
            matches = await uow.knowledge.similar_chunks(
                make_embedding(1.0),
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                limit=10,
            )

        assert [match.document.document_key for match in matches] == [
            "bearing-inspection-sop",
            "electrical-motor-safety",
        ]
        assert matches[0].score == pytest.approx(1.0)
        assert matches[1].score == pytest.approx(0.0)

    async def test_respects_the_limit(self, uow_factory: UnitOfWorkFactory) -> None:
        """The candidate bound is what keeps reranking affordable."""
        async with uow_factory() as uow:
            document = make_knowledge_document(is_active=True)
            await uow.knowledge.add_document(document)
            await uow.knowledge.add_chunks(
                [
                    make_knowledge_chunk(
                        document_id=document.document_id,
                        chunk_index=index,
                        content=f"Passage {index}.",
                    )
                    for index in range(5)
                ]
            )

        async with uow_factory() as uow:
            matches = await uow.knowledge.similar_chunks(
                make_embedding(1.0),
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                limit=3,
            )

        assert len(matches) == 3

    async def test_an_inactive_documents_chunks_are_never_returned(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Activation is a retrieval rule, not a display flag.

        The withdrawn version carries the *nearest* vector here, so an adapter
        that ranked first and filtered afterwards, or not at all, would return
        it.
        """
        withdrawn = make_knowledge_document(version="1.4", is_active=False)
        current = make_knowledge_document(version="2.0", is_active=True)
        async with uow_factory() as uow:
            await uow.knowledge.add_document(withdrawn)
            await uow.knowledge.add_document(current)
            await uow.knowledge.add_chunks(
                [
                    make_knowledge_chunk(
                        document_id=withdrawn.document_id,
                        content="Withdrawn procedure.",
                        embedding=make_embedding(1.0),
                    ),
                    make_knowledge_chunk(
                        document_id=current.document_id,
                        content="Current procedure.",
                        embedding=make_embedding(0.0, 1.0),
                    ),
                ]
            )

        async with uow_factory() as uow:
            matches = await uow.knowledge.similar_chunks(
                make_embedding(1.0),
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                limit=10,
            )

        assert [match.chunk.content for match in matches] == ["Current procedure."]

    async def test_filters_by_category(self, uow_factory: UnitOfWorkFactory) -> None:
        """A category filter narrows the corpus before ranking."""
        bearing = make_knowledge_document(is_active=True)
        safety = make_knowledge_document(
            document_key="electrical-motor-safety",
            category=DocumentCategory.ELECTRICAL_SAFETY,
            is_active=True,
        )
        async with uow_factory() as uow:
            await uow.knowledge.add_document(bearing)
            await uow.knowledge.add_document(safety)
            await uow.knowledge.add_chunks(
                [
                    make_knowledge_chunk(document_id=bearing.document_id, content="Bearing."),
                    make_knowledge_chunk(document_id=safety.document_id, content="Safety."),
                ]
            )

        async with uow_factory() as uow:
            matches = await uow.knowledge.similar_chunks(
                make_embedding(1.0),
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                limit=10,
                category=DocumentCategory.ELECTRICAL_SAFETY,
            )

        assert [match.chunk.content for match in matches] == ["Safety."]

    async def test_only_chunks_embedded_by_the_requested_model_are_returned(
        self, uow_factory: UnitOfWorkFactory
    ) -> None:
        """Vectors from two models are not comparable, so they are not mixed."""
        async with uow_factory() as uow:
            document = make_knowledge_document(is_active=True)
            await uow.knowledge.add_document(document)
            await uow.knowledge.add_chunks(
                [
                    make_knowledge_chunk(
                        document_id=document.document_id,
                        chunk_index=0,
                        content="Old model.",
                        embedding_model="sentence-transformers/all-MiniLM-L6-v2@old",
                    ),
                    make_knowledge_chunk(
                        document_id=document.document_id,
                        chunk_index=1,
                        content="Current model.",
                    ),
                ]
            )

        async with uow_factory() as uow:
            matches = await uow.knowledge.similar_chunks(
                make_embedding(1.0),
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                limit=10,
            )

        assert [match.chunk.content for match in matches] == ["Current model."]

    async def test_an_empty_corpus_returns_nothing(self, uow_factory: UnitOfWorkFactory) -> None:
        """Nothing stored is an empty result, not an error."""
        async with uow_factory() as uow:
            matches = await uow.knowledge.similar_chunks(
                make_embedding(1.0),
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                limit=10,
            )

        assert matches == []
