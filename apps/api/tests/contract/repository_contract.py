"""Shared behavioural contract for repository adapters.

Each `*Contract` class is a mixin: concrete test classes pair it with a fixture
supplying the unit-of-work factory under test. Both the in-memory and SQL
adapters must satisfy every assertion here.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from api.domain.errors import IncidentNotFoundError
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.risk_level import IncidentSeverity
from api.domain.value_objects.time_window import Aggregation
from tests.support.factories import (
    DEFAULT_NOW,
    make_incident,
    make_machine,
    make_prediction,
    make_reading,
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
