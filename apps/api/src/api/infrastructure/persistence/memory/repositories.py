"""In-memory repository implementations.

These satisfy the same domain ports as the SQL repositories and are held to
the same behaviour by the shared contract suite, so they are trustworthy as
application and presentation test doubles rather than merely convenient.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from datetime import UTC, datetime, timedelta

from api.domain.entities.incident import Incident
from api.domain.entities.machine import Machine
from api.domain.entities.prediction import Prediction
from api.domain.entities.telemetry import TelemetryRecord
from api.domain.errors import IncidentNotFoundError
from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.risk_level import IncidentSeverity
from api.domain.value_objects.sensor_reading import SensorReading
from api.domain.value_objects.time_window import Aggregation
from api.infrastructure.persistence.memory.store import InMemoryStore

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _bucket_start(moment: datetime, bucket_seconds: int) -> datetime:
    """Floor a timestamp onto a fixed bucket grid aligned to the epoch."""
    offset = int((moment - _EPOCH).total_seconds())
    return _EPOCH + timedelta(seconds=(offset // bucket_seconds) * bucket_seconds)


def _aggregate(values: Sequence[float], aggregation: Aggregation) -> float:
    """Combine the values in one bucket."""
    if aggregation is Aggregation.MEAN:
        return sum(values) / len(values)
    if aggregation is Aggregation.MIN:
        return min(values)
    if aggregation is Aggregation.MAX:
        return max(values)
    raise ValueError(f"Aggregation '{aggregation}' cannot be applied to a bucket.")


class InMemoryMachineRepository:
    """In-memory machine registry."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    async def list_all(self) -> Sequence[Machine]:
        """Return every registered machine, ordered by identifier."""
        machines = sorted(self._store.machines.values(), key=lambda item: item.id.value)
        return [deepcopy(machine) for machine in machines]

    async def get(self, machine_id: MachineId) -> Machine | None:
        """Return one machine, or None if it is not registered."""
        machine = self._store.machines.get(machine_id.value)
        return deepcopy(machine) if machine is not None else None

    async def add(self, machine: Machine) -> None:
        """Register a machine."""
        self._store.machines[machine.id.value] = deepcopy(machine)


class InMemoryTelemetryRepository:
    """In-memory telemetry store."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    async def add_many_idempotent(self, records: Sequence[TelemetryRecord]) -> int:
        """Insert records, skipping any whose `event_id` is already present."""
        inserted = 0
        for record in records:
            if record.event_id in self._store.telemetry:
                continue
            self._store.telemetry[record.event_id] = deepcopy(record)
            inserted += 1
        return inserted

    async def latest_for(self, machine_id: MachineId) -> TelemetryRecord | None:
        """Return the most recent record for a machine."""
        matching = [
            record for record in self._store.telemetry.values() if record.machine_id == machine_id
        ]
        if not matching:
            return None
        return deepcopy(max(matching, key=lambda record: record.recorded_at))

    async def latest_records(self, machine_id: MachineId, limit: int) -> Sequence[TelemetryRecord]:
        """Return the most recent `limit` measurements, oldest first."""
        matching = sorted(
            (
                record
                for record in self._store.telemetry.values()
                if record.machine_id == machine_id
            ),
            # `event_id` breaks ties, matching the SQL adapter, so both are
            # deterministic when two records share a timestamp.
            key=lambda record: (record.recorded_at, record.event_id),
        )
        return [deepcopy(record) for record in matching[-limit:]]

    async def count_for(self, machine_id: MachineId) -> int:
        """Return how many measurements are stored for a machine."""
        return sum(
            1 for record in self._store.telemetry.values() if record.machine_id == machine_id
        )

    async def window_raw(
        self,
        machine_id: MachineId,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> Sequence[TelemetryRecord]:
        """Return measurements in `[start, end]`, oldest first."""
        matching = sorted(
            (
                record
                for record in self._store.telemetry.values()
                if record.machine_id == machine_id and start <= record.recorded_at <= end
            ),
            key=lambda record: record.recorded_at,
        )
        # Keep the most recent `limit` points when the window holds more than
        # the caller asked for; the tail is what an operator cares about.
        return [deepcopy(record) for record in matching[-limit:]]

    async def window_bucketed(
        self,
        machine_id: MachineId,
        start: datetime,
        end: datetime,
        bucket_seconds: int,
        aggregation: Aggregation,
    ) -> Sequence[tuple[datetime, tuple[float, ...], int]]:
        """Aggregate measurements into fixed buckets, oldest first."""
        if bucket_seconds < 1:
            raise ValueError("Bucket size must be at least one second.")

        grouped: dict[datetime, list[SensorReading]] = {}
        for record in self._store.telemetry.values():
            if record.machine_id != machine_id or not start <= record.recorded_at <= end:
                continue
            key = _bucket_start(record.recorded_at, bucket_seconds)
            grouped.setdefault(key, []).append(record.reading)

        signal_names = SensorReading.signal_names()
        rows: list[tuple[datetime, tuple[float, ...], int]] = []
        for bucket in sorted(grouped):
            readings = grouped[bucket]
            signals = tuple(
                _aggregate([getattr(reading, name) for reading in readings], aggregation)
                for name in signal_names
            )
            rows.append((bucket, signals, len(readings)))
        return rows


class InMemoryPredictionRepository:
    """In-memory prediction store."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    async def add(self, prediction: Prediction) -> None:
        """Persist a prediction."""
        self._store.predictions[prediction.prediction_id] = deepcopy(prediction)

    async def latest_for(self, machine_id: MachineId) -> Prediction | None:
        """Return the most recent prediction for a machine."""
        matching = [
            prediction
            for prediction in self._store.predictions.values()
            if prediction.machine_id == machine_id
        ]
        if not matching:
            return None
        return deepcopy(max(matching, key=lambda prediction: prediction.predicted_at))

    async def history_for(self, machine_id: MachineId, limit: int) -> Sequence[Prediction]:
        """Return recent predictions for a machine, most recent first."""
        matching = sorted(
            (
                prediction
                for prediction in self._store.predictions.values()
                if prediction.machine_id == machine_id
            ),
            key=lambda prediction: prediction.predicted_at,
            reverse=True,
        )
        return [deepcopy(prediction) for prediction in matching[:limit]]


class InMemoryIncidentRepository:
    """In-memory incident store."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    async def add(self, incident: Incident) -> None:
        """Persist a new incident."""
        self._store.incidents[incident.incident_id] = deepcopy(incident)

    async def get(self, incident_id: str) -> Incident | None:
        """Return one incident, or None if unknown."""
        incident = self._store.incidents.get(incident_id)
        return deepcopy(incident) if incident is not None else None

    async def update(self, incident: Incident) -> None:
        """Persist a status change to an existing incident.

        Raises:
            IncidentNotFoundError: if no incident carries this identifier. An
                upsert would be wrong here: silently inserting would let a
                caller believe it had acknowledged an incident that was never
                recorded, and it would diverge from the SQL adapter, which
                cannot update a row that does not exist.
        """
        if incident.incident_id not in self._store.incidents:
            raise IncidentNotFoundError(incident.incident_id)
        self._store.incidents[incident.incident_id] = deepcopy(incident)

    async def list_for_machine(
        self,
        machine_id: MachineId,
        limit: int,
    ) -> Sequence[Incident]:
        """Return incidents for a machine, most recent first."""
        matching = sorted(
            (
                incident
                for incident in self._store.incidents.values()
                if incident.machine_id == machine_id
            ),
            key=lambda incident: incident.detected_at,
            reverse=True,
        )
        return [deepcopy(incident) for incident in matching[:limit]]

    async def list_filtered(
        self,
        status: IncidentStatus | None,
        severity: IncidentSeverity | None,
        limit: int,
    ) -> Sequence[Incident]:
        """Return incidents, optionally filtered, most recent first."""
        matching = sorted(
            (
                incident
                for incident in self._store.incidents.values()
                if (status is None or incident.status is status)
                and (severity is None or incident.severity is severity)
            ),
            key=lambda incident: incident.detected_at,
            reverse=True,
        )
        return [deepcopy(incident) for incident in matching[:limit]]
