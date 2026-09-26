"""In-memory repository implementations.

These satisfy the same domain ports as the SQL repositories and are held to
the same behaviour by the shared contract suite, so they are trustworthy as
application and presentation test doubles rather than merely convenient.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime, timedelta

from api.domain.entities.incident import Incident
from api.domain.entities.knowledge_document import KnowledgeChunk, KnowledgeDocument
from api.domain.entities.machine import Machine
from api.domain.entities.prediction import Prediction
from api.domain.entities.simulation_run import SimulationRun
from api.domain.entities.telemetry import TelemetryRecord
from api.domain.errors import (
    IncidentNotFoundError,
    KnowledgeDocumentNotFoundError,
    PersistenceError,
    SimulationRunNotFoundError,
)
from api.domain.services.similarity import cosine_similarity
from api.domain.value_objects.chunk_match import ChunkMatch
from api.domain.value_objects.document_category import DocumentCategory
from api.domain.value_objects.incident_status import OPEN_INCIDENT_STATUSES, IncidentStatus
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
        # `event_id` breaks ties, matching the SQL adapter and `latest_records`,
        # so `latest_for` and `latest_for_many` cannot disagree.
        return deepcopy(max(matching, key=lambda record: (record.recorded_at, record.event_id)))

    async def latest_for_many(
        self,
        machine_ids: Sequence[MachineId],
    ) -> Mapping[MachineId, TelemetryRecord]:
        """Return the most recent record for each machine."""
        wanted = set(machine_ids)
        newest: dict[MachineId, TelemetryRecord] = {}
        for record in self._store.telemetry.values():
            if record.machine_id not in wanted:
                continue
            current = newest.get(record.machine_id)
            # `event_id` breaks ties, matching the SQL adapter's ORDER BY, so
            # both agree on which of two same-instant records is the latest.
            if current is None or (record.recorded_at, record.event_id) > (
                current.recorded_at,
                current.event_id,
            ):
                newest[record.machine_id] = record
        return {machine_id: deepcopy(record) for machine_id, record in newest.items()}

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
        # Tie-broken on the identifier for the same reason telemetry is: so this
        # and `latest_for_many` cannot disagree.
        return deepcopy(
            max(
                matching, key=lambda prediction: (prediction.predicted_at, prediction.prediction_id)
            )
        )

    async def latest_for_many(
        self,
        machine_ids: Sequence[MachineId],
    ) -> Mapping[MachineId, Prediction]:
        """Return the most recent prediction for each machine."""
        wanted = set(machine_ids)
        newest: dict[MachineId, Prediction] = {}
        for prediction in self._store.predictions.values():
            if prediction.machine_id not in wanted:
                continue
            current = newest.get(prediction.machine_id)
            if current is None or (prediction.predicted_at, prediction.prediction_id) > (
                current.predicted_at,
                current.prediction_id,
            ):
                newest[prediction.machine_id] = prediction
        return {machine_id: deepcopy(item) for machine_id, item in newest.items()}

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

    async def open_counts_by_machine(
        self,
        machine_ids: Sequence[MachineId],
    ) -> Mapping[MachineId, int]:
        """Return how many incidents are open per machine."""
        wanted = set(machine_ids)
        counts: dict[MachineId, int] = {}
        for incident in self._store.incidents.values():
            if incident.machine_id not in wanted or incident.status not in OPEN_INCIDENT_STATUSES:
                continue
            counts[incident.machine_id] = counts.get(incident.machine_id, 0) + 1
        return counts


class InMemorySimulationRunRepository:
    """In-memory store of simulation runs."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    async def add(self, run: SimulationRun) -> None:
        """Record a new run.

        Refuses a second active run for a machine, mirroring the partial unique
        index the SQL adapter relies on. A real database enforces that in the
        schema, so an adapter that let it through would pass here and fail in
        production -- which is precisely the divergence the contract suite
        exists to catch.

        Raises `PersistenceError` rather than something more specific because
        that is what the SQL adapter produces: its driver raises a unique
        violation, which `translating_persistence_errors` converts. The two must
        agree or the contract suite is testing two different behaviours.

        This is a backstop, not the ordinary path. `StartSimulation` checks for
        an active run first and reports a clean conflict; reaching here means
        two requests raced, and the 500 that results is the honest symptom of
        that.
        """
        active = await self.active_for_machine(run.machine_id)
        if run.is_active and active is not None:
            raise PersistenceError(
                f"{run.machine_id} already has an active run ('{active.session_id}')."
            )
        self._store.simulations[run.session_id] = deepcopy(run)

    async def update(self, run: SimulationRun) -> None:
        """Persist changes to an existing run.

        Raises:
            SimulationRunNotFoundError: if no run carries this identifier.
        """
        if run.session_id not in self._store.simulations:
            raise SimulationRunNotFoundError(run.session_id)
        self._store.simulations[run.session_id] = deepcopy(run)

    async def delete(self, session_id: str) -> None:
        """Remove a run.

        Raises:
            SimulationRunNotFoundError: if no run carries this identifier.
        """
        if session_id not in self._store.simulations:
            raise SimulationRunNotFoundError(session_id)
        del self._store.simulations[session_id]

    async def get(self, session_id: str) -> SimulationRun | None:
        """Return one run, or None if unknown."""
        run = self._store.simulations.get(session_id)
        return deepcopy(run) if run is not None else None

    async def list_recent(self, limit: int) -> Sequence[SimulationRun]:
        """Return recent runs across the fleet, newest first."""
        ordered = sorted(
            self._store.simulations.values(),
            key=lambda run: run.created_at,
            reverse=True,
        )
        return [deepcopy(run) for run in ordered[:limit]]

    async def list_for_machine(
        self,
        machine_id: MachineId,
        limit: int,
    ) -> Sequence[SimulationRun]:
        """Return a machine's runs, newest first."""
        matching = sorted(
            (run for run in self._store.simulations.values() if run.machine_id == machine_id),
            key=lambda run: run.created_at,
            reverse=True,
        )
        return [deepcopy(run) for run in matching[:limit]]

    async def active_for_machine(self, machine_id: MachineId) -> SimulationRun | None:
        """Return the machine's active run, or None."""
        for run in self._store.simulations.values():
            if run.machine_id == machine_id and run.is_active:
                return deepcopy(run)
        return None

    async def count_active(self) -> int:
        """Return how many runs are active fleet-wide."""
        return sum(1 for run in self._store.simulations.values() if run.is_active)


class InMemoryKnowledgeRepository:
    """In-memory maintenance corpus.

    Ranking is computed in Python with the domain's `cosine_similarity`, which
    is what makes the in-memory adapter the one that can be held against
    pgvector: the postgres tier asserts the two agree on the same vectors.
    """

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    async def add_document(self, document: KnowledgeDocument) -> None:
        """Persist a document version.

        Raises:
            PersistenceError: another version of this key is already active, so
                the partial unique index the SQL adapter relies on would refuse
                the insert.
        """
        if document.is_active and any(
            stored.document_key == document.document_key and stored.is_active
            for stored in self._store.knowledge_documents.values()
        ):
            raise PersistenceError(f"'{document.document_key}' already has an active version.")
        self._store.knowledge_documents[document.document_id] = deepcopy(document)

    async def get_document(self, document_key: str, version: str) -> KnowledgeDocument | None:
        """Return one version of a document, or None if it was never ingested."""
        for document in self._store.knowledge_documents.values():
            if document.document_key == document_key and document.version == version:
                return deepcopy(document)
        return None

    async def list_documents(self, limit: int) -> Sequence[KnowledgeDocument]:
        """Return documents across every key, newest first."""
        ordered = sorted(
            self._store.knowledge_documents.values(),
            key=lambda document: (document.ingested_at, document.document_key, document.version),
            reverse=True,
        )
        return [deepcopy(document) for document in ordered[:limit]]

    async def add_chunks(self, chunks: Sequence[KnowledgeChunk]) -> None:
        """Persist a document's chunks."""
        for chunk in chunks:
            self._store.knowledge_chunks[chunk.chunk_id] = deepcopy(chunk)

    async def replace_content(
        self,
        document: KnowledgeDocument,
        chunks: Sequence[KnowledgeChunk],
    ) -> None:
        """Replace a version's stored content, in the caller's transaction."""
        stored = self._store.knowledge_documents.get(document.document_id)
        if stored is not None:
            stored.content_hash = document.content_hash
            stored.page_count = document.page_count
        self._store.knowledge_chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._store.knowledge_chunks.items()
            if chunk.document_id != document.document_id
        }
        for chunk in chunks:
            self._store.knowledge_chunks[chunk.chunk_id] = deepcopy(chunk)

    async def chunks_for(self, document_id: str) -> Sequence[KnowledgeChunk]:
        """Return a document's passages, in reading order."""
        matching = sorted(
            (
                chunk
                for chunk in self._store.knowledge_chunks.values()
                if chunk.document_id == document_id
            ),
            key=lambda chunk: chunk.chunk_index,
        )
        return [deepcopy(chunk) for chunk in matching]

    async def activate(self, document_key: str, version: str | None) -> None:
        """Make one version of a document active, or withdraw the document.

        Raises:
            KnowledgeDocumentNotFoundError: no such key, or no such version.
        """
        matching = [
            document
            for document in self._store.knowledge_documents.values()
            if document.document_key == document_key
        ]
        known_version = version is None or any(d.version == version for d in matching)
        if not matching or not known_version:
            raise KnowledgeDocumentNotFoundError(document_key, version)
        for document in matching:
            document.is_active = document.version == version

    async def similar_chunks(
        self,
        embedding: Sequence[float],
        *,
        embedding_model: str,
        limit: int,
        category: DocumentCategory | None = None,
    ) -> Sequence[ChunkMatch]:
        """Return the chunks nearest this vector, closest first."""
        active = {
            document.document_id: document
            for document in self._store.knowledge_documents.values()
            if document.is_active and (category is None or document.category is category)
        }
        scored = [
            (cosine_similarity(embedding, chunk.embedding), chunk)
            for chunk in self._store.knowledge_chunks.values()
            if chunk.document_id in active and chunk.embedding_model == embedding_model
        ]
        # `chunk_id` breaks ties, so the answer does not depend on dict order.
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            ChunkMatch(
                chunk=deepcopy(chunk),
                document=deepcopy(active[chunk.document_id]),
                score=score,
            )
            for score, chunk in scored[:limit]
        ]
