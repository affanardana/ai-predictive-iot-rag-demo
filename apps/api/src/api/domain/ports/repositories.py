"""Repository ports.

Storage contracts the domain owns. Implementations live in
`api.infrastructure.persistence`; application code depends only on these
Protocols, which is also what lets the in-memory test doubles be type-checked
against the same interface as the SQL repositories.

Identifiers are assigned by the entities themselves (see their `create`
factories) rather than by the store, so `add` methods return nothing and every
in-memory fake behaves identically to Postgres, where an INSERT need not be
followed by a SELECT.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from api.domain.entities.incident import Incident
from api.domain.entities.machine import Machine
from api.domain.entities.prediction import Prediction
from api.domain.entities.simulation_run import SimulationRun
from api.domain.entities.telemetry import TelemetryRecord
from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.risk_level import IncidentSeverity
from api.domain.value_objects.time_window import Aggregation


class MachineRepository(Protocol):
    """Storage for the machine registry."""

    async def list_all(self) -> Sequence[Machine]:
        """Return every registered machine."""
        ...

    async def get(self, machine_id: MachineId) -> Machine | None:
        """Return one machine, or None if it is not registered."""
        ...

    async def add(self, machine: Machine) -> None:
        """Register a machine."""
        ...


class TelemetryRepository(Protocol):
    """Storage for machine telemetry."""

    async def add_many_idempotent(self, records: Sequence[TelemetryRecord]) -> int:
        """Insert records, skipping any whose `event_id` already exists.

        Returns the number of records actually inserted. The transport may
        redeliver messages, and the non-functional requirements forbid that
        from creating duplicate rows, so duplicates are dropped rather than
        raising -- a redelivery is an expected event, not an error.
        """
        ...

    async def latest_for(self, machine_id: MachineId) -> TelemetryRecord | None:
        """Return the most recent record for a machine."""
        ...

    async def latest_for_many(
        self,
        machine_ids: Sequence[MachineId],
    ) -> Mapping[MachineId, TelemetryRecord]:
        """Return the most recent record for each machine, in one query.

        The bulk counterpart of `latest_for`, and the reason it exists: the
        fleet view needs every machine's newest reading at once, and asking per
        machine issues one query each. Three of those reads per machine is what
        made `GET /api/v1/machines` O(3n).

        **A machine with no telemetry is absent from the mapping rather than
        mapped to None.** The distinction is deliberate: `None` would suggest
        the store knows the machine has no reading, and the caller cannot then
        tell that apart from a machine it forgot to ask about. `dict.get`
        collapses both readings to `None` at the call site anyway.

        Ties on `recorded_at` are broken by `event_id`, matching
        `latest_records`, so which of two same-instant rows is "latest" does not
        depend on the query plan.
        """
        ...

    async def latest_records(self, machine_id: MachineId, limit: int) -> Sequence[TelemetryRecord]:
        """Return the most recent `limit` measurements, oldest first.

        Count-bounded rather than time-bounded, which is what makes it usable
        as a model input: the classifier needs 60 consecutive *readings*, and a
        60-minute window is a different thing whenever a reading is missing.

        It is also the only correct option under the simulator's playback mode.
        `recorded_at` advances 300x faster than the wall clock there, so a range
        computed from `clock.now()` selects nothing. This method never
        consults a clock.

        Returns fewer than `limit` when the history is short, rather than
        raising -- the caller decides whether that is enough.
        """
        ...

    async def count_for(self, machine_id: MachineId) -> int:
        """Return how many measurements are stored for a machine.

        Separate from `latest_records` because readiness is a question about a
        count, and answering it by fetching 60 rows to measure their length
        transfers data to compute a number the database already knows.
        """
        ...

    async def window_raw(
        self,
        machine_id: MachineId,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> Sequence[TelemetryRecord]:
        """Return measurements in `[start, end]`, oldest first.

        Chronological order because the result is plotted as a series; list
        queries that answer "what happened most recently" use `latest_for`.
        """
        ...

    async def window_bucketed(
        self,
        machine_id: MachineId,
        start: datetime,
        end: datetime,
        bucket_seconds: int,
        aggregation: Aggregation,
    ) -> Sequence[tuple[datetime, tuple[float, ...], int]]:
        """Aggregate measurements into fixed buckets.

        Each item is `(bucket_start, aggregated_signals, sample_count)`, where
        `aggregated_signals` follows `SensorReading.signal_names()` order and
        buckets are returned oldest first. The repository returns primitives
        rather than a domain type because the aggregation happens in SQL; the
        use case turns them into `TelemetryPoint`s.
        """
        ...


class PredictionRepository(Protocol):
    """Storage for model predictions."""

    async def add(self, prediction: Prediction) -> None:
        """Persist a prediction."""
        ...

    async def latest_for(self, machine_id: MachineId) -> Prediction | None:
        """Return the most recent prediction for a machine."""
        ...

    async def latest_for_many(
        self,
        machine_ids: Sequence[MachineId],
    ) -> Mapping[MachineId, Prediction]:
        """Return the most recent prediction for each machine, in one query.

        The bulk counterpart of `latest_for`; see `TelemetryRepository.
        latest_for_many` for why a machine with no predictions is absent from
        the mapping rather than mapped to None.
        """
        ...

    async def history_for(self, machine_id: MachineId, limit: int) -> Sequence[Prediction]:
        """Return recent predictions for a machine, most recent first."""
        ...


class SimulationRunRepository(Protocol):
    """Storage for controlled simulation runs.

    Unlike the other repositories, this one exists to answer questions the API
    asks about *itself*: whether a machine is already running, and how many runs
    are going at once. Those are the two limits that keep an unguarded start
    endpoint from costing more CPU than a one-core box has.
    """

    async def add(self, run: SimulationRun) -> None:
        """Record a new run."""
        ...

    async def update(self, run: SimulationRun) -> None:
        """Persist changes to an existing run.

        Raises:
            SimulationRunNotFoundError: if no row carries this identifier. An
                upsert would be wrong here for the reason the incident
                repository gives: it would let a caller believe it had updated a
                run that was never recorded.
        """
        ...

    async def delete(self, session_id: str) -> None:
        """Remove a run.

        Raises:
            SimulationRunNotFoundError: if no row carries this identifier.
        """
        ...

    async def get(self, session_id: str) -> SimulationRun | None:
        """Return one run, or None if unknown."""
        ...

    async def list_recent(self, limit: int) -> Sequence[SimulationRun]:
        """Return recent runs across the fleet, newest first."""
        ...

    async def list_for_machine(
        self,
        machine_id: MachineId,
        limit: int,
    ) -> Sequence[SimulationRun]:
        """Return a machine's runs, newest first."""
        ...

    async def active_for_machine(self, machine_id: MachineId) -> SimulationRun | None:
        """Return the machine's active run, or None.

        At most one can exist: the schema enforces it with a partial unique
        index over `ACTIVE_RUN_STATUSES`, because two runs interleaving readings
        for one machine would produce a risk band describing neither scenario and
        would leave no trace of why.
        """
        ...

    async def count_active(self) -> int:
        """Return how many runs are active fleet-wide.

        A count rather than a fetched list, for the reason `TelemetryRepository.
        count_for` gives: it is a question the database already knows the answer
        to, and fetching rows to measure their length transfers data to compute
        a number.
        """
        ...


class IncidentRepository(Protocol):
    """Storage for maintenance incidents."""

    async def add(self, incident: Incident) -> None:
        """Persist a new incident."""
        ...

    async def get(self, incident_id: str) -> Incident | None:
        """Return one incident, or None if unknown."""
        ...

    async def update(self, incident: Incident) -> None:
        """Persist a status change to an existing incident."""
        ...

    async def list_for_machine(
        self,
        machine_id: MachineId,
        limit: int,
    ) -> Sequence[Incident]:
        """Return incidents for a machine, most recent first."""
        ...

    async def list_filtered(
        self,
        status: IncidentStatus | None,
        severity: IncidentSeverity | None,
        limit: int,
    ) -> Sequence[Incident]:
        """Return incidents, optionally filtered, most recent first."""
        ...

    async def open_counts_by_machine(
        self,
        machine_ids: Sequence[MachineId],
    ) -> Mapping[MachineId, int]:
        """Return how many incidents are open per machine, in one query.

        Counts `OPEN_INCIDENT_STATUSES` -- open *and* acknowledged, which is
        `Incident.is_open`. Reading the rule from the domain rather than
        spelling the two values out in a `WHERE` clause is what keeps this
        count and the incident list from disagreeing.

        Replaces a per-machine `list_for_machine` call whose only purpose was
        to measure the length of the result, capped at 500 rows and fetching
        them all to count them. A machine with no open incidents is absent
        from the mapping; `dict.get(machine_id, 0)` is the intended reading.
        """
        ...
