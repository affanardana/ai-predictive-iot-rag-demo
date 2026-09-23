"""In-memory backing store."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from api.domain.entities.incident import Incident
from api.domain.entities.machine import Machine
from api.domain.entities.prediction import Prediction
from api.domain.entities.telemetry import TelemetryRecord


@dataclass
class InMemoryStore:
    """Mutable state shared by the repositories of an in-memory unit of work.

    Repositories copy on write and on read, so the store behaves like a real
    database with respect to object identity: mutating an entity returned from
    a read does not silently change stored state. Without that copy, a bug
    where a use case mutates an entity and forgets to persist it would pass in
    memory and fail against PostgreSQL -- exactly the divergence the contract
    suite exists to prevent.

    Telemetry is keyed by `event_id` rather than by a synthetic key, because
    that identifier *is* the uniqueness constraint being modelled.
    """

    machines: dict[str, Machine] = field(default_factory=dict)
    telemetry: dict[str, TelemetryRecord] = field(default_factory=dict)
    predictions: dict[str, Prediction] = field(default_factory=dict)
    incidents: dict[str, Incident] = field(default_factory=dict)

    def snapshot(self) -> InMemoryStore:
        """Return a deep copy of the current state."""
        return deepcopy(self)

    def restore(self, snapshot: InMemoryStore) -> None:
        """Replace the current state with `snapshot`."""
        self.machines = snapshot.machines
        self.telemetry = snapshot.telemetry
        self.predictions = snapshot.predictions
        self.incidents = snapshot.incidents
