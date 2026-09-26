"""In-memory unit of work."""

from __future__ import annotations

from api.infrastructure.persistence.memory.repositories import (
    InMemoryIncidentRepository,
    InMemoryKnowledgeRepository,
    InMemoryMachineRepository,
    InMemoryPredictionRepository,
    InMemorySimulationRunRepository,
    InMemoryTelemetryRepository,
)
from api.infrastructure.persistence.memory.store import InMemoryStore


class InMemoryUnitOfWork:
    """A transactional scope over an in-memory store.

    Writes are applied immediately, and the state is snapshotted on entry so a
    failure can restore it. That reproduces the observable behaviour of a
    database transaction -- uncommitted work disappears when the block raises --
    which is what lets the contract suite treat this adapter and the SQL one as
    interchangeable.
    """

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store
        self._snapshot: InMemoryStore | None = None

        self.machines = InMemoryMachineRepository(store)
        self.telemetry = InMemoryTelemetryRepository(store)
        self.predictions = InMemoryPredictionRepository(store)
        self.incidents = InMemoryIncidentRepository(store)
        self.simulations = InMemorySimulationRunRepository(store)
        self.knowledge = InMemoryKnowledgeRepository(store)

    async def __aenter__(self) -> InMemoryUnitOfWork:
        """Take a snapshot to roll back to."""
        self._snapshot = self._store.snapshot()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Restore the snapshot if the block raised."""
        raised = exc_info[0] is not None
        if raised and self._snapshot is not None:
            self._store.restore(self._snapshot)
        self._snapshot = None

    async def commit(self) -> None:
        """Finalise the transaction, discarding the rollback snapshot."""
        self._snapshot = None

    async def rollback(self) -> None:
        """Restore the state captured when the block was entered."""
        if self._snapshot is not None:
            self._store.restore(self._snapshot)
            self._snapshot = None


class InMemoryUnitOfWorkFactory:
    """Creates in-memory units of work over one shared store."""

    def __init__(self, store: InMemoryStore | None = None) -> None:
        self.store = store if store is not None else InMemoryStore()

    def __call__(self) -> InMemoryUnitOfWork:
        """Return a new unit of work."""
        return InMemoryUnitOfWork(self.store)
