"""Unit of work port.

Gives a use case one transactional boundary over all four repositories. Use
cases depend on this rather than on individual repositories so that a write
spanning several aggregates -- persisting a prediction and the incident it
raises -- either commits together or not at all.
"""

from __future__ import annotations

from typing import Protocol

from api.domain.ports.repositories import (
    IncidentRepository,
    MachineRepository,
    PredictionRepository,
    TelemetryRepository,
)


class UnitOfWork(Protocol):
    """A transactional scope exposing the repositories.

    The repositories are declared as read-only properties rather than plain
    attributes. A mutable attribute in a Protocol is invariant, so a concrete
    unit of work holding a `SqlMachineRepository` would not satisfy an attribute
    declared as `MachineRepository` -- the adapter would be rejected for being
    *more* specific. Read-only properties are covariant, which is what lets each
    adapter expose its own repository types.
    """

    @property
    def machines(self) -> MachineRepository:
        """Storage for the machine registry."""
        ...

    @property
    def telemetry(self) -> TelemetryRepository:
        """Storage for telemetry."""
        ...

    @property
    def predictions(self) -> PredictionRepository:
        """Storage for predictions."""
        ...

    @property
    def incidents(self) -> IncidentRepository:
        """Storage for incidents."""
        ...

    async def __aenter__(self) -> UnitOfWork:
        """Begin the transaction."""
        ...

    async def __aexit__(self, *exc_info: object) -> None:
        """Commit on success, roll back on exception."""
        ...

    async def commit(self) -> None:
        """Commit the current transaction."""
        ...

    async def rollback(self) -> None:
        """Discard the current transaction."""
        ...


class UnitOfWorkFactory(Protocol):
    """Creates independent units of work.

    A factory rather than a shared instance because a unit of work is
    per-request state; handing the same one to concurrent requests would let
    them share a transaction.
    """

    def __call__(self) -> UnitOfWork:
        """Return a new unit of work."""
        ...
