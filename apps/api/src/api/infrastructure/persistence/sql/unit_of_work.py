"""SQL unit of work."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from api.infrastructure.persistence.sql.repositories import (
    SqlIncidentRepository,
    SqlKnowledgeRepository,
    SqlMachineRepository,
    SqlPredictionRepository,
    SqlSimulationRunRepository,
    SqlTelemetryRepository,
)
from api.infrastructure.persistence.sql.session import create_session_factory


class SqlUnitOfWork:
    """A transactional scope over a single database session.

    The session is built eagerly, but SQLAlchemy opens a connection lazily on
    first use, so constructing this costs nothing until a query runs.

    Rollback happens on exception and the session is always closed, so a failed
    request cannot leak a checked-out connection back into the pool dirty.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        dialect_name: str,
    ) -> None:
        self._session = session_factory()
        self.machines = SqlMachineRepository(self._session)
        self.telemetry = SqlTelemetryRepository(self._session, dialect_name)
        self.predictions = SqlPredictionRepository(self._session)
        self.incidents = SqlIncidentRepository(self._session)
        self.simulations = SqlSimulationRunRepository(self._session)
        self.knowledge = SqlKnowledgeRepository(self._session, dialect_name)

    async def __aenter__(self) -> SqlUnitOfWork:
        """Enter the transactional scope."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Commit on success, roll back on failure, then close the session.

        Committing here is what makes the unit of work a real transactional
        boundary, and it is what the port documents. Without it, closing the
        session would discard every write from any use case that did not call
        `commit()` explicitly -- a failure mode that is silent, because the
        writes appear to succeed until the next session reads nothing back.
        """
        try:
            if exc_info[0] is not None:
                await self._session.rollback()
            else:
                await self._session.commit()
        finally:
            await self._session.close()

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self._session.commit()

    async def rollback(self) -> None:
        """Discard the current transaction."""
        await self._session.rollback()


class SqlUnitOfWorkFactory:
    """Creates SQL units of work over a single engine."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        #: Public so the readiness probe can share one session factory with the
        #: repositories rather than building a second one over the same engine.
        self.session_factory = create_session_factory(engine)
        self._dialect_name = engine.dialect.name

    def __call__(self) -> SqlUnitOfWork:
        """Return a new unit of work."""
        return SqlUnitOfWork(self.session_factory, self._dialect_name)
