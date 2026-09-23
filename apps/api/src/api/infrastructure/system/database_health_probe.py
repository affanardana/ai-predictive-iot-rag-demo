"""Readiness probe for the operational database."""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api.domain.ports.health import HealthStatus

logger = logging.getLogger(__name__)

#: Cheapest possible round trip that proves a connection works.
_PROBE_STATEMENT = text("SELECT 1")


class DatabaseHealthProbe:
    """Checks the database by issuing a trivial query."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def check(self) -> HealthStatus:
        """Return whether the database answers a query.

        The failure detail is intentionally generic. Driver messages can embed
        the connection string, which contains the password, so the specific
        error is logged (redacted) rather than returned to the caller.
        """
        try:
            async with self._session_factory() as session:
                await session.execute(_PROBE_STATEMENT)
        except SQLAlchemyError as exc:
            logger.warning(
                "infrastructure.health.database_unavailable",
                extra={"exception_type": type(exc).__name__},
            )
            return HealthStatus.down("The database is not reachable.")

        return HealthStatus.up("The database is reachable.")
