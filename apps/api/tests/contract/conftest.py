"""Fixtures for the contract suite."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool

from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from api.infrastructure.persistence.sql.base import Base
from api.infrastructure.persistence.sql.unit_of_work import SqlUnitOfWorkFactory

SQLITE_MEMORY_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
def in_memory_uow_factory() -> InMemoryUnitOfWorkFactory:
    """A unit-of-work factory over a fresh in-memory store."""
    return InMemoryUnitOfWorkFactory()


@pytest.fixture
async def sqlite_engine() -> AsyncIterator[AsyncEngine]:
    """An in-memory SQLite engine with the schema created.

    `StaticPool` is required: without it every connection would receive its own
    separate empty in-memory database, and the schema created here would be
    invisible to the repositories.

    Tables come from `metadata.create_all` rather than Alembic. Running
    migrations here would slow the suite down and would test the migration
    rather than the repository; the migration path is verified separately by
    `alembic upgrade head` and `alembic check` against PostgreSQL.
    """
    engine = create_async_engine(SQLITE_MEMORY_URL, poolclass=StaticPool)
    _enforce_foreign_keys(engine)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


def _enforce_foreign_keys(engine: AsyncEngine) -> None:
    """Make SQLite enforce foreign keys, as PostgreSQL does.

    **SQLite leaves them off by default**, which means the two dialects do not
    mean the same thing by the same schema: a row referencing a machine that was
    never registered is rejected by PostgreSQL in production and accepted by the
    SQLite tier in CI. The contract suite exists to hold both adapters to one
    behaviour, and this was a case where it could not -- the tests inserted
    orphan rows, passed locally, and failed the PostgreSQL job.

    Set per connection rather than once, because SQLite scopes the pragma to a
    connection rather than to the file.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragma(dbapi_connection: DBAPIConnection, _record: ConnectionPoolEntry) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


@pytest.fixture
def sqlite_uow_factory(sqlite_engine: AsyncEngine) -> SqlUnitOfWorkFactory:
    """A unit-of-work factory over the SQLite engine."""
    return SqlUnitOfWorkFactory(sqlite_engine)
