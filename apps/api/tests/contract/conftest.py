"""Fixtures for the contract suite."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool

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
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
def sqlite_uow_factory(sqlite_engine: AsyncEngine) -> SqlUnitOfWorkFactory:
    """A unit-of-work factory over the SQLite engine."""
    return SqlUnitOfWorkFactory(sqlite_engine)
