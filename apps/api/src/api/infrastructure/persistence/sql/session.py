"""Async engine and session construction."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_database_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Build an async engine for the configured database.

    `pool_pre_ping` is enabled because managed Postgres poolers drop idle
    connections without telling the client. Without a pre-flight check, the
    first query after an idle period fails with a stale-connection error that
    looks like an outage.

    No connections are opened here; SQLAlchemy connects lazily on first use.
    """
    return create_async_engine(database_url, echo=echo, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build a session factory bound to `engine`.

    `expire_on_commit` is disabled because refreshing attributes after a commit
    would trigger lazy I/O, which cannot happen implicitly in async code.
    """
    return async_sessionmaker(engine, expire_on_commit=False)
