"""Engine and unit-of-work construction."""

from __future__ import annotations

import pytest

from api.infrastructure.persistence.sql.session import create_database_engine
from api.infrastructure.persistence.sql.unit_of_work import SqlUnitOfWorkFactory

#: A host that cannot resolve, used to prove nothing connects eagerly.
UNREACHABLE_URL = "postgresql+psycopg://user:pass@127.0.0.1:1/nonexistent"


@pytest.mark.parametrize(
    ("url", "expected_dialect"),
    [
        (UNREACHABLE_URL, "postgresql"),
        ("sqlite+aiosqlite:///:memory:", "sqlite"),
    ],
)
async def test_engine_construction_does_not_connect(url: str, expected_dialect: str) -> None:
    """Building an engine performs no I/O, and reports its dialect.

    The composition root constructs the container at import time, so if engine
    creation connected, the process would fail to start during a brief database
    outage instead of reporting it through the readiness endpoint. The dialect
    name matters because the repositories branch on it to choose their
    `ON CONFLICT` implementation.
    """
    engine = create_database_engine(url)

    try:
        assert engine.dialect.name == expected_dialect
    finally:
        await engine.dispose()


async def test_unit_of_work_construction_does_not_connect() -> None:
    """A unit of work can be built against an unreachable database.

    The session is created eagerly but SQLAlchemy defers the connection until
    the first statement, which is what makes per-request wiring cheap and lets
    a write path be assembled before the store is reachable.
    """
    engine = create_database_engine(UNREACHABLE_URL)

    try:
        unit_of_work = SqlUnitOfWorkFactory(engine)()

        assert unit_of_work.machines is not None
        assert unit_of_work.telemetry is not None
        assert unit_of_work.predictions is not None
        assert unit_of_work.incidents is not None
    finally:
        await engine.dispose()


async def test_each_unit_of_work_is_independent() -> None:
    """The factory hands out a fresh unit of work per call.

    Sharing one would let concurrent requests share a transaction, so the
    factory exists precisely to prevent that.
    """
    engine = create_database_engine(UNREACHABLE_URL)

    try:
        factory = SqlUnitOfWorkFactory(engine)

        assert factory() is not factory()
    finally:
        await engine.dispose()
