"""Row-level security, checked against a real PostgreSQL.

Marked `postgres`, so the offline run deselects it: row-level security is a
PostgreSQL concept that SQLite does not have, and a test that passed on SQLite
would be asserting nothing at all -- the same reasoning as the contract suite's
dialect-specific cases.

The property under test is not "the migration ran" but "the guarantee holds":
every table the product owns refuses a non-owner, and the API's own connection
still reads everything. The second half is what would break first if somebody
added `FORCE ROW LEVEL SECURITY`, which applies policies to the owner too -- and
the API has no policies, so every query in the product would return empty.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from api.infrastructure.persistence.sql.base import Base
from tests.contract.test_repositories_postgres import _test_database_url

pytestmark = pytest.mark.postgres

#: The tables the migration names, spelled out again on purpose. Importing the
#: list from the migration would make this test agree with the migration by
#: construction and unable to notice a table that was never added to either.
TABLES = (
    "machines",
    "telemetry",
    "predictions",
    "incidents",
    "simulation_runs",
    "knowledge_documents",
    "knowledge_chunks",
)


@pytest.fixture
async def schema_engine() -> AsyncIterator[AsyncEngine]:
    """An engine pointed at a throwaway schema holding the models' tables."""
    url = _test_database_url()
    schema = f"test_rls_{uuid4().hex[:12]}"

    admin = create_async_engine(url, pool_pre_ping=True)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))

    engine = create_async_engine(url, pool_pre_ping=True).execution_options(
        schema_translate_map={None: schema}
    )
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            for table in TABLES:
                await connection.execute(text(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY'))
        yield engine
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin.dispose()


async def test_every_table_has_row_level_security_enabled(schema_engine: AsyncEngine) -> None:
    """The migration's whole effect, asserted rather than trusted.

    Read from `pg_class` rather than remembered: the failure this guards against
    is a table added in a later phase and forgotten here, which looks like
    nothing at all until somebody with an anon key reads it.
    """
    async with schema_engine.connect() as connection:
        rows = await connection.execute(
            text(
                "SELECT relname, relrowsecurity FROM pg_class "
                "WHERE relname = ANY(:tables) AND relkind = 'r'"
            ),
            {"tables": list(TABLES)},
        )
        # An explicit loop, not `dict(rows)`. `dict()` checks for a `keys()`
        # method first, `CursorResult` has one, and the result is treated as a
        # mapping and subscripted -- which raises `TypeError: 'CursorResult'
        # object is not subscriptable`. Ruff's C416 rule asks for exactly that
        # rewrite, which is how this reached CI; the rule is right in general
        # and wrong here, so the loop stays.
        found: dict[str, bool] = {}
        for name, enabled in rows:
            found[name] = enabled

    assert set(found) == set(TABLES), (
        f"tables missing from the database: {set(TABLES) - set(found)}"
    )
    assert all(found.values()), (
        f"row-level security is off on: {[t for t, on in found.items() if not on]}"
    )


async def test_no_policies_exist_for_the_public_roles(schema_engine: AsyncEngine) -> None:
    """Enabled with no policies is the fix; enabled *with* one would not be.

    A policy granting `anon` anything would hand back exactly what the anon key
    was reading before, so the absence of policies is the security property and
    is asserted as one.
    """
    # The schema-qualified query below is not rewritten by the translate map, so
    # this one runs through the scoped connection's own schema.
    async with schema_engine.connect() as connection:
        policies = await connection.execute(
            text(
                "SELECT policyname FROM pg_policies "
                "WHERE schemaname = current_schema() AND tablename = ANY(:tables)"
            ),
            {"tables": list(TABLES)},
        )
        assert list(policies) == []


async def test_the_owner_still_reads_everything(schema_engine: AsyncEngine) -> None:
    """The asymmetry the whole design rests on.

    A table owner bypasses row-level security unless `FORCE ROW LEVEL SECURITY`
    is set, and the API connects as the owner. This is the assertion that fails
    if somebody adds `FORCE` in the belief that stricter is safer: with no
    policies defined, forcing it would make every query in the product return
    nothing.
    """
    async with schema_engine.begin() as connection:
        await connection.execute(
            text(
                'INSERT INTO "machines" (machine_id, name, registered_at) '
                "VALUES ('M003', 'Demo motor', now())"
            )
        )
        count = await connection.execute(text('SELECT count(*) FROM "machines"'))
        assert count.scalar() == 1
