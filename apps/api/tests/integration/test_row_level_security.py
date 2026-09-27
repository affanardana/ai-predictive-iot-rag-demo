"""Row-level security, checked against the schema the migration produced.

Marked `postgres`, so the offline run deselects it: row-level security is a
PostgreSQL concept SQLite does not have, and a test that passed on SQLite would
be asserting nothing at all.

**It runs against the migrated database, not a throwaway schema.** The first
version created a schema of its own and hand-ran `ALTER TABLE ... ENABLE ROW
LEVEL SECURITY` on it, which was wrong twice over. Raw SQL bypasses
`schema_translate_map` -- the contract suite's own docstring says so, and the
statement landed on `public` instead -- and, worse, a test that enables the
security itself tests its own `ALTER`, not the migration. Asserting against what
`alembic upgrade head` produced is the only version that can fail when the
migration is wrong, which is the whole point of writing it.

The CI postgres job applies migrations before running this tier. Against a
database that has none, the first assertion fails naming the missing table,
which is the honest symptom rather than a skip.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from tests.contract.test_repositories_postgres import _test_database_url

pytestmark = pytest.mark.postgres

#: The tables the migration names, spelled out again on purpose. Importing the
#: list would make this agree with the migration by construction and unable to
#: notice a table that never made it into either.
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
async def engine() -> AsyncIterator[AsyncEngine]:
    """An engine over the migrated database, using the connection's own schema.

    No `schema_translate_map` here, deliberately: this tier is about what the
    migration did to the real tables, and translating the schema away would be
    testing something else.
    """
    database = create_async_engine(_test_database_url(), pool_pre_ping=True)
    try:
        yield database
    finally:
        await database.dispose()


async def test_every_table_has_row_level_security_enabled(engine: AsyncEngine) -> None:
    """The migration's whole effect, asserted rather than trusted.

    Read from `pg_class` rather than remembered: the failure this guards against
    is a table added in a later phase and left out of the migration, which looks
    like nothing at all until somebody with an anon key reads it.
    """
    async with engine.connect() as connection:
        rows = await connection.execute(
            text(
                "SELECT relname, relrowsecurity FROM pg_class "
                "WHERE relnamespace = 'public'::regnamespace AND relname = ANY(:tables)"
            ),
            {"tables": list(TABLES)},
        )
        # An explicit loop, not `dict(rows)`. `dict()` checks for a `keys()`
        # method first, `CursorResult` has one, and the result is treated as a
        # mapping and subscripted -- `TypeError: 'CursorResult' object is not
        # subscriptable`. Ruff's C416 rule asks for exactly that rewrite, which
        # is how it reached CI; the rule is right in general and wrong here.
        found: dict[str, bool] = {}
        for name, enabled in rows:
            found[name] = enabled

    missing = set(TABLES) - set(found)
    assert not missing, (
        f"tables absent from the database: {sorted(missing)}. This tier expects "
        "the migrations to have been applied -- the CI postgres job runs "
        "`alembic upgrade head` first."
    )
    assert all(found.values()), (
        f"row-level security is off on: {sorted(name for name, on in found.items() if not on)}"
    )


async def test_no_policies_exist(engine: AsyncEngine) -> None:
    """Enabled with no policies is the fix; enabled *with* one would not be.

    A policy granting `anon` anything would hand back exactly what the anon key
    was reading before, so the absence of policies is the security property and
    is asserted as one.
    """
    async with engine.connect() as connection:
        policies = await connection.execute(
            text(
                "SELECT policyname, tablename FROM pg_policies "
                "WHERE schemaname = 'public' AND tablename = ANY(:tables)"
            ),
            {"tables": list(TABLES)},
        )
        found = [f"{table}.{name}" for name, table in policies]

    assert found == [], (
        f"policies exist on tables that should have none: {found}. A policy that "
        "grants anything gives back the access row-level security was enabled to "
        "remove."
    )


async def test_the_owner_still_reads_everything(engine: AsyncEngine) -> None:
    """The asymmetry the whole design rests on.

    A table owner bypasses row-level security unless `FORCE ROW LEVEL SECURITY`
    is set, and the API connects as the owner. This is the assertion that fails
    if somebody adds `FORCE` believing stricter is safer: with no policies
    defined, forcing it would make every query in the product return nothing.

    Read-only on purpose. The first version inserted a machine to prove it could,
    which wrote to `public.machines` -- bypassing the schema isolation it
    believed it had -- and a monitoring test has no business leaving rows behind.
    """
    async with engine.connect() as connection:
        for table in TABLES:
            rows = (
                await connection.execute(
                    # The table name is a constant in this file and is quoted
                    # here. The rule being suppressed is a good one to apply to
                    # every other query in this repository; it cannot know that
                    # this string is not caller-supplied.
                    text(f'SELECT count(*) FROM public."{table}"')  # noqa: S608
                )
            ).all()
            # `count(*)` always yields exactly one row, so this asserts the query
            # was answered rather than refused. Under `FORCE ROW LEVEL SECURITY`
            # with no policies, every one of these raises instead.
            assert len(rows) == 1, f"the owner could not read {table}"
