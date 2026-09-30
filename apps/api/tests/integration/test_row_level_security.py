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

**The second version had a hole in the shape of its query.** It asked `pg_class`
for `relname = ANY(:tables)`, filtering *to* the seven names rather than asking
the schema what it held -- so a table in neither the migration's list nor this
file's copy of it was invisible to the check whose entire job was to notice it.
`alembic_version` was that table, and Supabase's security advisor found it
first, by email, months of deploys later. The sweep below asks the schema.

The CI postgres job applies migrations before running this tier. Against a
database that has none, `test_the_migrated_tables_are_present` fails naming the
missing tables, which is the honest symptom rather than a skip.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from tests.contract.test_repositories_postgres import _test_database_url

pytestmark = pytest.mark.postgres

#: The tables this application owns, spelled out rather than imported.
#:
#: Importing the migration's list would make this file agree with it by
#: construction, and so unable to notice a table that made it into neither.
#:
#: **This list is a vacuity guard, not the security assertion.** A schema with
#: no tables at all satisfies any "nothing unprotected" sweep, so something has
#: to assert the tables are there before the sweep means anything. Adding a name
#: here is therefore not how you fix a failure of the sweep below -- that needs
#: a migration.
TABLES = (
    "machines",
    "telemetry",
    "predictions",
    "incidents",
    "simulation_runs",
    "knowledge_documents",
    "knowledge_chunks",
    # Alembic's own, in `public` because that is the connection's default
    # schema. No model declares it, which is exactly why 0004 missed it and why
    # 0005 exists.
    "alembic_version",
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


async def test_the_migrated_tables_are_present(engine: AsyncEngine) -> None:
    """The guard against a sweep that passes because there is nothing to sweep.

    Trivial on its own, and load-bearing: every assertion below is of the form
    "nothing bad is in `public`", and an unmigrated database makes all of them
    true.
    """
    async with engine.connect() as connection:
        rows = await connection.execute(
            text(
                "SELECT relname FROM pg_class "
                "WHERE relnamespace = 'public'::regnamespace AND relname = ANY(:tables)"
            ),
            {"tables": list(TABLES)},
        )
        # An explicit loop, not `dict(rows)`. `dict()` checks for a `keys()`
        # method first, `CursorResult` has one, and the result is treated as a
        # mapping and subscripted -- `TypeError: 'CursorResult' object is not
        # subscriptable`. Ruff's C416 rule asks for exactly that rewrite, which
        # is how it reached CI; the rule is right in general and wrong here.
        found: set[str] = set()
        for (name,) in rows:
            found.add(name)

    missing = set(TABLES) - found
    assert not missing, (
        f"tables absent from the database: {sorted(missing)}. This tier expects "
        "the migrations to have been applied -- the CI postgres job runs "
        "`alembic upgrade head` first."
    )


async def test_nothing_in_the_public_schema_is_without_rls(engine: AsyncEngine) -> None:
    """The property, swept across the schema rather than filtered to a list.

    This is the assertion the previous version was trying to make and could not:
    it asked for the tables it already knew about, so it could only ever confirm
    that the tables somebody had remembered were protected. `alembic_version`
    was remembered by nobody -- not by `Base.metadata`, not by 0004, not by this
    file -- and sat publicly readable and writable until Supabase's advisor
    said so in an email. That detection path does not run on a pull request.

    `relkind IN ('r', 'p')` is ordinary and partitioned tables; indexes,
    sequences and views have no row-level security to enable. **A table
    appearing here that this project does not own is a finding rather than a
    false positive**: `public` is the schema the anon key reads, so anything in
    it is exposed whether or not a model declares it.
    """
    async with engine.connect() as connection:
        rows = await connection.execute(
            text(
                "SELECT relname FROM pg_class "
                "WHERE relnamespace = 'public'::regnamespace "
                "AND relkind IN ('r', 'p') "
                "AND NOT relrowsecurity "
                "ORDER BY relname"
            )
        )
        unprotected = [name for (name,) in rows]

    assert unprotected == [], (
        f"row-level security is off on: {unprotected}. Every table in `public` is "
        "readable, writable and deletable by anyone holding the project's anon key. "
        "The fix is a migration, not an edit to this file or to an applied one -- an "
        "applied migration is history and editing it changes nothing in a database "
        "that has already run it."
    )


async def test_no_policies_exist(engine: AsyncEngine) -> None:
    """Enabled with no policies is the fix; enabled *with* one would not be.

    A policy granting `anon` anything would hand back exactly what the anon key
    was reading before, so the absence of policies is the security property and
    is asserted as one.
    """
    async with engine.connect() as connection:
        policies = await connection.execute(
            text("SELECT policyname, tablename FROM pg_policies WHERE schemaname = 'public'")
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

    It matters more now that `alembic_version` is included: `alembic upgrade
    head` itself reads and writes that table as the owner, so an over-strict
    setting would not merely empty the dashboard, it would break the next
    deploy.

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
