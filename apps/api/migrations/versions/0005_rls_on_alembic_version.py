"""Row-level security on the one table no model owns.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-30

Migration 0004 enabled row-level security on the seven tables this application
owns, and it enumerated them rather than discovering them, for the reason its
own comment gives: a shared Supabase project's schema is not a migration's to
police.

**`alembic_version` is the table that reasoning missed.** Alembic creates it
itself, on the first upgrade, in the connection's default schema -- `public` on
Supabase, which is the schema PostgREST exposes to the anon key. No model
declares it, so `Base.metadata` never mentions it; no repository reads it; and
it appeared in neither 0004's list nor the test written to fail when that list
was incomplete. Supabase's security advisor found it instead:

    Table public.alembic_version is public, but RLS has not been enabled.

**Why a table of revision strings is worth a migration rather than a shrug.**
Reading it discloses which migrations are deployed, which is minor. *Writing* it
is not: `alembic_version` is the only record of what has been applied, so a
deleted row makes the next `alembic upgrade head` believe the database is empty
and try to apply 0001 onwards to a schema that already contains them -- failing
partway through, during a deploy, leaving the schema in an unknown state. One
publicly writable row, and the failure arrives at the worst moment.

PostgreSQL-only, like 0004 and for the same reason: SQLite has no such concept
and the offline tier runs on it.

**The name is deliberately unqualified.** Unqualified resolves through the
connection's `search_path`, which is the same path Alembic used to create and
read that table -- so this lands wherever the version table actually is, and
would follow it into a translated schema if one is ever configured. That is the
opposite of the raw-SQL trap the RLS test documents, and it is the correct
choice here because the target is *defined* by where Alembic put it. The table
always exists by this point: Alembic creates it before running any migration,
including the first.

The API is unaffected, exactly as in 0004: it connects as the owner, and an
owner bypasses row-level security unless `FORCE` is set -- deliberately not set,
because forcing it would apply the empty policy set to the API's own connection
and every query in the product would return nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Named as a tuple to keep the shape 0004 established, so that the next table
#: outside `Base.metadata` is an obvious addition rather than a rewrite. If this
#: ever grows, the reason each entry is here belongs beside it.
TABLES = ("alembic_version",)


def upgrade() -> None:
    """Enable row-level security, and grant nothing by default."""
    if op.get_bind().dialect.name != "postgresql":
        return
    for table in TABLES:
        op.execute(text(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY'))


def downgrade() -> None:
    """Turn it back off, restoring a public Supabase project's default."""
    if op.get_bind().dialect.name != "postgresql":
        return
    for table in TABLES:
        op.execute(text(f'ALTER TABLE "{table}" DISABLE ROW LEVEL SECURITY'))
