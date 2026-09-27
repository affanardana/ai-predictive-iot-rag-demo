"""Row-level security on every table.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-27

`docs/adr/0006` names this as Phase 11's work, and the reason it gives is worth
restating: the browser never receives a Supabase key, because the API is the
only path to this data. But Supabase projects are reachable without the API --
PostgREST answers at the project's own hostname with the project's *anon* key,
which is designed to be published in frontends and therefore leaks easily. With
row-level security off, that key reads every row of every table.

Enabling it with **no policies at all** is the whole fix. A table owner bypasses
row-level security unless `FORCE ROW LEVEL SECURITY` is set, and this API
connects as the owner; `anon` and `authenticated` are not owners, have no
policies granting them anything, and therefore see nothing. That asymmetry is
the security property, and it is why `FORCE` is deliberately **not** used here:
forcing it would apply the policies to the API's own connection, which has none,
and every query in the product would return empty.

PostgreSQL-only. SQLite has no such concept, and the offline tier runs on
SQLite; the dialect guard is the same one migration 0003 uses for the vector
extension.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Every table this application owns, spelled out rather than discovered.
#:
#: Discovered from `information_schema` would cover the next table added without
#: anybody remembering -- and would also silently enable row-level security on
#: anything else that happened to share the schema, which on a shared Supabase
#: project is not this application's to decide. A frozen list makes the next
#: table a deliberate addition, and `test_row_level_security.py` is what fails
#: when somebody forgets.
TABLES = (
    "machines",
    "telemetry",
    "predictions",
    "incidents",
    "simulation_runs",
    "knowledge_documents",
    "knowledge_chunks",
)


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
