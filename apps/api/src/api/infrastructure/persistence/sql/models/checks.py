"""Helpers for building schema CHECK constraints from domain enums."""

from __future__ import annotations

from collections.abc import Sequence


def in_clause(column: str, values: Sequence[str]) -> str:
    """Render a SQL `column IN (...)` expression from allowed values.

    Enum-backed columns are stored as text with a CHECK constraint rather than
    as a native PostgreSQL ENUM type. Adding a value to a Postgres enum requires
    `ALTER TYPE ... ADD VALUE`, which historically could not run inside a
    transaction and complicates migrations; widening a CHECK constraint is an
    ordinary migration.

    The allowed values are derived from the domain enums at import time, so a
    new enum member cannot be added without the database constraint following.
    """
    if not values:
        raise ValueError("A CHECK constraint needs at least one allowed value.")
    rendered = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({rendered})"
