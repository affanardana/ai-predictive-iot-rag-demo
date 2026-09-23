"""Timestamp guards.

Every instant in this system is timezone-aware. A naive datetime is a latent
bug: comparing it against an aware one raises `TypeError`, and persisting it
against a Postgres `timestamptz` column silently assumes the server's zone.
The domain rejects naive values at the boundary so neither can happen.
"""

from __future__ import annotations

from datetime import UTC, datetime

from api.domain.errors import DomainValidationError


def ensure_aware(value: datetime, field_name: str) -> datetime:
    """Return `value` if it is timezone-aware, otherwise raise."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise DomainValidationError(
            f"'{field_name}' must be timezone-aware; got a naive datetime. "
            "Use datetime.now(UTC) or attach a tzinfo."
        )
    return value


def utc_now() -> datetime:
    """Return the current time in UTC.

    Production code should inject a `Clock` port instead of calling this, so
    that time-dependent behaviour stays testable. This exists for the system
    clock adapter and for fixtures.
    """
    return datetime.now(UTC)
