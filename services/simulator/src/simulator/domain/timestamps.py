"""Timestamp guards.

Every instant in the generated telemetry is timezone-aware, matching the API's
requirement and the `timestamptz` columns it stores them in. A naive datetime
would be rejected on ingestion, so it is rejected here instead — at the point
where the value was created, where the cause is still visible.
"""

from __future__ import annotations

from datetime import UTC, datetime

from simulator.domain.errors import SimulationValidationError


def ensure_aware(value: datetime, field_name: str) -> datetime:
    """Return `value` if it is timezone-aware, otherwise raise."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise SimulationValidationError(
            f"'{field_name}' must be timezone-aware; got a naive datetime. "
            "Use datetime.now(UTC) or attach a tzinfo."
        )
    return value


def utc_now() -> datetime:
    """Return the current time in UTC.

    Used only by the command line, to stamp a run's start. The simulation model
    itself never reads the clock: a run's timestamps derive from its configured
    start, which is what makes it reproducible.
    """
    return datetime.now(UTC)
