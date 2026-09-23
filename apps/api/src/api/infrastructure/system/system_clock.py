"""Clock adapter backed by the operating system."""

from __future__ import annotations

from datetime import UTC, datetime


class SystemClock:
    """Returns the real current time.

    Always UTC and always timezone-aware, so nothing downstream has to guess a
    zone. Tests substitute a fixed clock rather than patching `datetime`.
    """

    def now(self) -> datetime:
        """Return the current instant in UTC."""
        return datetime.now(UTC)
