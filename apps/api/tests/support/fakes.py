"""Test doubles for domain ports.

Hand-written rather than produced by `unittest.mock`. A `Mock` accepts any
attribute and any call, so a renamed port method would let tests keep passing
while the real code fails at runtime; these implement the same Protocols and so
are type-checked against them.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from api.domain.ports.health import HealthStatus
from tests.support.factories import DEFAULT_NOW


class FixedClock:
    """A clock pinned to a chosen instant.

    Injecting time rather than patching `datetime` keeps time-dependent
    behaviour deterministic without a global monkeypatch, and makes `advance`
    read explicitly in the tests that need it.
    """

    def __init__(self, now: datetime | None = None) -> None:
        self._now = now or DEFAULT_NOW

    def now(self) -> datetime:
        """Return the pinned instant."""
        return self._now

    def advance(self, delta: timedelta) -> None:
        """Move the clock forward."""
        self._now += delta

    def set(self, moment: datetime) -> None:
        """Move the clock to a specific instant."""
        self._now = moment


class StubHealthProbe:
    """A health probe with a fixed verdict."""

    def __init__(self, healthy: bool = True, detail: str = "ok") -> None:
        self._healthy = healthy
        self._detail = detail
        self.check_count = 0

    async def check(self) -> HealthStatus:
        """Return the configured verdict, counting how often it was asked."""
        self.check_count += 1
        return HealthStatus(healthy=self._healthy, detail=self._detail)
