"""Health probe port, used by the readiness endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Outcome of a dependency check."""

    healthy: bool
    detail: str

    @classmethod
    def up(cls, detail: str = "ok") -> HealthStatus:
        """Report a healthy dependency."""
        return cls(healthy=True, detail=detail)

    @classmethod
    def down(cls, detail: str) -> HealthStatus:
        """Report an unhealthy dependency."""
        return cls(healthy=False, detail=detail)


class HealthProbe(Protocol):
    """Checks whether a dependency is usable."""

    async def check(self) -> HealthStatus:
        """Return the current health of the dependency."""
        ...
