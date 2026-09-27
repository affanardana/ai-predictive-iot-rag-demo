"""Health probe port, used by the readiness endpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


@dataclass(frozen=True, slots=True)
class DependencyReport:
    """What one dependency is, whether it answers, and what it is running.

    `models` carries the identities a service reports about itself. For the
    inference service that is the whole answer to "which model is deployed":
    the LSTM's run id, the two encoders, and the chat model's file name -- none
    of which this API can otherwise observe, because that container publishes no
    port and its artefacts are placed by hand on the host.
    """

    name: str
    reachable: bool
    detail: str
    models: Mapping[str, str]


class DependencyReporter(Protocol):
    """Reports on every service this API depends on.

    One call rather than one per dependency, because the caller that wants this
    wants all of it: a monitoring endpoint that answered for the database and
    left the other two unknown would be a monitoring endpoint nobody trusts.
    """

    async def report(self) -> Sequence[DependencyReport]:
        """Return one entry per dependency, in a stable order."""
        ...
