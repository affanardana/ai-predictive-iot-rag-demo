"""Output ports, and the control surface's interface.

Deliberately owned by the domain rather than by the implementations — the same
arrangement as the repository ports in `apps/api`.

`TelemetrySink` accepts only observations and `GroundTruthSink` only hidden
state, so no implementation can receive both and no observation can acquire a
hidden field on the way out. That is the structural half of the guarantee
`MASTERPLAN.md` §3.2 requires; the other half is that dataset mode writes them
to separate files.

`RunControl` is here for a layering reason rather than a conceptual one, and it
is worth stating: the HTTP surface and the run registry that implements it sit
on opposite sides of a contract boundary — infrastructure cannot import
application — so the interface between them has to live below both. The domain
is the only place that is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from simulator.domain.session import SimulationSession
from simulator.domain.state import GroundTruthState, TelemetrySample


class RunState(StrEnum):
    """Where a run has got to.

    `STOPPED` covers both an operator stopping a run and the process going away
    underneath it. They are the same thing from a caller's point of view -- the
    run is no longer advancing -- and separating them would require telling a
    clean stop from a lost container, which is not reliably knowable.
    """

    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class RunView:
    """One run's state, as anything outside the registry sees it."""

    session_id: str
    machine_id: str
    state: RunState
    completed_ticks: int
    total_ticks: int
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None


class RunReporter(Protocol):
    """Tells the API what a run is doing.

    The API is the source of truth for runs, and this is how it stays that way
    from a different container. It cannot be the other way round: the API's
    realtime broadcaster is in-process, so nothing outside it can put an event on
    the dashboard's stream. A run that finished without reporting would stay
    "Running" on screen until something polled.

    Implementations **must not raise**. A reporter sits inside a run loop, and a
    demo that aborts because a status update failed is worse than one whose
    progress bar is a few seconds stale.
    """

    def report(self, run: RunView) -> None:
        """Send one run's current state."""
        ...

    def close(self) -> None:
        """Release any resources."""
        ...


class RunControl(Protocol):
    """Starts and stops runs, and reports on them.

    Implemented by the run registry in the application layer and consumed by the
    HTTP surface in the infrastructure layer, which is why it is declared here.
    """

    def start(self, session: SimulationSession) -> RunView:
        """Begin a run.

        Raises:
            RunAlreadyActiveError: if this machine already has a run here.
        """
        ...

    def stop(self, session_id: str) -> RunView | None:
        """Ask a run to end, returning None if this process has never seen it."""
        ...

    def get(self, session_id: str) -> RunView | None:
        """Return one run, or None if this process has never seen it."""
        ...

    def list(self) -> Sequence[RunView]:
        """Return every run this process knows about, newest first."""
        ...


class TelemetrySink(Protocol):
    """Receives observable telemetry, and nothing else."""

    def write(self, sample: TelemetrySample) -> None:
        """Accept one observation."""
        ...

    def close(self) -> None:
        """Flush and release any resources."""
        ...


class GroundTruthSink(Protocol):
    """Receives hidden ground-truth state, and nothing else."""

    def write(self, state: GroundTruthState) -> None:
        """Accept one ground-truth record."""
        ...

    def close(self) -> None:
        """Flush and release any resources."""
        ...
