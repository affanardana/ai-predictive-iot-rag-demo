"""The port through which the API controls the simulator service.

Split the same way `Predictor` and `HttpPredictor` are, and for the same reason:
the API calls another deployable over HTTP, so the domain declares what it needs
and infrastructure decides how it is reached. That is what lets every use case
below be tested against a stub, with no container, no broker and no network.

**Scope.** Three operations, and deliberately no reconciliation sweep. A run
whose container vanished is reported as stopped rather than re-issued, which is
the honest answer and needs no background task. Re-issuing is possible -- the
engine is a pure function of the tick index and the run's plan is stored -- but
it is a recovery mechanism for a failure nobody has seen yet, and it belongs in
a later change rather than in this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.simulation_scenario import SimulationScenario


@dataclass(frozen=True, slots=True)
class SimulationPlan:
    """Everything the simulator needs to reproduce one run.

    Sent rather than a reference, because the simulator keeps no record of runs
    it has not been asked to execute -- the API is the source of truth, and this
    is the whole of what it has to hand over. `started_at` travels with it so a
    resumed run reproduces the original timestamps exactly.
    """

    session_id: str
    machine_id: MachineId
    scenario: SimulationScenario
    seed: int
    started_at: datetime
    duration: timedelta
    sample_interval: timedelta
    tick_seconds: float
    start_index: int = 0


class SimulationController(Protocol):
    """Starts and stops runs on the simulator service."""

    async def start(self, plan: SimulationPlan) -> None:
        """Ask the simulator to begin a run.

        Returns as soon as the run is accepted, not when it finishes -- a run
        lasts minutes to hours and the request that created it must not.

        Raises:
            SimulationUnavailableError: if the service could not be reached or
                refused. Raised as a dependency outage rather than a validation
                error, so a caller retries instead of hunting for a mistake.
        """
        ...

    async def stop(self, session_id: str) -> None:
        """Ask the simulator to end a run.

        Must not raise when the run is unknown to the service: a stop is an
        operator's escape hatch, and one that failed because the run had already
        ended would leave the caller unable to do the thing it was trying to do.
        A service that cannot be reached at all is still an error worth
        reporting -- but the caller decides what to do about it, which is why
        `StopSimulation` marks the run stopped either way.

        Raises:
            SimulationUnavailableError: if the service could not be reached.
        """
        ...

    async def aclose(self) -> None:
        """Release any held resources."""
        ...
