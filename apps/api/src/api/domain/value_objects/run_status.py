"""Lifecycle state of a simulation run, and its permitted transitions."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from api.domain.errors import InvalidRunTransitionError


class RunStatus(StrEnum):
    """Where a run has got to."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


#: The statuses that mean "still going". Declared here rather than left to
#: `SimulationRun.is_active` alone because the same rule is evaluated in SQL --
#: the active-run lookup and the partial unique index that enforces one active
#: run per machine. A second copy of the rule inside a query would be free to
#: drift from the domain's, and the drift would show as two runs interleaving
#: readings on one machine with nothing logged to explain it.
ACTIVE_RUN_STATUSES: frozenset[RunStatus] = frozenset({RunStatus.PENDING, RunStatus.RUNNING})

#: `COMPLETED`, `STOPPED` and `FAILED` are terminal: a demonstration that has
#: ended is not resumed by moving it back, it is replaced by a new run.
#:
#: The one exception is `FAILED -> RUNNING`, which is how a run re-issued after
#: its container died is recorded. It is the only resurrection edge, and
#: `SimulationRun.resume` guards it with a bounded attempt count so a run that
#: fails repeatedly cannot be retried forever.
ALLOWED_TRANSITIONS: Mapping[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.FAILED, RunStatus.STOPPED}),
    RunStatus.RUNNING: frozenset({RunStatus.COMPLETED, RunStatus.STOPPED, RunStatus.FAILED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.STOPPED: frozenset(),
    RunStatus.FAILED: frozenset({RunStatus.RUNNING}),
}


def ensure_run_transition_allowed(current: RunStatus, requested: RunStatus) -> None:
    """Raise if moving from `current` to `requested` is not permitted."""
    if requested not in ALLOWED_TRANSITIONS[current]:
        raise InvalidRunTransitionError(current=current.value, requested=requested.value)
