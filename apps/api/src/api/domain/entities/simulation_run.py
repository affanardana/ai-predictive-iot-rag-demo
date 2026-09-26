"""A controlled simulation run, as the API records it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

from api.domain.timestamps import ensure_aware
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.run_status import (
    ACTIVE_RUN_STATUSES,
    RunStatus,
    ensure_run_transition_allowed,
)
from api.domain.value_objects.simulation_scenario import SimulationScenario

#: How many times a run whose container died may be re-issued before it is
#: declared failed. One is enough for the case this exists for -- a container
#: restarted mid-demonstration -- and a bound is what stops a run that fails
#: instantly from being retried forever in a loop.
MAX_RESUME_ATTEMPTS = 1

#: The simulator's own ceiling, and the number that makes `session_id` safe to
#: embed in an event id: 32 for the session, one separator, 16 for the machine,
#: one separator, and eight digits of tick index comes to 58 of the 64 available.
MAX_SESSION_ID_LENGTH = 32


@dataclass
class SimulationRun:
    """One run: what it was asked to do, and how far it has got.

    The configuration is stored rather than referenced, because the API has to
    be able to re-issue a run without asking anyone. That is what makes a
    container that restarted mid-demonstration recoverable: the engine is a pure
    function of the tick index, so the same scenario, seed and duration replayed
    from `completed_ticks` produce the same series, and every tick already
    stored is dropped as a duplicate by the ingest endpoint's idempotency.
    """

    session_id: str
    machine_id: MachineId
    scenario: SimulationScenario
    seed: int
    started_at: datetime
    sample_interval: timedelta
    duration: timedelta
    created_at: datetime
    tick_seconds: float = 1.0
    status: RunStatus = RunStatus.PENDING
    completed_ticks: int = 0
    last_heartbeat_at: datetime | None = None
    finished_at: datetime | None = None
    detail: str | None = None
    resume_count: int = 0

    def __post_init__(self) -> None:
        """Validate the plan and the progress against it."""
        if not self.session_id.strip():
            raise ValueError("SimulationRun session_id must not be blank.")
        if len(self.session_id) > MAX_SESSION_ID_LENGTH:
            raise ValueError(
                f"session_id '{self.session_id}' exceeds {MAX_SESSION_ID_LENGTH} "
                "characters, which would push event ids past their column."
            )
        if self.sample_interval <= timedelta(0):
            raise ValueError("SimulationRun sample_interval must be positive.")
        if self.duration <= timedelta(0):
            raise ValueError("SimulationRun duration must be positive.")
        if self.tick_seconds < 0:
            raise ValueError("SimulationRun tick_seconds must not be negative.")
        if not 0 <= self.completed_ticks <= self.tick_count:
            raise ValueError(
                f"completed_ticks {self.completed_ticks} is outside the run, "
                f"which has {self.tick_count}."
            )
        for name in ("started_at", "created_at"):
            ensure_aware(getattr(self, name), name)
        for name in ("last_heartbeat_at", "finished_at"):
            moment = getattr(self, name)
            if moment is not None:
                ensure_aware(moment, name)

    @property
    def tick_count(self) -> int:
        """Return how many readings this run produces for its machine.

        Derived rather than stored. A stored copy would be a second source of
        truth, and the two would eventually disagree about how long a run is.
        """
        return int(self.duration / self.sample_interval)

    @property
    def is_active(self) -> bool:
        """Whether the run is still going."""
        return self.status in ACTIVE_RUN_STATUSES

    @property
    def is_terminal(self) -> bool:
        """Whether the run has finished, one way or another."""
        return not self.is_active

    def is_stale(self, now: datetime, timeout: timedelta) -> bool:
        """Whether an active run has stopped reporting.

        Derived from the clock rather than stored, so nothing has to be written
        to keep it true -- and so a container that dies is noticed without
        anything having to notice it. This is what stops a run whose simulator
        vanished from being displayed as live forever.
        """
        if not self.is_active:
            return False
        if self.last_heartbeat_at is None:
            # Never reported at all. Judged against when the API accepted it,
            # so a container that was down when the run was requested is caught
            # by the same rule rather than sitting at PENDING indefinitely.
            return now - self.created_at > timeout
        return now - self.last_heartbeat_at > timeout

    def mark_running(self) -> None:
        """Record that the simulator accepted the run.

        Deliberately does not touch `last_heartbeat_at`: `is_stale` falls back
        to `created_at` when no report has arrived, so a run that was accepted
        and then went quiet is caught by the same rule as one that never
        reported at all.
        """
        self._transition_to(RunStatus.RUNNING)

    def record_progress(self, ticks: int, at: datetime) -> None:
        """Record how far the run has got, and that it is still alive.

        Progress is monotonic. A report that moves it backwards is ignored
        rather than raising, because two in-flight reports can arrive out of
        order and the later one is not more true than the earlier one.
        """
        ensure_aware(at, "at")
        if ticks > self.completed_ticks:
            self.completed_ticks = min(ticks, self.tick_count)
        self.last_heartbeat_at = at

    def mark_completed(self, at: datetime) -> None:
        """Record that the run reached its final tick."""
        self._transition_to(RunStatus.COMPLETED)
        self.finished_at = at

    def mark_stopped(self, at: datetime, detail: str | None = None) -> None:
        """Record that the run ended early, or was ended."""
        self._transition_to(RunStatus.STOPPED)
        self.finished_at = at
        self.detail = detail

    def mark_failed(self, detail: str) -> None:
        """Record that the run could not be carried out."""
        self._transition_to(RunStatus.FAILED)
        self.detail = detail

    def resume(self) -> None:
        """Re-issue a run whose container went away.

        The only move out of a terminal state, and bounded by
        `MAX_RESUME_ATTEMPTS` so a run that fails immediately cannot be retried
        in a loop. Everything else that ended has ended.
        """
        if self.resume_count >= MAX_RESUME_ATTEMPTS:
            raise ValueError(
                f"Run '{self.session_id}' has already been resumed "
                f"{self.resume_count} time(s); it is not tried again."
            )
        self._transition_to(RunStatus.RUNNING)
        self.resume_count += 1
        self.finished_at = None

    def _transition_to(self, requested: RunStatus) -> None:
        """Apply a status change, raising if the lifecycle forbids it."""
        ensure_run_transition_allowed(self.status, requested)
        self.status = requested


def new_session_id(scenario: SimulationScenario) -> str:
    """Return an identifier for a freshly started run.

    **Unique per run, and that is a correctness requirement rather than a
    convenience.** A telemetry `event_id` is `{session_id}-{machine}-{index}`,
    and the ingest endpoint drops duplicates by primary key -- so two runs
    sharing a session id mint identical event ids, and the second one's readings
    are all discarded. The demo would publish 240 readings and persist none,
    reporting `accepted: 0`, which reads as a bug rather than as idempotency
    working.

    `simulator.domain.session.default_session_id` derives an id from the
    scenario and seed alone, which is right for the command line and wrong here.
    Reproducibility comes from the **seed**, which is stored and replayed;
    uniqueness comes from this. Conflating the two is the natural mistake, so
    they are deliberately separate.

    Readable rather than a bare UUID because the id appears in every event id
    and in log output, and `sim-bd-9f2c41ab` tells a person something. Bounded
    because it has to fit a column alongside a machine id and a tick index.
    """
    tag = "".join(part[0] for part in scenario.value.split("_")).lower()
    return f"sim-{tag}-{uuid4().hex[:8]}"
