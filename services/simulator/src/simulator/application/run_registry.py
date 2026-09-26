"""Tracks the runs this process is executing.

Framework-free: the HTTP surface above it decides how a start or stop request
arrives, and this decides what those mean. That split is what lets the whole
lifecycle be tested with no server, no broker and no thread -- the runner is
injected, so a test drives it synchronously.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from simulator.application.run_simulation import RunSummary, stream_session
from simulator.domain.errors import RunAlreadyActiveError
from simulator.domain.ports import (
    GroundTruthSink,
    RunReporter,
    RunState,
    RunView,
    TelemetrySink,
)
from simulator.domain.session import SimulationSession

#: Builds the two sinks for one run. Injected so a test can supply recording
#: sinks, and so this module never names the MQTT adapter -- constructing a sink
#: is a composition point's job, per the layering contract.
SinkFactory = Callable[[SimulationSession], tuple[TelemetrySink, GroundTruthSink]]

#: Runs one session. Injected so the registry can be tested without threads or
#: wall-clock waiting; defaults to the real paced loop.
Runner = Callable[..., RunSummary]

#: How many finished runs to remember. Bounded because this is a long-lived
#: process and nothing prunes it otherwise.
HISTORY_LIMIT = 32

#: How often a running run reports progress. Progress exists so the API can tell
#: a live run from an abandoned one, not to animate a progress bar -- the client
#: polls for that. Reporting every tick would be one HTTP request a second per
#: run, against an API sharing this box's single core.
DEFAULT_HEARTBEAT_SECONDS = 5.0


@dataclass
class _ActiveRun:
    """A run this process is executing, and the handle needed to stop it."""

    session: SimulationSession
    #: Set by `stop`, read by the run loop once per tick. A `threading.Event`
    #: because the loop runs on its own thread and the HTTP handler sets this
    #: from another.
    stop_requested: threading.Event = field(default_factory=threading.Event)
    state: RunState = RunState.RUNNING
    completed_ticks: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    error: str | None = None

    @property
    def machine_id(self) -> str:
        """Return the machine this run drives.

        A session may name several machines in principle; a controlled run names
        one, and the API creates one run per machine. Asserted rather than
        assumed, because a session carrying two machines under one run id would
        make stopping it ambiguous.
        """
        return self.session.machines[0].machine_id

    @property
    def is_active(self) -> bool:
        """Whether the run is still advancing."""
        return self.state is RunState.RUNNING

    def view(self) -> RunView:
        """Return this run's state, detached from the live object."""
        return RunView(
            session_id=self.session.session_id,
            machine_id=self.machine_id,
            state=self.state,
            completed_ticks=self.completed_ticks,
            total_ticks=self.session.tick_count,
            started_at=self.started_at,
            finished_at=self.finished_at,
            error=self.error,
        )


class RunRegistry:
    """The runs this process is executing, and how to end them."""

    def __init__(
        self,
        sink_factory: SinkFactory,
        runner: Runner = stream_session,
        tick_seconds: float | None = None,
        reporter: RunReporter | None = None,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    ) -> None:
        self._sink_factory = sink_factory
        self._runner = runner
        self._tick_seconds = tick_seconds
        self._reporter = reporter
        self._heartbeat_seconds = heartbeat_seconds
        self._runs: dict[str, _ActiveRun] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def start(self, session: SimulationSession) -> RunView:
        """Begin a run on its own thread, and return its state.

        Raises:
            RunAlreadyActiveError: if this machine already has an active run.
        """
        run = _ActiveRun(session=session)
        with self._lock:
            if any(
                existing.machine_id == run.machine_id and existing.is_active
                for existing in self._runs.values()
            ):
                raise RunAlreadyActiveError(
                    f"{run.machine_id} already has an active run in this process."
                )
            self._remember(run)

        # A thread rather than a task: the run loop sleeps on the wall clock and
        # is deliberately synchronous, and giving it an event loop would mean
        # rewriting the loop the whole package is built around.
        threading.Thread(
            target=self._execute,
            args=(run,),
            name=f"simulation-{session.session_id}",
            daemon=True,
        ).start()
        return run.view()

    def stop(self, session_id: str) -> RunView | None:
        """Ask a run to end. Idempotent: stopping a stopped run is not an error."""
        with self._lock:
            run = self._runs.get(session_id)
        if run is None:
            return None
        run.stop_requested.set()
        return run.view()

    def get(self, session_id: str) -> RunView | None:
        """Return one run, or None if this process has never seen it."""
        with self._lock:
            run = self._runs.get(session_id)
        return run.view() if run is not None else None

    def list(self) -> Sequence[RunView]:
        """Return every run this process knows about, newest first."""
        with self._lock:
            runs = [self._runs[session_id] for session_id in self._order]
        return [run.view() for run in runs]

    def _remember(self, run: _ActiveRun) -> None:
        """Record a run, dropping the oldest once the history is full."""
        self._runs[run.session.session_id] = run
        self._order.insert(0, run.session.session_id)
        for evicted in self._order[HISTORY_LIMIT:]:
            self._runs.pop(evicted, None)
        del self._order[HISTORY_LIMIT:]

    def _report(self, run: _ActiveRun) -> None:
        """Tell the API where this run has got to, if anyone is listening."""
        if self._reporter is not None:
            self._reporter.report(run.view())

    def _execute(self, run: _ActiveRun) -> None:
        """Run one session to completion, or to a stop, on this thread."""
        try:
            telemetry_sink, ground_truth_sink = self._sink_factory(run.session)
            last_report = 0.0

            def record(completed: int, _total: int) -> None:
                nonlocal last_report
                run.completed_ticks = completed
                # Rate-limited rather than per-tick: this runs on the loop's
                # critical path, and a run at one tick a second would otherwise
                # make one HTTP request a second for the whole demonstration.
                now = time.monotonic()
                if now - last_report >= self._heartbeat_seconds:
                    last_report = now
                    self._report(run)

            self._report(run)
            summary = self._runner(
                run.session,
                telemetry_sink,
                ground_truth_sink,
                tick_seconds=self._tick_seconds or _tick_seconds_from_env(),
                on_tick=record,
                should_stop=run.stop_requested.is_set,
            )
            # A run that reached its final tick completed; one that stopped short
            # was stopped, whether by an operator or by a shutdown.
            run.state = (
                RunState.STOPPED
                if summary.ticks_per_machine < run.session.tick_count
                else RunState.COMPLETED
            )
        except Exception as error:
            # Caught broadly and recorded rather than raised. This is a
            # background thread, so anything escaping would go to the thread's
            # default handler and leave the run at RUNNING forever -- the API
            # would report a healthy run that had actually died, which is worse
            # than reporting a failure with a message.
            run.state = RunState.FAILED
            run.error = f"{type(error).__name__}: {error}"
        finally:
            run.finished_at = datetime.now(UTC)
            # The final report, after the state and the finishing time are both
            # settled. Without it a run that ended between two heartbeats would
            # stay RUNNING in the API until its heartbeat went stale, and the
            # dashboard would show a finished run as live for the whole timeout.
            self._report(run)


def _tick_seconds_from_env() -> float:
    """Return the wall-clock delay between ticks.

    One second, matching the pace the runbook demonstrates: a 240-minute session
    at a one-minute sample interval is 240 ticks, so four minutes of watching
    replays four hours of degradation. Read from the environment because a faster
    pace is useful in testing and a slower one is useful on a loaded box.
    """
    return float(os.environ.get("SIMULATOR_TICK_SECONDS", "1.0"))
