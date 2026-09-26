"""The run registry: one run per machine, and how runs end.

The runner and the sinks are both injected, so every case here drives the real
lifecycle with a fake that returns immediately. The concurrency is genuine --
`start` really does spawn a thread -- so a few tests wait for a terminal state,
with a deadline rather than a fixed sleep.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

import pytest

from simulator.application.run_registry import HISTORY_LIMIT, RunRegistry
from simulator.application.run_simulation import RunSummary, TickCallback
from simulator.domain.errors import RunAlreadyActiveError
from simulator.domain.ports import GroundTruthSink, RunState, RunView, TelemetrySink
from simulator.domain.scenario import Scenario
from simulator.domain.session import SimulationSession
from simulator.tests.support import RecordingGroundTruthSink, RecordingTelemetrySink

#: A deadline, not a pace. Every fake runner here finishes in microseconds; this
#: exists so a bug is a failed assertion rather than a hung suite.
DEADLINE_SECONDS = 5.0

START = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)


def _session(
    machine_id: str = "M003",
    session_id: str = "sim-bd-test",
    minutes: int = 5,
) -> SimulationSession:
    """Build a session, with the identifiers a test cares about exposed."""
    return SimulationSession.create(
        machine_ids=[machine_id],
        scenario=Scenario.BEARING_DEGRADATION,
        seed=1,
        duration=timedelta(minutes=minutes),
        started_at=START,
        session_id=session_id,
    )


def _recording_sinks(_session: SimulationSession) -> tuple[TelemetrySink, GroundTruthSink]:
    """Build sinks that keep what they are given, and count as a real pair."""
    return RecordingTelemetrySink(), RecordingGroundTruthSink()


def _completing_runner(
    session: SimulationSession,
    telemetry: TelemetrySink,
    ground_truth: GroundTruthSink,
    *,
    tick_seconds: float,
    on_tick: TickCallback | None,
    should_stop: Callable[[], bool] | None,
) -> RunSummary:
    """Finish immediately, reporting every tick."""
    if on_tick is not None:
        for index in range(session.tick_count):
            on_tick(index + 1, session.tick_count)
    return RunSummary(session.session_id, 1, session.tick_count)


def _blocking_runner(
    session: SimulationSession,
    telemetry: TelemetrySink,
    ground_truth: GroundTruthSink,
    *,
    tick_seconds: float,
    on_tick: TickCallback | None,
    should_stop: Callable[[], bool] | None,
) -> RunSummary:
    """Wait until stopped, reporting a couple of ticks first."""
    if on_tick is not None:
        on_tick(1, session.tick_count)
    deadline = time.monotonic() + DEADLINE_SECONDS
    while should_stop is not None and not should_stop() and time.monotonic() < deadline:
        time.sleep(0.001)
    if on_tick is not None:
        on_tick(2, session.tick_count)
    return RunSummary(session.session_id, 1, 2)


def _wait_for_terminal(registry: RunRegistry, session_id: str) -> RunView:
    """Return a run once it has stopped moving."""
    deadline = time.monotonic() + DEADLINE_SECONDS
    while time.monotonic() < deadline:
        view = registry.get(session_id)
        if view is not None and view.state is not RunState.RUNNING:
            return view
        time.sleep(0.001)
    raise AssertionError(f"Run {session_id} never reached a terminal state.")


class _RecordingReporter:
    """Keeps every report, so a test can assert on the sequence."""

    def __init__(self) -> None:
        self.reports: list[RunView] = []

    def report(self, run: RunView) -> None:
        self.reports.append(run)

    def close(self) -> None:
        """Nothing to release."""


def test_a_completed_run_reports_completed() -> None:
    """Reaching the final tick is completion, not a stop."""
    registry = RunRegistry(sink_factory=_recording_sinks, runner=_completing_runner)
    registry.start(_session())

    view = _wait_for_terminal(registry, "sim-bd-test")

    assert view.state is RunState.COMPLETED
    assert view.completed_ticks == view.total_ticks
    assert view.finished_at is not None


def test_a_second_run_for_one_machine_is_refused() -> None:
    """Two runs on one machine would interleave two scenarios' readings.

    A machine's series is a pure function of its own session, so runs on
    *different* machines cannot interfere -- but two on the same machine would
    produce a risk band describing neither scenario, and no error anywhere.
    """
    registry = RunRegistry(sink_factory=_recording_sinks, runner=_blocking_runner)
    registry.start(_session(session_id="sim-first"))

    try:
        with pytest.raises(RunAlreadyActiveError):
            registry.start(_session(session_id="sim-second"))
    finally:
        registry.stop("sim-first")


def test_a_run_for_another_machine_is_allowed() -> None:
    """Concurrency is per machine, which is what the fleet view needs."""
    registry = RunRegistry(sink_factory=_recording_sinks, runner=_blocking_runner)
    registry.start(_session(machine_id="M003", session_id="sim-m3"))
    registry.start(_session(machine_id="M004", session_id="sim-m4"))

    try:
        assert registry.get("sim-m3") is not None
        assert registry.get("sim-m4") is not None
    finally:
        registry.stop("sim-m3")
        registry.stop("sim-m4")


def test_stopping_a_run_ends_it_as_stopped() -> None:
    """A stopped run is not a failed one, and not a completed one."""
    registry = RunRegistry(sink_factory=_recording_sinks, runner=_blocking_runner)
    registry.start(_session())

    registry.stop("sim-bd-test")
    view = _wait_for_terminal(registry, "sim-bd-test")

    assert view.state is RunState.STOPPED
    assert view.completed_ticks < view.total_ticks


def test_stopping_twice_is_not_an_error() -> None:
    """Stop is idempotent, so a double-click needs no special handling.

    The API's `DELETE` may also be retried by a client that did not see the
    first response, and neither case should read as a fault.
    """
    registry = RunRegistry(sink_factory=_recording_sinks, runner=_blocking_runner)
    registry.start(_session())

    assert registry.stop("sim-bd-test") is not None
    assert registry.stop("sim-bd-test") is not None
    _wait_for_terminal(registry, "sim-bd-test")


def test_stopping_an_unknown_run_returns_none() -> None:
    """A run this process has never seen is absent, not an error."""
    registry = RunRegistry(sink_factory=_recording_sinks, runner=_completing_runner)

    assert registry.stop("never-existed") is None
    assert registry.get("never-existed") is None


def test_a_failing_runner_is_reported_rather_than_lost() -> None:
    """An exception on the run thread must surface as a status.

    Without the catch this would go to the thread's default handler and leave the
    run at RUNNING forever: the API would report a healthy run that had died,
    which is worse than reporting a failure with a message.
    """

    def exploding_runner(
        session: SimulationSession,
        telemetry: TelemetrySink,
        ground_truth: GroundTruthSink,
        **options: object,
    ) -> RunSummary:
        raise RuntimeError("the broker refused the connection")

    registry = RunRegistry(sink_factory=_recording_sinks, runner=exploding_runner)
    registry.start(_session())

    view = _wait_for_terminal(registry, "sim-bd-test")

    assert view.state is RunState.FAILED
    assert view.error is not None
    assert "broker refused" in view.error


def test_the_reporter_sees_the_run_start_and_finish() -> None:
    """The API learns about a run without polling for it.

    The API's broadcaster is in-process, so a container that never reported would
    leave a finished run displayed as running until something happened to ask.
    """
    reporter = _RecordingReporter()
    registry = RunRegistry(
        sink_factory=_recording_sinks,
        runner=_completing_runner,
        reporter=reporter,
    )

    registry.start(_session())
    _wait_for_terminal(registry, "sim-bd-test")

    assert reporter.reports[0].state is RunState.RUNNING
    assert reporter.reports[-1].state is RunState.COMPLETED


def test_history_is_bounded() -> None:
    """A long-lived process does not accumulate every run it ever executed."""
    registry = RunRegistry(sink_factory=_recording_sinks, runner=_completing_runner)

    for index in range(HISTORY_LIMIT + 5):
        registry.start(_session(machine_id=f"M{index:03d}", session_id=f"sim-{index}"))
    deadline = time.monotonic() + DEADLINE_SECONDS
    while time.monotonic() < deadline and len(registry.list()) < HISTORY_LIMIT:
        time.sleep(0.001)

    assert len(registry.list()) == HISTORY_LIMIT


def test_runs_are_listed_newest_first() -> None:
    """The order a UI wants, and stable regardless of thread scheduling."""
    registry = RunRegistry(sink_factory=_recording_sinks, runner=_completing_runner)

    registry.start(_session(machine_id="M001", session_id="sim-old"))
    time.sleep(0.002)
    registry.start(_session(machine_id="M002", session_id="sim-new"))

    listed: Sequence[RunView] = registry.list()
    assert [run.session_id for run in listed] == ["sim-new", "sim-old"]
