"""Run orchestration: what reaches the sinks, and in what shape."""

from __future__ import annotations

from datetime import timedelta

import pytest

from simulator.application import run_session, stream_session
from simulator.domain.session import SimulationSession
from simulator.domain.state import GroundTruthState, TelemetrySample
from simulator.tests.support import RecordingGroundTruthSink, RecordingTelemetrySink


class ExplodingGroundTruthSink:
    """Fails on its first write, to exercise the cleanup path."""

    def __init__(self) -> None:
        self.closed = False

    def write(self, state: GroundTruthState) -> None:
        """Raise, simulating a failure part way through a run."""
        raise RuntimeError("destination failed")

    def close(self) -> None:
        """Record that the sink was closed."""
        self.closed = True


def test_every_tick_reaches_both_sinks(session: SimulationSession) -> None:
    """One observation and one ground-truth record per tick, inseparably paired."""
    telemetry = RecordingTelemetrySink()
    ground_truth = RecordingGroundTruthSink()

    summary = run_session(session, telemetry, ground_truth)

    assert len(telemetry.samples) == summary.ticks_per_machine
    assert len(ground_truth.states) == summary.ticks_per_machine
    assert [sample.recorded_at for sample in telemetry.samples] == [
        state.recorded_at for state in ground_truth.states
    ]


def test_the_summary_describes_what_was_written(session: SimulationSession) -> None:
    """Counts come from the run, so a caller can report them without counting."""
    summary = run_session(session, RecordingTelemetrySink(), RecordingGroundTruthSink())

    assert summary.session_id == session.session_id
    assert summary.machines == 1
    assert summary.total_samples == session.tick_count


def test_a_multi_machine_run_writes_one_snapshot_per_instant(
    session: SimulationSession,
) -> None:
    """Machines are interleaved tick-major, not written one history at a time.

    A dataset is meant to read as a sequence of fleet snapshots. Each machine's
    whole history in one block would make a cross-machine comparison at a single
    instant impossible without re-sorting the file.
    """
    fleet = SimulationSession.create(
        machine_ids=["M001", "M002"],
        scenario=session.scenario,
        seed=session.seed,
        duration=timedelta(minutes=3),
        started_at=session.started_at,
    )
    telemetry = RecordingTelemetrySink()

    run_session(fleet, telemetry, RecordingGroundTruthSink())

    assert [sample.machine_id for sample in telemetry.samples] == [
        "M001",
        "M002",
        "M001",
        "M002",
        "M001",
        "M002",
    ]


def test_sinks_are_closed_even_when_generation_fails(session: SimulationSession) -> None:
    """A partial dataset is still closed, which for Parquet means it stays readable.

    Without the footer a Parquet file is unopenable, so losing a run half way
    would lose all of it rather than the tail.
    """
    telemetry = RecordingTelemetrySink()
    ground_truth = ExplodingGroundTruthSink()

    with pytest.raises(RuntimeError):
        run_session(session, telemetry, ground_truth)

    assert telemetry.closed
    assert ground_truth.closed


def test_streaming_paces_itself_between_ticks(session: SimulationSession) -> None:
    """The wall clock and the simulated clock are independent.

    `tick_seconds` is the delay; the session's `sample_interval` is how much
    simulated time each tick advances. That separation is what lets an hour of
    degradation play back in seconds.
    """
    delays: list[float] = []

    summary = stream_session(
        session,
        RecordingTelemetrySink(),
        RecordingGroundTruthSink(),
        tick_seconds=0.25,
        sleep=delays.append,
    )

    # One delay between each pair of ticks; none after the last.
    assert delays == [0.25] * (summary.ticks_per_machine - 1)


def test_streaming_without_a_delay_never_sleeps(session: SimulationSession) -> None:
    """A zero pace means as fast as possible, not a zero-length sleep loop."""
    delays: list[float] = []

    stream_session(
        session,
        RecordingTelemetrySink(),
        RecordingGroundTruthSink(),
        tick_seconds=0.0,
        sleep=delays.append,
    )

    assert delays == []


def test_the_two_sinks_receive_different_types(session: SimulationSession) -> None:
    """The split is enforced by type, not by convention."""
    telemetry = RecordingTelemetrySink()
    ground_truth = RecordingGroundTruthSink()

    run_session(session, telemetry, ground_truth)

    assert all(isinstance(sample, TelemetrySample) for sample in telemetry.samples)
    assert all(isinstance(state, GroundTruthState) for state in ground_truth.states)
