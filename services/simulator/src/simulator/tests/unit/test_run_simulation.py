"""Run orchestration: what reaches the sinks, and in what shape."""

from __future__ import annotations

from datetime import timedelta

import pytest

from simulator.application import run_session, stream_session
from simulator.domain.errors import SimulationValidationError
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


def test_streaming_stops_early_when_asked(session: SimulationSession) -> None:
    """A run can be ended from outside, which is what `stop` needs.

    Before this the only way to end a run was `KeyboardInterrupt` reaching the
    process -- there was no flag, no event and no cancellation anywhere in the
    package. PRD section 20.5 requires a stop control, so the loop had to learn
    to be interrupted.

    Five ticks in, out of a session of many more.
    """
    telemetry = RecordingTelemetrySink()
    remaining = 5

    def stop_after_five() -> bool:
        nonlocal remaining
        remaining -= 1
        return remaining < 0

    summary = stream_session(
        session,
        telemetry,
        RecordingGroundTruthSink(),
        should_stop=stop_after_five,
    )

    # Six ticks emitted, then the stop was honoured: the check runs *after* a
    # tick, so the run never ends between a machine's two sinks.
    assert summary.ticks_per_machine == 6
    assert len(telemetry.samples) == 6


def test_a_stopped_run_closes_its_sinks(session: SimulationSession) -> None:
    """Stopping early still runs the cleanup an exception would.

    The columnar sinks write their file footer in `close`, so a run that ended
    without it would leave an unreadable file rather than a partial one.
    """
    telemetry = RecordingTelemetrySink()
    ground_truth = RecordingGroundTruthSink()

    stream_session(
        session,
        telemetry,
        ground_truth,
        should_stop=lambda: True,
    )

    assert telemetry.closed
    assert ground_truth.closed


def test_resuming_at_an_index_produces_the_tail_of_the_same_run(
    session: SimulationSession,
) -> None:
    """A resumed run emits exactly what the original would have, from there on.

    The property the whole recovery story rests on. `MachineSimulator` seeds a
    fresh generator per tick from the tick index, so tick `k` does not depend on
    ticks 0 through `k-1` having been requested -- which means a run interrupted
    by a container restart can be re-issued from where it stopped and produce
    the same series, not a different one that happens to start late.
    """
    whole = RecordingTelemetrySink()
    stream_session(session, whole, RecordingGroundTruthSink())

    resumed = RecordingTelemetrySink()
    summary = stream_session(
        session,
        resumed,
        RecordingGroundTruthSink(),
        start_index=10,
    )

    assert [sample.event_id for sample in resumed.samples] == [
        sample.event_id for sample in whole.samples[10:]
    ]
    # Absolute progress, not this call's count: a caller resuming at 10 has
    # completed 11 ticks of the session, and a progress bar reading "1 of 120"
    # would be worse than showing none.
    assert summary.ticks_per_machine == session.tick_count


def test_resuming_at_the_end_emits_nothing_and_still_closes(
    session: SimulationSession,
) -> None:
    """A run resumed at its final tick is finished, not broken.

    Reachable in practice: a container that died after the last tick but before
    reporting it would be re-issued at exactly this index.
    """
    telemetry = RecordingTelemetrySink()
    ground_truth = RecordingGroundTruthSink()

    summary = stream_session(
        session,
        telemetry,
        ground_truth,
        start_index=session.tick_count,
    )

    assert telemetry.samples == []
    assert telemetry.closed
    assert ground_truth.closed
    assert summary.ticks_per_machine == session.tick_count


def test_resuming_outside_the_run_is_refused(session: SimulationSession) -> None:
    """An index past the end is a caller error, not an empty run.

    Silently emitting nothing would look identical to a run that had already
    finished, and the two need different responses.
    """
    with pytest.raises(SimulationValidationError):
        stream_session(
            session,
            RecordingTelemetrySink(),
            RecordingGroundTruthSink(),
            start_index=session.tick_count + 1,
        )


def test_a_stopped_run_reports_what_it_emitted(session: SimulationSession) -> None:
    """The summary describes the run that happened, not the one that was planned.

    An early stop is not an error -- it is the ordinary way a demonstration
    ends -- so it returns normally, and a caller reading `total_samples` gets
    the truth rather than the session's full length.
    """
    telemetry = RecordingTelemetrySink()

    summary = stream_session(
        session,
        telemetry,
        RecordingGroundTruthSink(),
        should_stop=lambda: True,
    )

    assert summary.total_samples == len(telemetry.samples)
    assert summary.total_samples < session.tick_count


def test_the_two_sinks_receive_different_types(session: SimulationSession) -> None:
    """The split is enforced by type, not by convention."""
    telemetry = RecordingTelemetrySink()
    ground_truth = RecordingGroundTruthSink()

    run_session(session, telemetry, ground_truth)

    assert all(isinstance(sample, TelemetrySample) for sample in telemetry.samples)
    assert all(isinstance(state, GroundTruthState) for state in ground_truth.states)
