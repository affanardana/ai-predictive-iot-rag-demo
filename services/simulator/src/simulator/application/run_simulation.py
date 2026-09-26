"""Orchestrates a simulation run.

Drives the simulation model and hands its two output channels to their sinks.
The sinks arrive as two separate arguments and are never combined, so the
separation between observation and ground truth survives all the way to the
destination — this layer could not merge them even if it wanted to.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from simulator.domain.engine import MachineSimulator
from simulator.domain.errors import SimulationValidationError
from simulator.domain.ports import GroundTruthSink, TelemetrySink
from simulator.domain.session import SimulationSession
from simulator.domain.state import SimulationTick

#: Called after each simulated instant, with `(completed, total)`.
TickCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class RunSummary:
    """What a run produced."""

    session_id: str
    machines: int
    ticks_per_machine: int

    @property
    def total_samples(self) -> int:
        """Return the number of observations written."""
        return self.machines * self.ticks_per_machine


def run_session(
    session: SimulationSession,
    telemetry_sink: TelemetrySink,
    ground_truth_sink: GroundTruthSink,
    *,
    on_tick: TickCallback | None = None,
) -> RunSummary:
    """Generate every tick for every machine in `session`.

    Machines are emitted tick-major — every machine's reading for instant zero,
    then every machine's reading for instant one — so a multi-machine dataset is
    a sequence of coherent fleet snapshots rather than each machine's history
    end to end.

    Sinks are closed even when generation fails part way. For the columnar sinks
    that matters: closing writes the file footer, which is the difference
    between an inspectable partial dataset and an unreadable one.
    """
    simulators = [MachineSimulator(session, profile) for profile in session.machines]
    ticks = session.tick_count

    try:
        for index in range(ticks):
            for simulator in simulators:
                _emit(simulator.tick(index), telemetry_sink, ground_truth_sink)
            if on_tick is not None:
                on_tick(index + 1, ticks)
    finally:
        telemetry_sink.close()
        ground_truth_sink.close()

    return RunSummary(session.session_id, len(simulators), ticks)


def stream_session(
    session: SimulationSession,
    telemetry_sink: TelemetrySink,
    ground_truth_sink: GroundTruthSink,
    *,
    tick_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    on_tick: TickCallback | None = None,
    should_stop: Callable[[], bool] | None = None,
    start_index: int = 0,
) -> RunSummary:
    """Emit ticks at a wall-clock pace, for watching a machine run.

    The simulated clock and the wall clock are deliberately independent:
    `tick_seconds` is how long to wait between samples, while the session's
    `sample_interval` is how much *simulated* time each sample advances. A tick
    every 0.2 seconds over a session sampling once a minute therefore plays back
    at 300x — which is what makes an hour of degradation watchable in a
    demonstration.

    `sleep` is injectable so a test can drive a paced stream without waiting, and
    `should_stop` is consulted once per tick so a caller can end a run early.
    Both are callables rather than a thread or a signal because that is the shape
    the rest of this module already uses, and because a caller holding a stop
    flag knows better than this function what "stop" means for it.

    A run that stops early returns the same `RunSummary` a completed one does,
    reporting what it actually emitted. It is not an error: an operator pressing
    stop is the ordinary way a demonstration ends.

    `start_index` resumes a run from where it reached. This is safe rather than
    merely convenient, and `MachineSimulator`'s docstring is why: a machine's
    condition is a pure function of the index, so tick `k` is the same whether
    or not ticks 0 through `k-1` were ever requested. A resumed run therefore
    produces exactly the tail of the original series, and every tick it re-emits
    carries an `event_id` the ingest endpoint already holds -- so the duplicate
    is dropped rather than counted twice.

    Raises:
        SimulationValidationError: if `start_index` is outside the run.
    """
    if not 0 <= start_index <= session.tick_count:
        raise SimulationValidationError(
            f"start_index {start_index} is outside the run, which has {session.tick_count} ticks."
        )

    simulators = [MachineSimulator(session, profile) for profile in session.machines]
    ticks = session.tick_count
    # Seeded from `start_index`, not zero: this counts the run's absolute
    # progress. A run resumed at 100 has completed 101 ticks, and one resumed at
    # the final tick -- a container that died after its last tick but before
    # reporting it -- has completed them all, not none.
    emitted = start_index

    try:
        for index in range(start_index, ticks):
            for simulator in simulators:
                _emit(simulator.tick(index), telemetry_sink, ground_truth_sink)
            # Counting the run's absolute progress, not this call's: a caller
            # resuming at 100 has completed 101 ticks of the session, and a
            # progress bar reading 1 of 240 would be worse than none.
            emitted = index + 1
            if on_tick is not None:
                on_tick(emitted, ticks)
            # Checked after the tick rather than before, so a stop request that
            # arrives mid-tick is honoured at the next boundary and the run
            # never ends between a machine's two sinks.
            if should_stop is not None and should_stop():
                break
            if tick_seconds > 0.0 and emitted < ticks:
                sleep(tick_seconds)
    finally:
        telemetry_sink.close()
        ground_truth_sink.close()

    return RunSummary(session.session_id, len(simulators), emitted)


def _emit(
    tick: SimulationTick,
    telemetry_sink: TelemetrySink,
    ground_truth_sink: GroundTruthSink,
) -> None:
    """Route one tick's two halves to their respective sinks.

    The only place the two channels are handled together, and it separates them
    immediately. Kept as one function so that fact is greppable.
    """
    telemetry_sink.write(tick.telemetry)
    ground_truth_sink.write(tick.ground_truth)
