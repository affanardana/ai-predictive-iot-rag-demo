"""Running the fleet, and writing what it produces.

One telemetry file and one ground-truth file for the whole fleet, as Phase 2
established: separate files with disjoint columns, so a feature builder that
reaches for a label has nothing to reach for.

## Why the sinks here are unusual

`simulator.application.run_session` closes its sinks when it returns, which is
right for a session and wrong for a dataset — a dataset is several thousand
sessions and wants one file. The obvious workaround is to drive the simulator
directly from this module, and it was rejected: `run_session` contains the only
place in the codebase where the two output channels are handled together, and
`_emit` is deliberately one greppable function. A second copy of that loop here
would be a second place for the two channels to be combined, which is the exact
structure the Phase 2 separation tests protect.

So the sinks are session-scoped facades over a writer that outlives them. Their
`close` does nothing, on purpose, and says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import pyarrow as pa
import pyarrow.parquet as pq

from ml.dataset.lives import Life, MachinePlan
from simulator.application import run_session
from simulator.domain.session import DEFAULT_SAMPLE_INTERVAL, SimulationSession
from simulator.domain.state import GroundTruthState, TelemetrySample
from simulator.infrastructure.sinks import GROUND_TRUTH_SCHEMA, TELEMETRY_SCHEMA

#: Rows held in memory before a row group is written. Large enough that the
#: write overhead disappears, small enough that a crash near the end of a long
#: generation loses seconds rather than minutes.
FLUSH_THRESHOLD = 200_000

TELEMETRY_FILENAME = "telemetry.parquet"
GROUND_TRUTH_FILENAME = "ground_truth.parquet"


@dataclass(frozen=True, slots=True)
class GenerationReport:
    """What a generation run produced."""

    machines: int
    lives: int
    rows: int
    output: Path

    @property
    def rows_per_machine(self) -> int:
        """Return the mean number of observations per machine."""
        return self.rows // self.machines if self.machines else 0


class _ChannelAppender:
    """Appends rows to one Parquet file, across as many sessions as it takes."""

    def __init__(self, path: Path, schema: pa.Schema, flush_threshold: int) -> None:
        self._path = path
        self._schema = schema
        self._flush_threshold = flush_threshold
        self._buffer: list[dict[str, object]] = []
        self._writer: pq.ParquetWriter | None = None

    def append(self, row: dict[str, object]) -> None:
        """Buffer one row, writing a row group once the buffer is full."""
        self._buffer.append(row)
        if len(self._buffer) >= self._flush_threshold:
            self._flush()

    def close(self) -> None:
        """Write whatever remains and close the file."""
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _flush(self) -> None:
        if not self._buffer:
            return
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self._path, self._schema, compression="snappy")
        self._writer.write_table(pa.Table.from_pylist(self._buffer, schema=self._schema))
        self._buffer.clear()


class _TelemetryChannel:
    """One session's view of the fleet's telemetry file."""

    def __init__(self, appender: _ChannelAppender) -> None:
        self._appender = appender

    def write(self, sample: TelemetrySample) -> None:
        """Accept one observation."""
        self._appender.append(sample.as_row())

    def close(self) -> None:
        """Do nothing: the file outlives this session and is closed by its owner.

        `run_session` closes both sinks when it returns. For a dataset that is
        thousands of sessions into a single file, honouring that would write a
        footer after the first life and leave the rest unreadable.
        """


class _GroundTruthChannel:
    """The same arrangement for the hidden channel, in its own file."""

    def __init__(self, appender: _ChannelAppender) -> None:
        self._appender = appender

    def write(self, state: GroundTruthState) -> None:
        """Accept one ground-truth record."""
        self._appender.append(state.as_row())

    def close(self) -> None:
        """Do nothing. See `_TelemetryChannel.close`."""


def build_session(plan: MachinePlan, life: Life) -> SimulationSession:
    """Build the session that runs one life of one machine.

    Constructed directly rather than through `SimulationSession.create`, and the
    reason is the profile. `create` derives every machine's nominal point from
    the session seed, so a fresh session per life would give the same machine a
    different rated speed every time it was repaired — and a machine-level split
    only means something if a machine is the same machine throughout.

    Passing the profile in keeps that identity stable while each life still gets
    its own seed, so the noise differs between lives rather than repeating.
    """
    return SimulationSession(
        session_id=life.life_id,
        scenario=life.scenario,
        seed=life.seed,
        started_at=life.started_at,
        sample_interval=DEFAULT_SAMPLE_INTERVAL,
        duration=life.duration,
        machines=(plan.profile,),
    )


def generate(
    plans: Sequence[MachinePlan],
    *,
    output: Path,
    progress: TextIO | None = None,
) -> GenerationReport:
    """Run every life of every machine, writing both channels to Parquet.

    Machines are generated in the order given and each machine's lives in
    sequence, so the telemetry file is grouped by life. `ml.dataset.artifacts`
    relies on that grouping — and verifies it rather than trusting it.
    """
    telemetry_appender = _ChannelAppender(
        output / TELEMETRY_FILENAME, TELEMETRY_SCHEMA, FLUSH_THRESHOLD
    )
    truth_appender = _ChannelAppender(
        output / GROUND_TRUTH_FILENAME, GROUND_TRUTH_SCHEMA, FLUSH_THRESHOLD
    )
    telemetry = _TelemetryChannel(telemetry_appender)
    truth = _GroundTruthChannel(truth_appender)

    lives = 0
    rows = 0
    try:
        for plan in plans:
            for life in plan.lives:
                summary = run_session(build_session(plan, life), telemetry, truth)
                lives += 1
                rows += summary.total_samples
                _announce(progress, plan, life, lives, rows)
    finally:
        # Closed here, not by `run_session`. Closing in a `finally` matters for
        # the same reason it does there: an interrupted generation should leave
        # an inspectable partial dataset rather than an unreadable one.
        telemetry_appender.close()
        truth_appender.close()

    return GenerationReport(machines=len(plans), lives=lives, rows=rows, output=output)


def _announce(
    progress: TextIO | None,
    plan: MachinePlan,
    life: Life,
    lives: int,
    rows: int,
) -> None:
    """Report progress once per machine, not once per life.

    Flushed explicitly. Reporting on a job that takes minutes is only useful if
    it arrives while the job is running, and a pipe — which is how this is
    usually run — makes stdout block-buffered, so without this the whole run's
    progress appears at once, at the end, having reported nothing.
    """
    if progress is None or life.index != 0:
        return
    print(
        f"  {plan.machine_id}  lives={len(plan.lives):>3}  "
        f"scenario={plan.primary_scenario.value:<20} running total={rows:>9,}",
        file=progress,
        flush=True,
    )
