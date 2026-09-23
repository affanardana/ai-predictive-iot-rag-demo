"""Parquet sinks: columnar, compressed, and built for volume.

The schema is declared explicitly rather than inferred, so the column types a
Phase 3 loader sees are fixed by this file rather than by whatever happened to
be in the first batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from simulator.domain.readings import SIGNAL_NAMES
from simulator.domain.state import CHANNEL_NAMES, GroundTruthState, TelemetrySample
from simulator.infrastructure.sinks.base import DEFAULT_FLUSH_THRESHOLD, BufferedSink

#: Microsecond precision, UTC. Matches the `timestamptz` columns the API stores
#: the same values in, so a round trip through either does not shift an instant.
_TIMESTAMP = pa.timestamp("us", tz="UTC")

TELEMETRY_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string()),
        pa.field("machine_id", pa.string()),
        pa.field("recorded_at", _TIMESTAMP),
        pa.field("session_id", pa.string()),
        *[pa.field(name, pa.float64()) for name in SIGNAL_NAMES],
    ]
)

#: Written to its own file, never joined to the telemetry table. Phase 3's
#: feature builder reads telemetry; labels come from here. Keeping them apart
#: is what makes leaking a ground-truth column into the features impossible
#: rather than merely discouraged.
GROUND_TRUTH_SCHEMA = pa.schema(
    [
        pa.field("machine_id", pa.string()),
        pa.field("recorded_at", _TIMESTAMP),
        pa.field("scenario", pa.string()),
        *[pa.field(name, pa.float64()) for name in CHANNEL_NAMES],
        pa.field("health_index", pa.float64()),
        pa.field("failure_imminent", pa.bool_()),
    ]
)


class _ParquetRowWriter(BufferedSink[dict[str, object]]):
    """Buffers flattened rows and writes them as Parquet row groups.

    The file is opened lazily on the first flush and fed a row group per batch,
    rather than holding every row until the end. A Phase 3 dataset runs to
    millions of rows: keeping them all in memory before the first write would be
    a poor use of it, and a crash halfway through would lose the lot.
    """

    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        flush_threshold: int = DEFAULT_FLUSH_THRESHOLD,
    ) -> None:
        super().__init__(flush_threshold)
        self._path = path
        self._schema = schema
        self._writer: pq.ParquetWriter | None = None

    def _flush(self, items: Sequence[dict[str, object]]) -> None:
        writer = self._open()
        writer.write_table(pa.Table.from_pylist(list(items), schema=self._schema))

    def _close_target(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _open(self) -> pq.ParquetWriter:
        """Open the destination on first use, creating parent directories."""
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self._path, self._schema, compression="snappy")
        return self._writer


class ParquetTelemetrySink:
    """Writes observations to a Parquet file."""

    def __init__(self, path: Path, flush_threshold: int = DEFAULT_FLUSH_THRESHOLD) -> None:
        self._writer = _ParquetRowWriter(path, TELEMETRY_SCHEMA, flush_threshold)

    def write(self, sample: TelemetrySample) -> None:
        """Write one observation."""
        self._writer.write(sample.as_row())

    def close(self) -> None:
        """Flush and close the file."""
        self._writer.close()


class ParquetGroundTruthSink:
    """Writes ground-truth records to a Parquet file."""

    def __init__(self, path: Path, flush_threshold: int = DEFAULT_FLUSH_THRESHOLD) -> None:
        self._writer = _ParquetRowWriter(path, GROUND_TRUTH_SCHEMA, flush_threshold)

    def write(self, state: GroundTruthState) -> None:
        """Write one ground-truth record."""
        self._writer.write(state.as_row())

    def close(self) -> None:
        """Flush and close the file."""
        self._writer.close()
