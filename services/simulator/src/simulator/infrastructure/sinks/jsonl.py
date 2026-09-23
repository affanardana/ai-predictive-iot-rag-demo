"""JSON Lines sinks: one JSON object per line.

The shape a message would take on the wire, so this is what a stream looks like
before Phase 6 puts it on MQTT.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from simulator.domain.state import GroundTruthState, TelemetrySample
from simulator.infrastructure.sinks.base import DEFAULT_FLUSH_THRESHOLD, BufferedSink


def json_default(value: object) -> str:
    """Serialise values `json` cannot handle natively.

    Raising for anything else rather than falling back to `str()`, so a new
    field type fails loudly instead of being written as an unparseable repr.
    """
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable.")


class _JsonLinesRowWriter(BufferedSink[dict[str, object]]):
    """Buffers flattened rows and appends them to a file, one per line."""

    def __init__(self, path: Path, flush_threshold: int = DEFAULT_FLUSH_THRESHOLD) -> None:
        super().__init__(flush_threshold)
        self._path = path
        self._handle: TextIO | None = None

    def _flush(self, items: Sequence[dict[str, object]]) -> None:
        handle = self._open()
        for row in items:
            handle.write(json.dumps(row, default=json_default) + "\n")

    def _close_target(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _open(self) -> TextIO:
        """Open the destination on first use, creating parent directories."""
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("w", encoding="utf-8")
        return self._handle


class JsonLinesTelemetrySink:
    """Writes observations to a JSON Lines file."""

    def __init__(self, path: Path, flush_threshold: int = DEFAULT_FLUSH_THRESHOLD) -> None:
        self._writer = _JsonLinesRowWriter(path, flush_threshold)

    def write(self, sample: TelemetrySample) -> None:
        """Write one observation."""
        self._writer.write(sample.as_row())

    def close(self) -> None:
        """Flush and close the file."""
        self._writer.close()


class JsonLinesGroundTruthSink:
    """Writes ground-truth records to a JSON Lines file."""

    def __init__(self, path: Path, flush_threshold: int = DEFAULT_FLUSH_THRESHOLD) -> None:
        self._writer = _JsonLinesRowWriter(path, flush_threshold)

    def write(self, state: GroundTruthState) -> None:
        """Write one ground-truth record."""
        self._writer.write(state.as_row())

    def close(self) -> None:
        """Flush and close the file."""
        self._writer.close()
