"""Shared plumbing for sinks.

The `TelemetrySink` and `GroundTruthSink` interfaces themselves live in
`simulator.domain.ports`, alongside the other output ports. Only the batching
helper an implementation may want lives here.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Rows are accumulated before a write. A dataset run produces millions of them,
#: and flushing per row would spend nearly all its time in the filesystem.
DEFAULT_FLUSH_THRESHOLD = 50_000


class BufferedSink[ItemT]:
    """Collects items and hands them to `_flush` in batches.

    Subclasses implement `_flush` and, if they hold a resource open between
    writes, `_close_target`. Neither `write` nor `close` is abstract, so a sink
    that only needs batching gets it for free.
    """

    def __init__(self, flush_threshold: int = DEFAULT_FLUSH_THRESHOLD) -> None:
        if flush_threshold < 1:
            raise ValueError("flush_threshold must be at least 1.")
        self._buffer: list[ItemT] = []
        self._flush_threshold = flush_threshold

    def write(self, item: ItemT) -> None:
        """Buffer one item, flushing if the batch is full."""
        self._buffer.append(item)
        if len(self._buffer) >= self._flush_threshold:
            self._drain()

    def close(self) -> None:
        """Flush what remains and release the target."""
        self._drain()
        self._close_target()

    def _drain(self) -> None:
        """Flush and clear the buffer."""
        if self._buffer:
            batch = self._buffer
            self._buffer = []
            self._flush(batch)

    def _flush(self, items: Sequence[ItemT]) -> None:
        """Write one batch. Subclasses must implement this."""
        raise NotImplementedError

    def _close_target(self) -> None:
        """Release any open resource. Default: nothing to release."""
