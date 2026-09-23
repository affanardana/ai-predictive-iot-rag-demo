"""Human-readable sinks, for watching a run.

Intended to be used as a pair: the telemetry line and the ground-truth line for
one tick are printed consecutively, so the truth line does not repeat the
machine and timestamp it belongs to. That is a presentation convenience, not a
merging of the two channels — they remain separate sinks that happen to write to
the same stream.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import TextIO

from simulator.domain.state import GroundTruthState, TelemetrySample

#: Prefix for a ground-truth line, indented under the observation it belongs to.
_TRUTH_PREFIX = " " * 40 + "truth  "


def _stamp(moment: datetime) -> str:
    """Format an instant as UTC ISO 8601, seconds precision."""
    return moment.astimezone(UTC).isoformat(timespec="seconds")


class ConsoleTelemetrySink:
    """Prints one line per observation."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def write(self, sample: TelemetrySample) -> None:
        """Print one observation."""
        reading = sample.reading
        self._stream.write(
            f"{sample.machine_id} {_stamp(sample.recorded_at)}  "
            f"T {reading.temperature:6.1f}C  "
            f"V {reading.vibration:5.2f}mm/s  "
            f"N {reading.rpm:6.1f}rpm  "
            f"I {reading.current:5.2f}A  "
            f"L {reading.load:5.3f}  "
            f"U {reading.voltage:6.1f}V\n"
        )

    def close(self) -> None:
        """Flush the stream."""
        self._stream.flush()


class ConsoleGroundTruthSink:
    """Prints one line per ground-truth record.

    Deliberately prints the hidden channels next to the observation they explain,
    because seeing vibration climb *and* the bearing wear behind it is the point
    of a demonstration — while the two remain separate streams that could as
    easily go to separate files.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def write(self, state: GroundTruthState) -> None:
        """Print one ground-truth record."""
        degradation = state.degradation
        marker = "  <-- FAILURE IMMINENT" if state.failure_imminent else ""
        self._stream.write(
            f"{_TRUTH_PREFIX}"
            f"health {state.health_index:5.2f}  "
            f"bearing {degradation.bearing_wear:5.2f}  "
            f"thermal {degradation.thermal_stress:5.2f}  "
            f"load {degradation.load_stress:5.2f}"
            f"{marker}\n"
        )

    def close(self) -> None:
        """Flush the stream."""
        self._stream.flush()
