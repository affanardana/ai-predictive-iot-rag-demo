"""Output ports.

Two interfaces, deliberately separate, and owned by the domain rather than by
the sinks that implement them — the same arrangement as the repository ports in
`apps/api`.

`TelemetrySink` accepts only observations and `GroundTruthSink` only hidden
state, so no implementation can receive both and no observation can acquire a
hidden field on the way out. That is the structural half of the guarantee
`MASTERPLAN.md` §3.2 requires; the other half is that dataset mode writes them
to separate files.
"""

from __future__ import annotations

from typing import Protocol

from simulator.domain.state import GroundTruthState, TelemetrySample


class TelemetrySink(Protocol):
    """Receives observable telemetry, and nothing else."""

    def write(self, sample: TelemetrySample) -> None:
        """Accept one observation."""
        ...

    def close(self) -> None:
        """Flush and release any resources."""
        ...


class GroundTruthSink(Protocol):
    """Receives hidden ground-truth state, and nothing else."""

    def write(self, state: GroundTruthState) -> None:
        """Accept one ground-truth record."""
        ...

    def close(self) -> None:
        """Flush and release any resources."""
        ...
