"""Helpers shared across the simulator's tests."""

from __future__ import annotations

from collections.abc import Sequence

from simulator.domain.engine import MachineSimulator
from simulator.domain.machine import MachineProfile
from simulator.domain.readings import SIGNAL_NAMES, SensorReading
from simulator.domain.state import GroundTruthState, TelemetrySample

#: A plausible four-pole motor, for tests that need a profile but do not care
#: about the fleet variation `build_profile` introduces.
PLAIN_PROFILE = MachineProfile(machine_id="M001", name="Motor M001")


def readings_between(
    simulator: MachineSimulator,
    start: int,
    stop: int,
) -> list[SensorReading]:
    """Collect readings for a half-open range of tick indices."""
    return [simulator.tick(index).telemetry.reading for index in range(start, stop)]


def mean_signals(readings: Sequence[SensorReading]) -> dict[str, float]:
    """Average each signal over a series.

    Tests compare trend rather than individual samples: the measurement noise is
    small but not zero, so asserting on a single reading would test the
    generator's randomness rather than the physics behind it.
    """
    return {
        name: sum(getattr(reading, name) for reading in readings) / len(readings)
        for name in SIGNAL_NAMES
    }


class RecordingTelemetrySink:
    """Captures observations, for assertions."""

    def __init__(self) -> None:
        self.samples: list[TelemetrySample] = []
        self.closed = False

    def write(self, sample: TelemetrySample) -> None:
        """Record one observation."""
        self.samples.append(sample)

    def close(self) -> None:
        """Mark the sink closed."""
        self.closed = True


class RecordingGroundTruthSink:
    """Captures ground-truth records, for assertions."""

    def __init__(self) -> None:
        self.states: list[GroundTruthState] = []
        self.closed = False

    def write(self, state: GroundTruthState) -> None:
        """Record one ground-truth record."""
        self.states.append(state)

    def close(self) -> None:
        """Mark the sink closed."""
        self.closed = True
