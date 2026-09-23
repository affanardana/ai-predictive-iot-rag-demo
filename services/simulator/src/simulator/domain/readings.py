"""The observable telemetry signals.

Mirrors the shape `apps/api` defines for its `SensorReading`: the same field
names, the same declaration order, and the same rejection of non-finite and
physically impossible values.

The two services deliberately do not share code — see the Phase 2 plan. The API
validates values it *receives*; the simulator must produce values that are
*physically plausible*, which is a stricter obligation, and coupling a batch
generator to FastAPI and SQLAlchemy to save eleven lines would be a poor trade.
What stops the two drifting apart is a test: the emitted readings are validated
against the API's own `SensorReading`, so a change to either side that breaks
the contract fails the build.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from simulator.domain.errors import SimulationValidationError

#: Signal order is part of the wire contract. `apps/api` uses the identical
#: order for its `SensorReading.signal_names()` and for the column order of its
#: bucketed telemetry query.
SIGNAL_NAMES = ("temperature", "vibration", "rpm", "current", "load", "voltage")

#: Signals that cannot be negative. `temperature` and `current` are excluded
#: deliberately, matching the API: current may be reported signed by some
#: meters, and a temperature below zero is legitimate in a cold environment.
NON_NEGATIVE_SIGNALS = ("vibration", "rpm", "load", "voltage")


def signal_names() -> tuple[str, ...]:
    """Return the canonical signal order."""
    return SIGNAL_NAMES


@dataclass(frozen=True, slots=True)
class SensorReading:
    """The six telemetry signals captured at one instant."""

    temperature: float
    vibration: float
    rpm: float
    current: float
    load: float
    voltage: float

    def __post_init__(self) -> None:
        """Reject non-finite values and negative readings where impossible."""
        for name in SIGNAL_NAMES:
            value = getattr(self, name)
            if math.isnan(value) or math.isinf(value):
                raise SimulationValidationError(
                    f"Signal '{name}' must be a finite number, got {value}."
                )

        for name in NON_NEGATIVE_SIGNALS:
            value = getattr(self, name)
            if value < 0.0:
                raise SimulationValidationError(f"Signal '{name}' cannot be negative, got {value}.")

    def as_row(self) -> dict[str, float]:
        """Return the signals as a mapping, for serialisation."""
        return {name: getattr(self, name) for name in SIGNAL_NAMES}
