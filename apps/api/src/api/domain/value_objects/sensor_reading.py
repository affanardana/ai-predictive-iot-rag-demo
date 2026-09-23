"""Sensor reading value object.

The six signals are modelled as one object rather than six loose fields
because they are a single measurement taken at a single instant. Separating
them would allow a record to exist with a temperature from one moment and a
vibration from another, which is meaningless for a temporal model.

Validation is deliberately limited to invariants that hold for any electric
motor -- finiteness, and non-negativity for quantities that cannot go
negative. Physically plausible *ranges* are a property of the simulation and
belong to the simulator (Phase 2), not to the domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from api.domain.errors import DomainValidationError

#: Signals that cannot physically be negative. `current` and `temperature` are
#: omitted deliberately: current may be reported signed by some meters, and
#: temperature is legitimately negative in a cold environment.
NON_NEGATIVE_SIGNALS = ("vibration", "rpm", "load", "voltage")


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
        for name in (
            "temperature",
            "vibration",
            "rpm",
            "current",
            "load",
            "voltage",
        ):
            value = getattr(self, name)
            if math.isnan(value) or math.isinf(value):
                raise DomainValidationError(f"Sensor signal '{name}' must be a finite number.")

        for name in NON_NEGATIVE_SIGNALS:
            value = getattr(self, name)
            if value < 0.0:
                raise DomainValidationError(
                    f"Sensor signal '{name}' cannot be negative, got {value}."
                )

    @classmethod
    def signal_names(cls) -> tuple[str, ...]:
        """Return the canonical signal order used by the model and the API."""
        return ("temperature", "vibration", "rpm", "current", "load", "voltage")
