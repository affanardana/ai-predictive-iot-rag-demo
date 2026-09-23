"""Simulation state: what is observed, and what is actually true.

Two kinds of state exist here and they must never be confused:

* **observable** — what the sensors report. This becomes telemetry, then a model
  input.
* **ground truth** — what is actually happening inside the machine. This labels
  training data and evaluates the model, and must never be an input.

`MASTERPLAN.md` §3.2 requires that ground truth never be exposed to the model as
a feature, and §25.15 repeats it as a product constraint. Keeping the two in
separate objects, flowing into separate sinks, is how that is enforced rather
than remembered — a field added to an observation cannot be a hidden one by
accident, because there is nowhere for it to come from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from simulator.domain.errors import SimulationValidationError
from simulator.domain.readings import SensorReading
from simulator.domain.scenario import Scenario
from simulator.domain.timestamps import ensure_aware

CHANNEL_NAMES = ("bearing_wear", "thermal_stress", "load_stress")


@dataclass(frozen=True, slots=True)
class DegradationState:
    """How far each failure mechanism has progressed, on `[0, 1]`."""

    bearing_wear: float = 0.0
    thermal_stress: float = 0.0
    load_stress: float = 0.0

    def __post_init__(self) -> None:
        """Validate that every channel is a real fraction."""
        for name in CHANNEL_NAMES:
            value = getattr(self, name)
            if math.isnan(value) or math.isinf(value):
                raise SimulationValidationError(f"Degradation channel '{name}' must be finite.")
            if not 0.0 <= value <= 1.0:
                raise SimulationValidationError(
                    f"Degradation channel '{name}' must be within 0.0-1.0, got {value}."
                )

    def worst(self) -> float:
        """Return the most advanced mechanism.

        A machine is only as healthy as its worst problem, which is why health
        is derived from this rather than from an average that would let a badly
        degraded bearing hide behind two healthy channels.
        """
        # Written out rather than looped over CHANNEL_NAMES: a dynamic
        # `getattr` returns `Any`, which would silently erase the return type.
        return max(self.bearing_wear, self.thermal_stress, self.load_stress)


@dataclass(frozen=True, slots=True)
class GroundTruthState:
    """The hidden condition behind a reading.

    Never written to the telemetry stream, never a model input. The scenario is
    recorded here — and only here — so that Phase 3 can compose a training set
    by failure mode without the product layer ever learning which mode a
    machine is running.
    """

    machine_id: str
    recorded_at: datetime
    scenario: Scenario
    degradation: DegradationState
    health_index: float
    failure_imminent: bool

    def __post_init__(self) -> None:
        """Validate the timestamp and health index."""
        ensure_aware(self.recorded_at, "recorded_at")
        if not 0.0 <= self.health_index <= 1.0:
            raise SimulationValidationError(
                f"health_index must be within 0.0-1.0, got {self.health_index}."
            )

    def as_row(self) -> dict[str, object]:
        """Return the state as a flat mapping, for serialisation.

        Mirrors `TelemetrySample.as_row`, and is intentionally the only place
        ground-truth fields are named together — which makes the split between
        the two channels easy to audit.
        """
        return {
            "machine_id": self.machine_id,
            "recorded_at": self.recorded_at,
            "scenario": self.scenario.value,
            **{name: getattr(self.degradation, name) for name in CHANNEL_NAMES},
            "health_index": self.health_index,
            "failure_imminent": self.failure_imminent,
        }


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    """One observation, as a sensor would report it.

    Deliberately carries no ground-truth field. The field set here is the
    telemetry contract, and a test asserts it stays that way.
    """

    event_id: str
    machine_id: str
    recorded_at: datetime
    session_id: str
    reading: SensorReading

    def __post_init__(self) -> None:
        """Validate the identifiers and timestamp."""
        for field in ("event_id", "machine_id", "session_id"):
            if not getattr(self, field).strip():
                raise SimulationValidationError(f"Telemetry '{field}' must not be blank.")
        ensure_aware(self.recorded_at, "recorded_at")

    def as_row(self) -> dict[str, object]:
        """Return the sample as a flat mapping, for serialisation."""
        return {
            "event_id": self.event_id,
            "machine_id": self.machine_id,
            "recorded_at": self.recorded_at,
            "session_id": self.session_id,
            **self.reading.as_row(),
        }


@dataclass(frozen=True, slots=True)
class SimulationTick:
    """One simulation step: an observation, and the truth behind it.

    Both halves are produced together and then separated immediately — the
    telemetry goes to a `TelemetrySink`, the ground truth to a
    `GroundTruthSink`. Nothing downstream receives both.
    """

    telemetry: TelemetrySample
    ground_truth: GroundTruthState
