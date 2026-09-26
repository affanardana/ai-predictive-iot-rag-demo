"""Port for model inference.

The model runs in its own service so that torch stays out of the API's
environment and container image. This port is what makes that a detail: the
application calls `Predictor`, and whether the answer comes from a network
service or a fake in a test is decided in the composition root.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from api.domain.value_objects.sensor_reading import SensorReading


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """What the model returned: a probability, and which model said so."""

    failure_probability: float
    model_version: str


class Predictor(Protocol):
    """Scores a window of telemetry."""

    async def predict(self, readings: Sequence[SensorReading]) -> ModelOutput:
        """Return the model's estimate for one window of readings.

        `readings` are ordered oldest first and must span the sequence window.
        Implementations raise `PredictionUnavailableError` when the model cannot
        be reached, rather than letting a transport error escape as a 500.

        Raises:
            PredictionUnavailableError: if inference could not be performed.
        """
        ...
