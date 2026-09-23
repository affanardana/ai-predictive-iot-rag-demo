"""Failure probability value object."""

from __future__ import annotations

import math
from dataclasses import dataclass

from api.domain.errors import DomainValidationError

PROBABILITY_MIN = 0.0
PROBABILITY_MAX = 1.0


@dataclass(frozen=True, slots=True)
class FailureProbability:
    """The model's estimated probability of failure within the prediction horizon.

    A value outside ``[0.0, 1.0]`` is meaningless rather than dangerous, and a
    neural network can drift a hair past the bounds through floating-point
    error, so out-of-range input is clamped instead of rejected.

    NaN is *not* clamped -- it is rejected. NaN survives every comparison
    silently, so a NaN probability would flow through the risk classifier and
    land on whichever band the comparison order happened to select.
    """

    value: float

    def __post_init__(self) -> None:
        """Reject non-finite input and clamp to the unit interval."""
        if math.isnan(self.value):
            raise DomainValidationError("Failure probability must be a number, not NaN.")
        if math.isinf(self.value):
            raise DomainValidationError("Failure probability must be finite.")
        object.__setattr__(
            self,
            "value",
            min(PROBABILITY_MAX, max(PROBABILITY_MIN, self.value)),
        )

    def as_percentage(self) -> float:
        """Return the probability as a percentage in ``[0.0, 100.0]``."""
        return self.value * 100.0

    def __float__(self) -> float:
        """Allow direct numeric use."""
        return self.value
