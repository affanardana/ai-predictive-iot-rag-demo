"""Correcting for the prior shift that rebalancing training introduces.

Training is the only split that may be rebalanced — validation and test keep
their natural 4.4% — so a model trained on a resampled set comes out calibrated
to the *sampled* prevalence, not the real one. Its probabilities are then wrong
by a constant factor in log-odds, and every calibration figure measured on
validation would be reporting the sampler rather than the model.

## Why a constant offset is exactly the right correction

Resampling changes which rows the model sees, but it does not change what a
failing machine looks like. That is **label shift**: `p(y)` moves, `p(x | y)`
does not. Under label shift the optimal decision function changes only by a
constant in log-odds, so

    logit(p_corrected) = logit(p_model) + logit(pi_natural) - logit(pi_trained)

is the exact correction rather than an approximation.

## What it cannot do

If the model ranks badly, no offset fixes it — a constant shift moves every
score the same way and cannot reorder anything. So this is a calibration
correction and never a performance one, and the reliability table in
`ml.evaluation.curves` is where a model that is miscalibrated for a *structural*
reason rather than a prior-shift one will show up regardless of this being
applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ml.evaluation.errors import EvaluationError

Float64Array = npt.NDArray[np.float64]

#: Probabilities are clipped away from 0 and 1 before the log-odds transform.
#: A model that emits an exact 0 or 1 is over-confident rather than correct, and
#: `logit` would return an infinity that propagates through every downstream
#: number instead of showing up as the one bad row it is.
_EPSILON = 1e-12


def logit(probabilities: Float64Array) -> Float64Array:
    """Return the log-odds of `probabilities`, clipped to stay finite."""
    clipped = np.clip(probabilities, _EPSILON, 1.0 - _EPSILON)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: Float64Array) -> Float64Array:
    """Return the logistic function of `values`."""
    return 1.0 / (1.0 + np.exp(-values))


@dataclass(frozen=True, slots=True)
class PriorShift:
    """The log-odds offset between a sampled prevalence and the real one."""

    trained_rate: float
    natural_rate: float

    def __post_init__(self) -> None:
        """Reject rates that are not probabilities."""
        for name, value in (
            ("trained_rate", self.trained_rate),
            ("natural_rate", self.natural_rate),
        ):
            if not 0.0 < value < 1.0:
                raise EvaluationError(f"{name} must be strictly between 0 and 1, got {value}.")

    @property
    def offset(self) -> float:
        """Return the constant to add in log-odds space."""
        return float(logit(np.array([self.natural_rate]))[0]) - float(
            logit(np.array([self.trained_rate]))[0]
        )

    def apply(self, probabilities: Float64Array) -> Float64Array:
        """Return `probabilities` re-calibrated to the natural prevalence."""
        return sigmoid(logit(probabilities) + self.offset)

    def describe(self) -> str:
        """Return a one-line summary, for the report."""
        return (
            f"trained at {self.trained_rate:.2%}, natural {self.natural_rate:.2%}, "
            f"offset {self.offset:+.3f} in log-odds"
        )
