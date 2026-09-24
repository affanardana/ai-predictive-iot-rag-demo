"""Calibration: whether a probability of 0.8 means eight times in ten.

The product tells an operator that a machine has a 0.81 probability of failing
within the hour, and `PRD.md` §9 turns that number into a risk band with hard
edges at 0.30, 0.60 and 0.80. Every one of those is a claim about frequency, so
a model whose ranking is excellent and whose probabilities are meaningless would
still make the product lie to its user. That is what this module measures.

## Two things that make the naive measurement useless here

**Expected calibration error over the full range is dominated by the bulk.** At
4.6% prevalence, roughly 95% of rows sit near `p = 0.01`, and a model can be
badly wrong in the top 5% while ECE stays excellent, because ECE averages the
error over rows rather than over the region anyone acts on. So the reliability
curve is reported in full and the calibration figures that matter are computed
on the **high-risk slice** as well as overall.

**Bins near the top are small, so normal-approximation intervals lie.** A bin
holding 200 rows with 40 positives gives a confidence interval that the normal
approximation reports as symmetric and possibly negative; Wilson's interval does
not. Bin counts are printed beside every interval, because an interval over nine
rows is not an interval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from ml.evaluation.errors import EvaluationError

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

#: Two-sided z for a 95% interval. Passed as a constant rather than computed
#: from a distribution function, because scipy is not a dependency of this
#: package and one hard-coded constant is clearer than a hand-rolled inverse
#: normal CDF.
_Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, *, z: float = _Z_95) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion.

    Preferred to the normal approximation because it stays inside `[0, 1]` and
    behaves sanely when `successes` is 0 or `total` is small — both common in
    the high-score bins this module cares most about.
    """
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = (
        z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4 * total * total))
    ) / denominator
    return float(max(0.0, centre - spread)), float(min(1.0, centre + spread))


def brier_score(labels: Int64Array, scores: Float64Array) -> float:
    """Return the mean squared error of the probabilities.

    Reported beside the reliability curve because it is the one calibration
    number that is a proper score — it cannot be improved by hedging — so it is
    the honest scalar when a single figure is wanted.
    """
    return float(np.mean((scores - labels) ** 2))


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bin of the reliability curve."""

    lower: float
    upper: float
    count: int
    positives: int
    mean_score: float
    observed_rate: float
    lower_ci: float
    upper_ci: float

    @property
    def gap(self) -> float:
        """Return how far the bin's predictions are from reality."""
        return self.mean_score - self.observed_rate

    @property
    def overlaps_identity(self) -> bool:
        """Whether the interval contains a perfectly calibrated bin."""
        return self.lower_ci <= self.mean_score <= self.upper_ci


@dataclass(frozen=True, slots=True)
class Reliability:
    """A reliability curve and the scalars derived from it."""

    bins: tuple[ReliabilityBin, ...]
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    rows: int
    positives: int


def reliability(
    labels: Int64Array,
    scores: Float64Array,
    *,
    bins: int = 15,
    strategy: Literal["quantile", "uniform"] = "quantile",
) -> Reliability:
    """Return the reliability curve of `scores` against `labels`.

    Bins are quantile-spaced by default. Uniform bins would put nine tenths of
    the rows in the first bin at this prevalence and leave the top bins holding
    a handful of rows each — which is precisely where the interesting
    miscalibration lives.
    """
    if labels.shape != scores.shape:
        raise EvaluationError("labels and scores must describe the same rows.")
    if labels.shape[0] == 0:
        raise EvaluationError("Cannot measure calibration on an empty set of rows.")
    if bins < 1:
        raise EvaluationError(f"bins must be at least 1, got {bins}.")

    edges = _bin_edges(scores, bins=bins, strategy=strategy)
    assignments = np.clip(np.digitize(scores, edges[1:-1]), 0, len(edges) - 2)

    members: list[ReliabilityBin] = []
    weighted_error = 0.0
    worst = 0.0
    for index in range(len(edges) - 1):
        chosen = assignments == index
        count = int(np.count_nonzero(chosen))
        if count == 0:
            continue
        positives = int(np.count_nonzero(labels[chosen] == 1))
        mean_score = float(np.mean(scores[chosen]))
        observed = positives / count
        lower, upper = wilson_interval(positives, count)
        members.append(
            ReliabilityBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                positives=positives,
                mean_score=mean_score,
                observed_rate=observed,
                lower_ci=lower,
                upper_ci=upper,
            )
        )
        gap = abs(mean_score - observed)
        weighted_error += gap * count
        worst = max(worst, gap)

    total = int(labels.shape[0])
    return Reliability(
        bins=tuple(members),
        expected_calibration_error=weighted_error / total if total else 0.0,
        maximum_calibration_error=worst,
        brier_score=brier_score(labels, scores),
        rows=total,
        positives=int(np.count_nonzero(labels == 1)),
    )


def _bin_edges(
    scores: Float64Array, *, bins: int, strategy: Literal["quantile", "uniform"]
) -> Float64Array:
    """Return `bins + 1` edges, deduplicated.

    Deduplicated because the clock baseline emits one score per elapsed minute:
    a fifteen-quantile split over it can produce repeated edges, and repeated
    edges produce empty bins and a curve with fewer points than asked for.
    """
    inner = (
        np.quantile(scores, np.linspace(0.0, 1.0, bins + 1)[1:-1])
        if strategy == "quantile"
        else np.linspace(0.0, 1.0, bins + 1)[1:-1]
    )
    inner = np.unique(inner)
    return np.concatenate(([-np.inf], inner, [np.inf]))


def reliability_from_slice(
    labels: Int64Array,
    scores: Float64Array,
    *,
    share: float,
    bins: int = 10,
) -> Reliability:
    """Return calibration restricted to the highest-scoring `share` of rows.

    The top-of-the-range check. A model that is well calibrated overall and
    over-confident in the region where it raises alarms is one that will cost an
    operator real time, and averaging over the other ninety-five percent of rows
    hides it completely.
    """
    if not 0.0 < share <= 1.0:
        raise EvaluationError(f"share must be in (0, 1], got {share}.")
    count = max(1, round(labels.shape[0] * share))
    chosen = np.argsort(-scores, kind="stable")[:count]
    return reliability(labels[chosen], scores[chosen], bins=bins)


def risk_bands(
    labels: Int64Array,
    scores: Float64Array,
    *,
    edges: Sequence[float],
) -> tuple[dict[str, float], ...]:
    """Return per-band counts and observed rates for the product's own bands.

    `PRD.md` §9 defines NORMAL/WARNING/HIGH/CRITICAL at 0.30, 0.60 and 0.80.
    Those bands are the interface between this model and everything built on
    it, so the report answers "when the product says CRITICAL, how often is it
    right" directly rather than leaving it to be inferred from a curve.
    """
    boundaries = [-np.inf, *edges, np.inf]
    bands: list[dict[str, float]] = []
    for index in range(len(boundaries) - 1):
        chosen = (scores >= boundaries[index]) & (scores < boundaries[index + 1])
        count = int(np.count_nonzero(chosen))
        positives = int(np.count_nonzero(labels[chosen] == 1)) if count else 0
        bands.append(
            {
                "lower": float(boundaries[index]),
                "upper": float(boundaries[index + 1]),
                "rows": float(count),
                "share": count / labels.shape[0] if labels.shape[0] else 0.0,
                "observed_rate": positives / count if count else 0.0,
            }
        )
    return tuple(bands)
