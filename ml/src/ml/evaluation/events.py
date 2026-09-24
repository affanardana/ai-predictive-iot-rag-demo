"""Grouping scores by the thing that actually varies: the machine and the life.

Sixty consecutive windows before a failure share 59 of their 60 timesteps and
are about 98% pairwise identical. Treating them as sixty samples understates a
confidence interval by roughly `sqrt(60)`, and lets one hard life contribute
sixty false positives — so per-window F1 penalises a hard life sixty times
harder than an easy one. Everything in this module exists to undo that.

## Clustering is by machine, not by life

Lives of the same machine are not independent draws. A machine's nominal
operating point and its susceptibility are stable across all of its lives, so
lives within a machine are correlated in a way lives across machines are not.
Resampling lives would therefore understate the interval just as badly as
resampling windows. The cluster is the machine.

## The interval at this scale is weak, and says so

There are 15 machine clusters in validation and 12 in `test_shift`. A bootstrap
over twelve clusters produces a lumpy, erratic interval. It is more honest than
a window-level interval and it is *not* reliable, which is why the per-machine
distribution is the primary display and the bootstrap is reported second with
that caveat attached. A number with a caveat is worth more than a number without
one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ml.evaluation.errors import EvaluationError

Float64Array = npt.NDArray[np.float64]
Int32Array = npt.NDArray[np.int32]
Int64Array = npt.NDArray[np.int64]

#: A function turning labels and scores into one number.
Metric = Callable[[Int64Array, Float64Array], float]

#: How many windows a life contributes to a life-level statistic. Sixty is one
#: hour, matching the prediction horizon.
DEFAULT_PER_LIFE = 60


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate and a resampled interval around it."""

    point: float
    low: float
    high: float
    clusters: int
    resamples: int

    @property
    def width(self) -> float:
        """Return the interval's width."""
        return self.high - self.low


@dataclass(frozen=True, slots=True)
class LifeScores:
    """One score per life, and whether that life ever failed."""

    life: Int32Array
    label: Int64Array
    score: Float64Array
    sampled: Int32Array

    @property
    def positives(self) -> int:
        """Return how many lives entered failure."""
        return int(np.count_nonzero(self.label == 1))

    @property
    def prevalence(self) -> float:
        """Return the share of lives that failed."""
        return self.positives / self.life.shape[0] if self.life.shape[0] else 0.0


def life_scores(
    life: Int32Array,
    failed: Int64Array,
    score: Float64Array,
    *,
    per_life: int = DEFAULT_PER_LIFE,
) -> LifeScores:
    """Reduce each life's window scores to a single number.

    Args:
        life: the life each window belongs to.
        failed: one flag per life, indexed by life id — whether that life ever
            entered failure. Per life rather than per row, because every window
            of a life shares the answer and 400,000 copies of it would be a
            second source of truth.
        score: one score per window.
        per_life: how many windows a life contributes, at most.

    The reduction is the maximum over a **fixed number** of evenly spaced
    windows, and both halves of that matter.

    *Fixed* because the obvious alternative — the maximum over everything —
    is length-biased. A 48-hour life holds roughly eight times as many windows
    as a 6-hour one and so gets eight times as many chances to be alarming, and
    `test_shift` is exactly the split where 48-hour and 6-hour lives sit side by
    side. A length-biased statistic there would be systematically generous to
    the split whose whole purpose is to catch a model that cheats.

    *Evenly spaced* rather than randomly sampled, because it is deterministic
    without a seed to record and cannot be quietly re-rolled. The alternative
    needs a seed in the manifest and still moves between runs if anyone forgets
    to pass it.

    Raises:
        EvaluationError: if the arrays disagree in length.
    """
    if life.shape != score.shape:
        raise EvaluationError(
            f"life and score must describe the same rows: {life.shape} against {score.shape}."
        )
    if per_life < 1:
        raise EvaluationError(f"per_life must be at least 1, got {per_life}.")
    if failed.ndim != 1 or (life.shape[0] and int(life.max()) >= failed.shape[0]):
        raise EvaluationError(
            f"failed must hold one flag per life, indexed by life id; it has "
            f"{failed.shape[0]} entries and the lives reach "
            f"{int(life.max()) if life.shape[0] else -1}."
        )
    # A 0/1 flag, not the raw onset column. `life_onset` marks a life that never
    # failed with `NEVER_FAILS`, which is -32768 and therefore *truthy* — so
    # passing it here would label every healthy life as having failed, silently
    # and in the worst possible direction.
    if not bool(np.isin(failed, (0, 1)).all()):
        raise EvaluationError(
            "failed must be 0/1 per life. To build it from an onset column use "
            "`life_onset != NEVER_FAILS`, not the column itself."
        )

    order = np.argsort(life, kind="stable")
    ordered_life = life[order]
    ordered_score = score[order]

    boundaries = np.flatnonzero(np.diff(ordered_life, append=-1) != 0)
    starts = np.concatenate(([0], boundaries + 1))
    stops = np.concatenate((boundaries + 1, [ordered_life.shape[0]]))

    lives: list[int] = []
    labels: list[int] = []
    values: list[float] = []
    counts: list[int] = []

    for start, stop in zip(starts, stops, strict=True):
        count = int(stop - start)
        if count == 0:
            continue
        take = min(per_life, count)
        picks = np.linspace(0, count - 1, take).round().astype(np.int64)
        block = ordered_score[start:stop][picks]
        identifier = int(ordered_life[start])
        lives.append(identifier)
        labels.append(1 if bool(failed[identifier]) else 0)
        values.append(float(block.max()))
        counts.append(take)

    return LifeScores(
        life=np.asarray(lives, dtype=np.int32),
        label=np.asarray(labels, dtype=np.int64),
        score=np.asarray(values, dtype=np.float64),
        sampled=np.asarray(counts, dtype=np.int32),
    )


def per_machine_metric(
    labels: Int64Array,
    scores: Float64Array,
    machine: Int32Array,
    *,
    metric: Metric,
) -> tuple[float, ...]:
    """Return the metric computed within each machine separately.

    The primary display, and the reason is sample size: a cluster bootstrap over
    fifteen machines is erratic, while the fifteen numbers it is resampling are
    not. A reader can see whether the model is uniformly decent or carried by
    two machines, which an interval over the pooled rows cannot show.

    Machines with no positives are reported rather than skipped. A machine whose
    lives never fail is a real outcome of the split, and silently dropping it
    would make the distribution look better than it is.
    """
    return tuple(
        metric(labels[machine == index], scores[machine == index]) for index in np.unique(machine)
    )


def cluster_bootstrap(
    labels: Int64Array,
    scores: Float64Array,
    clusters: Int32Array,
    *,
    metric: Metric,
    resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Resample whole clusters with replacement and return a percentile interval.

    Clusters are drawn, not rows, so the correlation between rows of the same
    machine is carried into the interval instead of being averaged away. The
    resampled set contains each drawn cluster in full, however many rows that is
    — which is the point, and also why the interval is wide.

    Raises:
        EvaluationError: if there are fewer than two clusters or `level` is not
            in `(0, 1)`.
    """
    unique = np.unique(clusters)
    if unique.shape[0] < 2:
        raise EvaluationError(
            f"A cluster bootstrap needs at least two clusters, got {unique.shape[0]}."
        )
    if not 0.0 < level < 1.0:
        raise EvaluationError(f"level must be in (0, 1), got {level}.")
    if resamples < 1:
        raise EvaluationError(f"resamples must be at least 1, got {resamples}.")

    rows_by_cluster = [np.flatnonzero(clusters == value) for value in unique]
    generator = np.random.default_rng(seed)

    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        # `np.asarray` rather than using the return value directly: numpy's stub
        # for `integers` advertises a scalar-or-array union that mypy cannot
        # narrow from the `size` argument, even though the array branch is the
        # only one reachable here.
        chosen = np.asarray(
            generator.integers(0, unique.shape[0], size=unique.shape[0]), dtype=np.int64
        )
        rows = np.concatenate([rows_by_cluster[value] for value in chosen])
        draws[index] = metric(labels[rows], scores[rows])

    tail = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return Interval(
        point=metric(labels, scores),
        low=float(low),
        high=float(high),
        clusters=int(unique.shape[0]),
        resamples=resamples,
    )


def describe(values: Sequence[float]) -> dict[str, float]:
    """Return the five-number summary of a per-machine distribution."""
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "machines": float(array.shape[0]),
        "minimum": float(array.min()),
        "first_quartile": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "third_quartile": float(np.quantile(array, 0.75)),
        "maximum": float(array.max()),
    }
