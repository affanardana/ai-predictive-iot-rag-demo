"""Threshold metrics, ranking metrics, and the argument about ROC-AUC.

## The two ranking metrics that matter, and the one the spec asks for

At the prevalence this dataset has — 4.6% — the false-positive rate stays tiny
until a model is badly wrong, so ROC-AUC reads about 0.99 for almost anything.
`ml.dataset.baseline` refuses to compute it on those grounds and that refusal
stands; this module computes it anyway, because `MASTERPLAN.md` §6 names it as a
required Phase 4 metric, and a number the spec demands is better published with
its caveat than quietly omitted.

What it does not get to do is appear alone. `ml.evaluation.report` refuses to
render a model's figures without the clock baseline's on the same rows at the
same thresholds, so what a reader sees is always "the model scored 0.994 and so
did a clock". Alongside it go the statistics that carry the same information
honestly — the partial AUC over the low-false-positive region, and the true
positive rate at fixed false-positive rates — because at this prevalence those
are where the discrimination actually shows.

## Reused, not reimplemented

`average_precision` and `recall_at_precision` are re-exported from
`ml.dataset.baseline`. The clock baseline's published average precision and the
model's are then ranked by literally the same arithmetic, which is the only way
a comparison between them means anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ml.dataset.baseline import average_precision, recall_at_precision
from ml.evaluation.errors import EvaluationError

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

__all__ = [
    "Confusion",
    "average_precision",
    "best_f1",
    "confusion",
    "partial_roc_auc",
    "recall_at_precision",
    "roc_auc",
    "summarise",
    "threshold_for_precision",
    "true_positive_rate_at",
]


@dataclass(frozen=True, slots=True)
class Confusion:
    """A confusion matrix, and the rates derived from it."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def support(self) -> int:
        """Return how many rows were classified."""
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def positives(self) -> int:
        """Return how many rows are actually positive."""
        return self.true_positive + self.false_negative

    @property
    def precision(self) -> float:
        """Return the share of predicted positives that are positive.

        Zero when nothing was predicted positive, which is the honest reading:
        a model that predicts nothing has no precision rather than perfect
        precision, and a caller dividing by it wants to know that.
        """
        predicted = self.true_positive + self.false_positive
        return self.true_positive / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        """Return the share of positives that were found."""
        return self.true_positive / self.positives if self.positives else 0.0

    @property
    def f1(self) -> float:
        """Return the harmonic mean of precision and recall."""
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def specificity(self) -> float:
        """Return the share of negatives that were correctly left alone."""
        negatives = self.true_negative + self.false_positive
        return self.true_negative / negatives if negatives else 0.0

    @property
    def predicted_positive_rate(self) -> float:
        """Return the share of rows the model flagged.

        Reported beside prevalence everywhere, because a model that flags
        everything has perfect recall and is useless, and that is invisible in
        recall alone.
        """
        return (self.true_positive + self.false_positive) / self.support if self.support else 0.0


def confusion(labels: Int64Array, scores: Float64Array, *, threshold: float) -> Confusion:
    """Return the confusion matrix at `threshold`.

    The comparison is `score >= threshold`, which is the same direction
    `threshold_for_precision` and `best_f1` select against, so a threshold
    chosen by either reproduces its own numbers here.
    """
    _check(labels, scores)
    predicted = scores >= threshold
    actual = labels == 1
    return Confusion(
        true_positive=int(np.sum(predicted & actual)),
        false_positive=int(np.sum(predicted & ~actual)),
        true_negative=int(np.sum(~predicted & ~actual)),
        false_negative=int(np.sum(~predicted & actual)),
    )


def _average_ranks(values: Float64Array) -> Float64Array:
    """Return each value's rank, averaging the ranks of tied values.

    Ties are not a corner case here. The clock baseline emits one score per
    elapsed minute, so hundreds of thousands of rows share a score; ranking them
    by position instead of averaging would invent an ordering the model does not
    have and quietly inflate whichever class happened to sort last.
    """
    _unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    first = cumulative - counts + 1
    return ((first + cumulative) / 2.0)[inverse]


def roc_auc(labels: Int64Array, scores: Float64Array) -> float:
    """Return the area under the ROC curve.

    Computed from ranks rather than by sweeping thresholds: the rank form is
    exact, tie-aware, and `O(n log n)` instead of quadratic in the number of
    distinct scores.

    **Read this beside the clock baseline, never on its own.** At this
    prevalence almost any ranking scores above 0.99, so the number is
    uninformative in isolation and is required by the specification anyway. See
    the module docstring.
    """
    _check(labels, scores)
    positives = int(np.sum(labels == 1))
    negatives = labels.shape[0] - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ranks = _average_ranks(scores)
    rank_sum = float(np.sum(ranks[labels == 1]))
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _roc_points(labels: Int64Array, scores: Float64Array) -> tuple[Float64Array, Float64Array]:
    """Return the ROC curve as ascending false- and true-positive rates.

    The curve starts at the origin and steps only at distinct scores, so tied
    scores move it by a whole group at once rather than one row at a time.
    """
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    cumulative_positive = np.cumsum(ranked)
    cumulative_negative = np.cumsum(1 - ranked)

    # Keep the last position of every run of equal scores.
    distinct = np.flatnonzero(np.diff(scores[order], append=np.inf) != 0)
    true_positive = cumulative_positive[distinct]
    false_positive = cumulative_negative[distinct]

    positives = float(cumulative_positive[-1]) if ranked.shape[0] else 0.0
    negatives = float(cumulative_negative[-1]) if ranked.shape[0] else 0.0
    if positives == 0 or negatives == 0:
        return np.zeros(1), np.zeros(1)

    return (
        np.concatenate(([0.0], false_positive / negatives)),
        np.concatenate(([0.0], true_positive / positives)),
    )


def partial_roc_auc(labels: Int64Array, scores: Float64Array, *, max_fpr: float = 0.05) -> float:
    """Return the ROC area restricted to false-positive rates at or below `max_fpr`.

    Normalised by `max_fpr`, so an uninformative ranking scores 0.5 rather than
    `max_fpr / 2`. This is the standard remedy for the low-prevalence problem
    above: it looks only at the region where the false-positive rate has not yet
    been swamped, which is the region an operator actually works in.

    Raises:
        EvaluationError: if `max_fpr` is not in `(0, 1]`.
    """
    _check(labels, scores)
    if not 0.0 < max_fpr <= 1.0:
        raise EvaluationError(f"max_fpr must be in (0, 1], got {max_fpr}.")
    fpr, tpr = _roc_points(labels, scores)
    if fpr.shape[0] < 2:
        return 0.0

    # Interpolate the height of the curve exactly at `max_fpr`, so the
    # truncation is not quantised to whichever threshold happened to fall
    # nearby. Without it the area depends on the score grid rather than on the
    # model, and two models with the same curve would score differently.
    inside = fpr < max_fpr
    height = float(np.interp(max_fpr, fpr, tpr))
    area = float(
        np.trapezoid(
            np.concatenate((tpr[inside], [height])),
            np.concatenate((fpr[inside], [max_fpr])),
        )
    )

    # McClish normalisation, which is what makes the number readable. Dividing
    # the raw partial area by `max_fpr` also rescales it, but leaves an
    # uninformative ranking at `max_fpr / 2` — 0.025 at a 5% window — so the
    # figure would move with the window chosen and could not be compared against
    # anything. This maps a random ranking to 0.5, a perfect one to 1.0, and
    # agrees with the full AUC when `max_fpr` is 1.
    minimum = max_fpr**2 / 2.0
    return 0.5 * (1.0 + (area - minimum) / (max_fpr - minimum))


def true_positive_rate_at(
    labels: Int64Array, scores: Float64Array, *, false_positive_rate: float
) -> float:
    """Return the best recall achievable without exceeding a false-positive rate.

    The question an operator asks: "if I can tolerate one false alarm in a
    thousand negatives, how many failures do I catch?" It is a more useful
    number than ROC-AUC at this prevalence and it is reported beside it.
    """
    _check(labels, scores)
    fpr, tpr = _roc_points(labels, scores)
    if fpr.shape[0] < 2:
        return 0.0
    allowed = np.flatnonzero(fpr <= false_positive_rate)
    if allowed.size == 0:
        return 0.0
    return float(tpr[allowed[-1]])


def _threshold_sweep(
    labels: Int64Array, scores: Float64Array
) -> tuple[Float64Array, Float64Array, Float64Array]:
    """Return thresholds, and precision and recall at each, in descending order."""
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    true_positive = np.cumsum(ranked)
    predicted = np.arange(1, ranked.shape[0] + 1)
    precision = true_positive / predicted
    total = true_positive[-1] if ranked.shape[0] else 0
    recall = true_positive / total if total else np.zeros_like(precision)
    return scores[order], precision, recall


def best_f1(labels: Int64Array, scores: Float64Array) -> tuple[float, float]:
    """Return the threshold maximising F1, and that F1.

    Ties are broken towards the *higher* threshold. Two thresholds with equal F1
    are not equivalent in this domain: the higher one alarms less often and
    fires later, so it is the safer of the two to adopt.
    """
    _check(labels, scores)
    if not bool(labels.any()):
        return 1.0, 0.0
    thresholds, precision, recall = _threshold_sweep(labels, scores)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def threshold_for_precision(labels: Int64Array, scores: Float64Array, *, target: float) -> float:
    """Return the lowest threshold whose precision is at least `target`.

    Lowest rather than highest, because a lower threshold is more recall, and
    the point of naming a precision target is to buy as much recall as it will
    pay for.
    """
    _check(labels, scores)
    if not bool(labels.any()):
        return 1.0
    thresholds, precision, _ = _threshold_sweep(labels, scores)
    eligible = np.flatnonzero(precision >= target)
    if eligible.size == 0:
        return 1.0
    return float(thresholds[eligible[-1]])


def summarise(
    labels: Int64Array,
    scores: Float64Array,
    *,
    thresholds: Sequence[float],
    target_precision: float = 0.8,
) -> dict[str, float]:
    """Return every published figure for one set of scores.

    The threshold-dependent metrics are reported at each of `thresholds`, which
    the caller supplies as the product's own risk-band edges — so the report
    answers "what happens if we alarm at WARNING" rather than only "what happens
    at whichever threshold happens to maximise F1".
    """
    _check(labels, scores)
    positives = int(np.sum(labels == 1))
    summary: dict[str, float] = {
        "rows": float(labels.shape[0]),
        "positives": float(positives),
        "prevalence": positives / labels.shape[0] if labels.shape[0] else 0.0,
        "average_precision": average_precision(labels, scores),
        "roc_auc": roc_auc(labels, scores),
        "partial_roc_auc_0.05": partial_roc_auc(labels, scores, max_fpr=0.05),
        "tpr_at_fpr_1e-3": true_positive_rate_at(labels, scores, false_positive_rate=1e-3),
        "tpr_at_fpr_1e-2": true_positive_rate_at(labels, scores, false_positive_rate=1e-2),
        "recall_at_precision_target": recall_at_precision(labels, scores, target_precision)[0],
    }

    f1_threshold, f1 = best_f1(labels, scores)
    summary["best_f1"] = f1
    summary["best_f1_threshold"] = f1_threshold

    for threshold in thresholds:
        matrix = confusion(labels, scores, threshold=threshold)
        prefix = f"at_{threshold:.2f}"
        summary[f"precision_{prefix}"] = matrix.precision
        summary[f"recall_{prefix}"] = matrix.recall
        summary[f"f1_{prefix}"] = matrix.f1
        summary[f"predicted_positive_rate_{prefix}"] = matrix.predicted_positive_rate

    return summary


def _check(labels: Int64Array, scores: Float64Array) -> None:
    """Reject arrays that cannot be scored together.

    Raises:
        EvaluationError: if the shapes disagree, the arrays are empty, or the
            labels are not binary.
    """
    if labels.shape != scores.shape:
        raise EvaluationError(
            f"labels has {labels.shape[0]:,} rows and scores has {scores.shape[0]:,}. "
            "They must describe the same rows."
        )
    if labels.shape[0] == 0:
        raise EvaluationError("Cannot score an empty set of rows.")
    if not bool(np.isin(labels, (0, 1)).all()):
        raise EvaluationError("labels must be binary 0/1.")
