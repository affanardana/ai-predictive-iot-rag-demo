"""The clock baseline: how well you do knowing only when the machine was repaired.

This module exists to answer an uncomfortable question about the dataset before
Phase 4 answers it by accident.

The simulator's degradation is a deterministic function of ``elapsed /
duration``. Nothing about it is random once that ratio is fixed. So a model does
not have to read a single signal to score well — it can infer how far into its
life a machine is, and answer from that alone. A number like "PR-AUC 0.94" means
something quite different if a model that has never seen a reading reaches 0.90.

So that model is fitted and reported. It sees one number per timestep: minutes
since the life began. No signals, no machine identity, nothing else. It is not a
candidate — `elapsed` is deliberately absent from `FEATURE_COLUMNS` and a test
says so — it is a floor, and Phase 4's headline is only meaningful as a margin
above it.

The estimator is a discrete hazard: for an elapsed time ``t``, the share of
training lives still running at ``t`` whose onset falls in the next sixty
minutes. Lives that never fail stay in the denominator forever, which is what
makes the estimate collapse late in a long healthy life — correctly, since a
machine that has not failed after thirty hours is probably not going to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ml.dataset.artifacts import Dataset
from ml.dataset.labelling import HORIZON_MINUTES, NEVER_FAILS
from ml.dataset.splits import Split

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class HazardBaseline:
    """A lookup from elapsed minutes to a failure probability."""

    by_elapsed: Float64Array

    @property
    def horizon(self) -> int:
        """Return how many elapsed minutes the table covers."""
        return int(self.by_elapsed.shape[0])

    def score(self, elapsed: Int64Array) -> Float64Array:
        """Return a failure probability for each elapsed time.

        Times past the table's end score zero: no training life was still
        running that long, so there is nothing to estimate from.
        """
        clipped = np.clip(elapsed, 0, self.horizon - 1)
        inside = elapsed < self.horizon
        return np.where(inside, self.by_elapsed[clipped], 0.0)


def fit(dataset: Dataset) -> HazardBaseline:
    """Fit the baseline from training lives only.

    Training lives, not all lives: a baseline fitted on the test set would be
    measuring the test set, and the whole point of quoting it is that it is an
    honest floor.
    """
    training = {index for index, split in enumerate(dataset.splits) if split is Split.TRAIN}
    onsets = _onsets_by_life(dataset, training)

    # A life that never fails is still running at every elapsed time, so it
    # belongs in the denominator of every hazard estimate and the numerator of
    # none.
    longest = int(max((np.diff(dataset.life_offsets)).max(), HORIZON_MINUTES)) + 1
    table = np.zeros(longest + HORIZON_MINUTES, dtype=np.float64)

    for elapsed in range(table.shape[0]):
        still_running = onsets > elapsed
        denominator = int(still_running.sum())
        if denominator == 0:
            continue
        failing_soon = (onsets > elapsed) & (onsets <= elapsed + HORIZON_MINUTES)
        table[elapsed] = failing_soon.sum() / denominator

    return HazardBaseline(by_elapsed=table)


def _onsets_by_life(dataset: Dataset, training: set[int]) -> npt.NDArray[np.float64]:
    """Return each training life's onset minute, with infinity for never-failing."""
    selected = [
        index for index, machine in enumerate(dataset.life_machine) if int(machine) in training
    ]
    onsets = np.full(len(selected), np.inf, dtype=np.float64)
    for position, life_index in enumerate(selected):
        value = int(dataset.life_onset[life_index])
        if value != NEVER_FAILS:
            onsets[position] = value
    return onsets


def evaluate(
    labels: Int64Array,
    scores: Float64Array,
    *,
    target_precision: float = 0.8,
) -> dict[str, float]:
    """Summarise a scoring model against prevalence.

    `prevalence` is the number to beat and the one most often omitted. A
    positive rate of 0.03 makes a precision of 0.30 look poor and a positive
    rate of 0.30 makes it look excellent, so precision without prevalence is
    unreadable.
    """
    positive = int(labels.sum())
    recall, precision = recall_at_precision(labels, scores, target_precision)
    return {
        "prevalence": positive / labels.shape[0] if labels.shape[0] else 0.0,
        "average_precision": average_precision(labels, scores),
        "recall_at_target_precision": recall,
        "precision_at_target": precision,
        "positives": float(positive),
    }


def average_precision(labels: Int64Array, scores: Float64Array) -> float:
    """Return average precision — the area under the precision-recall curve.

    ROC-AUC is not reported anywhere in this pipeline. At a prevalence near 3%
    the false-positive rate stays tiny until the model is badly wrong, so
    ROC-AUC reads about 0.99 for almost anything and distinguishes nothing.
    """
    if labels.shape[0] == 0 or not labels.any():
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    true_positive = np.cumsum(ranked)
    false_positive = np.cumsum(1 - ranked)
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    recall = true_positive / true_positive[-1]
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def recall_at_precision(
    labels: Int64Array, scores: Float64Array, target: float
) -> tuple[float, float]:
    """Return the highest recall reachable with precision at least `target`."""
    if labels.shape[0] == 0 or not labels.any():
        return 0.0, 0.0
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    true_positive = np.cumsum(ranked)
    precision = true_positive / np.arange(1, ranked.shape[0] + 1)
    recall = true_positive / true_positive[-1]
    eligible = np.flatnonzero(precision >= target)
    if eligible.size == 0:
        return 0.0, 0.0
    best = int(eligible[-1])
    return float(recall[best]), float(precision[best])


def split_rows(
    dataset: Dataset, split: Split, *, trainable_only: bool = True
) -> npt.NDArray[np.int64]:
    """Return the row positions belonging to one split.

    Validation and test keep their *natural* prevalence. Rebalancing is applied
    to training only, because a model's probabilities are calibrated to whatever
    prevalence it was trained at — so measuring calibration or PR-AUC on a
    rebalanced split measures the rebalancing rather than the model.
    """
    wanted = np.array(
        [index for index, value in enumerate(dataset.splits) if value is split],
        dtype=np.int64,
    )
    lives = np.flatnonzero(np.isin(dataset.life_machine, wanted))
    return _rows_for_lives(dataset, lives, trainable_only=trainable_only)


def _rows_for_lives(
    dataset: Dataset, lives: npt.NDArray[np.int64], *, trainable_only: bool
) -> npt.NDArray[np.int64]:
    """Return every row position held by the given lives."""
    if lives.size == 0:
        return np.empty(0, dtype=np.int64)
    lengths = np.diff(dataset.life_offsets)[lives]
    starts = dataset.life_offsets[lives]
    # Each life's rows are contiguous, so the positions are its start repeated
    # for its length plus a counter that resets at each life.
    base = np.repeat(starts, lengths)
    within = np.arange(base.shape[0], dtype=np.int64) - np.repeat(
        np.cumsum(lengths) - lengths, lengths
    )
    rows: Int64Array = base + within
    if not trainable_only:
        return rows
    return rows[dataset.trainable()[rows]]


def elapsed_minutes(dataset: Dataset) -> Int64Array:
    """Return minutes since each row's life began.

    **A diagnostic, never a feature.** It is not in `FEATURE_COLUMNS`, and it
    must stay out: on this simulator, elapsed time is close to the answer, and a
    model given it would be reporting the composition back rather than reading a
    machine. It exists so the baseline above can be fitted and so the report can
    describe the dataset's difficulty honestly.
    """
    # `life_offsets[:-1]` is already one start per life, in life order. It must
    # not be indexed by `life_machine`, which holds *machine* indices — doing so
    # reads the offsets of the first N lives and silently misplaces every row of
    # every later machine, scoring all of them as past the table's end.
    starts = dataset.life_offsets[:-1]
    lengths = np.diff(dataset.life_offsets)
    row_offsets = np.repeat(starts, lengths)
    return np.arange(dataset.rows, dtype=np.int64) - row_offsets
