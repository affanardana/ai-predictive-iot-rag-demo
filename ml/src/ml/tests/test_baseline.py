"""The clock baseline, and the diagnostic it is built on.

This file exists because the baseline reported chance-level skill on the first
full run, and the cause was not the simulator being hard to read — it was
`elapsed_minutes` reading the offsets of the first N lives and scoring every row
of every later machine as past the table's end. One assertion on the shape of
that array would have caught it, so here it is.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ml.dataset import baseline
from ml.dataset.artifacts import Dataset
from ml.dataset.labelling import NEVER_FAILS
from ml.dataset.splits import Split


def test_elapsed_restarts_at_every_life_and_counts_up_by_one(dataset: Dataset) -> None:
    """Minutes since this row's life began, and nothing else.

    The whole value of the baseline rests on this being right: an offset read
    from the wrong index space puts every row of every later machine outside the
    table, where the score is zero, and the baseline silently reports chance.
    """
    elapsed = baseline.elapsed_minutes(dataset)

    assert elapsed.shape[0] == dataset.rows
    for life_index in range(dataset.lives):
        start = int(dataset.life_offsets[life_index])
        stop = int(dataset.life_offsets[life_index + 1])

        assert elapsed[start] == 0, f"life {life_index} does not start at zero."
        assert np.array_equal(elapsed[start:stop], np.arange(stop - start)), (
            f"life {life_index} does not count up by one."
        )


def test_elapsed_never_exceeds_its_life(dataset: Dataset) -> None:
    """A stale offset would let a row claim to be older than its own life."""
    elapsed = baseline.elapsed_minutes(dataset)
    longest = int(dataset.life_lengths().max())

    assert int(elapsed.max()) < longest
    assert int(elapsed.min()) == 0


def test_scoring_past_the_table_is_zero(dataset: Dataset) -> None:
    """An elapsed time no training life reached carries no information."""
    hazard = baseline.fit(dataset)
    far = np.array([hazard.horizon, hazard.horizon + 10_000], dtype=np.int64)

    assert hazard.score(far).tolist() == [0.0, 0.0]
    inside = hazard.score(np.array([900, 1000], dtype=np.int64))
    assert bool((inside > 0.0).any())


def test_the_table_is_fitted_from_training_lives_only(dataset: Dataset) -> None:
    """Moving a test machine's onset must not move the floor.

    A baseline fitted on the split it is quoted against would flatter itself,
    which defeats the point of quoting it.
    """
    original = baseline.fit(dataset).by_elapsed

    # Rewrite the onset of the first non-training life and refit.
    altered_onset = dataset.life_onset.copy()
    victim = next(
        index
        for index in range(dataset.lives)
        if dataset.splits[int(dataset.life_machine[index])] is not Split.TRAIN
        and int(altered_onset[index]) != NEVER_FAILS
    )
    altered_onset[victim] = 1
    altered = dataclasses.replace(dataset, life_onset=altered_onset)

    assert np.array_equal(baseline.fit(altered).by_elapsed, original)


def test_averaged_precision_scores_a_perfect_ordering_at_one() -> None:
    labels = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.95])

    assert baseline.average_precision(labels, scores) == pytest.approx(1.0)


def test_averaged_precision_does_not_collapse_on_a_reversed_ordering() -> None:
    """The floor for this shape is about 0.48, not zero.

    With three positives among five items, reversing the order puts the two
    negatives first and leaves every positive position forced correct, so even
    the worst possible ranking scores highly. A test expecting near-zero here
    would be asserting something false about average precision — worth writing
    down rather than rediscovering as a surprise later.
    """
    labels = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.95])

    assert baseline.average_precision(labels, -scores) == pytest.approx(0.4778, abs=1e-4)


def test_averaged_precision_of_a_skill_free_ranking_is_the_prevalence() -> None:
    """The property the report's clock baseline is read against.

    "Average precision 0.04 beside a prevalence of 0.045" only means "no skill"
    if a model with no information scores the prevalence. This is that claim,
    checked, on a realistic base rate.
    """
    rng = np.random.default_rng(0)
    labels = (rng.random(20_000) < 0.05).astype(np.int64)
    scores = rng.random(20_000)

    assert baseline.average_precision(labels, scores) == pytest.approx(0.05, abs=0.01)


def test_average_precision_with_no_positives_is_zero() -> None:
    """Rather than a division by zero that reads as a perfect score."""
    labels = np.zeros(10, dtype=np.int64)

    assert baseline.average_precision(labels, np.random.default_rng(0).random(10)) == 0.0


def test_recall_at_a_precision_nobody_reaches_is_zero() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.int64)

    assert baseline.recall_at_precision(labels, np.zeros(4), target=0.99) == (0.0, 0.0)


def test_split_rows_stay_inside_their_split(dataset: Dataset) -> None:
    """Training rows must not be reachable from a test query, or the reverse."""
    train = baseline.split_rows(dataset, Split.TRAIN)
    test = baseline.split_rows(dataset, Split.TEST)

    assert train.size > 0 and test.size > 0
    assert not set(train.tolist()) & set(test.tolist())

    train_machines = {
        int(dataset.life_machine[index])
        for index in range(dataset.lives)
        if dataset.splits[int(dataset.life_machine[index])] is Split.TRAIN
    }
    for row in train[:1000]:
        life = int(np.searchsorted(dataset.life_offsets, row, side="right") - 1)
        assert int(dataset.life_machine[life]) in train_machines


def test_split_rows_exclude_the_post_onset_tail(dataset: Dataset) -> None:
    rows = baseline.split_rows(dataset, Split.TEST)

    assert bool(dataset.trainable()[rows].all())
