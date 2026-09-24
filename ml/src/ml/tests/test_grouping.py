"""Grouping by life and by machine, and the intervals that follow from it."""

from __future__ import annotations

import numpy as np
import pytest

from ml.dataset.labelling import NEVER_FAILS
from ml.evaluation import metrics
from ml.evaluation.errors import EvaluationError
from ml.evaluation.events import (
    cluster_bootstrap,
    describe,
    life_scores,
    per_machine_metric,
)


def _failing_life(length: int, onset: int = 2_000) -> tuple[np.ndarray, np.ndarray]:
    """One life's windows, with the last few scoring high."""
    positions = np.arange(length, dtype=np.int32)
    scores = np.where(positions >= length - 10, 0.9, 0.1).astype(np.float64)
    del onset
    return positions, scores


def test_a_long_life_and_a_short_life_contribute_equally() -> None:
    """Length neutrality, which is the entire point of sampling a fixed count.

    A 48-hour life holds roughly eight times the windows of a 6-hour one, so a
    maximum over everything would give it eight times the chances to look
    alarming — in exactly the split built to catch a model that cheats.
    """
    life = np.array([0] * 400 + [1] * 3_200, dtype=np.int32)
    score = np.concatenate([np.linspace(0.0, 0.4, 400), np.linspace(0.0, 0.4, 3_200)])
    failed = np.array([0, 0], dtype=np.int64)

    result = life_scores(life, failed, score, per_life=50)

    assert result.sampled.tolist() == [50, 50]
    assert result.score[0] == pytest.approx(result.score[1], abs=1e-9)


def test_a_life_shorter_than_the_sample_is_taken_whole() -> None:
    life = np.zeros(12, dtype=np.int32)
    score = np.arange(12, dtype=np.float64)

    result = life_scores(life, np.array([0], dtype=np.int64), score, per_life=50)

    assert result.sampled.tolist() == [12]
    assert result.score[0] == 11.0


def test_the_life_label_comes_from_whether_it_failed() -> None:
    life = np.array([0] * 10 + [1] * 10, dtype=np.int32)
    score = np.zeros(20)
    failed = np.array([1, 0], dtype=np.int64)

    result = life_scores(life, failed, score)

    assert result.label.tolist() == [1, 0]
    assert result.positives == 1
    assert result.prevalence == 0.5


def test_the_reduction_is_a_maximum() -> None:
    """So a single alarming window is enough, which is what detection means."""
    life = np.zeros(100, dtype=np.int32)
    score = np.zeros(100)
    score[37] = 0.99

    result = life_scores(life, np.array([0], dtype=np.int64), score, per_life=100)

    assert result.score[0] == pytest.approx(0.99)


def test_life_scores_rejects_mismatched_arrays() -> None:
    with pytest.raises(EvaluationError, match="same rows"):
        life_scores(np.zeros(4, dtype=np.int32), np.zeros(1, dtype=np.int64), np.zeros(3))


def test_life_scores_rejects_a_flag_array_short_of_the_lives() -> None:
    """One flag per life, indexed by life id — a short array is a mapping error."""
    life = np.array([0, 1, 2], dtype=np.int32)

    with pytest.raises(EvaluationError, match="one flag per life"):
        life_scores(life, np.zeros(2, dtype=np.int64), np.zeros(3))


def test_per_machine_metric_returns_one_number_per_machine() -> None:
    labels = np.array([0, 1, 0, 1, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    machine = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)

    values = per_machine_metric(labels, scores, machine, metric=metrics.average_precision)

    assert len(values) == 3


def test_a_machine_with_no_positives_is_reported_not_skipped() -> None:
    """A machine whose lives never fail is a real outcome of the split.

    Dropping it would make the distribution look better than it is, and the
    whole reason for showing the distribution rather than an interval is
    honesty about spread.
    """
    labels = np.array([0, 0, 0, 1], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    machine = np.array([0, 0, 1, 1], dtype=np.int32)

    values = per_machine_metric(labels, scores, machine, metric=metrics.average_precision)

    assert len(values) == 2
    assert values[0] == 0.0


def test_describe_summarises_a_machine_distribution() -> None:
    summary = describe([0.1, 0.2, 0.3, 0.4, 0.5])

    assert summary["machines"] == 5
    assert summary["median"] == pytest.approx(0.3)
    assert summary["minimum"] == pytest.approx(0.1)
    assert summary["maximum"] == pytest.approx(0.5)


def test_describe_of_nothing_is_empty() -> None:
    assert describe([]) == {}


def test_clustering_widens_the_interval() -> None:
    """Resampling machines must give a strictly wider interval than resampling rows.

    The synthetic fleet is built so half the machines are easily separable and
    half are not, which is the correlation the clustering exists for: whether a
    draw happens to contain the easy machines dominates the result. Data whose
    rows were independent would show no difference at all, and the test would
    pass while proving nothing — which is how the first version of it was
    written.
    """
    rng = np.random.default_rng(17)
    per_machine, machines = 400, 10
    parts_label: list[np.ndarray] = []
    parts_score: list[np.ndarray] = []
    for index in range(machines):
        separable = index % 2 == 0
        labels = (rng.random(per_machine) < 0.1).astype(np.int64)
        noise = rng.random(per_machine)
        parts_label.append(labels)
        parts_score.append(np.where(labels == 1, noise + (0.6 if separable else 0.0), noise))

    labels = np.concatenate(parts_label)
    scores = np.concatenate(parts_score)
    cluster = np.repeat(np.arange(machines, dtype=np.int32), per_machine)
    rows = np.arange(labels.shape[0], dtype=np.int32)

    by_machine = cluster_bootstrap(
        labels, scores, cluster, metric=metrics.average_precision, resamples=200, seed=1
    )
    by_row = cluster_bootstrap(
        labels, scores, rows, metric=metrics.average_precision, resamples=200, seed=1
    )

    assert by_machine.width > by_row.width
    # Not merely wider: the row-level interval is the one that would have been
    # published, and it understates by a factor of several.
    assert by_machine.width > 2 * by_row.width


def test_bootstrapping_needs_more_than_one_cluster() -> None:
    with pytest.raises(EvaluationError, match="two clusters"):
        cluster_bootstrap(
            np.array([0, 1], dtype=np.int64),
            np.array([0.1, 0.9]),
            np.zeros(2, dtype=np.int32),
            metric=metrics.average_precision,
        )


def test_the_bootstrap_is_reproducible_from_its_seed() -> None:
    """A published interval that moves between runs is not a published interval."""
    rng = np.random.default_rng(19)
    labels = (rng.random(600) < 0.15).astype(np.int64)
    scores = rng.random(600)
    cluster = np.repeat(np.arange(6, dtype=np.int32), 100)

    first = cluster_bootstrap(labels, scores, cluster, metric=metrics.average_precision, seed=3)
    second = cluster_bootstrap(labels, scores, cluster, metric=metrics.average_precision, seed=3)

    assert (first.low, first.high) == (second.low, second.high)


def test_a_life_that_never_fails_still_gets_a_score() -> None:
    """Never-failing lives are the negatives of the life-level question."""
    life = np.zeros(100, dtype=np.int32)
    score = np.full(100, 0.02)

    result = life_scores(life, np.array([0], dtype=np.int64), score, per_life=60)

    assert result.label.tolist() == [0]
    assert result.score[0] == pytest.approx(0.02)


def test_the_raw_onset_column_is_refused_as_a_flag() -> None:
    """`NEVER_FAILS` is -32768, which is truthy.

    A caller who passes `life_onset` where a 0/1 flag belongs would label every
    healthy life as having failed — silently, and in the direction that makes
    the model look better.
    """
    life = np.zeros(10, dtype=np.int32)
    onset_column = np.array([NEVER_FAILS], dtype=np.int64)

    with pytest.raises(EvaluationError, match="0/1 per life"):
        life_scores(life, onset_column, np.zeros(10), per_life=10)
