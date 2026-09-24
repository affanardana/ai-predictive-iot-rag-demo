"""Threshold and ranking metrics, checked against values known by construction."""

from __future__ import annotations

import numpy as np
import pytest

from ml.evaluation import metrics
from ml.evaluation.errors import EvaluationError


def _separable() -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    return labels, scores


def test_roc_auc_at_the_extremes() -> None:
    labels, scores = _separable()

    assert metrics.roc_auc(labels, scores) == pytest.approx(1.0)
    assert metrics.roc_auc(labels, -scores) == pytest.approx(0.0)


def test_roc_auc_of_an_uninformative_ranking_is_one_half() -> None:
    """Including the all-ties case, which is what the clock baseline produces.

    The hazard baseline emits one score per elapsed minute, so hundreds of
    thousands of rows share a score. A rank function that breaks ties by
    position would invent an ordering the model does not have.
    """
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    assert metrics.roc_auc(labels, np.full(4, 0.5)) == pytest.approx(0.5)


def test_roc_auc_matches_a_brute_force_count() -> None:
    """The rank formula against the definition, on data with no structure."""
    rng = np.random.default_rng(7)
    labels = (rng.random(400) < 0.2).astype(np.int64)
    scores = rng.random(400)

    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    wins = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    expected = (wins + 0.5 * ties) / (positives.shape[0] * negatives.shape[0])

    assert metrics.roc_auc(labels, scores) == pytest.approx(float(expected))


def test_partial_auc_over_the_whole_range_is_the_full_auc() -> None:
    """The partial statistic has to agree with the full one at its own limit."""
    rng = np.random.default_rng(3)
    labels = (rng.random(500) < 0.15).astype(np.int64)
    scores = rng.random(500)

    assert metrics.partial_roc_auc(labels, scores, max_fpr=1.0) == pytest.approx(
        metrics.roc_auc(labels, scores)
    )


def test_partial_auc_of_an_uninformative_ranking_is_one_half() -> None:
    """Normalised by max_fpr, so a random ranking scores 0.5 rather than 0.025.

    Without the normalisation the number would scale with the window chosen and
    could not be compared between two reports.
    """
    rng = np.random.default_rng(11)
    labels = (rng.random(20_000) < 0.05).astype(np.int64)
    scores = rng.random(20_000)

    assert metrics.partial_roc_auc(labels, scores, max_fpr=0.05) == pytest.approx(0.5, abs=0.1)


def test_partial_auc_rejects_an_impossible_window() -> None:
    labels, scores = _separable()

    with pytest.raises(EvaluationError, match="max_fpr"):
        metrics.partial_roc_auc(labels, scores, max_fpr=0.0)


def test_true_positive_rate_at_a_zero_false_positive_rate() -> None:
    """A perfectly separable ranking catches everything without a single false alarm."""
    labels, scores = _separable()

    assert metrics.true_positive_rate_at(labels, scores, false_positive_rate=0.0) == 1.0
    assert metrics.true_positive_rate_at(labels, -scores, false_positive_rate=0.0) == 0.0


def test_confusion_rates() -> None:
    labels, scores = _separable()
    matrix = metrics.confusion(labels, scores, threshold=0.5)

    assert (matrix.true_positive, matrix.false_positive) == (2, 0)
    assert (matrix.true_negative, matrix.false_negative) == (2, 0)
    assert matrix.precision == 1.0
    assert matrix.recall == 1.0
    assert matrix.f1 == 1.0


def test_a_model_that_predicts_nothing_has_no_precision() -> None:
    """Zero rather than undefined, and certainly not perfect.

    Dividing by "we predicted nothing" is how a useless model reports a
    flawless precision, so the property returns 0.0 and a caller that cares can
    see it from `predicted_positive_rate`.
    """
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    matrix = metrics.confusion(labels, np.zeros(4), threshold=0.5)

    assert matrix.precision == 0.0
    assert matrix.recall == 0.0
    assert matrix.predicted_positive_rate == 0.0


def test_best_f1_breaks_ties_towards_the_higher_threshold() -> None:
    """Two thresholds with equal F1 are not equivalent in this domain.

    The higher one alarms less often and fires later, so it is the safer of the
    two to adopt, and ties should resolve that way rather than arbitrarily.
    """
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    scores = np.array([0.1, 0.9, 0.2, 0.8])

    threshold, f1 = metrics.best_f1(labels, scores)

    assert f1 == pytest.approx(1.0)
    assert metrics.confusion(labels, scores, threshold=threshold).f1 == pytest.approx(f1)


def test_threshold_for_precision_reproduces_its_own_precision() -> None:
    """The returned threshold has to round-trip, or the report and the matrix disagree.

    The scores carry real signal on purpose. A *random* ranking cannot reach a
    precision of 0.5 at 30% prevalence — its top rows score about the base rate,
    so the target is unreachable and the fallback fires, which is correct
    behaviour and would make this test assert nothing.
    """
    rng = np.random.default_rng(5)
    labels = (rng.random(2_000) < 0.3).astype(np.int64)
    scores = np.clip(labels * 0.5 + rng.random(2_000) * 0.5, 0.0, 1.0)

    threshold = metrics.threshold_for_precision(labels, scores, target=0.5)
    matrix = metrics.confusion(labels, scores, threshold=threshold)

    assert matrix.precision >= 0.5
    assert threshold < 1.0, "a reachable target must not fall back to certainty."


def test_a_target_precision_nobody_reaches_falls_back_to_certainty() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.int64)

    assert metrics.threshold_for_precision(labels, np.zeros(4), target=0.9) == 1.0


def test_summarise_reports_every_product_risk_band() -> None:
    """PRD §9's edges, because that is the interface everything else uses."""
    rng = np.random.default_rng(13)
    labels = (rng.random(1_000) < 0.1).astype(np.int64)
    scores = rng.random(1_000)

    summary = metrics.summarise(labels, scores, thresholds=(0.30, 0.60, 0.80))

    for edge in ("0.30", "0.60", "0.80"):
        assert f"precision_at_{edge}" in summary
        assert f"recall_at_{edge}" in summary
        assert f"predicted_positive_rate_at_{edge}" in summary
    for key in ("average_precision", "roc_auc", "partial_roc_auc_0.05", "best_f1"):
        assert key in summary


def test_mismatched_inputs_are_refused() -> None:
    with pytest.raises(EvaluationError, match="same rows"):
        metrics.roc_auc(np.array([0, 1], dtype=np.int64), np.array([0.5]))


def test_non_binary_labels_are_refused() -> None:
    """A silently-ignored class would corrupt every rate computed from it."""
    with pytest.raises(EvaluationError, match="binary"):
        metrics.roc_auc(np.array([0, 2], dtype=np.int64), np.array([0.5, 0.5]))


def test_an_empty_set_is_refused() -> None:
    with pytest.raises(EvaluationError, match="empty"):
        metrics.roc_auc(np.array([], dtype=np.int64), np.array([]))
