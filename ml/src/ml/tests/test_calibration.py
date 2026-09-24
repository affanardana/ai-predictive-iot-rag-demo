"""Calibration and the prior shift: whether a probability means anything."""

from __future__ import annotations

import numpy as np
import pytest

from ml.evaluation import curves, prior
from ml.evaluation.errors import EvaluationError


def test_wilson_interval_stays_inside_the_unit_range() -> None:
    """The normal approximation does not, and that is why it is not used.

    A bin with no positives is common in the high-score region, and a symmetric
    interval around zero would report a negative probability.
    """
    low, high = curves.wilson_interval(0, 10)

    assert low == 0.0
    assert 0.0 < high < 1.0


def test_wilson_interval_contains_the_estimate() -> None:
    low, high = curves.wilson_interval(40, 200)

    assert low < 0.2 < high
    assert low < high


def test_wilson_interval_of_nothing_is_the_whole_range() -> None:
    """No observations means no information, which is not the same as zero."""
    assert curves.wilson_interval(0, 0) == (0.0, 1.0)


def test_brier_score_of_a_perfect_predictor_is_zero() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.int64)

    assert curves.brier_score(labels, labels.astype(np.float64)) == 0.0


def test_brier_score_of_a_confident_loser_is_one() -> None:
    labels = np.array([0, 1], dtype=np.int64)

    assert curves.brier_score(labels, np.array([1.0, 0.0])) == 1.0


def test_reliability_error_is_the_weighted_gap() -> None:
    """Hand-computed: two bins, each two rows, gaps 0.15 and 0.20."""
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.7, 0.9])
    result = curves.reliability(labels, scores, bins=2)

    assert len(result.bins) == 2
    assert result.expected_calibration_error == pytest.approx(0.175)
    assert result.maximum_calibration_error == pytest.approx(0.20)


def test_quantile_bins_are_not_uniform_bins_at_this_prevalence() -> None:
    """Uniform bins would put almost everything in the first bin.

    That is the whole reason quantile spacing is the default: the interesting
    miscalibration is in the top of the range, and uniform edges leave that
    region holding a handful of rows.
    """
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.uniform(0, 0.05, 9_500), rng.uniform(0.05, 1.0, 500)])
    labels = (rng.random(10_000) < 0.05).astype(np.int64)

    quantile = curves.reliability(labels, scores, bins=5, strategy="quantile")
    uniform = curves.reliability(labels, scores, bins=5, strategy="uniform")

    assert len(quantile.bins) == 5
    assert min(bin.count for bin in quantile.bins) > min(bin.count for bin in uniform.bins)


def test_high_risk_slice_is_restricted_to_the_top_rows() -> None:
    rng = np.random.default_rng(4)
    labels = (rng.random(1_000) < 0.1).astype(np.int64)
    scores = rng.random(1_000)

    top = curves.reliability_from_slice(labels, scores, share=0.05)

    assert top.rows == 50
    assert top.rows < 1_000


def test_high_risk_slice_rejects_an_impossible_share() -> None:
    with pytest.raises(EvaluationError, match="share"):
        curves.reliability_from_slice(
            np.array([0, 1], dtype=np.int64), np.array([0.1, 0.9]), share=0.0
        )


def test_risk_bands_cover_every_row_exactly_once() -> None:
    """PRD §9's bands are the product's interface; a gap would lose machines."""
    rng = np.random.default_rng(6)
    labels = (rng.random(500) < 0.2).astype(np.int64)
    scores = rng.random(500)

    bands = curves.risk_bands(labels, scores, edges=(0.30, 0.60, 0.80))

    assert len(bands) == 4
    assert sum(band["rows"] for band in bands) == 500


def test_the_prior_offset_is_the_log_odds_difference() -> None:
    shift = prior.PriorShift(trained_rate=0.20, natural_rate=0.045)

    expected = np.log(0.045 / 0.955) - np.log(0.20 / 0.80)

    assert shift.offset == pytest.approx(float(expected))


def test_the_prior_correction_moves_a_score_to_the_natural_rate() -> None:
    """A model trained at 20% that says 0.20 means 4.5% in the real world."""
    shift = prior.PriorShift(trained_rate=0.20, natural_rate=0.045)

    corrected = shift.apply(np.array([0.20]))

    assert corrected[0] == pytest.approx(0.045, abs=1e-6)


def test_the_prior_correction_preserves_the_ranking() -> None:
    """It is a calibration fix and never a performance one.

    A constant shift in log-odds cannot reorder anything, and a test that
    asserted otherwise would be asserting a bug.
    """
    shift = prior.PriorShift(trained_rate=0.20, natural_rate=0.045)
    scores = np.array([0.05, 0.2, 0.5, 0.9])

    corrected = shift.apply(scores)

    assert list(np.argsort(corrected)) == list(np.argsort(scores))


def test_a_prior_shift_that_is_not_a_shift_changes_nothing() -> None:
    shift = prior.PriorShift(trained_rate=0.05, natural_rate=0.05)
    scores = np.array([0.1, 0.5, 0.9])

    assert shift.apply(scores) == pytest.approx(scores)


def test_logit_clips_rather_than_returning_infinity() -> None:
    """An exactly-0 or exactly-1 prediction is over-confident, not correct.

    Returning an infinity would propagate through every downstream figure
    instead of showing up as the one bad row it is.
    """
    result = prior.logit(np.array([0.0, 1.0]))

    assert np.isfinite(result).all()


def test_impossible_rates_are_refused() -> None:
    with pytest.raises(EvaluationError, match="natural_rate"):
        prior.PriorShift(trained_rate=0.2, natural_rate=0.0)
    with pytest.raises(EvaluationError, match="trained_rate"):
        prior.PriorShift(trained_rate=1.0, natural_rate=0.1)
