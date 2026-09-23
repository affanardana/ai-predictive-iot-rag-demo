"""Risk band classification."""

from __future__ import annotations

import pytest

from api.domain.services.risk_level_classifier import RiskLevelClassifier
from api.domain.value_objects.failure_probability import FailureProbability
from api.domain.value_objects.risk_level import RiskLevel
from api.domain.value_objects.risk_thresholds import RiskThresholds


def _classify(probability: float, thresholds: RiskThresholds | None = None) -> RiskLevel:
    classifier = RiskLevelClassifier(thresholds=thresholds or RiskThresholds())
    return classifier.classify(FailureProbability(probability))


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        # Exactly the PRD section 9 table, including both edges of every band.
        (0.00, RiskLevel.NORMAL),
        (0.29, RiskLevel.NORMAL),
        # Each threshold is the *lower* bound of its band, so 0.30 is WARNING
        # rather than NORMAL. These four values are the ones an off-by-one
        # would silently flip.
        (0.30, RiskLevel.WARNING),
        (0.59, RiskLevel.WARNING),
        (0.60, RiskLevel.HIGH),
        (0.79, RiskLevel.HIGH),
        (0.80, RiskLevel.CRITICAL),
        (1.00, RiskLevel.CRITICAL),
    ],
)
def test_classifies_exactly_at_prd_boundaries(probability: float, expected: RiskLevel) -> None:
    """Every band edge maps to the band the PRD specifies."""
    assert _classify(probability) == expected


@pytest.mark.parametrize("probability", [0.299999, 0.599999, 0.799999])
def test_values_just_below_a_threshold_stay_in_the_lower_band(probability: float) -> None:
    """A hair under a threshold is not promoted into the next band."""
    expected = {
        0.299999: RiskLevel.NORMAL,
        0.599999: RiskLevel.WARNING,
        0.799999: RiskLevel.HIGH,
    }[probability]
    assert _classify(probability) == expected


def test_classification_is_monotonic() -> None:
    """Risk never decreases as probability rises.

    Swept densely rather than sampled, because a classifier built from
    independent `if` statements can be locally correct at each threshold and
    still be non-monotonic in between.
    """
    ranks = {RiskLevel.NORMAL: 0, RiskLevel.WARNING: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}
    previous = -1
    for step in range(1001):
        rank = ranks[_classify(step / 1000)]
        assert rank >= previous, f"risk decreased at probability {step / 1000}"
        previous = rank


def test_uses_configured_thresholds_not_hardcoded_values() -> None:
    """The bands follow injected configuration.

    PRD section 9 is explicit that these numbers are product configuration
    rather than universal standards, so a classifier with different thresholds
    must classify differently.
    """
    lenient = RiskThresholds(warning=0.10, high=0.20, critical=0.30)

    assert _classify(0.30, lenient) == RiskLevel.CRITICAL
    assert _classify(0.30) == RiskLevel.WARNING
