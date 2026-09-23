"""Maps a failure probability onto an application risk level."""

from __future__ import annotations

from dataclasses import dataclass

from api.domain.value_objects.failure_probability import FailureProbability
from api.domain.value_objects.risk_level import RiskLevel
from api.domain.value_objects.risk_thresholds import RiskThresholds


@dataclass(frozen=True, slots=True)
class RiskLevelClassifier:
    """Classifies a failure probability into a configured risk band.

    Bands are inclusive of their lower bound, matching the PRD table:

        0.00 - 0.29  NORMAL
        0.30 - 0.59  WARNING
        0.60 - 0.79  HIGH
        0.80 - 1.00  CRITICAL

    So 0.30 is WARNING, not NORMAL. The boundaries are the interesting cases
    and are covered exhaustively by the domain tests.
    """

    thresholds: RiskThresholds

    def classify(self, probability: FailureProbability) -> RiskLevel:
        """Return the risk band the probability falls into."""
        value = probability.value
        if value >= self.thresholds.critical:
            return RiskLevel.CRITICAL
        if value >= self.thresholds.high:
            return RiskLevel.HIGH
        if value >= self.thresholds.warning:
            return RiskLevel.WARNING
        return RiskLevel.NORMAL
