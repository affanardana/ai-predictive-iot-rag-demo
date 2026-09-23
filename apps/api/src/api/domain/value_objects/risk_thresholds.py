"""Risk band thresholds.

Exposed as a value object rather than constants buried in the classifier so
that PRD section 9 stays literally true: these numbers are *product
configuration for the synthetic demonstration, not universal industrial
standards*. They are injected from settings at composition time, which makes
that claim structural rather than a comment.
"""

from __future__ import annotations

from dataclasses import dataclass

from api.domain.errors import DomainValidationError

DEFAULT_WARNING_THRESHOLD = 0.30
DEFAULT_HIGH_THRESHOLD = 0.60
DEFAULT_CRITICAL_THRESHOLD = 0.80


@dataclass(frozen=True, slots=True)
class RiskThresholds:
    """Lower bounds of the WARNING, HIGH, and CRITICAL risk bands.

    Each threshold is the *lower* bound of its band, so a probability is
    classified into the highest band whose threshold it reaches.
    """

    warning: float = DEFAULT_WARNING_THRESHOLD
    high: float = DEFAULT_HIGH_THRESHOLD
    critical: float = DEFAULT_CRITICAL_THRESHOLD

    def __post_init__(self) -> None:
        """Enforce that thresholds are ordered and within the unit interval."""
        for name, value in (
            ("warning", self.warning),
            ("high", self.high),
            ("critical", self.critical),
        ):
            if not 0.0 <= value <= 1.0:
                raise DomainValidationError(
                    f"Risk threshold '{name}' must be within 0.0-1.0, got {value}."
                )

        if not self.warning < self.high < self.critical:
            raise DomainValidationError(
                "Risk thresholds must satisfy warning < high < critical, got "
                f"{self.warning} < {self.high} < {self.critical}."
            )
