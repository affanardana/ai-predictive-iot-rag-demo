"""Risk threshold configuration."""

from __future__ import annotations

import pytest

from api.domain.errors import DomainValidationError
from api.domain.value_objects.risk_thresholds import (
    DEFAULT_CRITICAL_THRESHOLD,
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_WARNING_THRESHOLD,
    RiskThresholds,
)


def test_defaults_match_the_prd_table() -> None:
    """The shipped defaults are the PRD section 9 boundaries."""
    thresholds = RiskThresholds()

    assert thresholds.warning == DEFAULT_WARNING_THRESHOLD == 0.30
    assert thresholds.high == DEFAULT_HIGH_THRESHOLD == 0.60
    assert thresholds.critical == DEFAULT_CRITICAL_THRESHOLD == 0.80


@pytest.mark.parametrize(
    ("warning", "high", "critical"),
    [
        (0.60, 0.30, 0.80),  # warning above high
        (0.30, 0.80, 0.60),  # high above critical
        (0.30, 0.30, 0.80),  # warning equals high
        (0.30, 0.80, 0.80),  # high equals critical
    ],
)
def test_rejects_unordered_thresholds(warning: float, high: float, critical: float) -> None:
    """Thresholds must increase strictly.

    An equal pair would make one band unreachable, which looks like a
    classifier bug rather than a configuration mistake.
    """
    with pytest.raises(DomainValidationError):
        RiskThresholds(warning=warning, high=high, critical=critical)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_rejects_thresholds_outside_the_unit_interval(value: float) -> None:
    """Thresholds are probabilities, so they live in [0, 1]."""
    with pytest.raises(DomainValidationError):
        RiskThresholds(warning=value)

    with pytest.raises(DomainValidationError):
        RiskThresholds(critical=value)


def test_accepts_a_valid_custom_configuration() -> None:
    """A well-formed alternative configuration is allowed."""
    thresholds = RiskThresholds(warning=0.10, high=0.50, critical=0.90)

    assert thresholds.warning == 0.10
    assert thresholds.critical == 0.90
