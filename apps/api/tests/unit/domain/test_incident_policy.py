"""Incident creation policy."""

from __future__ import annotations

import pytest

from api.domain.services.incident_policy import DefaultIncidentPolicy
from api.domain.value_objects.incident_type import IncidentType
from api.domain.value_objects.risk_level import IncidentSeverity, RiskLevel
from tests.support.factories import make_reading

POLICY = DefaultIncidentPolicy()


@pytest.mark.parametrize(
    ("risk_level", "expected"),
    [
        (RiskLevel.NORMAL, False),
        # WARNING is deliberately excluded. A warning is the system saying
        # "watch this"; raising an incident for every one trains operators to
        # ignore incidents.
        (RiskLevel.WARNING, False),
        (RiskLevel.HIGH, True),
        (RiskLevel.CRITICAL, True),
    ],
)
def test_raises_incidents_only_at_high_or_critical(risk_level: RiskLevel, expected: bool) -> None:
    """Only HIGH and CRITICAL risk produce an incident."""
    assert POLICY.should_raise(risk_level) is expected


@pytest.mark.parametrize(
    ("risk_level", "expected"),
    [
        (RiskLevel.NORMAL, IncidentSeverity.LOW),
        (RiskLevel.WARNING, IncidentSeverity.MEDIUM),
        (RiskLevel.HIGH, IncidentSeverity.HIGH),
        (RiskLevel.CRITICAL, IncidentSeverity.CRITICAL),
    ],
)
def test_severity_tracks_the_risk_band(risk_level: RiskLevel, expected: IncidentSeverity) -> None:
    """A newly raised incident inherits the risk band as its severity."""
    assert POLICY.severity_for(risk_level) is expected


def test_every_risk_level_has_a_severity() -> None:
    """The mapping is total, so no risk level can raise a KeyError."""
    for risk_level in RiskLevel:
        assert POLICY.severity_for(risk_level) in IncidentSeverity


@pytest.mark.parametrize("reading", [None, make_reading(), make_reading(vibration=9.9)])
def test_type_is_always_unclassified(reading: object) -> None:
    """Failure-mode classification is a documented stub, not a guess.

    The predictive model is a binary 60-minute failure classifier: it answers
    "will this machine fail soon?", not "what is failing?". Returning a
    plausible-looking type would be inventing information, so every incident is
    UNCLASSIFIED until that question is settled.
    """
    assert POLICY.type_for(reading) is IncidentType.UNCLASSIFIED  # type: ignore[arg-type]
