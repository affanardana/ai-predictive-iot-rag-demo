"""Decides what kind of incident, if any, a prediction should raise."""

from __future__ import annotations

from typing import Protocol

from api.domain.value_objects.incident_type import IncidentType
from api.domain.value_objects.risk_level import IncidentSeverity, RiskLevel
from api.domain.value_objects.sensor_reading import SensorReading

#: Risk levels at or above this one produce an incident. WARNING is
#: deliberately excluded: a warning is the system saying "watch this", and
#: raising an incident for every warning would train operators to ignore them.
INCIDENT_RAISING_RISK_LEVELS = frozenset({RiskLevel.HIGH, RiskLevel.CRITICAL})

#: Severity follows risk for a newly raised incident. They diverge afterwards:
#: acknowledging or resolving an incident changes its status but not the
#: severity it was raised with.
_SEVERITY_BY_RISK: dict[RiskLevel, IncidentSeverity] = {
    RiskLevel.NORMAL: IncidentSeverity.LOW,
    RiskLevel.WARNING: IncidentSeverity.MEDIUM,
    RiskLevel.HIGH: IncidentSeverity.HIGH,
    RiskLevel.CRITICAL: IncidentSeverity.CRITICAL,
}


class IncidentPolicy(Protocol):
    """Decides incident creation and classification from a prediction."""

    def should_raise(self, risk_level: RiskLevel) -> bool:
        """Whether this risk level warrants a new incident."""
        ...

    def severity_for(self, risk_level: RiskLevel) -> IncidentSeverity:
        """The severity a newly raised incident should carry."""
        ...

    def type_for(self, reading: SensorReading | None) -> IncidentType:
        """Classify the failure mode implied by the machine's signals."""
        ...


class DefaultIncidentPolicy:
    """Phase 1 incident policy.

    `type_for` always returns `UNCLASSIFIED`. This is a deliberate stub, not
    an oversight -- the predictive model is a binary failure classifier and
    cannot say *what* is failing. See `domain.value_objects.incident_type`
    for the three ways that could be resolved later.
    """

    def should_raise(self, risk_level: RiskLevel) -> bool:
        """Raise an incident at HIGH or CRITICAL risk."""
        return risk_level in INCIDENT_RAISING_RISK_LEVELS

    def severity_for(self, risk_level: RiskLevel) -> IncidentSeverity:
        """Map the risk band onto an initial severity."""
        return _SEVERITY_BY_RISK[risk_level]

    def type_for(self, reading: SensorReading | None) -> IncidentType:
        """Return `UNCLASSIFIED`; failure-mode classification is unresolved."""
        return IncidentType.UNCLASSIFIED
