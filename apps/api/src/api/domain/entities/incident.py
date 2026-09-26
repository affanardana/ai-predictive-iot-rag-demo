"""Incident entity and its lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from api.domain.timestamps import ensure_aware
from api.domain.value_objects.failure_probability import FailureProbability
from api.domain.value_objects.incident_status import (
    OPEN_INCIDENT_STATUSES,
    IncidentStatus,
    ensure_transition_allowed,
)
from api.domain.value_objects.incident_type import IncidentType
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.risk_level import IncidentSeverity


@dataclass
class Incident:
    """A maintenance incident raised when predictive risk crosses a threshold.

    Status changes go through methods rather than direct assignment so the
    transition table in `incident_status` cannot be bypassed. PRD FR-006
    requires the lifecycle to be maintained, and an invalid transition is a
    domain error rather than a database problem.
    """

    machine_id: MachineId
    incident_type: IncidentType
    severity: IncidentSeverity
    probability: FailureProbability
    detected_at: datetime
    incident_id: str
    status: IncidentStatus = IncidentStatus.OPEN
    prediction_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the detection timestamp and identifier."""
        ensure_aware(self.detected_at, "detected_at")
        if not self.incident_id.strip():
            raise ValueError("Incident incident_id must not be blank.")

    @property
    def is_open(self) -> bool:
        """Whether the incident still requires attention."""
        # The set lives in `incident_status` because the fleet summary counts it
        # in SQL, and one rule with two implementations is one rule too many.
        return self.status in OPEN_INCIDENT_STATUSES

    def acknowledge(self) -> None:
        """Mark the incident as seen by an operator."""
        self._transition_to(IncidentStatus.ACKNOWLEDGED)

    def resolve(self) -> None:
        """Mark the incident as addressed."""
        self._transition_to(IncidentStatus.RESOLVED)

    def dismiss(self) -> None:
        """Close the incident without action, e.g. a false positive."""
        self._transition_to(IncidentStatus.DISMISSED)

    def transition_to(self, requested: IncidentStatus) -> None:
        """Move to `requested`, for a caller that already knows the target.

        The three named methods above read better at a call site that knows
        what it means to do. This one serves a caller that is handed a status
        -- an HTTP request naming one -- and needs the same guard without a
        dispatch table restating the lifecycle somewhere outside the domain.

        Note that the transition table permits no self-transition, so passing
        the status an incident already has raises. That is deliberate:
        re-acknowledging an acknowledged incident means someone else acted on
        it in the meantime, and reporting success would hide that.
        """
        self._transition_to(requested)

    def _transition_to(self, requested: IncidentStatus) -> None:
        """Apply a status change, raising if the transition is not permitted."""
        ensure_transition_allowed(self.status, requested)
        self.status = requested

    @classmethod
    def create(
        cls,
        machine_id: MachineId,
        incident_type: IncidentType,
        severity: IncidentSeverity,
        probability: FailureProbability,
        detected_at: datetime,
        prediction_id: str | None = None,
    ) -> Incident:
        """Build an incident in the OPEN state, assigning a fresh identifier."""
        return cls(
            machine_id=machine_id,
            incident_type=incident_type,
            severity=severity,
            probability=probability,
            detected_at=detected_at,
            incident_id=str(uuid4()),
            prediction_id=prediction_id,
        )
