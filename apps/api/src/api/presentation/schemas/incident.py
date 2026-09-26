"""Incident response schema."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.incident_type import IncidentType
from api.domain.value_objects.risk_level import IncidentSeverity


class IncidentSchema(BaseModel):
    """A maintenance incident."""

    model_config = ConfigDict(frozen=True)

    incident_id: str
    machine_id: str
    incident_type: IncidentType = Field(
        description=(
            "Failure mode. Currently always UNCLASSIFIED: the predictive model "
            "detects rising failure risk but does not identify its cause."
        ),
    )
    severity: IncidentSeverity = Field(
        description="Severity at the time the incident was raised.",
    )
    failure_probability: float = Field(ge=0.0, le=1.0)
    detected_at: datetime
    status: IncidentStatus
    prediction_id: str | None = Field(
        default=None,
        description="The prediction that triggered this incident, when known.",
    )


class UpdateIncidentStatusRequest(BaseModel):
    """A requested lifecycle transition.

    The target is a status rather than an action, mirroring the domain: the
    permitted moves are a table over statuses, and expressing the request the
    same way keeps the router from holding a second copy of that table.

    `OPEN` is a valid value of this enum but never a valid target -- nothing
    transitions back to it. That is a 409 rather than a 422, because the request
    is well formed and the incident's current state is what refuses it.
    """

    model_config = ConfigDict(frozen=True)

    status: IncidentStatus
