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
