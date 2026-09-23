"""Machine response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from api.domain.value_objects.risk_level import RiskLevel
from api.presentation.schemas.incident import IncidentSchema
from api.presentation.schemas.prediction import PredictionSchema
from api.presentation.schemas.telemetry import TelemetryRecordSchema


class MachineSchema(BaseModel):
    """A machine's registry identity."""

    model_config = ConfigDict(frozen=True)

    machine_id: str
    name: str
    registered_at: datetime


class MachineSummarySchema(BaseModel):
    """A machine with its latest observed and predicted condition.

    `latest_telemetry` and `latest_prediction` are independently nullable: a
    machine that is registered but silent has neither, and one that has stopped
    reporting keeps its last known values so staleness is visible rather than
    presenting as health.
    """

    model_config = ConfigDict(frozen=True)

    machine: MachineSchema
    latest_telemetry: TelemetryRecordSchema | None = None
    latest_prediction: PredictionSchema | None = None
    risk_level: RiskLevel | None = Field(
        default=None,
        description="Risk band from the latest prediction; null if none exists yet.",
    )
    open_incident_count: int = Field(ge=0)
    is_reporting: bool = Field(
        description="Whether any telemetry has ever been received for this machine.",
    )


class MachineDetailSchema(BaseModel):
    """Consolidated view of one machine."""

    model_config = ConfigDict(frozen=True)

    summary: MachineSummarySchema
    recent_incidents: list[IncidentSchema]
