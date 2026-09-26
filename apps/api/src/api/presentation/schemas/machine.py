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


class RegisterMachineRequest(BaseModel):
    """A machine to add to the registry.

    The length bounds mirror the columns. `Machine.__post_init__` validates the
    timestamp but not the name -- only `rename` does -- so an empty name would
    otherwise reach the database and persist.
    """

    machine_id: str = Field(
        min_length=1,
        max_length=16,
        description="Fleet-unique identifier, as published in telemetry.",
    )
    name: str = Field(min_length=1, max_length=128, description="Human-readable label.")


class MachineSummarySchema(BaseModel):
    """A machine with its latest observed and predicted condition.

    `latest_telemetry` and `latest_prediction` are independently nullable: a
    machine that is registered but silent has neither, and one that has stopped
    reporting keeps its last known values so staleness is visible rather than
    presenting as health.

    **Nullable but not optional**, and the distinction reaches the frontend.
    Pydantic gives a field with a default a `null` type *and* removes it from
    OpenAPI's `required` list, so a generated client sees `T | null | undefined`
    and a check for `null` alone leaves `undefined` in the else-branch -- which
    compiles and then throws on the first machine that has not been scored. No
    route uses `exclude_none`, so these are always sent; declaring them required
    makes the generated types say what the wire already does.
    """

    model_config = ConfigDict(frozen=True)

    machine: MachineSchema
    latest_telemetry: TelemetryRecordSchema | None
    latest_prediction: PredictionSchema | None
    risk_level: RiskLevel | None = Field(
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
