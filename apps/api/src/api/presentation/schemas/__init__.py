"""Response schemas.

These are the API's contract with the frontend. Kept separate from domain
entities so that renaming a domain concept is not automatically a breaking API
change, and so nothing framework-shaped can leak inward.
"""

from api.presentation.schemas.errors import ErrorDetail, ErrorResponse
from api.presentation.schemas.incident import IncidentSchema
from api.presentation.schemas.machine import (
    MachineDetailSchema,
    MachineSchema,
    MachineSummarySchema,
)
from api.presentation.schemas.prediction import PredictionSchema
from api.presentation.schemas.telemetry import (
    SensorReadingSchema,
    SeriesResolutionSchema,
    TelemetryPointSchema,
    TelemetryRecordSchema,
    TelemetrySeriesSchema,
)

__all__ = [
    "ErrorDetail",
    "ErrorResponse",
    "IncidentSchema",
    "MachineDetailSchema",
    "MachineSchema",
    "MachineSummarySchema",
    "PredictionSchema",
    "SensorReadingSchema",
    "SeriesResolutionSchema",
    "TelemetryPointSchema",
    "TelemetryRecordSchema",
    "TelemetrySeriesSchema",
]
