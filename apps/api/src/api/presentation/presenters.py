"""Domain-to-schema translation.

Kept out of the routers so route handlers stay thin, and out of the schemas so
the schemas describe shape rather than know about domain objects. Every
function here is total and side-effect free.
"""

from __future__ import annotations

from collections.abc import Sequence

from api.application.read_models import MachineDetail, MachineSummary
from api.domain.entities.incident import Incident
from api.domain.entities.machine import Machine
from api.domain.entities.prediction import Prediction
from api.domain.entities.telemetry import TelemetryRecord
from api.domain.read_models import TelemetrySeries
from api.domain.value_objects.sensor_reading import SensorReading
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


def to_sensor_reading(reading: SensorReading) -> SensorReadingSchema:
    """Translate a sensor reading."""
    return SensorReadingSchema(
        temperature=reading.temperature,
        vibration=reading.vibration,
        rpm=reading.rpm,
        current=reading.current,
        load=reading.load,
        voltage=reading.voltage,
    )


def to_telemetry_record(record: TelemetryRecord) -> TelemetryRecordSchema:
    """Translate a stored telemetry record."""
    return TelemetryRecordSchema(
        event_id=record.event_id,
        machine_id=record.machine_id.value,
        recorded_at=record.recorded_at,
        reading=to_sensor_reading(record.reading),
    )


def to_prediction(prediction: Prediction) -> PredictionSchema:
    """Translate a prediction."""
    return PredictionSchema(
        prediction_id=prediction.prediction_id,
        machine_id=prediction.machine_id.value,
        predicted_at=prediction.predicted_at,
        failure_probability=prediction.probability.value,
        risk_level=prediction.risk_level,
        horizon_seconds=int(prediction.horizon.total_seconds()),
        model_version=prediction.model_version,
    )


def to_incident(incident: Incident) -> IncidentSchema:
    """Translate an incident."""
    return IncidentSchema(
        incident_id=incident.incident_id,
        machine_id=incident.machine_id.value,
        incident_type=incident.incident_type,
        severity=incident.severity,
        failure_probability=incident.probability.value,
        detected_at=incident.detected_at,
        status=incident.status,
        prediction_id=incident.prediction_id,
    )


def to_machine(machine: Machine) -> MachineSchema:
    """Translate a machine's registry identity."""
    return MachineSchema(
        machine_id=machine.id.value,
        name=machine.name,
        registered_at=machine.registered_at,
    )


def to_machine_summary(summary: MachineSummary) -> MachineSummarySchema:
    """Translate an assembled machine summary."""
    return MachineSummarySchema(
        machine=to_machine(summary.machine),
        latest_telemetry=(
            to_telemetry_record(summary.latest_reading)
            if summary.latest_reading is not None
            else None
        ),
        latest_prediction=(
            to_prediction(summary.latest_prediction)
            if summary.latest_prediction is not None
            else None
        ),
        risk_level=summary.risk_level,
        open_incident_count=summary.open_incident_count,
        is_reporting=summary.is_reporting,
    )


def to_machine_detail(detail: MachineDetail) -> MachineDetailSchema:
    """Translate a consolidated machine view."""
    return MachineDetailSchema(
        summary=to_machine_summary(detail.summary),
        recent_incidents=[to_incident(incident) for incident in detail.recent_incidents],
    )


def to_telemetry_series(series: TelemetrySeries) -> TelemetrySeriesSchema:
    """Translate a telemetry series."""
    bucket = series.resolution.bucket
    return TelemetrySeriesSchema(
        machine_id=series.machine_id.value,
        window=series.window,
        resolution=SeriesResolutionSchema(
            is_raw=series.resolution.is_raw,
            bucket_seconds=int(bucket.total_seconds()) if bucket is not None else None,
            aggregation=series.resolution.aggregation,
        ),
        points=[
            TelemetryPointSchema(
                timestamp=point.timestamp,
                reading=to_sensor_reading(point.reading),
                sample_count=point.sample_count,
            )
            for point in series.points
        ],
    )


def to_incident_list(incidents: Sequence[Incident]) -> list[IncidentSchema]:
    """Translate a sequence of incidents."""
    return [to_incident(incident) for incident in incidents]
