"""Mappers between domain entities and persistence models.

This module is the only place that knows both shapes. Mappers live in
infrastructure because they necessarily import SQLAlchemy; the domain stays
unaware that a relational representation exists.

Every mapping is explicit rather than reflective or automated, so adding a
field to either side is a visible edit here instead of a silent behavioural
change.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from api.domain.entities.incident import Incident
from api.domain.entities.machine import Machine
from api.domain.entities.prediction import Prediction
from api.domain.entities.simulation_run import SimulationRun
from api.domain.entities.telemetry import TelemetryRecord
from api.domain.value_objects.failure_probability import FailureProbability
from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.incident_type import IncidentType
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.risk_level import IncidentSeverity, RiskLevel
from api.domain.value_objects.run_status import RunStatus
from api.domain.value_objects.sensor_reading import SensorReading
from api.domain.value_objects.simulation_scenario import SimulationScenario
from api.infrastructure.persistence.sql.models import (
    IncidentModel,
    MachineModel,
    PredictionModel,
    SimulationRunModel,
    TelemetryModel,
)


def as_aware(value: datetime) -> datetime:
    """Return an aware UTC datetime, attaching UTC to a naive value.

    PostgreSQL `timestamptz` columns round-trip the offset, but SQLite does not
    persist one, so the same row reads back naive under SQLite. Everything this
    system writes is UTC, so a naive value can only have come from a store that
    dropped the offset -- interpreting it as UTC is therefore correct rather
    than a guess. The domain rejects naive datetimes outright, so without this
    the SQLite test tier could not exercise the domain at all.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=UTC)
    return value


# ---------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------


def machine_to_model(machine: Machine) -> MachineModel:
    """Build a persistence model from a machine entity."""
    return MachineModel(
        machine_id=machine.id.value,
        name=machine.name,
        registered_at=machine.registered_at,
    )


def machine_from_model(model: MachineModel) -> Machine:
    """Build a machine entity from a persistence model."""
    return Machine(
        id=MachineId(model.machine_id),
        name=model.name,
        registered_at=as_aware(model.registered_at),
    )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def telemetry_to_model(record: TelemetryRecord) -> TelemetryModel:
    """Build a persistence model from a telemetry record."""
    return TelemetryModel(
        event_id=record.event_id,
        machine_id=record.machine_id.value,
        recorded_at=record.recorded_at,
        temperature=record.reading.temperature,
        vibration=record.reading.vibration,
        rpm=record.reading.rpm,
        current=record.reading.current,
        load=record.reading.load,
        voltage=record.reading.voltage,
        simulation_session_id=record.simulation_session_id,
    )


def telemetry_from_model(model: TelemetryModel) -> TelemetryRecord:
    """Build a telemetry record from a persistence model."""
    return TelemetryRecord(
        event_id=model.event_id,
        machine_id=MachineId(model.machine_id),
        recorded_at=as_aware(model.recorded_at),
        reading=SensorReading(
            temperature=model.temperature,
            vibration=model.vibration,
            rpm=model.rpm,
            current=model.current,
            load=model.load,
            voltage=model.voltage,
        ),
        simulation_session_id=model.simulation_session_id,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def prediction_to_model(prediction: Prediction) -> PredictionModel:
    """Build a persistence model from a prediction entity."""
    return PredictionModel(
        prediction_id=prediction.prediction_id,
        machine_id=prediction.machine_id.value,
        predicted_at=prediction.predicted_at,
        failure_probability=prediction.probability.value,
        risk_level=prediction.risk_level.value,
        horizon_seconds=int(prediction.horizon.total_seconds()),
        model_version=prediction.model_version,
    )


def prediction_from_model(model: PredictionModel) -> Prediction:
    """Build a prediction entity from a persistence model."""
    return Prediction(
        machine_id=MachineId(model.machine_id),
        predicted_at=as_aware(model.predicted_at),
        probability=FailureProbability(model.failure_probability),
        risk_level=RiskLevel(model.risk_level),
        model_version=model.model_version,
        prediction_id=model.prediction_id,
        horizon=timedelta(seconds=model.horizon_seconds),
    )


# ---------------------------------------------------------------------------
# Incident
# ---------------------------------------------------------------------------


def incident_to_model(incident: Incident) -> IncidentModel:
    """Build a persistence model from an incident entity."""
    return IncidentModel(
        incident_id=incident.incident_id,
        machine_id=incident.machine_id.value,
        incident_type=incident.incident_type.value,
        severity=incident.severity.value,
        failure_probability=incident.probability.value,
        detected_at=incident.detected_at,
        status=incident.status.value,
        prediction_id=incident.prediction_id,
    )


def incident_from_model(model: IncidentModel) -> Incident:
    """Build an incident entity from a persistence model."""
    return Incident(
        machine_id=MachineId(model.machine_id),
        incident_type=IncidentType(model.incident_type),
        severity=IncidentSeverity(model.severity),
        probability=FailureProbability(model.failure_probability),
        detected_at=as_aware(model.detected_at),
        incident_id=model.incident_id,
        status=IncidentStatus(model.status),
        prediction_id=model.prediction_id,
    )


def simulation_run_to_model(run: SimulationRun) -> SimulationRunModel:
    """Build a persistence model from a simulation run entity.

    Durations are stored as whole seconds, which is what makes the tick count
    recoverable: `duration_seconds // sample_interval_seconds` must equal
    `int(duration / sample_interval)`, and both are exact for the whole-second
    values the API accepts.
    """
    return SimulationRunModel(
        session_id=run.session_id,
        machine_id=run.machine_id.value,
        scenario=run.scenario.value,
        seed=run.seed,
        started_at=run.started_at,
        sample_interval_seconds=int(run.sample_interval.total_seconds()),
        duration_seconds=int(run.duration.total_seconds()),
        tick_seconds=run.tick_seconds,
        status=run.status.value,
        created_at=run.created_at,
        completed_ticks=run.completed_ticks,
        last_heartbeat_at=run.last_heartbeat_at,
        finished_at=run.finished_at,
        detail=run.detail,
        resume_count=run.resume_count,
    )


def simulation_run_from_model(model: SimulationRunModel) -> SimulationRun:
    """Build a simulation run entity from a persistence model."""
    return SimulationRun(
        session_id=model.session_id,
        machine_id=MachineId(model.machine_id),
        scenario=SimulationScenario(model.scenario),
        seed=model.seed,
        started_at=as_aware(model.started_at),
        sample_interval=timedelta(seconds=model.sample_interval_seconds),
        duration=timedelta(seconds=model.duration_seconds),
        tick_seconds=model.tick_seconds,
        status=RunStatus(model.status),
        created_at=as_aware(model.created_at),
        completed_ticks=model.completed_ticks,
        last_heartbeat_at=(
            as_aware(model.last_heartbeat_at) if model.last_heartbeat_at is not None else None
        ),
        finished_at=as_aware(model.finished_at) if model.finished_at is not None else None,
        detail=model.detail,
        resume_count=model.resume_count,
    )
