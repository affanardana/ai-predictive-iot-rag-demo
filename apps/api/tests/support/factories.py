"""Builders for domain objects.

Tests state only the fields they care about and let everything else default to
a valid, readable value. That keeps each test's intent visible instead of
buried in a dozen constructor arguments.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from api.domain.entities.incident import Incident
from api.domain.entities.machine import Machine
from api.domain.entities.prediction import (
    DEFAULT_PREDICTION_HORIZON,
    Prediction,
)
from api.domain.entities.simulation_run import SimulationRun
from api.domain.entities.telemetry import TelemetryRecord
from api.domain.value_objects.failure_probability import FailureProbability
from api.domain.value_objects.incident_type import IncidentType
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.risk_level import IncidentSeverity, RiskLevel
from api.domain.value_objects.run_status import RunStatus
from api.domain.value_objects.sensor_reading import SensorReading
from api.domain.value_objects.simulation_scenario import SimulationScenario

#: A fixed instant used across the suite so assertions read as constants.
#: Deliberately not "now": tests that depend on the real clock are the ones
#: that turn flaky at midnight.
DEFAULT_NOW = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)

DEFAULT_TEMPERATURE = 65.0
DEFAULT_VIBRATION = 1.4
DEFAULT_RPM = 1480.0
DEFAULT_CURRENT = 12.5
DEFAULT_LOAD = 0.72
DEFAULT_VOLTAGE = 400.0


def make_reading(
    temperature: float = DEFAULT_TEMPERATURE,
    vibration: float = DEFAULT_VIBRATION,
    rpm: float = DEFAULT_RPM,
    current: float = DEFAULT_CURRENT,
    load: float = DEFAULT_LOAD,
    voltage: float = DEFAULT_VOLTAGE,
) -> SensorReading:
    """Build a sensor reading with plausible defaults."""
    return SensorReading(
        temperature=temperature,
        vibration=vibration,
        rpm=rpm,
        current=current,
        load=load,
        voltage=voltage,
    )


def make_machine(
    machine_id: str = "M001",
    name: str | None = None,
    registered_at: datetime | None = None,
) -> Machine:
    """Build a machine."""
    return Machine(
        id=MachineId(machine_id),
        name=name if name is not None else f"Motor {machine_id}",
        registered_at=registered_at or DEFAULT_NOW,
    )


def make_telemetry(
    event_id: str = "evt-1",
    machine_id: str = "M001",
    recorded_at: datetime | None = None,
    reading: SensorReading | None = None,
    simulation_session_id: str | None = None,
) -> TelemetryRecord:
    """Build a telemetry record."""
    return TelemetryRecord(
        event_id=event_id,
        machine_id=MachineId(machine_id),
        recorded_at=recorded_at or DEFAULT_NOW,
        reading=reading or make_reading(),
        simulation_session_id=simulation_session_id,
    )


def make_prediction(
    machine_id: str = "M001",
    predicted_at: datetime | None = None,
    probability: float = 0.85,
    risk_level: RiskLevel = RiskLevel.CRITICAL,
    model_version: str = "lstm-v1",
    prediction_id: str = "pred-1",
    horizon: timedelta = DEFAULT_PREDICTION_HORIZON,
) -> Prediction:
    """Build a prediction."""
    return Prediction(
        machine_id=MachineId(machine_id),
        predicted_at=predicted_at or DEFAULT_NOW,
        probability=FailureProbability(probability),
        risk_level=risk_level,
        model_version=model_version,
        prediction_id=prediction_id,
        horizon=horizon,
    )


def make_simulation_run(
    session_id: str = "sim-bd-test",
    machine_id: str = "M001",
    scenario: SimulationScenario = SimulationScenario.BEARING_DEGRADATION,
    seed: int = 1,
    minutes: int = 120,
    created_at: datetime | None = None,
    started_at: datetime | None = None,
    status: RunStatus = RunStatus.PENDING,
    completed_ticks: int = 0,
) -> SimulationRun:
    """Build a simulation run.

    Defaults to 120 minutes rather than the ten `MASTERPLAN.md` uses as its
    example: at a one-minute sample interval, ten minutes is ten readings, which
    is below the sixty the model needs and so could never be scored. The
    default here is one a test can actually reason about.
    """
    return SimulationRun(
        session_id=session_id,
        machine_id=MachineId(machine_id),
        scenario=scenario,
        seed=seed,
        started_at=started_at or DEFAULT_NOW,
        sample_interval=timedelta(minutes=1),
        duration=timedelta(minutes=minutes),
        created_at=created_at or DEFAULT_NOW,
        status=status,
        completed_ticks=completed_ticks,
    )


def make_incident(
    machine_id: str = "M001",
    detected_at: datetime | None = None,
    incident_type: IncidentType = IncidentType.UNCLASSIFIED,
    severity: IncidentSeverity = IncidentSeverity.HIGH,
    probability: float = 0.85,
    incident_id: str = "inc-1",
    prediction_id: str | None = None,
) -> Incident:
    """Build an incident in the OPEN state."""
    return Incident(
        machine_id=MachineId(machine_id),
        incident_type=incident_type,
        severity=severity,
        probability=FailureProbability(probability),
        detected_at=detected_at or DEFAULT_NOW,
        incident_id=incident_id,
        prediction_id=prediction_id,
    )
