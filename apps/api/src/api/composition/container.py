"""Dependency wiring.

This is the one place that names concrete adapters. Every seam the later
phases add -- a PyTorch predictor, an MQTT consumer, a pgvector retriever, an
LLM provider -- plugs in here by swapping or adding a constructor argument,
with no application or presentation file changing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine

from api.application.use_cases import (
    GetMachineDetail,
    GetPredictionHistory,
    GetTelemetryHistory,
    IngestTelemetry,
    ListIncidents,
    ListMachines,
    RecordPrediction,
    RegisterMachine,
)
from api.domain.entities.prediction import PREDICTION_WINDOW_READINGS
from api.domain.errors import PredictionUnavailableError
from api.domain.ports.clock import Clock
from api.domain.ports.health import HealthProbe, HealthStatus
from api.domain.ports.predictor import ModelOutput, Predictor
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.services.incident_policy import DefaultIncidentPolicy
from api.domain.services.risk_level_classifier import RiskLevelClassifier
from api.domain.value_objects.risk_thresholds import RiskThresholds
from api.domain.value_objects.sensor_reading import SensorReading
from api.infrastructure.config import Settings
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from api.infrastructure.persistence.sql.session import create_database_engine
from api.infrastructure.persistence.sql.unit_of_work import SqlUnitOfWorkFactory
from api.infrastructure.prediction import HttpPredictor
from api.infrastructure.system.database_health_probe import DatabaseHealthProbe
from api.infrastructure.system.system_clock import SystemClock


@dataclass(frozen=True, slots=True)
class Container:
    """Every dependency the presentation layer needs, already wired.

    Use cases are constructed once here rather than resolved per request. They
    hold no per-request state -- the unit of work supplies that inside each
    call -- so sharing one instance across requests is safe.
    """

    settings: Settings
    unit_of_work_factory: UnitOfWorkFactory
    clock: Clock
    health_probe: HealthProbe

    list_machines: ListMachines
    get_machine_detail: GetMachineDetail
    get_telemetry_history: GetTelemetryHistory
    get_prediction_history: GetPredictionHistory
    list_incidents: ListIncidents
    record_prediction: RecordPrediction
    ingest_telemetry: IngestTelemetry
    register_machine: RegisterMachine

    #: Present only when a database engine was created; the in-memory wiring
    #: has nothing to dispose.
    engine: AsyncEngine | None = None

    async def aclose(self) -> None:
        """Release resources held by the container.

        Called from the application's lifespan shutdown so the connection pool
        is drained rather than abandoned.
        """
        if self.engine is not None:
            await self.engine.dispose()


def build_container(settings: Settings) -> Container:
    """Wire the application against the operational database.

    No connection is opened here. The engine is lazy, so building the container
    succeeds even when the database is briefly unreachable -- which is what lets
    the readiness endpoint report an outage instead of the process refusing to
    start.
    """
    engine = create_database_engine(settings.database_url)
    unit_of_work_factory = SqlUnitOfWorkFactory(engine)

    return _assemble(
        settings=settings,
        unit_of_work_factory=unit_of_work_factory,
        health_probe=DatabaseHealthProbe(unit_of_work_factory.session_factory),
        predictor=HttpPredictor(
            base_url=settings.inference_service_url, client=httpx.AsyncClient()
        ),
        engine=engine,
        clock=SystemClock(),
    )


def build_in_memory_container(
    settings: Settings,
    unit_of_work_factory: InMemoryUnitOfWorkFactory | None = None,
    clock: Clock | None = None,
    health_probe: HealthProbe | None = None,
    predictor: Predictor | None = None,
) -> Container:
    """Wire the application against an in-memory store.

    Used by the test suite. It lives in the composition root rather than in the
    tests so the wiring itself is exercised in one place: a typo in a
    constructor argument surfaces whichever backend is selected.

    The optional arguments let a test share one store across the container and
    its own assertions, and pin time. Everything else about the wiring is
    identical to production.
    """
    return _assemble(
        settings=settings,
        unit_of_work_factory=unit_of_work_factory or InMemoryUnitOfWorkFactory(),
        health_probe=health_probe or AlwaysHealthyProbe(),
        predictor=predictor or UnavailablePredictor(),
        engine=None,
        clock=clock or SystemClock(),
    )


class AlwaysHealthyProbe:
    """Reports healthy unconditionally, for the in-memory wiring."""

    async def check(self) -> HealthStatus:
        """Return a healthy status."""
        return HealthStatus.up("In-memory store; no external dependency to check.")


def _assemble(
    settings: Settings,
    unit_of_work_factory: UnitOfWorkFactory,
    health_probe: HealthProbe,
    predictor: Predictor,
    engine: AsyncEngine | None,
    clock: Clock,
) -> Container:
    """Build the container from already-chosen adapters."""
    return Container(
        settings=settings,
        unit_of_work_factory=unit_of_work_factory,
        clock=clock,
        health_probe=health_probe,
        list_machines=ListMachines(unit_of_work_factory=unit_of_work_factory),
        get_machine_detail=GetMachineDetail(unit_of_work_factory=unit_of_work_factory),
        get_telemetry_history=GetTelemetryHistory(
            unit_of_work_factory=unit_of_work_factory,
            clock=clock,
        ),
        get_prediction_history=GetPredictionHistory(unit_of_work_factory=unit_of_work_factory),
        list_incidents=ListIncidents(unit_of_work_factory=unit_of_work_factory),
        record_prediction=RecordPrediction(
            unit_of_work_factory=unit_of_work_factory,
            predictor=predictor,
            clock=clock,
            classifier=RiskLevelClassifier(
                thresholds=RiskThresholds(
                    warning=settings.risk_warning_threshold,
                    high=settings.risk_high_threshold,
                    critical=settings.risk_critical_threshold,
                )
            ),
            incident_policy=DefaultIncidentPolicy(),
        ),
        ingest_telemetry=IngestTelemetry(
            unit_of_work_factory=unit_of_work_factory,
            window_readings=PREDICTION_WINDOW_READINGS,
        ),
        register_machine=RegisterMachine(unit_of_work_factory=unit_of_work_factory, clock=clock),
        engine=engine,
    )


class UnavailablePredictor:
    """A predictor that always refuses, for wiring with no model behind it.

    The in-memory container is used by tests that never call inference, and by
    a deployment that has not been pointed at a service yet. Failing loudly here
    is better than a placeholder that returns a fixed number: a fabricated
    probability would be persisted as a prediction.
    """

    async def predict(self, readings: Sequence[SensorReading]) -> ModelOutput:
        """Refuse, whatever was asked.

        Raises:
            PredictionUnavailableError: always.
        """
        del readings
        raise PredictionUnavailableError(
            "No inference service is configured. Set INFERENCE_SERVICE_URL."
        )
