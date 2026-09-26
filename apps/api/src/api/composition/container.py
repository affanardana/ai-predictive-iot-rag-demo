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
    IngestKnowledgeDocument,
    IngestTelemetry,
    ListIncidents,
    ListKnowledgeDocuments,
    ListMachines,
    ListSimulations,
    RecordPrediction,
    RegisterMachine,
    ReportSimulationState,
    ResetSimulation,
    SearchMaintenanceKnowledge,
    SetActiveDocumentVersion,
    StartSimulation,
    StopSimulation,
    UpdateIncidentStatus,
)
from api.domain.entities.prediction import PREDICTION_WINDOW_READINGS
from api.domain.errors import (
    PredictionUnavailableError,
    RetrievalUnavailableError,
    SimulationUnavailableError,
)
from api.domain.ports.clock import Clock
from api.domain.ports.embedder import Embedder
from api.domain.ports.events import EventSubscriber
from api.domain.ports.health import HealthProbe, HealthStatus
from api.domain.ports.predictor import ModelOutput, Predictor
from api.domain.ports.reranker import Reranker, RerankResult
from api.domain.ports.simulation import SimulationController, SimulationPlan
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.services.incident_policy import DefaultIncidentPolicy
from api.domain.services.risk_level_classifier import RiskLevelClassifier
from api.domain.value_objects.risk_thresholds import RiskThresholds
from api.domain.value_objects.sensor_reading import SensorReading
from api.infrastructure.config import Settings
from api.infrastructure.embedding import HttpEmbedder
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from api.infrastructure.persistence.sql.session import create_database_engine
from api.infrastructure.persistence.sql.unit_of_work import SqlUnitOfWorkFactory
from api.infrastructure.prediction import HttpPredictor
from api.infrastructure.realtime import InProcessEventBroadcaster
from api.infrastructure.reranking import HttpReranker
from api.infrastructure.simulation import HttpSimulationController
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
    #: The subscriber half of the event bus, for the streaming endpoint. Typed
    #: as the domain protocol rather than the concrete broadcaster so the
    #: presentation layer never names an adapter -- which is what the
    #: "presentation does not import infrastructure" contract requires.
    event_subscriber: EventSubscriber

    list_machines: ListMachines
    get_machine_detail: GetMachineDetail
    get_telemetry_history: GetTelemetryHistory
    get_prediction_history: GetPredictionHistory
    list_incidents: ListIncidents
    record_prediction: RecordPrediction
    ingest_telemetry: IngestTelemetry
    register_machine: RegisterMachine
    update_incident_status: UpdateIncidentStatus
    list_simulations: ListSimulations
    start_simulation: StartSimulation
    stop_simulation: StopSimulation
    reset_simulation: ResetSimulation
    report_simulation_state: ReportSimulationState
    list_knowledge_documents: ListKnowledgeDocuments
    ingest_knowledge_document: IngestKnowledgeDocument
    search_maintenance_knowledge: SearchMaintenanceKnowledge
    set_active_document_version: SetActiveDocumentVersion

    #: Held for its `aclose`, which shuts down the HTTP client the simulation
    #: controller borrows. Present on both wirings, so the type is not optional
    #: -- unlike the engine, which exists only when there is a database.
    simulation_controller: SimulationController

    #: Present only when a database engine was created; the in-memory wiring
    #: has nothing to dispose.
    engine: AsyncEngine | None = None
    #: Present only when a broadcaster was built, so `aclose` has something to
    #: stop the keepalive timer on.
    broadcaster: InProcessEventBroadcaster | None = None
    #: The client the retrieval adapters share. Held here rather than by either
    #: adapter so the two use one connection pool and it is closed once.
    inference_client: httpx.AsyncClient | None = None

    async def aclose(self) -> None:
        """Release resources held by the container.

        Called from the application's lifespan shutdown so the connection pool
        is drained and the broadcaster's keepalive timer is stopped rather than
        abandoned.
        """
        if self.broadcaster is not None:
            await self.broadcaster.aclose()
        await self.simulation_controller.aclose()
        if self.inference_client is not None:
            await self.inference_client.aclose()
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
    # One client for every call to the inference service. Three adapters share
    # the connection pool, and the container closes it once.
    inference_client = httpx.AsyncClient()

    return _assemble(
        settings=settings,
        unit_of_work_factory=unit_of_work_factory,
        health_probe=DatabaseHealthProbe(unit_of_work_factory.session_factory),
        predictor=HttpPredictor(base_url=settings.inference_service_url, client=inference_client),
        embedder=HttpEmbedder(
            base_url=settings.inference_service_url,
            client=inference_client,
            model_id=settings.knowledge_embedding_model,
        ),
        reranker=HttpReranker(base_url=settings.inference_service_url, client=inference_client),
        engine=engine,
        clock=SystemClock(),
        simulation_controller=HttpSimulationController(
            base_url=settings.simulation_service_url, client=httpx.AsyncClient()
        ),
        inference_client=inference_client,
    )


def build_in_memory_container(
    settings: Settings,
    unit_of_work_factory: InMemoryUnitOfWorkFactory | None = None,
    clock: Clock | None = None,
    health_probe: HealthProbe | None = None,
    predictor: Predictor | None = None,
    simulation_controller: SimulationController | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
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
        embedder=embedder or UnavailableEmbedder(),
        reranker=reranker or UnavailableReranker(),
        engine=None,
        clock=clock or SystemClock(),
        # No service to call in the in-memory wiring, so a run cannot be
        # started. The tests that exercise simulation control supply their own
        # stub through this seam.
        simulation_controller=simulation_controller or UnavailableSimulationController(),
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
    embedder: Embedder,
    reranker: Reranker,
    engine: AsyncEngine | None,
    clock: Clock,
    simulation_controller: SimulationController,
    inference_client: httpx.AsyncClient | None = None,
) -> Container:
    """Build the container from already-chosen adapters."""
    # Built here rather than passed in: every wiring path wants the same one,
    # and the publishing use cases below hold the same object the streaming
    # endpoint subscribes to. Two instances would mean a stream that never
    # fires.
    broadcaster = InProcessEventBroadcaster()

    return Container(
        settings=settings,
        unit_of_work_factory=unit_of_work_factory,
        clock=clock,
        health_probe=health_probe,
        event_subscriber=broadcaster,
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
            events=broadcaster,
        ),
        ingest_telemetry=IngestTelemetry(
            unit_of_work_factory=unit_of_work_factory,
            window_readings=PREDICTION_WINDOW_READINGS,
            events=broadcaster,
        ),
        register_machine=RegisterMachine(unit_of_work_factory=unit_of_work_factory, clock=clock),
        update_incident_status=UpdateIncidentStatus(
            unit_of_work_factory=unit_of_work_factory,
            events=broadcaster,
        ),
        list_simulations=ListSimulations(
            unit_of_work_factory=unit_of_work_factory,
            clock=clock,
            heartbeat_timeout_seconds=settings.simulation_heartbeat_timeout_seconds,
        ),
        start_simulation=StartSimulation(
            unit_of_work_factory=unit_of_work_factory,
            controller=simulation_controller,
            clock=clock,
            events=broadcaster,
            max_concurrent_runs=settings.simulation_max_concurrent_runs,
        ),
        stop_simulation=StopSimulation(
            unit_of_work_factory=unit_of_work_factory,
            controller=simulation_controller,
            clock=clock,
            events=broadcaster,
        ),
        reset_simulation=ResetSimulation(
            unit_of_work_factory=unit_of_work_factory,
            events=broadcaster,
        ),
        report_simulation_state=ReportSimulationState(
            unit_of_work_factory=unit_of_work_factory,
            clock=clock,
            events=broadcaster,
        ),
        list_knowledge_documents=ListKnowledgeDocuments(
            unit_of_work_factory=unit_of_work_factory,
        ),
        ingest_knowledge_document=IngestKnowledgeDocument(
            unit_of_work_factory=unit_of_work_factory,
            embedder=embedder,
            clock=clock,
        ),
        search_maintenance_knowledge=SearchMaintenanceKnowledge(
            unit_of_work_factory=unit_of_work_factory,
            embedder=embedder,
            reranker=reranker,
            minimum_score=settings.knowledge_minimum_score,
        ),
        set_active_document_version=SetActiveDocumentVersion(
            unit_of_work_factory=unit_of_work_factory,
        ),
        simulation_controller=simulation_controller,
        engine=engine,
        broadcaster=broadcaster,
        inference_client=inference_client,
    )


class UnavailableSimulationController:
    """A controller that always refuses, for wiring with no simulator service.

    Failing loudly here is the same choice `UnavailablePredictor` makes: a
    placeholder that reported success would leave the API believing a run was
    going while nothing produced telemetry, and the dashboard would show a run
    that never moves with no error anywhere.
    """

    async def start(self, plan: SimulationPlan) -> None:
        """Refuse, whatever was asked.

        Raises:
            SimulationUnavailableError: always.
        """
        raise SimulationUnavailableError(
            f"No simulator service is configured, so '{plan.session_id}' cannot be "
            "started. Set SIMULATION_SERVICE_URL."
        )

    async def stop(self, session_id: str) -> None:
        """Refuse, whatever was asked.

        Raises:
            SimulationUnavailableError: always.
        """
        raise SimulationUnavailableError(
            f"No simulator service is configured, so '{session_id}' cannot be stopped."
        )

    async def aclose(self) -> None:
        """Nothing to release."""


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


class UnavailableEmbedder:
    """An embedder that always refuses, for wiring with no model behind it.

    The knowledge endpoints report a dependency outage rather than an empty
    result: an empty result would read as "the documentation does not cover
    this", which is a claim about the corpus rather than about the deployment.
    """

    @property
    def model_id(self) -> str:
        """The identifier a stored vector would carry, which is none of them."""
        return "unavailable"

    async def embed(self, texts: Sequence[str]) -> Sequence[tuple[float, ...]]:
        """Refuse, whatever was asked.

        Raises:
            RetrievalUnavailableError: always.
        """
        del texts
        raise RetrievalUnavailableError(
            "No inference service is configured, so text cannot be embedded. "
            "Set INFERENCE_SERVICE_URL."
        )


class UnavailableReranker:
    """A reranker that always refuses, for wiring with no model behind it."""

    async def rank(
        self,
        query: str,
        documents: Sequence[str],
        limit: int,
    ) -> Sequence[RerankResult]:
        """Refuse, whatever was asked.

        Raises:
            RetrievalUnavailableError: always.
        """
        del query, documents, limit
        raise RetrievalUnavailableError(
            "No inference service is configured, so candidates cannot be reranked. "
            "Set INFERENCE_SERVICE_URL."
        )
