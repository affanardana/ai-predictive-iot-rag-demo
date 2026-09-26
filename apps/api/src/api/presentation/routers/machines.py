"""Machine monitoring endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.time_window import TimeWindow
from api.presentation.dependencies import (
    GetMachineDetailDep,
    GetPredictionHistoryDep,
    GetTelemetryHistoryDep,
    IngestTokenDep,
    ListIncidentsDep,
    ListMachinesDep,
    ListSimulationsDep,
    RecordPredictionDep,
    RegisterMachineDep,
)
from api.presentation.presenters import (
    to_incident_list,
    to_machine,
    to_machine_detail,
    to_machine_summary,
    to_prediction,
    to_simulation_run_list,
    to_telemetry_series,
)
from api.presentation.schemas.incident import IncidentSchema
from api.presentation.schemas.machine import (
    MachineDetailSchema,
    MachineSchema,
    MachineSummarySchema,
    RegisterMachineRequest,
)
from api.presentation.schemas.prediction import PredictionSchema
from api.presentation.schemas.simulation import SimulationRunSchema
from api.presentation.schemas.telemetry import TelemetrySeriesSchema

router = APIRouter(prefix="/machines", tags=["machines"])

DEFAULT_TELEMETRY_WINDOW = TimeWindow.ONE_HOUR

#: Declared as an annotation rather than a default-value call. FastAPI supports
#: both, but a `Query(...)` in the default position is a function call in a
#: default argument, which flake8-bugbear flags (B008) for good reason.
WindowQuery = Annotated[
    TimeWindow,
    Query(description="History window to return."),
]


@router.get("", response_model=list[MachineSummarySchema], summary="List machines")
async def list_machines(use_case: ListMachinesDep) -> list[MachineSummarySchema]:
    """Return the fleet with each machine's latest observed and predicted state.

    Returns an empty list when no machines are registered, which is the
    expected state before the simulator exists.
    """
    summaries = await use_case.execute()
    return [to_machine_summary(summary) for summary in summaries]


@router.get(
    "/{machine_id}",
    response_model=MachineDetailSchema,
    summary="Get machine detail",
)
async def get_machine(
    machine_id: str,
    use_case: GetMachineDetailDep,
) -> MachineDetailSchema:
    """Return one machine's current state and recent incidents.

    The identifier is parsed into a domain value object here rather than
    validated by the framework, so a malformed identifier produces the same
    error shape as any other domain validation failure.
    """
    detail = await use_case.execute(MachineId(machine_id))
    return to_machine_detail(detail)


@router.get(
    "/{machine_id}/telemetry",
    response_model=TelemetrySeriesSchema,
    summary="Get telemetry history",
)
async def get_telemetry(
    machine_id: str,
    use_case: GetTelemetryHistoryDep,
    window: WindowQuery = DEFAULT_TELEMETRY_WINDOW,
) -> TelemetrySeriesSchema:
    """Return telemetry for a machine over a window.

    Longer windows are aggregated server-side before transmission. The response
    states the resolution it was reduced to, so a caller can always tell
    measurements from aggregates.
    """
    series = await use_case.execute(MachineId(machine_id), window)
    return to_telemetry_series(series)


@router.get(
    "/{machine_id}/predictions",
    response_model=list[PredictionSchema],
    summary="Get prediction history",
)
async def get_predictions(
    machine_id: str,
    use_case: GetPredictionHistoryDep,
) -> list[PredictionSchema]:
    """Return how the predicted failure risk has changed over time."""
    predictions = await use_case.execute(MachineId(machine_id))
    return [to_prediction(prediction) for prediction in predictions]


@router.get(
    "/{machine_id}/incidents",
    response_model=list[IncidentSchema],
    summary="Get incidents for a machine",
)
async def get_machine_incidents(
    machine_id: str,
    use_case: ListIncidentsDep,
) -> list[IncidentSchema]:
    """Return incidents raised for one machine, most recent first."""
    incidents = await use_case.execute(machine_id=MachineId(machine_id))
    return to_incident_list(incidents)


@router.post(
    "",
    response_model=MachineSchema,
    status_code=status.HTTP_201_CREATED,
    dependencies=[IngestTokenDep],
    summary="Register a machine",
)
async def register_machine(
    payload: RegisterMachineRequest,
    response: Response,
    use_case: RegisterMachineDep,
) -> MachineSchema:
    """Add a machine to the fleet, or return the one already registered.

    Idempotent, because the pipeline declares its fleet on start without
    tracking what it declared last time. The status code distinguishes the two
    outcomes: 201 when this call created the machine, 200 when it already
    existed.
    """
    result = await use_case.execute(MachineId(payload.machine_id), payload.name)
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return to_machine(result.machine)


@router.post(
    "/{machine_id}/predictions",
    response_model=PredictionSchema,
    status_code=status.HTTP_201_CREATED,
    dependencies=[IngestTokenDep],
    summary="Score a machine from its stored telemetry",
)
async def record_prediction(
    machine_id: str,
    use_case: RecordPredictionDep,
) -> PredictionSchema:
    """Ask the model about a machine, and store what it said.

    Takes no body. The window comes from telemetry already stored for the
    machine, which is what the pipeline persists before anything scores it --
    so the ordering the model depends on is the API's responsibility rather
    than a caller's.
    """
    result = await use_case.execute(MachineId(machine_id))
    # Only the prediction is a response concern. Whether an incident was raised
    # alongside it is announced on the event stream, and is already visible at
    # `GET /machines/{id}`.
    return to_prediction(result.prediction)


@router.get(
    "/{machine_id}/simulations",
    response_model=list[SimulationRunSchema],
    summary="List a machine's simulation runs",
)
async def list_machine_simulations(
    machine_id: str,
    use_case: ListSimulationsDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[SimulationRunSchema]:
    """Return a machine's runs, newest first.

    A sub-resource rather than a field on `GET /machines/{id}`, so the machine's
    simulation panel can poll at its own cadence -- several seconds while a run
    is active -- without dragging the whole detail response with it.
    """
    views = await use_case.execute(MachineId(machine_id), limit=limit)
    return to_simulation_run_list(views)
