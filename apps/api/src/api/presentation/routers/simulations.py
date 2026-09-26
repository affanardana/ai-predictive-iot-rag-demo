"""Simulation control endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from api.application.read_models import SimulationRunView
from api.domain.entities.simulation_run import SimulationRun
from api.domain.value_objects.machine_id import MachineId
from api.presentation.dependencies import (
    IngestTokenDep,
    ListSimulationsDep,
    ReportSimulationStateDep,
    ResetSimulationDep,
    StartSimulationDep,
    StopSimulationDep,
)
from api.presentation.presenters import to_simulation_run, to_simulation_run_list
from api.presentation.schemas.simulation import (
    ReportSimulationStateRequest,
    SimulationRunSchema,
    StartSimulationRequest,
)

router = APIRouter(prefix="/simulations", tags=["simulations"])

#: Annotated rather than a default-value call, so the declaration is not a
#: function call in a default argument (flake8-bugbear B008).
LimitQuery = Annotated[int, Query(ge=1, le=200)]


@router.get("", response_model=list[SimulationRunSchema], summary="List simulation runs")
async def list_simulations(
    use_case: ListSimulationsDep,
    limit: LimitQuery = 50,
) -> list[SimulationRunSchema]:
    """Return recent runs across the fleet, newest first.

    A run's `is_active` and `is_stale` are computed by the API. A client must
    never derive either: staleness needs a clock and a timeout, and a browser
    reading its own clock would disagree with the server about whether a run is
    alive.
    """
    return to_simulation_run_list(await use_case.execute(limit=limit))


@router.post(
    "",
    response_model=SimulationRunSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Start a simulation run",
)
async def start_simulation(
    request: StartSimulationRequest,
    use_case: StartSimulationDep,
) -> SimulationRunSchema:
    """Begin a run on a machine.

    **Unguarded**, like the incident actions and for the same reason: the
    dashboard calls it from a browser, and a token in a browser bundle is not a
    boundary but the appearance of one. The exposure is larger here than for an
    incident, though -- a run costs CPU on a box that already shares one core
    between the API, the orchestrator and the model service -- so it is bounded
    instead: one active run per machine, a ceiling on concurrent runs, and a
    floor on duration that refuses a run too short to produce anything. See ADR
    0008.
    """
    run = await use_case.execute(
        MachineId(request.machine_id),
        request.scenario,
        demo=request.demo,
        seed=request.seed,
        duration_minutes=request.duration_minutes,
    )
    # The use case returns the entity it built; `is_stale` is false by
    # construction for a run that was just accepted, which is what the view
    # field would say.
    return to_simulation_run(_as_view(run))


@router.post(
    "/{session_id}/stop",
    response_model=SimulationRunSchema,
    summary="Stop a simulation run",
)
async def stop_simulation(
    session_id: str,
    use_case: StopSimulationDep,
) -> SimulationRunSchema:
    """End a run.

    A separate route from `DELETE` because the two mean different things:
    stopping ends a run and keeps the record of it, and resetting discards the
    record once the run has ended. Collapsing them would make "stop" silently
    destroy the history a demonstration is worth reviewing.

    Idempotent, and it terminalises the run whatever the simulator says -- an
    escape hatch that could fail to close would leave a run stuck at RUNNING
    with no way out but the database.
    """
    run = await use_case.execute(session_id)
    return to_simulation_run(_as_view(run))


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Reset a simulation run",
)
async def reset_simulation(session_id: str, use_case: ResetSimulationDep) -> None:
    """Discard a finished run's record so the machine is free for another.

    It does **not** delete the telemetry the run produced, and cannot: deleting
    stored readings is a destructive write, which would have to be guarded by
    the ingest token, which a browser cannot hold. So a second run appends to a
    machine's charts rather than replacing them -- which the windowed charts
    hide in practice, and which the runbook says out loud.
    """
    await use_case.execute(session_id)


@router.patch(
    "/{session_id}",
    response_model=SimulationRunSchema,
    dependencies=[IngestTokenDep],
    summary="Report a run's state",
)
async def report_simulation_state(
    session_id: str,
    request: ReportSimulationStateRequest,
    use_case: ReportSimulationStateDep,
) -> SimulationRunSchema:
    """Accept a state report from the simulator service.

    **The one guarded route here**, and the guard is on the route rather than
    the router because this router mixes browser traffic with machine traffic.
    That is not a weakness in the usual arrangement -- it is the point: the
    caller is a service, exactly like n8n, and `ingest_api_token`'s own
    docstring calls it "one machine proving to another that it is the expected
    caller". A browser is structurally unable to forge a completion, which the
    three dashboard routes above cannot say.

    Progress reports are accepted without publishing an event. The service
    reports every few seconds; announcing each one would make every open
    dashboard refetch on that cadence for a progress bar.
    """
    run = await use_case.execute(session_id, request.status, request.completed_ticks)
    return to_simulation_run(_as_view(run))


def _as_view(run: SimulationRun) -> SimulationRunView:
    """Wrap a run with the derived facts the schema carries.

    `is_stale=False` is not a shortcut: a run is stale only when it is active
    *and* has stopped reporting, and these routes have just accepted or just
    ended one. Recomputing it by re-reading the row would be a second round trip
    for an answer already in hand -- and for a run that was just stopped, a
    wrong one, because a stopped run is never stale.
    """
    return SimulationRunView(run=run, is_active=run.is_active, is_stale=False)
