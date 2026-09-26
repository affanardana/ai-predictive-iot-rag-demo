"""Fleet-wide incident endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.risk_level import IncidentSeverity
from api.presentation.dependencies import ListIncidentsDep, UpdateIncidentStatusDep
from api.presentation.presenters import to_incident, to_incident_list
from api.presentation.schemas.incident import IncidentSchema, UpdateIncidentStatusRequest

router = APIRouter(prefix="/incidents", tags=["incidents"])

#: Annotated rather than default-value calls, so the filter declarations are not
#: function calls in default arguments (flake8-bugbear B008).
StatusQuery = Annotated[
    IncidentStatus | None,
    Query(description="Return only incidents in this lifecycle state."),
]
SeverityQuery = Annotated[
    IncidentSeverity | None,
    Query(description="Return only incidents of this severity."),
]


@router.get("", response_model=list[IncidentSchema], summary="List incidents")
async def list_incidents(
    use_case: ListIncidentsDep,
    status: StatusQuery = None,
    severity: SeverityQuery = None,
) -> list[IncidentSchema]:
    """Return incidents across the fleet, most recent first.

    Both filters are optional and combine conjunctively. The enum types are
    validated by the framework, so an unknown status yields a 422 rather than an
    empty list that would look like "no incidents".
    """
    incidents = await use_case.execute(status=status, severity=severity)
    return to_incident_list(incidents)


@router.patch(
    "/{incident_id}",
    response_model=IncidentSchema,
    summary="Change an incident's lifecycle status",
)
async def update_incident_status(
    incident_id: str,
    request: UpdateIncidentStatusRequest,
    use_case: UpdateIncidentStatusDep,
) -> IncidentSchema:
    """Move an incident to a new status.

    The only write in this API that is **not** behind `X-Ingest-Token`, and
    that is a decision rather than an oversight. The dashboard calls it from a
    browser, so guarding it would mean shipping the shared secret to every
    client -- the thing unguarded reads exist to avoid. The exposure is bounded:
    incidents are the only mutable resource reachable without a token, and
    closing one cannot inject telemetry, register a machine, or alter a
    prediction. Phase 11 owns operator authentication and closes it. See ADR
    0007.

    This is also what keeps incident suppression from being permanent. An
    incident is raised only when a machine has none open, so until one can be
    resolved, no later excursion on that machine can ever file another.
    """
    incident = await use_case.execute(incident_id, request.status)
    return to_incident(incident)
