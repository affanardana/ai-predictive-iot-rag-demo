"""Fleet-wide incident endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.risk_level import IncidentSeverity
from api.presentation.dependencies import ListIncidentsDep
from api.presentation.presenters import to_incident_list
from api.presentation.schemas.incident import IncidentSchema

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
