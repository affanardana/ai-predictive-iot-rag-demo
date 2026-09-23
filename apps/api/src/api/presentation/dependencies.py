"""Dependency resolution for route handlers.

Route handlers declare the use case they need and FastAPI resolves it from the
container attached to the application. Handlers never construct a use case or
reach for a repository themselves.
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from api.application.use_cases import (
    GetMachineDetail,
    GetPredictionHistory,
    GetTelemetryHistory,
    ListIncidents,
    ListMachines,
)
from api.composition.container import Container


def get_container(request: Request) -> Container:
    """Return the wired container attached to the running application."""
    return cast(Container, request.app.state.container)


ContainerDep = Annotated[Container, Depends(get_container)]


def get_list_machines(container: ContainerDep) -> ListMachines:
    """Resolve the fleet listing use case."""
    return container.list_machines


def get_machine_detail(container: ContainerDep) -> GetMachineDetail:
    """Resolve the machine detail use case."""
    return container.get_machine_detail


def get_telemetry_history(container: ContainerDep) -> GetTelemetryHistory:
    """Resolve the telemetry history use case."""
    return container.get_telemetry_history


def get_prediction_history(container: ContainerDep) -> GetPredictionHistory:
    """Resolve the prediction history use case."""
    return container.get_prediction_history


def get_list_incidents(container: ContainerDep) -> ListIncidents:
    """Resolve the incident listing use case."""
    return container.list_incidents


ListMachinesDep = Annotated[ListMachines, Depends(get_list_machines)]
GetMachineDetailDep = Annotated[GetMachineDetail, Depends(get_machine_detail)]
GetTelemetryHistoryDep = Annotated[GetTelemetryHistory, Depends(get_telemetry_history)]
GetPredictionHistoryDep = Annotated[GetPredictionHistory, Depends(get_prediction_history)]
ListIncidentsDep = Annotated[ListIncidents, Depends(get_list_incidents)]
