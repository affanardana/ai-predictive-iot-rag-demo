"""Dependency resolution for route handlers.

Route handlers declare the use case they need and FastAPI resolves it from the
container attached to the application. Handlers never construct a use case or
reach for a repository themselves.
"""

from __future__ import annotations

import hmac
from typing import Annotated, cast

from fastapi import Depends, Header, Request

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
from api.composition.container import Container
from api.presentation.errors.authentication import AuthenticationError


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


def get_record_prediction(container: ContainerDep) -> RecordPrediction:
    """Resolve the record-prediction use case."""
    return container.record_prediction


def get_list_incidents(container: ContainerDep) -> ListIncidents:
    """Resolve the incident listing use case."""
    return container.list_incidents


def get_ingest_telemetry(container: ContainerDep) -> IngestTelemetry:
    """Resolve the telemetry ingestion use case."""
    return container.ingest_telemetry


def get_register_machine(container: ContainerDep) -> RegisterMachine:
    """Resolve the machine registration use case."""
    return container.register_machine


def require_ingest_token(
    container: ContainerDep,
    x_ingest_token: Annotated[str | None, Header()] = None,
) -> None:
    """Refuse a write unless the caller presents the configured shared secret.

    Applied to every route that writes, not only to ingestion: a prediction and
    a machine registration are equally things only the pipeline should be able
    to create.

    Compared as bytes rather than as `str`, because `hmac.compare_digest` raises
    `TypeError` on a `str` holding any non-ASCII character -- which would turn a
    malformed header into a 500 rather than a 401.

    Raises:
        AuthenticationError: if the header is missing or does not match.
    """
    expected = container.settings.ingest_api_token
    if expected is None:
        # No token configured, so there is nothing to check. The settings
        # validator makes that state unreachable outside `local`.
        return

    presented = x_ingest_token or ""
    if not hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
        raise AuthenticationError


ListMachinesDep = Annotated[ListMachines, Depends(get_list_machines)]
GetMachineDetailDep = Annotated[GetMachineDetail, Depends(get_machine_detail)]
GetTelemetryHistoryDep = Annotated[GetTelemetryHistory, Depends(get_telemetry_history)]
GetPredictionHistoryDep = Annotated[GetPredictionHistory, Depends(get_prediction_history)]
ListIncidentsDep = Annotated[ListIncidents, Depends(get_list_incidents)]
RecordPredictionDep = Annotated[RecordPrediction, Depends(get_record_prediction)]
IngestTelemetryDep = Annotated[IngestTelemetry, Depends(get_ingest_telemetry)]
RegisterMachineDep = Annotated[RegisterMachine, Depends(get_register_machine)]

#: Declared on a router rather than per route, so a new write cannot be added
#: to it and silently go unauthenticated.
IngestTokenDep = Depends(require_ingest_token)
