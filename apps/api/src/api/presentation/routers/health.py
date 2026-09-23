"""Liveness and readiness endpoints.

Deliberately mounted outside `/api/v1`. A probe that depends on API versioning
would break the moment the version changes, and orchestrators should never have
to track that.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, ConfigDict, Field

from api.presentation.dependencies import ContainerDep

router = APIRouter(tags=["health"])

DEGRADED_STATUS_CODE = status.HTTP_503_SERVICE_UNAVAILABLE


class HealthResponse(BaseModel):
    """Result of a health check."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "degraded"]
    detail: str
    app_env: str = Field(description="Which environment this process is running in.")


@router.get("/health", summary="Liveness probe")
async def liveness(container: ContainerDep) -> HealthResponse:
    """Report that the process is running.

    Deliberately does not touch the database: a liveness probe answers "should
    this process be restarted?", and a database outage is not fixed by
    restarting the API.
    """
    return HealthResponse(
        status="ok",
        detail="The service is running.",
        app_env=container.settings.app_env,
    )


@router.get("/health/ready", summary="Readiness probe")
async def readiness(container: ContainerDep, response: Response) -> HealthResponse:
    """Report whether the service can serve requests.

    Unlike liveness, this does check the database -- readiness answers "should
    this instance receive traffic?", and an instance that cannot reach its
    database should be taken out of rotation rather than restarted.
    """
    health = await container.health_probe.check()
    if not health.healthy:
        response.status_code = DEGRADED_STATUS_CODE

    return HealthResponse(
        status="ok" if health.healthy else "degraded",
        detail=health.detail,
        app_env=container.settings.app_env,
    )
