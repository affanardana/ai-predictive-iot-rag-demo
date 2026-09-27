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
    commit: str = Field(
        default="unknown",
        description=(
            "The git commit this image was built from. `unknown` when the build did not pass one."
        ),
    )


class DependencyStatusSchema(BaseModel):
    """One dependency, and what it says it is running."""

    model_config = ConfigDict(frozen=True)

    name: str
    reachable: bool
    detail: str
    models: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Identities the service reported about itself. For the model service "
            "this is the deployed artefacts: the LSTM's run id, the two encoders, "
            "and the Copilot's model file."
        ),
    )


class DependencyReportResponse(BaseModel):
    """Every dependency, in a stable order."""

    model_config = ConfigDict(frozen=True)

    dependencies: list[DependencyStatusSchema]
    app_env: str


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
        commit=container.settings.source_commit,
    )


@router.get("/health/dependencies", summary="What every dependency is running")
async def dependencies(container: ContainerDep) -> DependencyReportResponse:
    """Report each service this API depends on, and the models behind it.

    **Not part of readiness, deliberately.** Readiness gates traffic -- n8n
    checks it before writing telemetry -- and the inference container spends
    twenty seconds loading its chat model at startup. Folding this into
    readiness would mean that during every deploy the API declares itself
    unready and the pipeline stops storing readings: a real outage caused by
    more information.

    It is also the only way to see the two side services at all. Neither
    publishes a host port, so from outside the Compose bridge there is nothing
    to ask -- and their model identities, which are placed by hand on the host
    and change without a rebuild, exist nowhere else.
    """
    reports = await container.dependency_reporter.report()
    return DependencyReportResponse(
        dependencies=[
            DependencyStatusSchema(
                name=report.name,
                reachable=report.reachable,
                detail=report.detail,
                models=dict(report.models),
            )
            for report in reports
        ],
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
