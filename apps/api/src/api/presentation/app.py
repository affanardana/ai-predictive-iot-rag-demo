"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import __version__
from api.composition.container import Container
from api.presentation.errors import register_error_handlers
from api.presentation.middleware import RequestContextMiddleware
from api.presentation.routers import health_router, incidents_router, machines_router

API_V1_PREFIX = "/api/v1"

DESCRIPTION = """
Monitoring, prediction, and incident APIs for the AI Predictive Maintenance
platform.

**On predictions.** Failure probabilities are model output, not certainty. Risk
bands (NORMAL / WARNING / HIGH / CRITICAL) are product configuration for this
synthetic demonstration and are not universal industrial standards.
""".strip()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Release the container's resources when the application stops."""
    yield
    await app.state.container.aclose()


def create_app(container: Container) -> FastAPI:
    """Build the application around an already-wired container.

    Taking the container as an argument rather than building it here is what
    keeps this layer free of infrastructure knowledge: the presentation layer
    never learns which persistence adapter it is running against.
    """
    app = FastAPI(
        title="AI Predictive Maintenance API",
        description=DESCRIPTION,
        version=__version__,
        lifespan=_lifespan,
        openapi_url=f"{API_V1_PREFIX}/openapi.json",
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.container = container

    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)

    app.include_router(health_router)
    app.include_router(machines_router, prefix=API_V1_PREFIX)
    app.include_router(incidents_router, prefix=API_V1_PREFIX)

    return app
