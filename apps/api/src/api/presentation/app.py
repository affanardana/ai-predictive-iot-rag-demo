"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import __version__
from api.composition.container import Container
from api.presentation.errors import register_error_handlers
from api.presentation.middleware import RequestContextMiddleware
from api.presentation.routers import (
    events_router,
    health_router,
    incidents_router,
    machines_router,
    simulations_router,
    telemetry_router,
)
from api.request_context import REQUEST_ID_HEADER

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
    # Added second, and that ordering is load-bearing: Starlette inserts each
    # middleware at the front of the stack, so the last one added is the
    # outermost. CORS has to be outermost to answer a preflight without the
    # request reaching the router, and so that an error raised deeper still
    # carries the headers a browser needs in order to read it.
    app.add_middleware(
        CORSMiddleware,
        # A wildcard, deliberately, and not a settings field.
        #
        # Reads are unguarded by design and writes are guarded by a header
        # token rather than a cookie, so this grants a browser nothing that
        # `curl` cannot already do. `allow_credentials=False` means the
        # wildcard cannot be combined with ambient credentials later, so it
        # cannot quietly become a confused-deputy problem.
        #
        # The deciding reason is Vercel: every preview deployment gets its own
        # generated origin, so a fixed allowlist would break previews on every
        # push, and the failure surfaces as an opaque browser CORS error rather
        # than anything the server logs. Phase 11 tightens this alongside the
        # authentication that would make tightening mean something.
        allow_origins=["*"],
        # `DELETE` is here for resetting a simulation run. Omitting it fails in
        # the browser as an opaque preflight rejection rather than as anything
        # the server logs, which is the kind of gap that costs an afternoon.
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Ingest-Token"],
        allow_credentials=False,
        # Without this the browser cannot read the correlation id on a
        # cross-origin response, which is the only handle a dashboard bug
        # report would have on a specific request.
        expose_headers=[REQUEST_ID_HEADER],
    )
    register_error_handlers(app)

    app.include_router(health_router)
    app.include_router(machines_router, prefix=API_V1_PREFIX)
    app.include_router(incidents_router, prefix=API_V1_PREFIX)
    app.include_router(telemetry_router, prefix=API_V1_PREFIX)
    app.include_router(simulations_router, prefix=API_V1_PREFIX)
    app.include_router(events_router, prefix=API_V1_PREFIX)

    return app
