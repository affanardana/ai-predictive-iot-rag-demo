"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import __version__
from api.composition.container import Container
from api.observability.metrics import record_build
from api.presentation.client_ip import TrustedProxies
from api.presentation.errors import register_error_handlers
from api.presentation.errors.handlers import RETRY_AFTER_HEADER
from api.presentation.middleware import RateLimitMiddleware, RequestContextMiddleware
from api.presentation.rate_limit import ClientRateLimiter, Cost, Limit
from api.presentation.routers import (
    copilot_router,
    events_router,
    health_router,
    incidents_router,
    knowledge_router,
    machines_router,
    metrics_router,
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


def _rate_limiter(container: Container) -> ClientRateLimiter:
    """Build the limiter from configuration.

    Takes the container rather than `Settings`, because `Settings` lives in
    `api.infrastructure` and this layer may not import it -- a rule the
    architecture test enforces by parsing this file, which is how the first
    version of this function was caught.

    Bursts are constants rather than settings -- a burst is a shape decision
    (how many clicks in a row a person makes), while the rate is a property of
    the host -- but they are named here, next to the rates they modify, so the
    two are read together.
    """
    settings = container.settings
    return ClientRateLimiter(
        limits={
            Cost.GLOBAL: Limit(per_minute=settings.rate_limit_global_per_minute, burst=120),
            Cost.COPILOT: Limit(per_minute=settings.rate_limit_copilot_per_minute, burst=2),
            Cost.SIMULATION: Limit(per_minute=settings.rate_limit_simulation_per_minute, burst=2),
            Cost.SEARCH: Limit(per_minute=settings.rate_limit_search_per_minute, burst=5),
        },
        max_tracked_clients=settings.rate_limit_max_tracked_clients,
    )


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
    # Which build this is, as a metric rather than a line in a log: a dashboard
    # that can graph `pdm_build_info` can answer "when did behaviour change"
    # without anyone correlating deploy times against a spike by eye. Read from
    # the environment because only the image build knows it.
    record_build(version=__version__, app_env=container.settings.app_env)

    # Built once and parsed once: an invalid entry fails here, at startup,
    # rather than on the first request that needs to resolve an address.
    trusted_proxies = TrustedProxies.parse(container.settings.rate_limit_trusted_proxies)
    app.state.trusted_proxies = trusted_proxies
    # On `app.state` as well as in the middleware, because the per-route budget
    # dependencies reach it through `request.app.state` -- the same way they
    # reach the container.
    app.state.rate_limiter = _rate_limiter(container)

    # Added first, so it is the *innermost* middleware: a 429 from the global
    # ceiling then still passes back out through the access log and the CORS
    # layer, which is what makes it readable by a browser and visible in the
    # logs rather than a bare rejection.
    app.add_middleware(
        RateLimitMiddleware,
        limiter=app.state.rate_limiter,
        trusted_proxies=trusted_proxies,
    )
    app.add_middleware(RequestContextMiddleware, trusted_proxies=trusted_proxies)
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
        # Both, and the second is not optional: without it a cross-origin
        # browser cannot read the countdown on a 429, so the dashboard would
        # have a rate-limited state with no way to say how long. Same class of
        # silent gap as the request id, which is why it is exposed for the same
        # reason.
        expose_headers=[REQUEST_ID_HEADER, RETRY_AFTER_HEADER],
    )
    register_error_handlers(app)

    app.include_router(health_router)
    # Beside `/health` and outside the versioned prefix, for the same reason:
    # it reports on the process rather than on the fleet, and it is not part
    # of the API a client is written against.
    app.include_router(metrics_router)
    app.include_router(machines_router, prefix=API_V1_PREFIX)
    app.include_router(incidents_router, prefix=API_V1_PREFIX)
    app.include_router(telemetry_router, prefix=API_V1_PREFIX)
    app.include_router(simulations_router, prefix=API_V1_PREFIX)
    app.include_router(knowledge_router, prefix=API_V1_PREFIX)
    app.include_router(copilot_router, prefix=API_V1_PREFIX)
    app.include_router(events_router, prefix=API_V1_PREFIX)

    return app
