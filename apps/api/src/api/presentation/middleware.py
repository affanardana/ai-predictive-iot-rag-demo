"""Request-scoped middleware."""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.observability.metrics import (
    record_request,
    record_request_finished,
    record_request_started,
)
from api.presentation.client_ip import TrustedProxies, bucket_key, resolve_client_ip
from api.presentation.errors.handlers import rate_limited_response
from api.presentation.errors.rate_limit import RateLimitError
from api.presentation.rate_limit import ClientRateLimiter, Cost
from api.request_context import REQUEST_ID_HEADER, reset_request_id, set_request_id

logger = logging.getLogger(__name__)


#: The media type of a response that stays open. Its duration is the length of
#: the visitor's visit, so timing it would add one sample to the top bucket per
#: connection and answer nothing.
_STREAMING_MEDIA_TYPE = "text/event-stream"


class RequestContextMiddleware:
    """Assigns a request id, emits one access log per request, records metrics.

    Written against the raw ASGI interface rather than `BaseHTTPMiddleware`,
    which buffers responses and interferes with streaming -- which this API
    does, on two routes.

    An inbound `x-request-id` is honoured so a trace can span the client, this
    API, and any future upstream caller; otherwise one is generated.

    Metrics are recorded here rather than in a middleware of their own because
    this is already the one place that sees every request's start, its status
    and its end, and a second pass over the same scope would be a second place
    for `status_code` to be defaulted wrongly.
    """

    def __init__(self, app: ASGIApp, *, trusted_proxies: TrustedProxies) -> None:
        self.app = app
        self._trusted_proxies = trusted_proxies

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI call."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _inbound_request_id(scope) or str(uuid4())
        token = set_request_id(request_id)
        started = time.perf_counter()
        status_code = 500
        content_type = ""

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code, content_type
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
                content_type = headers.get("content-type", "")
            await send(message)

        record_request_started()
        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration = time.perf_counter() - started
            route = _route_template(scope)
            streaming = content_type.startswith(_STREAMING_MEDIA_TYPE)
            record_request(
                route=route,
                method=str(scope.get("method", "")),
                status=status_code,
                duration_seconds=None if streaming else duration,
            )
            record_request_finished()
            logger.info(
                "http.request",
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "route": route,
                    "status_code": status_code,
                    "duration_ms": round(duration * 1000, 2),
                    # The address the rate limiter charges this request to, in
                    # the log so that a misconfigured proxy chain is visible
                    # rather than inferred. Without it the only symptom is a
                    # budget shared by every visitor on earth, and no error
                    # anywhere to explain it.
                    "client_ip": resolve_client_ip(scope, self._trusted_proxies),
                },
            )
            reset_request_id(token)


#: Paths the global ceiling never charges.
#:
#: `/health` because the container's own healthcheck calls it every thirty
#: seconds from inside the network, and a limiter that can fail a healthcheck
#: is a limiter that can restart the container. `/metrics` because a scraper is
#: a machine on a fixed schedule, and throttling it would fail as "target down".
#: The event stream because `EventSource` reconnects on its own every three
#: seconds: a 429 there turns a dropped connection into a storm that cannot
#: recover, and the stream's *establishment* is one request that costs nothing.
_EXEMPT_PATHS = ("/health", "/metrics")
_EXEMPT_EXACT = ("/api/v1/events",)


class RateLimitMiddleware:
    """The per-address ceiling that covers every route, including new ones.

    The three expensive routes carry their own budgets as dependencies, because
    their limits differ. This is the other half: a ceiling that cannot be
    forgotten when somebody adds a route, applied to whatever arrives.

    **It charges only callers from outside.** The pipeline is not a visitor:
    n8n posts one telemetry batch per reading, roughly one a second per machine
    for as long as a run lasts, from an address inside the Compose network. A
    ceiling that metered that would start returning 429 to the ingest route --
    dropped readings, holes in the charts, a simulation whose stored data is
    wrong -- which is a far worse failure than a slow dashboard. Callers from
    inside are recognised by the *absence* of a forwarding header, which Caddy
    always adds and an in-network caller never does; the routes they use are
    token-guarded separately, so exempting them here costs nothing.

    Added innermost of the middleware stack, so a 429 still passes back out
    through the access log and the CORS layer.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: ClientRateLimiter,
        trusted_proxies: TrustedProxies,
    ) -> None:
        self.app = app
        self._limiter = limiter
        self._trusted_proxies = trusted_proxies

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Charge the request, or refuse it before it reaches the router."""
        if scope["type"] != "http" or not _is_chargeable(scope):
            await self.app(scope, receive, send)
            return

        try:
            self._limiter.charge(
                bucket_key(resolve_client_ip(scope, self._trusted_proxies)), Cost.GLOBAL
            )
        except RateLimitError as refused:
            # Rendered here rather than raised onwards: this middleware sits
            # *outside* Starlette's `ExceptionMiddleware`, so an exception from
            # it never reaches the handlers registered on the app -- it would
            # leave as a 500 with the real reason only in the log.
            response = rate_limited_response(refused, path=str(scope.get("path", "")))
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def _is_chargeable(scope: Scope) -> bool:
    """Whether the global ceiling applies to this request."""
    path = str(scope.get("path", ""))
    if path in _EXEMPT_EXACT or path.startswith(_EXEMPT_PATHS):
        return False
    # No forwarded header means the caller is inside the Compose network.
    return bool(Headers(scope=scope).getlist("x-forwarded-for"))


def _route_template(scope: Scope) -> str:
    """Return the matched route's template, or `"unmatched"`.

    A *template* -- `/machines/{machine_id}` -- never the requested path. The
    path is one series per machine, per typo, per probe, which is how a metrics
    endpoint becomes a memory leak; the template is bounded by the number of
    routes this application has.

    **The template is missing its mount prefix**, which is `/api/v1` for
    everything under it: FastAPI keeps `include_router(prefix=...)` on the
    router rather than on the route, so the matched `APIRoute` reports
    `/machines/{machine_id}` and the scope exposes the prefix nowhere. Verified
    rather than assumed, and left alone rather than reconstructed by string
    surgery on the requested path -- that would be a worse bug than a label
    that reads one segment short. It stays unique because each router is
    included exactly once.

    `scope["route"]` is set by FastAPI while matching, not by Starlette, so a
    test asserts it rather than trusting it to survive a version bump.
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else "unmatched"


def _inbound_request_id(scope: Scope) -> str | None:
    """Return the caller-supplied request id, if any."""
    return Headers(scope=scope).get(REQUEST_ID_HEADER)
