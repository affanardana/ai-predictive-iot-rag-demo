"""Which requests the global ceiling charges.

Asserted at the predicate rather than by sending requests, for one path in
particular: `/api/v1/events` is a stream that is designed never to end, so a
test that opens one waits for ever. The integration tier covers the rest of the
behaviour through the real routes.
"""

from __future__ import annotations

import pytest
from starlette.types import Message, Receive, Scope, Send

from api.presentation.client_ip import TrustedProxies
from api.presentation.middleware import RateLimitMiddleware, _is_chargeable
from api.presentation.rate_limit import ClientRateLimiter, Cost, Limit

TRUSTED = TrustedProxies.parse("127.0.0.1,::1,172.16.0.0/12")

LIMITS = {cost: Limit(per_minute=1, burst=1) for cost in Cost}


async def receive() -> dict[str, object]:
    """Minimal receive callable: nothing here reads a request body."""
    return {"type": "http.request"}


def a_scope(path: str, *, forwarded: str | None = "203.0.113.9") -> Scope:
    """Build a scope for `path`, optionally with Caddy's header."""
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return {"type": "http", "path": path, "headers": headers, "client": ("172.17.0.1", 1)}


@pytest.mark.parametrize(
    "path",
    ["/api/v1/machines", "/api/v1/telemetry", "/api/v1/copilot/chat", "/docs"],
)
def test_an_external_caller_is_charged(path: str) -> None:
    """Everything a browser or a scraper reaches from outside."""
    assert _is_chargeable(a_scope(path))


@pytest.mark.parametrize("path", ["/health", "/health/ready", "/health/dependencies", "/metrics"])
def test_health_and_metrics_are_exempt(path: str) -> None:
    """A limiter that can fail a healthcheck is a limiter that restarts things.

    The container checks readiness every thirty seconds and a scraper hits
    metrics on its own schedule; both are machines on a fixed cadence, and
    throttling either fails as "target down" rather than as a clear error.
    """
    assert not _is_chargeable(a_scope(path))


def test_the_event_stream_is_exempt() -> None:
    """`EventSource` reconnects every three seconds on its own.

    A 429 here turns a dropped connection into a storm that cannot recover --
    the one place where refusing a request is worse than serving it.
    """
    assert not _is_chargeable(a_scope("/api/v1/events"))


def test_a_caller_from_inside_the_network_is_exempt() -> None:
    """No forwarded header means the caller is the pipeline, not a visitor.

    n8n posts one telemetry batch per reading for as long as a run lasts.
    Metering that would start refusing the ingest route -- dropped readings,
    holes in the charts -- which is a worse failure than a slow dashboard.
    """
    assert not _is_chargeable(a_scope("/api/v1/telemetry", forwarded=None))


async def test_the_middleware_renders_its_own_refusal() -> None:
    """It answers rather than raising, and the request never reaches the app.

    **It has to render rather than raise.** This middleware sits outside
    Starlette's `ExceptionMiddleware`, so an exception raised here never reaches
    the handlers registered on the app: the first version of this raised, and a
    rate-limited caller got `500 internal_error` with the real reason only in
    the log. Asserting the status here is what stops that coming back.
    """
    reached: list[str] = []
    sent: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        reached.append(str(scope["path"]))

    async def send(message: Message) -> None:
        sent.append(message)

    limiter = ClientRateLimiter(limits=LIMITS, max_tracked_clients=10)
    middleware = RateLimitMiddleware(app, limiter=limiter, trusted_proxies=TRUSTED)

    await middleware(a_scope("/api/v1/machines"), receive, send)
    assert reached == ["/api/v1/machines"]
    assert sent == []

    await middleware(a_scope("/api/v1/machines"), receive, send)

    assert reached == ["/api/v1/machines"], "the refused request reached the application"
    start = next(message for message in sent if message["type"] == "http.response.start")
    assert start["status"] == 429
    assert dict(start["headers"])[b"retry-after"] == b"60"
