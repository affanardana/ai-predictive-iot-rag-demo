"""Cross-origin access for the dashboard.

The frontend is served from Vercel and calls this API on a VPS host, so every
request it makes is cross-origin. None of this existed before Phase 7 and the
failure it prevents is total rather than partial: without these headers the
browser refuses every response, and nothing appears in the server's logs to say
why.
"""

from __future__ import annotations

from httpx import AsyncClient

#: Stands in for a deployed frontend. Any origin behaves the same under a
#: wildcard, which is the point of the assertion below.
DASHBOARD_ORIGIN = "https://pdm-dashboard.vercel.app"


async def test_a_preflight_is_answered(client: AsyncClient) -> None:
    """An `OPTIONS` preflight succeeds without reaching the router.

    `PATCH /incidents/{id}` is the request that needs this: a `PATCH` with a
    JSON body is not a CORS-simple request, so the browser asks first and
    abandons the write if the answer is missing.
    """
    response = await client.options(
        "/api/v1/incidents/inc-1",
        headers={
            "Origin": DASHBOARD_ORIGIN,
            "Access-Control-Request-Method": "PATCH",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


async def test_a_read_carries_the_origin_header(client: AsyncClient) -> None:
    """A plain `GET` is readable from the dashboard's origin.

    Simple requests get no preflight, so this header is the only thing standing
    between the fleet list and an opaque browser error.
    """
    response = await client.get("/api/v1/machines", headers={"Origin": DASHBOARD_ORIGIN})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


async def test_the_correlation_id_is_readable(client: AsyncClient) -> None:
    """`x-request-id` is exposed to cross-origin JavaScript.

    Without `expose_headers` the browser hides every non-safelisted response
    header, so the one handle a dashboard bug report could carry -- the id that
    ties a screen to a server log line -- would be invisible exactly when it is
    needed.
    """
    response = await client.get("/api/v1/machines", headers={"Origin": DASHBOARD_ORIGIN})

    exposed = response.headers["access-control-expose-headers"].lower()
    assert "x-request-id" in exposed


async def test_an_error_response_is_readable_too(client: AsyncClient) -> None:
    """The headers are on failures, not only on successes.

    This is why CORS is added after the request-context middleware: Starlette
    makes the last-added middleware outermost, so a 404 produced deep in the
    router still passes back out through it. A browser that could read the
    success but not the failure would report every problem as "network error".
    """
    response = await client.get("/api/v1/machines/M999", headers={"Origin": DASHBOARD_ORIGIN})

    assert response.status_code == 404
    assert response.headers["access-control-allow-origin"] == "*"
    assert response.json()["error"]["code"] == "machine_not_found"
