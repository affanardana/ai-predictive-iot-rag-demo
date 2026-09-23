"""Cross-cutting error behaviour."""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from api.composition import build_in_memory_container
from api.infrastructure.config import Settings
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from api.presentation.app import create_app
from api.request_context import REQUEST_ID_HEADER
from tests.support.fakes import StubHealthProbe


async def test_error_responses_share_one_envelope(client: AsyncClient) -> None:
    """Every failure uses the same JSON shape."""
    response = await client.get("/api/v1/machines/M999")

    assert set(response.json()) == {"error"}
    assert set(response.json()["error"]) == {"code", "message", "details"}


async def test_response_carries_a_request_id(client: AsyncClient) -> None:
    """Every response is traceable to a log line."""
    response = await client.get("/health")

    assert response.headers.get(REQUEST_ID_HEADER)


async def test_an_inbound_request_id_is_honoured(client: AsyncClient) -> None:
    """A caller-supplied correlation id is preserved.

    This is what lets one trace span the client, this API, and any upstream
    caller rather than each generating its own id.
    """
    response = await client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc-123"})

    assert response.headers[REQUEST_ID_HEADER] == "trace-abc-123"


async def test_unknown_routes_return_404(client: AsyncClient) -> None:
    """An unrouted path is a 404, not a crash."""
    response = await client.get("/api/v1/nonexistent")

    assert response.status_code == 404


async def test_openapi_schema_is_served_under_the_api_prefix(
    client: AsyncClient,
) -> None:
    """The OpenAPI document is versioned alongside the API it describes."""
    response = await client.get("/api/v1/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["version"]
    assert "/api/v1/machines" in schema["paths"]


async def test_health_lives_outside_the_versioned_prefix(client: AsyncClient) -> None:
    """Probes never depend on API versioning.

    An orchestrator should not have to track the API version to check that a
    process is alive.
    """
    assert (await client.get("/health")).status_code == 200
    assert (await client.get("/api/v1/health")).status_code == 404


async def test_startup_does_not_require_a_reachable_database(
    settings: Settings,
) -> None:
    """Building and serving the app works while the store is unreachable.

    The container is constructed eagerly at import time in `api.main`, so if
    construction tried to connect, the process would refuse to start during a
    brief database outage instead of reporting it through readiness.
    """
    container = build_in_memory_container(
        settings,
        unit_of_work_factory=InMemoryUnitOfWorkFactory(),
        health_probe=StubHealthProbe(healthy=False, detail="down"),
    )
    app = create_app(container)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/api/v1/machines")).status_code == 200
