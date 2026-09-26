"""The internal token guarding the write endpoints.

Configured directly rather than through the environment, so these run whatever
`APP_ENV` the rest of the suite is pinned to.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from api.composition.container import build_in_memory_container
from api.infrastructure.config import Settings
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from api.presentation.app import create_app
from tests.support.factories import DEFAULT_NOW, make_machine
from tests.support.fakes import FixedClock, StubHealthProbe

TOKEN = "a-shared-secret"
HEADER = "X-Ingest-Token"


def a_sample() -> dict[str, object]:
    return {
        "event_id": "sim-bd-20260923-M003-00000000",
        "machine_id": "M003",
        "recorded_at": DEFAULT_NOW.isoformat(),
        "session_id": "sim-bd-20260923",
        "temperature": 61.5,
        "vibration": 1.42,
        "rpm": 1480.0,
        "current": 12.5,
        "load": 0.72,
        "voltage": 400.0,
    }


@pytest.fixture
async def guarded(
    settings: Settings,
    uow_factory: InMemoryUnitOfWorkFactory,
    clock: FixedClock,
    health_probe: StubHealthProbe,
) -> AsyncIterator[AsyncClient]:
    """A client whose API requires the token, over a store holding one machine."""
    guarded_settings = Settings(_env_file=None, ingest_api_token=TOKEN)
    container = build_in_memory_container(
        guarded_settings,
        unit_of_work_factory=uow_factory,
        clock=clock,
        health_probe=health_probe,
    )
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))

    transport = ASGITransport(app=create_app(container))
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


WRITE_ROUTES = [
    ("POST", "/api/v1/telemetry", {"records": [a_sample()]}),
    ("POST", "/api/v1/machines", {"machine_id": "M004", "name": "Pump"}),
    ("POST", "/api/v1/machines/M003/predictions", None),
]


@pytest.mark.parametrize(("method", "path", "body"), WRITE_ROUTES)
async def test_a_write_without_the_token_is_refused(
    guarded: AsyncClient, method: str, path: str, body: dict | None
) -> None:
    """Every route that writes, not only ingestion.

    A prediction and a machine registration are as much the pipeline's to
    create as a reading is, so guarding one and not the others would be an
    arbitrary line rather than a boundary.
    """
    response = await guarded.request(method, path, json=body)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


@pytest.mark.parametrize(("method", "path", "body"), WRITE_ROUTES)
async def test_a_write_with_the_wrong_token_is_refused(
    guarded: AsyncClient, method: str, path: str, body: dict | None
) -> None:
    response = await guarded.request(method, path, json=body, headers={HEADER: "not-the-token"})

    assert response.status_code == 401


async def test_the_refusal_uses_the_shared_error_envelope(guarded: AsyncClient) -> None:
    """Not FastAPI's `{"detail": ...}`, which is a different shape.

    A client parsing this API's errors should not have to special-case the one
    route that happens to be guarded.
    """
    response = await guarded.post("/api/v1/telemetry", json={"records": [a_sample()]})

    assert set(response.json()) == {"error"}
    assert set(response.json()["error"]) == {"code", "message", "details"}


async def test_a_non_ascii_token_is_refused_at_startup(
    settings: Settings,
) -> None:
    """Caught when configuration is read, not when a request arrives.

    An HTTP header value is a byte string, so a client cannot even encode a
    non-ASCII token -- `httpx` refuses before the request leaves. Such a token
    would therefore refuse every caller forever, which is a misconfiguration
    worth one clear message rather than a wall of 401s.
    """
    with pytest.raises(ValidationError, match="ASCII"):
        Settings(_env_file=None, ingest_api_token="sécret")


async def test_a_blank_token_is_refused_at_startup(settings: Settings) -> None:
    """A blank secret is not a secret.

    `compare_digest("", "")` is true, so an empty token would authenticate a
    request that carried no header at all -- the opposite of what configuring
    one is for.
    """
    with pytest.raises(ValidationError, match="blank"):
        Settings(_env_file=None, ingest_api_token="   ")


async def test_the_right_token_is_accepted(guarded: AsyncClient) -> None:
    response = await guarded.post(
        "/api/v1/telemetry",
        json={"records": [a_sample()]},
        headers={HEADER: TOKEN},
    )

    assert response.status_code == 201, response.text
    assert response.json()["accepted"] == 1


async def test_reads_are_not_guarded(guarded: AsyncClient) -> None:
    """The token is for writing.

    The dashboard and the copilot read this API from a browser, and requiring a
    shared secret there would mean shipping it to every client -- which is a
    different problem, and Phase 11's.
    """
    response = await guarded.get("/api/v1/machines")

    assert response.status_code == 200


async def test_no_token_configured_means_no_check(
    settings: Settings,
    uow_factory: InMemoryUnitOfWorkFactory,
    clock: FixedClock,
    health_probe: StubHealthProbe,
) -> None:
    """Local wiring runs unauthenticated, and only local is allowed to.

    The settings validator refuses to construct a non-local environment without
    a token, so this state is unreachable in a deployment.
    """
    assert settings.ingest_api_token is None
    container = build_in_memory_container(
        settings,
        unit_of_work_factory=uow_factory,
        clock=clock,
        health_probe=health_probe,
    )
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))

    transport = ASGITransport(app=create_app(container))
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/v1/telemetry", json={"records": [a_sample()]})

    assert response.status_code == 201


async def test_the_token_is_not_echoed_in_the_response(guarded: AsyncClient) -> None:
    """A secret in a response body or header is a secret leaked to logs."""
    response = await guarded.post(
        "/api/v1/telemetry",
        json={"records": [a_sample()]},
        headers={HEADER: "wrong"},
    )

    assert TOKEN not in response.text
    assert "wrong" not in response.text
