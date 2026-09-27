"""The rate limits, over the real routes.

The peer address is set explicitly on every client here, and it is the **Docker
bridge gateway** rather than `127.0.0.1`. That is what the container sees in
production, and httpx's default peer is loopback -- so a test that omits it
would pass while the deployment keyed every visitor into one bucket.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from api.composition.container import build_in_memory_container
from api.infrastructure.config import Settings
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from api.presentation.app import create_app
from tests.support.factories import make_machine
from tests.support.fakes import FixedClock, StubHealthProbe

#: What the API sees as its peer in production: the bridge, not the client.
GATEWAY = ("172.17.0.1", 31337)

#: The address Caddy forwards on behalf of.
CLIENT = "203.0.113.9"


def client_settings(
    *,
    global_per_minute: int = 1,
    copilot_per_minute: int = 1,
    simulation_per_minute: int = 1,
    search_per_minute: int = 1,
) -> Settings:
    """Settings with small limits.

    Named parameters rather than `**overrides`: `Settings` takes a `_env_file`
    keyword that `**kwargs` cannot be typed against, and mypy is right to
    complain about the ambiguity.
    """
    return Settings(
        _env_file=None,
        rate_limit_global_per_minute=global_per_minute,
        rate_limit_copilot_per_minute=copilot_per_minute,
        rate_limit_simulation_per_minute=simulation_per_minute,
        rate_limit_search_per_minute=search_per_minute,
    )


@pytest.fixture
async def limited(
    uow_factory: InMemoryUnitOfWorkFactory,
    clock: FixedClock,
    health_probe: StubHealthProbe,
) -> AsyncIterator[AsyncClient]:
    """A client over a container whose budgets are tiny and known."""
    container = build_in_memory_container(
        client_settings(),
        unit_of_work_factory=uow_factory,
        clock=clock,
        health_probe=health_probe,
    )
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))

    async with AsyncClient(
        transport=ASGITransport(app=create_app(container), client=GATEWAY),
        base_url="http://testserver",
    ) as client:
        yield client


def forwarded(address: str = CLIENT) -> dict[str, str]:
    """The header Caddy adds, which is how a caller is recognised as external."""
    return {"X-Forwarded-For": address}


#: The global burst, from `_rate_limiter` in `app.py`. A constant rather than a
#: setting, because a burst is a shape decision -- how many requests a page load
#: fires at once -- while the rate is a property of the host. Spelled out here
#: so that changing it is a decision rather than a surprise in a test.
GLOBAL_BURST = 120


async def test_an_exhausted_budget_is_a_429_in_the_shared_envelope(
    limited: AsyncClient,
) -> None:
    """Same envelope, same fields, one extra header."""
    for _ in range(GLOBAL_BURST):
        assert (await limited.get("/api/v1/machines", headers=forwarded())).status_code == 200

    response = await limited.get("/api/v1/machines", headers=forwarded())

    assert response.status_code == 429
    body = response.json()
    assert set(body["error"]) == {"code", "message", "details"}
    assert body["error"]["code"] == "rate_limited"
    assert body["error"]["details"]["limit"] == "global"
    assert int(response.headers["retry-after"]) >= 1


async def test_a_different_address_has_its_own_budget(limited: AsyncClient) -> None:
    """The reason the whole client-IP chain exists."""
    for _ in range(GLOBAL_BURST + 1):
        await limited.get("/api/v1/machines", headers=forwarded())
    assert (await limited.get("/api/v1/machines", headers=forwarded())).status_code == 429

    assert (
        await limited.get("/api/v1/machines", headers=forwarded("198.51.100.4"))
    ).status_code == 200


async def test_the_three_expensive_routes_have_independent_budgets(
    limited: AsyncClient,
) -> None:
    """A spent Copilot budget does not refuse a search."""
    question = {"question": "What should I inspect on M003?"}

    for _ in range(2):
        await limited.post("/api/v1/copilot/chat", json=question, headers=forwarded())
    refused = await limited.post("/api/v1/copilot/chat", json=question, headers=forwarded())
    assert refused.status_code == 429
    assert refused.json()["error"]["details"]["limit"] == "copilot"

    searched = await limited.post(
        "/api/v1/knowledge/search", json={"query": "bearing inspection"}, headers=forwarded()
    )
    assert searched.status_code != 429


async def test_the_pipeline_is_never_charged(limited: AsyncClient) -> None:
    """n8n writes telemetry from inside the network, at about one a second.

    Metering that would return 429 to the ingest route -- dropped readings and
    holes in the charts -- which is a far worse failure than a slow dashboard.
    An in-network caller is recognised by the *absence* of a forwarded header,
    which Caddy always adds; the routes it uses are token-guarded separately.

    Asserted past the burst rather than at it, because the failure this guards
    against appears only after a few minutes of a running simulation.
    """
    for _ in range(200):
        response = await limited.post(
            "/api/v1/telemetry",
            json={"records": []},
            headers={"X-Ingest-Token": "unused-here"},
        )
        assert response.status_code != 429


async def test_health_and_metrics_are_never_charged(limited: AsyncClient) -> None:
    """A limiter that can fail a healthcheck is a limiter that restarts things.

    The container checks `/health/ready` every thirty seconds from inside, and a
    scraper hits `/metrics` on its own schedule. Both are exempt.
    """
    for _ in range(50):
        assert (await limited.get("/health/ready")).status_code == 200
        assert (await limited.get("/metrics")).status_code == 200


# The event stream's exemption is asserted in
# `tests/unit/presentation/test_rate_limit_middleware.py`, not here: opening a
# stream from a test means awaiting a response that is designed never to end.
# The first version of this file did exactly that and hung.
