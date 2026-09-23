"""Health endpoints."""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from api.composition import build_in_memory_container
from api.composition.container import Container
from api.infrastructure.config import Settings
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from api.presentation.app import create_app
from tests.support.fakes import StubHealthProbe


async def test_liveness_reports_ok(client: AsyncClient) -> None:
    """The process reports itself alive."""
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_env"] == "local"


async def test_readiness_reports_ok_when_dependencies_are_healthy(
    client: AsyncClient, health_probe: StubHealthProbe
) -> None:
    """Readiness checks the dependency, unlike liveness."""
    response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert health_probe.check_count == 1


async def test_readiness_reports_degraded_when_the_database_is_unreachable(
    settings: Settings, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """An unreachable dependency yields 503 rather than 200.

    Readiness answers "should this instance receive traffic?"; an instance that
    cannot reach its database should be taken out of rotation, not restarted.
    """
    container: Container = build_in_memory_container(
        settings,
        unit_of_work_factory=uow_factory,
        health_probe=StubHealthProbe(healthy=False, detail="The database is not reachable."),
    )
    app = create_app(container)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["detail"] == "The database is not reachable."


async def test_liveness_does_not_depend_on_the_database(
    settings: Settings, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Liveness stays green during a database outage.

    A database outage is not fixed by restarting the API, so liveness must not
    fail alongside it -- otherwise an orchestrator restarts a healthy process
    in a loop.
    """
    probe = StubHealthProbe(healthy=False, detail="down")
    container = build_in_memory_container(
        settings, unit_of_work_factory=uow_factory, health_probe=probe
    )
    app = create_app(container)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert probe.check_count == 0
