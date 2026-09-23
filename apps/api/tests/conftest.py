"""Shared fixtures for the whole suite."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from api.composition import build_in_memory_container
from api.composition.container import Container
from api.infrastructure.config import Settings
from api.infrastructure.event_loop import use_psycopg_compatible_event_loop
from api.infrastructure.persistence.memory.store import InMemoryStore
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from api.presentation.app import create_app
from tests.support.fakes import FixedClock, StubHealthProbe

# Import time, before pytest-asyncio builds any loop. Without this the
# `postgres` tier cannot run on Windows: psycopg 3 rejects the Proactor loop
# that Windows uses by default.
use_psycopg_compatible_event_loop()

#: An in-memory URL so constructing `Settings` never reaches for a real
#: database, and so the suite runs with no credentials and no network.
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(autouse=True)
def _hermetic_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin configuration for every test in the suite.

    Environment variables outrank `.env` in pydantic-settings' source order, so
    setting them here means a developer's local `.env` cannot leak into a test
    run -- which is what makes `Settings()` behave the same on a laptop as on
    CI.

    Autouse because a single test that forgot it would read a real connection
    string, and the failure would look like a flaky test rather than a missing
    fixture.

    `TEST_DATABASE_URL` is deliberately left alone. It is the opt-in that enables
    the `postgres`-marked tier, so clearing it here would silently skip every one
    of those tests -- which is precisely what happened before this was noticed:
    the whole tier reported as skipped and looked like a pass. A test that needs
    it unset clears it locally instead.
    """
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)


@pytest.fixture
def settings() -> Settings:
    """Settings built through the real loading path, with `.env` excluded.

    The autouse fixture supplies what this needs through the environment, which
    is read exactly as in production. Excluding `.env` is what stops the suite
    depending on a file that only exists on a developer's machine -- without it
    the tests pass locally and fail in CI, for reasons that look unrelated.
    """
    return Settings(_env_file=None)


@pytest.fixture
def store() -> InMemoryStore:
    """An empty in-memory store."""
    return InMemoryStore()


@pytest.fixture
def uow_factory(store: InMemoryStore) -> InMemoryUnitOfWorkFactory:
    """A unit-of-work factory over the shared in-memory store."""
    return InMemoryUnitOfWorkFactory(store)


@pytest.fixture
def clock() -> FixedClock:
    """A clock pinned to a deterministic instant."""
    return FixedClock()


@pytest.fixture
def health_probe() -> StubHealthProbe:
    """A health probe reporting healthy."""
    return StubHealthProbe()


@pytest.fixture
def container(
    settings: Settings,
    uow_factory: InMemoryUnitOfWorkFactory,
    clock: FixedClock,
    health_probe: StubHealthProbe,
) -> Container:
    """A fully wired container backed by the in-memory store."""
    return build_in_memory_container(
        settings,
        unit_of_work_factory=uow_factory,
        clock=clock,
        health_probe=health_probe,
    )


@pytest.fixture
async def client(container: Container) -> AsyncIterator[AsyncClient]:
    """An HTTP client bound to the application.

    The container is attached directly to `app.state` rather than letting the
    lifespan build it, so tests never open a database engine and the shutdown
    hook has nothing to dispose.
    """
    app = create_app(container)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        yield http_client
