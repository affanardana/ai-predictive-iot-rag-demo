"""The two services, talking to each other.

Nothing here is faked at the seam being tested: the API's real `HttpPredictor`
posts real JSON to the inference service's real FastAPI app, through ASGI
rather than a socket. That keeps it repeatable and port-free while exercising
the thing a fake cannot — that both sides agree on the wire format.

The model is faked, because a checkpoint cannot live in version control and
loading one would drag 58 MB and a trained artefact into the test suite.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

torch = pytest.importorskip("torch")

from api.composition.container import Container, build_in_memory_container  # noqa: E402
from api.domain.entities.prediction import PREDICTION_WINDOW_READINGS  # noqa: E402
from api.infrastructure.config import Settings  # noqa: E402
from api.infrastructure.persistence.memory.store import InMemoryStore  # noqa: E402
from api.infrastructure.persistence.memory.unit_of_work import (  # noqa: E402
    InMemoryUnitOfWorkFactory,
)
from api.infrastructure.prediction import HttpPredictor  # noqa: E402
from api.presentation.app import create_app  # noqa: E402
from inference.app import create_app as create_inference_app  # noqa: E402
from inference.scorer import Prediction  # noqa: E402
from tests.support.factories import (  # noqa: E402
    DEFAULT_NOW,
    make_machine,
    make_reading,
    make_telemetry,
)
from tests.support.fakes import FixedClock, StubHealthProbe  # noqa: E402

pytestmark = pytest.mark.torch

MODEL_VERSION = "run-20260924-003429"


class StubScorer:
    """Stands in for a loaded checkpoint."""

    def __init__(self, probability: float) -> None:
        self._probability = probability

    @property
    def model_version(self) -> str:
        return MODEL_VERSION

    @property
    def window(self) -> int:
        return PREDICTION_WINDOW_READINGS

    def score(self, readings):
        return Prediction(failure_probability=self._probability, model_version=self.model_version)


def a_window(count: int = PREDICTION_WINDOW_READINGS) -> list[dict[str, float]]:
    return [
        {
            "temperature": 61.5,
            "vibration": 1.42,
            "rpm": 1480.0,
            "current": 12.5,
            "load": 0.72,
            "voltage": 400.0,
        }
        for _ in range(count)
    ]


async def seed_telemetry(factory: InMemoryUnitOfWorkFactory) -> None:
    """Store a full window, the way the pipeline would have before scoring."""
    async with factory() as uow:
        await uow.telemetry.add_many_idempotent(
            [
                make_telemetry(
                    event_id=f"evt-{index:04d}",
                    machine_id="M003",
                    recorded_at=DEFAULT_NOW + timedelta(minutes=index),
                    reading=make_reading(),
                )
                for index in range(PREDICTION_WINDOW_READINGS)
            ]
        )


def an_inference_app(probability: float) -> FastAPI:
    """The service, with a stub scorer already attached.

    `ASGITransport` does not run lifespan events — those belong to a server —
    so the scorer is attached directly rather than by startup. Reaching into
    `state` here is the test doing what the lifespan would have done.
    """
    application = create_inference_app()
    application.state.scorer = StubScorer(probability)
    return application


@pytest.fixture
async def wired(
    settings: Settings,
) -> AsyncIterator[tuple[AsyncClient, float]]:
    """An API client whose predictor is the real adapter, over the real service."""
    probability = 0.83
    inference = an_inference_app(probability)

    # The real adapter, with a client whose transport is the other app.
    predictor = HttpPredictor(
        base_url="http://inference",
        client=AsyncClient(transport=ASGITransport(app=inference)),
    )

    factory = InMemoryUnitOfWorkFactory(store=InMemoryStore())
    container: Container = build_in_memory_container(
        settings,
        unit_of_work_factory=factory,
        clock=FixedClock(),
        health_probe=StubHealthProbe(),
        predictor=predictor,
    )
    async with factory() as uow:
        await uow.machines.add(make_machine("M003"))
    await seed_telemetry(factory)

    transport = ASGITransport(app=create_app(container))
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, probability


async def test_a_stored_window_becomes_a_stored_prediction(
    wired: tuple[AsyncClient, float],
) -> None:
    """The whole path, across both services.

    Telemetry already in the database, read back as a window, out to the model
    service over HTTP, a probability back, classified into a risk band by the
    domain, and persisted.
    """
    client, probability = wired

    response = await client.post("/api/v1/machines/M003/predictions")

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["failure_probability"] == pytest.approx(probability)
    assert body["risk_level"] == "CRITICAL"
    assert body["model_version"] == MODEL_VERSION
    assert body["machine_id"] == "M003"


async def test_the_stored_record_is_readable_afterwards(
    wired: tuple[AsyncClient, float],
) -> None:
    client, _ = wired
    await client.post("/api/v1/machines/M003/predictions")

    history = (await client.get("/api/v1/machines/M003/predictions")).json()

    assert len(history) == 1
    assert history[0]["model_version"] == MODEL_VERSION


async def test_the_service_refuses_a_window_the_api_would_have_caught(
    settings: Settings,
) -> None:
    """Belt and braces across the boundary.

    The API refuses a machine with too little history before it calls the
    service. This posts a short window to the service directly, proving the
    service would also refuse it if something else called it.
    """
    inference = an_inference_app(0.5)
    async with AsyncClient(
        transport=ASGITransport(app=inference), base_url="http://inference"
    ) as client:
        response = await client.post(
            "/predict",
            json={
                "machine_id": "M003",
                "readings": a_window(count=PREDICTION_WINDOW_READINGS - 1),
            },
        )

    assert response.status_code == 422
    assert str(PREDICTION_WINDOW_READINGS) in response.text


async def test_the_service_reports_which_model_is_loaded(
    settings: Settings,
) -> None:
    """The value the API persists comes from here, not from configuration."""
    del settings
    inference = an_inference_app(0.5)
    async with AsyncClient(
        transport=ASGITransport(app=inference), base_url="http://inference"
    ) as client:
        response = await client.get("/health")

    assert response.json()["model_version"] == MODEL_VERSION
