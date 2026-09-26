"""Recording a prediction, end to end through the API.

The predictor is faked, so the API's behaviour is tested without the inference
service running, without a model on disk and without torch. That is the whole
reason the use case depends on a port.

The window is no longer posted: it is read from stored telemetry, so these
tests seed readings the way the pipeline does and then ask for a prediction.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from api.composition.container import Container, build_in_memory_container
from api.domain.entities.prediction import PREDICTION_WINDOW_READINGS
from api.domain.errors import PredictionUnavailableError
from api.domain.ports.predictor import ModelOutput
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.sensor_reading import SensorReading
from api.infrastructure.config import Settings
from api.infrastructure.persistence.memory.store import InMemoryStore
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from api.presentation.app import create_app
from tests.support.factories import DEFAULT_NOW, make_machine, make_reading, make_telemetry
from tests.support.fakes import FixedClock, StubHealthProbe


@dataclass
class FakePredictor:
    """A predictor that returns what it is told, and records what it was asked."""

    probability: float = 0.42
    model_version: str = "run-test"
    fail: bool = False
    seen: list[list[SensorReading]] = field(default_factory=list)

    async def predict(self, readings):
        self.seen.append(list(readings))
        if self.fail:
            raise PredictionUnavailableError("inference is down")
        return ModelOutput(failure_probability=self.probability, model_version=self.model_version)


@dataclass
class Wired:
    """A client over a seedable store, with the fake behind it."""

    client: AsyncClient
    predictor: FakePredictor
    factory: InMemoryUnitOfWorkFactory


async def seed_telemetry(
    factory: InMemoryUnitOfWorkFactory,
    count: int = PREDICTION_WINDOW_READINGS,
    machine_id: str = "M003",
) -> None:
    """Store `count` readings for `machine_id`, oldest first.

    Temperature carries the index so a test can assert the order the model
    received them in.
    """
    async with factory() as uow:
        await uow.telemetry.add_many_idempotent(
            [
                make_telemetry(
                    event_id=f"evt-{index:04d}",
                    machine_id=machine_id,
                    recorded_at=DEFAULT_NOW + timedelta(minutes=index),
                    reading=make_reading(temperature=float(index)),
                )
                for index in range(count)
            ]
        )


async def wire(
    settings: Settings,
    predictor: FakePredictor,
    *,
    telemetry: int = PREDICTION_WINDOW_READINGS,
    register: bool = True,
) -> Wired:
    """Build a client over an in-memory store holding a machine and its history."""
    factory = InMemoryUnitOfWorkFactory(store=InMemoryStore())
    container = build_in_memory_container(
        settings,
        unit_of_work_factory=factory,
        clock=FixedClock(),
        health_probe=StubHealthProbe(),
        predictor=predictor,
    )
    if register:
        async with factory() as uow:
            await uow.machines.add(make_machine("M003"))
        await seed_telemetry(factory, telemetry)

    transport = ASGITransport(app=create_app(container))
    return Wired(
        client=AsyncClient(transport=transport, base_url="http://testserver"),
        predictor=predictor,
        factory=factory,
    )


@pytest.fixture
async def wired(settings: Settings) -> AsyncIterator[Wired]:
    """A client whose container is wired to a fake predictor, with a full window."""
    instance = await wire(settings, FakePredictor())
    async with instance.client:
        yield instance


async def test_a_prediction_is_recorded_and_returned(wired: Wired) -> None:
    response = await wired.client.post("/api/v1/machines/M003/predictions")

    assert response.status_code == 201
    body = response.json()
    assert body["machine_id"] == "M003"
    assert body["failure_probability"] == pytest.approx(0.42)
    assert body["model_version"] == "run-test"
    assert body["horizon_seconds"] == 3600


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.29, "NORMAL"),
        (0.30, "WARNING"),
        (0.59, "WARNING"),
        (0.60, "HIGH"),
        (0.79, "HIGH"),
        (0.80, "CRITICAL"),
        (1.00, "CRITICAL"),
    ],
)
async def test_the_risk_band_comes_from_the_domains_own_classifier(
    settings: Settings, probability: float, expected: str
) -> None:
    """PRD section 9's edges, exercised where they are applied.

    The model returns a probability and nothing else; the band is the product's
    decision, and this is the only place the two meet.
    """
    instance = await wire(settings, FakePredictor(probability=probability))

    async with instance.client as client:
        response = await client.post("/api/v1/machines/M003/predictions")

    assert response.json()["risk_level"] == expected


async def test_the_record_matches_the_accepted_contract(wired: Wired) -> None:
    """AC-002's six fields, which the dashboard and the copilot read later."""
    body = (await wired.client.post("/api/v1/machines/M003/predictions")).json()

    assert set(body) >= {
        "machine_id",
        "predicted_at",
        "failure_probability",
        "risk_level",
        "model_version",
    }
    assert body["horizon_seconds"] == 3600


async def test_it_is_queryable_through_the_history_endpoint(wired: Wired) -> None:
    """Written and read back through the paths a client actually uses."""
    await wired.client.post("/api/v1/machines/M003/predictions")

    history = await wired.client.get("/api/v1/machines/M003/predictions")

    assert history.status_code == 200
    assert len(history.json()) == 1
    assert history.json()[0]["model_version"] == "run-test"


async def test_an_unknown_machine_is_a_404(settings: Settings) -> None:
    """Refused before inference runs, so a typo costs no model call."""
    instance = await wire(settings, FakePredictor(), register=False)

    async with instance.client as client:
        response = await client.post("/api/v1/machines/M999/predictions")

    assert response.status_code == 404
    assert instance.predictor.seen == []


async def test_a_short_history_is_refused_before_inference(settings: Settings) -> None:
    """The cold-start rule, now applied to stored telemetry.

    A machine with 59 readings has nothing to score. Padding would invent data
    and scoring a short window would feed the model an input shape it was never
    trained on, so the request is refused with the counts in the message.
    """
    instance = await wire(settings, FakePredictor(), telemetry=PREDICTION_WINDOW_READINGS - 1)

    async with instance.client as client:
        response = await client.post("/api/v1/machines/M003/predictions")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "insufficient_history"
    assert f"{PREDICTION_WINDOW_READINGS - 1} arrived" in response.json()["error"]["message"]
    assert instance.predictor.seen == []


async def test_an_inference_outage_is_a_503_not_a_500(settings: Settings) -> None:
    """A dependency outage, reported as one.

    The request was well formed and the machine exists; the system could not
    answer. 500 would say the API is broken, which would send a caller looking
    in the wrong place and would not suggest a retry.
    """
    instance = await wire(settings, FakePredictor(fail=True))

    async with instance.client as client:
        response = await client.post("/api/v1/machines/M003/predictions")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "inference_unavailable"


async def test_the_window_reaches_the_predictor_in_order(wired: Wired) -> None:
    """Sixty readings, oldest first, in the order they were recorded.

    Ordering is the API's responsibility now rather than a caller's, which is
    the point of reading the window from storage: a caller cannot get it wrong,
    so the only question is whether the API does.
    """
    await wired.client.post("/api/v1/machines/M003/predictions")

    sent = wired.predictor.seen[0]
    assert len(sent) == PREDICTION_WINDOW_READINGS
    assert [reading.temperature for reading in sent] == [
        float(index) for index in range(PREDICTION_WINDOW_READINGS)
    ]


async def test_the_default_container_refuses_rather_than_inventing_a_number(
    container: Container,
) -> None:
    """With no service configured, a prediction is refused.

    A placeholder that returned a fixed probability would be persisted as a
    real prediction, which is worse than an error.
    """
    from api.application.use_cases import RecordPrediction

    use_case: RecordPrediction = container.record_prediction
    with pytest.raises(PredictionUnavailableError, match="INFERENCE_SERVICE_URL"):
        await use_case.predictor.predict([])


def test_the_machine_id_type_is_the_one_the_api_uses() -> None:
    """A reminder that the route converts at the boundary, not inside."""
    assert str(MachineId("M003")) == "M003"
