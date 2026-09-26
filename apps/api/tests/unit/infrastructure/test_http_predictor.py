"""The HTTP adapter, against a mock transport.

No network and no service running: `httpx.MockTransport` answers in-process.
What is being checked is the wire format going out and the error translation
coming back, which are the two things a fake `Predictor` cannot exercise.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from api.domain.errors import PredictionUnavailableError
from api.domain.value_objects.sensor_reading import SensorReading
from api.infrastructure.prediction import HttpPredictor


def a_reading(**overrides: float) -> SensorReading:
    """Return one reading with distinct signal values, so a transposition shows."""
    values = dict.fromkeys(SensorReading.signal_names(), 1.0)
    values.update({"temperature": 61.5, "vibration": 1.42, "rpm": 1480.0})
    values.update(overrides)
    return SensorReading(**values)


def a_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"failure_probability": 0.81, "model_version": "run-test"})


async def test_a_successful_call_returns_the_output() -> None:
    predictor = HttpPredictor(base_url="http://model", client=a_client(ok))

    result = await predictor.predict([a_reading()])

    assert result.failure_probability == pytest.approx(0.81)
    assert result.model_version == "run-test"


async def test_the_payload_uses_the_domains_signal_names() -> None:
    """The wire format comes from `signal_names`, so it cannot drift.

    It is the same tuple the API validates against, so a transposed
    temperature — which would be silent and would poison every score — cannot
    happen by editing one side only.
    """
    seen: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return ok(request)

    await HttpPredictor(base_url="http://model", client=a_client(capture)).predict(
        [a_reading(), a_reading(vibration=2.0)]
    )

    readings = seen[0]["readings"]
    assert len(readings) == 2
    assert readings[0]["temperature"] == pytest.approx(61.5)
    assert readings[1]["vibration"] == pytest.approx(2.0)
    assert set(readings[0]) == set(SensorReading.signal_names())


async def test_it_posts_to_the_models_predict_path() -> None:
    seen: list[str] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return ok(request)

    await HttpPredictor(base_url="http://model/", client=a_client(capture)).predict([a_reading()])

    assert seen == ["http://model/predict"]


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(lambda r: httpx.Response(500, text="boom"), id="server-error"),
        pytest.param(lambda r: httpx.Response(422, json={"detail": "bad"}), id="unprocessable"),
        pytest.param(lambda r: httpx.Response(200, json={"unexpected": 1}), id="wrong-fields"),
        pytest.param(lambda r: httpx.Response(200, text="not json"), id="not-json"),
    ],
)
async def test_every_bad_answer_becomes_one_domain_error(
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    """The application should not have to know what httpx raises."""
    predictor = HttpPredictor(base_url="http://model", client=a_client(handler))

    with pytest.raises(PredictionUnavailableError):
        await predictor.predict([a_reading()])


async def test_a_connection_failure_becomes_the_same_domain_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    predictor = HttpPredictor(base_url="http://model", client=a_client(refuse))

    with pytest.raises(PredictionUnavailableError, match="could not be reached"):
        await predictor.predict([a_reading()])


async def test_a_probability_outside_the_unit_range_is_refused() -> None:
    """A model that returns 1.7 has not returned a probability.

    Catching it here means the domain never sees it, and the database CHECK
    constraint is a backstop rather than the first line of defence.
    """
    handler = lambda r: httpx.Response(  # noqa: E731 - one-line test double
        200, json={"failure_probability": 1.7, "model_version": "run-test"}
    )
    predictor = HttpPredictor(base_url="http://model", client=a_client(handler))

    with pytest.raises(PredictionUnavailableError, match="not a probability"):
        await predictor.predict([a_reading()])
