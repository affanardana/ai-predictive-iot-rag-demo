"""Talks to the inference service over HTTP.

Every failure — a refused connection, a timeout, a 422, a body that is not a
probability — becomes `PredictionUnavailableError`. The application should not
have to know what `httpx` raises, and the request handler should not turn
someone else's outage into a 500 on our side.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

from api.domain.errors import PredictionUnavailableError
from api.domain.ports.predictor import ModelOutput
from api.domain.value_objects.sensor_reading import SensorReading

logger = logging.getLogger(__name__)

#: Generous, because the first request after a cold start loads a model.
DEFAULT_TIMEOUT_SECONDS = 15.0


class HttpPredictor:
    """A `Predictor` backed by the inference service."""

    def __init__(
        self,
        *,
        base_url: str,
        client: httpx.AsyncClient,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    async def predict(self, readings: Sequence[SensorReading]) -> ModelOutput:
        """Score one window of readings.

        Raises:
            PredictionUnavailableError: if the service cannot be reached, or
                answers with anything that is not a probability.
        """
        payload = {"readings": [_serialise(reading) for reading in readings]}

        try:
            response = await self._client.post(
                f"{self._base_url}/predict", json=payload, timeout=self._timeout
            )
            response.raise_for_status()
            body = response.json()
            probability = float(body["failure_probability"])
            model_version = str(body["model_version"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Inference call failed: %s", type(exc).__name__)
            raise PredictionUnavailableError(
                "The model service could not be reached or returned an unusable response."
            ) from exc

        if not 0.0 <= probability <= 1.0:
            raise PredictionUnavailableError(
                f"The model returned {probability}, which is not a probability."
            )
        return ModelOutput(failure_probability=probability, model_version=model_version)


def _serialise(reading: SensorReading) -> dict[str, float]:
    """Return one reading as the service expects it.

    The field names come from the domain's own `signal_names`, so the wire
    format cannot drift from the signals the API accepts.
    """
    return {name: float(getattr(reading, name)) for name in SensorReading.signal_names()}
