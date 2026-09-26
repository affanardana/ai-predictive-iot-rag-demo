"""The ASGI app.

A plain FastAPI application, so it runs under uvicorn locally and can be wrapped
by whichever host is chosen. Nothing here is specific to a platform.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status

from inference.schemas import PredictRequest, PredictResponse
from inference.scorer import InsufficientHistoryError, Scorable, Scorer
from inference.settings import Settings


@dataclass(frozen=True, slots=True)
class Health:
    """What the health endpoint reports."""

    status: str
    model_version: str
    window: int


def create_app(scorer: Scorable | None = None) -> FastAPI:
    """Build the application.

    `scorer` exists so tests can inject one without a checkpoint on disk. When
    it is None the real one is built at startup from the environment, which is
    what a deployment does.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.scorer = scorer or Scorer(Settings.from_environment())
        yield

    application = FastAPI(
        title="Predictive maintenance inference",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    async def health(request: Request) -> Health:
        """Report readiness, including which model is loaded."""
        loaded: Scorable = request.app.state.scorer
        return Health(status="ok", model_version=loaded.model_version, window=loaded.window)

    @application.post("/predict", response_model=PredictResponse)
    async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        """Score one window of telemetry."""
        loaded: Scorable = request.app.state.scorer
        rows = [reading.as_row() for reading in payload.readings]
        try:
            result = loaded.score(rows)
        except InsufficientHistoryError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        return PredictResponse(
            failure_probability=result.failure_probability,
            model_version=result.model_version,
        )

    return application


def app_factory() -> FastAPI:
    """Entry point for `uvicorn inference.app:app_factory --factory`."""
    return create_app()
