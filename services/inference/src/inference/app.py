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

from inference.embedder import EMBEDDING_DIMENSIONS, Embedder, SentenceEmbedder
from inference.reranker import CrossEncoderReranker, Reranker
from inference.schemas import (
    EmbedRequest,
    EmbedResponse,
    PredictRequest,
    PredictResponse,
    RerankHit,
    RerankRequest,
    RerankResponse,
)
from inference.scorer import InsufficientHistoryError, Scorable, Scorer
from inference.settings import Settings


@dataclass(frozen=True, slots=True)
class Health:
    """What the health endpoint reports."""

    status: str
    model_version: str
    window: int
    embed_model: str
    rerank_model: str


def create_app(
    scorer: Scorable | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    """Build the application.

    The three models are injected independently so tests can supply stubs. A
    test that injects one must inject all three: any left as None is loaded from
    disk at startup, which is what a deployment does and what a test does not
    want.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        provided = (scorer, embedder, reranker)
        if all(model is not None for model in provided):
            application.state.scorer, application.state.embedder, application.state.reranker = (
                provided
            )
        else:
            settings = Settings.from_environment()
            application.state.scorer = scorer or Scorer(settings)
            application.state.embedder = embedder or SentenceEmbedder(settings)
            application.state.reranker = reranker or CrossEncoderReranker(settings)
        yield

    application = FastAPI(
        title="Predictive maintenance inference",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    async def health(request: Request) -> Health:
        """Report readiness, including which models are loaded."""
        loaded: Scorable = request.app.state.scorer
        encoder: Embedder = request.app.state.embedder
        orderer: Reranker = request.app.state.reranker
        return Health(
            status="ok",
            model_version=loaded.model_version,
            window=loaded.window,
            embed_model=encoder.model_id,
            rerank_model=orderer.model_id,
        )

    @application.post("/embed", response_model=EmbedResponse)
    async def embed(payload: EmbedRequest, request: Request) -> EmbedResponse:
        """Embed a batch of texts, in order.

        An empty batch is an empty answer rather than an error: the caller
        decides what an empty document means.
        """
        loaded: Embedder = request.app.state.embedder
        try:
            vectors = loaded.embed(payload.texts)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        return EmbedResponse(
            model=loaded.model_id,
            # For an empty batch there is nothing to measure, so the model's own
            # width is reported rather than a zero the caller might store.
            dimensions=len(vectors[0]) if vectors else EMBEDDING_DIMENSIONS,
            embeddings=vectors,
        )

    @application.post("/rerank", response_model=RerankResponse)
    async def rerank(payload: RerankRequest, request: Request) -> RerankResponse:
        """Order candidate passages against a query, best first."""
        loaded: Reranker = request.app.state.reranker
        try:
            results = loaded.rank(payload.query, payload.documents, payload.limit)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        return RerankResponse(
            results=[RerankHit(index=result.index, score=result.score) for result in results]
        )

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
