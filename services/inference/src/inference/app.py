"""The ASGI app.

A plain FastAPI application, so it runs under uvicorn locally and can be wrapped
by whichever host is chosen. Nothing here is specific to a platform.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from inference.chat import ChatModel, LlamaChat, Turn
from inference.embedder import EMBEDDING_DIMENSIONS, Embedder, SentenceEmbedder
from inference.reranker import CrossEncoderReranker, Reranker
from inference.schemas import (
    ChatRequest,
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
    """What the health endpoint reports.

    `chat_model` is nullable where the other three are not, because the
    Copilot's model is the one this service can run without — and a deployment
    that has not configured it should be able to say so rather than reporting a
    model it does not have.
    """

    status: str
    model_version: str
    window: int
    embed_model: str
    rerank_model: str
    chat_model: str | None


def create_app(
    scorer: Scorable | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    chat_model: ChatModel | None = None,
) -> FastAPI:
    """Build the application.

    The models are injected independently so tests can supply stubs. A test that
    injects one must inject all of them: any left as None is loaded from disk at
    startup, which is what a deployment does and what a test does not want.

    The chat model is the exception, and it is a difference in kind rather than
    an oversight: it is optional, so a deployment without one starts normally
    and says so at `/chat`. That is deliberate — it is the hungriest model here
    and the least essential, and a service that refused to start without it
    would take prediction and retrieval down with it on a box with no swap.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        provided = (scorer, embedder, reranker)
        if all(model is not None for model in provided):
            application.state.scorer, application.state.embedder, application.state.reranker = (
                provided
            )
            application.state.chat = chat_model
        else:
            settings = Settings.from_environment()
            application.state.scorer = scorer or Scorer(settings)
            application.state.embedder = embedder or SentenceEmbedder(settings)
            application.state.reranker = reranker or CrossEncoderReranker(settings)
            application.state.chat = chat_model or (
                LlamaChat(settings) if settings.chat_model is not None else None
            )
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
        writer: ChatModel | None = request.app.state.chat
        return Health(
            status="ok",
            model_version=loaded.model_version,
            window=loaded.window,
            embed_model=encoder.model_id,
            rerank_model=orderer.model_id,
            chat_model=writer.model_id if writer is not None else None,
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

    @application.post("/chat")
    async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
        """Write an answer from the evidence it is given, as it is written.

        **Newline-delimited JSON rather than server-sent events**, and the
        difference is the audience. This hop has exactly one consumer — the API,
        over a connection it opened — so the format wants to be trivial to parse
        and to validate: one complete JSON object per line, handed to
        `json.loads` with no frame grammar to get wrong. The browser-facing hop
        speaks SSE because `EventSource` is the browser's own vocabulary, and the
        API translates between them.

        The stream ends at the last line. Nothing is buffered: on this box the
        answer takes about twenty seconds to write, and holding it until the end
        would throw away the only reason to stream it.
        """
        loaded: ChatModel | None = request.app.state.chat
        if loaded is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "No chat model is configured on this service, so the Copilot "
                    "cannot write an answer. Set INFERENCE_CHAT_MODEL."
                ),
            )

        turns = [Turn(role=turn.role, content=turn.content) for turn in payload.messages]
        return StreamingResponse(
            _chunks(loaded, turns, payload.max_tokens),
            media_type="application/x-ndjson",
            headers={
                # Nothing between here and the API may hold a partially written
                # answer, for the same reason the event stream says so.
                "Cache-Control": "no-cache, no-store",
                "X-Accel-Buffering": "no",
            },
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


def _chunks(model: ChatModel, turns: Sequence[Turn], max_tokens: int) -> Iterator[str]:
    """Render the answer as one JSON object per line.

    Errors are not caught here. A model that fails mid-answer closes the stream,
    and the API -- which has already been streaming frames to a browser -- is
    the layer that knows what a reader should be told about that. Reporting it
    from inside the generator would mean inventing a second error shape on a
    channel that has no status codes left.
    """
    for piece in model.stream(turns, max_tokens=max_tokens):
        yield json.dumps({"text": piece}, ensure_ascii=False) + "\n"
