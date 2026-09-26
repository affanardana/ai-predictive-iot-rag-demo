"""The embedding adapter, against a mock transport.

No network and no service: `httpx.MockTransport` answers in-process. What is
being checked is the wire format going out and the error translation coming
back, which a stub `Embedder` cannot exercise.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from api.domain.entities.knowledge_document import EMBEDDING_DIMENSIONS
from api.domain.errors import EmbeddingModelMismatchError, RetrievalUnavailableError
from api.infrastructure.embedding import HttpEmbedder

MODEL = "sentence-transformers/all-MiniLM-L6-v2@abc123"


def a_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def a_vector(first: float = 1.0) -> list[float]:
    """Return one vector of the width the corpus is stored at."""
    return [first, *([0.0] * (EMBEDDING_DIMENSIONS - 1))]


def ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"model": MODEL, "dimensions": 384, "embeddings": [a_vector()]})


def an_embedder(client: httpx.AsyncClient) -> HttpEmbedder:
    return HttpEmbedder(base_url="http://model", client=client, model_id=MODEL)


async def test_a_successful_call_returns_the_vectors() -> None:
    embedder = an_embedder(a_client(ok))

    vectors = await embedder.embed(["a passage"])

    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIMENSIONS


async def test_the_texts_go_out_as_a_batch() -> None:
    """One request for the whole batch: the cost is the forward pass, not the items."""
    seen: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "dimensions": 384,
                "embeddings": [a_vector(0.0), a_vector(1.0)],
            },
        )

    embedder = an_embedder(a_client(capture))

    await embedder.embed(["first", "second"])

    assert seen == [{"texts": ["first", "second"]}]


async def test_an_empty_batch_is_not_sent() -> None:
    """Nothing to embed is nothing to ask, and no call to fail."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made for an empty batch")

    embedder = an_embedder(a_client(refuse))

    assert await embedder.embed([]) == ()


async def test_an_unreachable_service_is_a_retrieval_outage() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    embedder = an_embedder(a_client(fail))

    with pytest.raises(RetrievalUnavailableError):
        await embedder.embed(["a passage"])


async def test_an_error_status_is_a_retrieval_outage() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "longer than the model can read"})

    embedder = an_embedder(a_client(refuse))

    with pytest.raises(RetrievalUnavailableError):
        await embedder.embed(["a passage"])


async def test_a_different_model_is_refused_rather_than_stored() -> None:
    """Vectors from two models are not comparable, and mixing them is silent.

    This is the failure the identity guard exists for: swapping the model would
    otherwise produce plausible-looking rankings with nothing logged.
    """

    def other_model(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "some-other-encoder@x", "dimensions": 384, "embeddings": [a_vector()]},
        )

    embedder = an_embedder(a_client(other_model))

    with pytest.raises(EmbeddingModelMismatchError):
        await embedder.embed(["a passage"])


async def test_a_short_vector_is_refused() -> None:
    """The stored column is `vector(384)`, so a narrower one cannot be written."""

    def narrow(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"model": MODEL, "dimensions": 3, "embeddings": [[1.0, 0.0, 0.0]]}
        )

    embedder = an_embedder(a_client(narrow))

    with pytest.raises(RetrievalUnavailableError):
        await embedder.embed(["a passage"])


async def test_a_missing_vector_is_refused() -> None:
    """Two texts and one vector cannot be matched up, so neither is stored."""

    def too_few(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"model": MODEL, "dimensions": 384, "embeddings": [a_vector()]}
        )

    embedder = an_embedder(a_client(too_few))

    with pytest.raises(RetrievalUnavailableError):
        await embedder.embed(["first", "second"])


async def test_the_model_identity_is_reported_for_storage() -> None:
    """Every chunk records it, so retrieval can refuse to mix models."""
    assert an_embedder(a_client(ok)).model_id == MODEL
