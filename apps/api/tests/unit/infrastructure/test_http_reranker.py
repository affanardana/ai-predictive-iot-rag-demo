"""The reranking adapter, against a mock transport."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from api.domain.errors import RetrievalUnavailableError
from api.infrastructure.reranking import HttpReranker


def a_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"results": [{"index": 1, "score": 4.5}]})


async def test_a_successful_call_returns_indices_and_scores() -> None:
    reranker = HttpReranker(base_url="http://model", client=a_client(ok))

    results = await reranker.rank("vibration", ["isolation", "bearing"], limit=1)

    assert [(result.index, result.score) for result in results] == [(1, 4.5)]


async def test_the_request_carries_the_query_candidates_and_limit() -> None:
    """The limit is sent rather than applied here, so the model scores only what is needed."""
    seen: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"results": []})

    reranker = HttpReranker(base_url="http://model", client=a_client(capture))

    await reranker.rank("vibration", ["isolation", "bearing"], limit=2)

    assert seen == [{"query": "vibration", "documents": ["isolation", "bearing"], "limit": 2}]


async def test_no_candidates_is_not_a_call() -> None:
    """Nothing to order, and nothing to fail."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made with no candidates")

    reranker = HttpReranker(base_url="http://model", client=a_client(refuse))

    assert await reranker.rank("vibration", [], limit=5) == ()


async def test_an_unreachable_service_is_a_retrieval_outage() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    reranker = HttpReranker(base_url="http://model", client=a_client(fail))

    with pytest.raises(RetrievalUnavailableError):
        await reranker.rank("vibration", ["isolation"], limit=1)


async def test_an_index_outside_the_batch_is_refused() -> None:
    """An index with no candidate would score the wrong passage, or crash later.

    Refused here rather than mapped onto something: a mismatched index would
    attach a relevance score to a passage the model never read.
    """

    def out_of_range(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 7, "score": 1.0}]})

    reranker = HttpReranker(base_url="http://model", client=a_client(out_of_range))

    with pytest.raises(RetrievalUnavailableError):
        await reranker.rank("vibration", ["isolation"], limit=1)
