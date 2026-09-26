"""Talks to the inference service's reranking endpoint.

A reranker outage is reported as an outage rather than absorbed. Falling back to
the vector order would hide the failure and quietly make "reranking is
implemented" untrue -- the same choice `UnavailablePredictor` documents.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

from api.domain.errors import RetrievalUnavailableError
from api.domain.ports.reranker import RerankResult

logger = logging.getLogger(__name__)

#: Longer than a prediction call: the cross-encoder scores every candidate, in
#: one batch, on a machine that has one core.
DEFAULT_TIMEOUT_SECONDS = 60.0


class HttpReranker:
    """A `Reranker` backed by the inference service."""

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

    async def rank(
        self,
        query: str,
        documents: Sequence[str],
        limit: int,
    ) -> Sequence[RerankResult]:
        """Score `documents` against `query`, best first.

        Raises:
            RetrievalUnavailableError: the service could not be reached, or
                answered with indices that do not name the candidates sent.
        """
        if not documents or limit <= 0:
            return ()

        try:
            response = await self._client.post(
                f"{self._base_url}/rerank",
                json={"query": query, "documents": list(documents), "limit": limit},
                timeout=self._timeout,
            )
            response.raise_for_status()
            body = response.json()
            results = [
                RerankResult(index=int(item["index"]), score=float(item["score"]))
                for item in body["results"]
            ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Reranking call failed: %s", type(exc).__name__)
            raise RetrievalUnavailableError(
                "The reranking service could not be reached or returned an unusable response."
            ) from exc

        # An index outside the batch would map a score onto the wrong passage,
        # which is worse than failing: it looks like a result.
        for result in results:
            if not 0 <= result.index < len(documents):
                raise RetrievalUnavailableError(
                    f"The reranking service scored passage {result.index}, "
                    f"but {len(documents)} were sent."
                )
        return results
