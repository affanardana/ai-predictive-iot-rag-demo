"""Reranking port."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RerankResult:
    """One document's relevance, and where it sat in the request."""

    index: int
    score: float


class Reranker(Protocol):
    """Orders candidate passages against a query.

    Text in, indices out: the port never sees a chunk, so the ordering rule can
    be tested with no model and no corpus behind it.
    """

    async def rank(
        self,
        query: str,
        documents: Sequence[str],
        limit: int,
    ) -> Sequence[RerankResult]:
        """Score `documents` against `query`, best first, at most `limit` of them.

        Raises:
            RetrievalUnavailableError: the service could not be reached, or
                refused the request.
        """
        ...
