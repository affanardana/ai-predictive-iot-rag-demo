"""Cross-encoder reranking.

A cross-encoder reads the query and one passage together, so it orders
candidates better than the vector search that produced them -- at a cost per
candidate, which is why the caller sends a bounded list.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from inference.settings import Settings

#: Longest query/passage pair the cross-encoder reads. A chunk is capped at
#: ~1000 characters by the API and a query is a sentence, so this is a ceiling
#: rather than a limit anything is expected to reach.
MAX_PAIR_TOKENS = 512


@dataclass(frozen=True, slots=True)
class RerankResult:
    """One passage's relevance, and where it sat in the request."""

    index: int
    score: float


class Reranker(Protocol):
    """What the app needs from a reranker, so tests can supply their own."""

    @property
    def model_id(self) -> str:
        """Identity of the loaded model, reported by the health endpoint."""
        ...

    def rank(self, query: str, documents: Sequence[str], limit: int) -> list[RerankResult]:
        """Score `documents` against `query`, best first, at most `limit` of them."""
        ...


class CrossEncoderReranker:
    """A cross-encoder, loaded once at startup.

    The import lives in the constructor rather than at module scope: nothing
    else in this service needs `sentence_transformers`, and paying for it at
    import time would slow every start for a code path that may never be called.
    """

    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            settings.rerank_model,
            revision=settings.rerank_revision,
            max_length=MAX_PAIR_TOKENS,
            cache_folder=str(settings.model_cache) if settings.model_cache else None,
            device="cpu",
        )
        self._model_id = f"{settings.rerank_model}@{settings.rerank_revision}"

    @property
    def model_id(self) -> str:
        """Identity of the loaded model, name and revision."""
        return self._model_id

    def rank(
        self,
        query: str,
        documents: Sequence[str],
        limit: int,
    ) -> list[RerankResult]:
        """Score every candidate against the query, best first."""
        if not documents or limit <= 0:
            return []
        scores = self._model.predict(
            [(query, document) for document in documents],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Ties keep the order the caller sent, which is the vector ranking's --
        # so a tie is broken by the better first-stage result rather than by
        # whichever way the sort happened to fall.
        ranked = sorted(range(len(documents)), key=lambda index: (-float(scores[index]), index))
        return [RerankResult(index=index, score=float(scores[index])) for index in ranked[:limit]]
