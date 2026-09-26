"""Use case: find the documented evidence a question needs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from api.domain.errors import DomainValidationError
from api.domain.ports.embedder import Embedder
from api.domain.ports.reranker import Reranker
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.services.evidence_sufficiency import DEFAULT_MINIMUM_SCORE, Sufficiency
from api.domain.services.evidence_sufficiency import assess_sufficiency as assess
from api.domain.value_objects.chunk_match import ChunkMatch
from api.domain.value_objects.document_category import DocumentCategory

#: How many passages the vector search hands to the reranker. The cross-encoder
#: scores every candidate on a one-core box, so this is the knob that trades
#: recall against latency -- and at this corpus size the recall is already
#: saturated well below it.
CANDIDATE_LIMIT = 20

#: How many passages an answer is given to work from.
RESULT_LIMIT = 5


@dataclass(frozen=True, slots=True)
class KnowledgeSearchResult:
    """What was retrieved, and whether it is enough to answer from."""

    matches: Sequence[ChunkMatch]
    sufficiency: Sufficiency


@dataclass(frozen=True, slots=True)
class SearchMaintenanceKnowledge:
    """Retrieve documented evidence for a question.

    Two stages, and both are needed. The vector search is exact but shallow: it
    compares a query embedding against every stored passage. The reranker reads
    the query and each candidate together, which orders them better than a
    distance between two independently computed vectors can -- and it costs a
    model call per candidate, which is why the candidates are bounded first.
    """

    unit_of_work_factory: UnitOfWorkFactory
    embedder: Embedder
    reranker: Reranker
    minimum_score: float = DEFAULT_MINIMUM_SCORE
    candidate_limit: int = CANDIDATE_LIMIT
    result_limit: int = RESULT_LIMIT

    async def execute(
        self,
        query: str,
        *,
        limit: int | None = None,
        category: DocumentCategory | None = None,
        rerank: bool = True,
    ) -> KnowledgeSearchResult:
        """Return the passages that best support an answer to `query`.

        `rerank=False` returns the vector ranking as it stands. It exists so the
        evaluation can measure what the cross-encoder contributes on the same
        questions, and it is a caller's explicit choice rather than a fallback:
        an outage is still an outage.

        Raises:
            DomainValidationError: if the query is blank.
            RetrievalUnavailableError: if embedding or reranking could not be
                performed. A reranker outage is not absorbed by falling back to
                the vector order: that would hide the outage and make the
                ranking silently worse.
        """
        question = query.strip()
        if not question:
            raise DomainValidationError("A search needs a question.")

        vectors = await self.embedder.embed([question])
        async with self.unit_of_work_factory() as uow:
            candidates = await uow.knowledge.similar_chunks(
                vectors[0],
                embedding_model=self.embedder.model_id,
                limit=self.candidate_limit,
                category=category,
            )

        if not candidates:
            return KnowledgeSearchResult(
                matches=(),
                sufficiency=assess((), minimum_score=self.minimum_score),
            )

        wanted = limit or self.result_limit
        matches = (
            await self._reranked(question, candidates, wanted)
            if rerank
            else list(candidates[:wanted])
        )
        return KnowledgeSearchResult(
            matches=matches,
            sufficiency=assess(
                [match.score for match in matches],
                minimum_score=self.minimum_score,
            ),
        )

    async def _reranked(
        self,
        question: str,
        candidates: Sequence[ChunkMatch],
        limit: int,
    ) -> list[ChunkMatch]:
        """Order the candidates with the cross-encoder, best first."""
        ranked = await self.reranker.rank(
            question,
            [match.chunk.content for match in candidates],
            limit=limit,
        )
        return [replace(candidates[result.index], score=result.score) for result in ranked]
