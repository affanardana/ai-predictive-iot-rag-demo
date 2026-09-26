"""The retrieval use case."""

from __future__ import annotations

import pytest

from api.application.use_cases import SearchMaintenanceKnowledge
from api.domain.errors import DomainValidationError
from api.domain.value_objects.document_category import DocumentCategory
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.factories import (
    DEFAULT_EMBEDDING_MODEL,
    make_embedding,
    make_knowledge_chunk,
    make_knowledge_document,
)
from tests.support.fakes import StubEmbedder, StubReranker

QUERY = "Vibration is rising on M003. What should I inspect?"

BEARING = "Inspect the bearing housing for discoloured grease."
SAFETY = "Isolate the supply before opening the terminal box."


@pytest.fixture
def embedder() -> StubEmbedder:
    """An encoder whose query vector matches everything equally."""
    return StubEmbedder(model_id=DEFAULT_EMBEDDING_MODEL)


@pytest.fixture
def reranker() -> StubReranker:
    """A reranker that prefers the bearing passage."""
    return StubReranker(scores={BEARING: 0.9, SAFETY: 0.1})


@pytest.fixture
def use_case(
    uow_factory: InMemoryUnitOfWorkFactory,
    embedder: StubEmbedder,
    reranker: StubReranker,
) -> SearchMaintenanceKnowledge:
    """The use case under test, with both models stubbed."""
    return SearchMaintenanceKnowledge(
        unit_of_work_factory=uow_factory,
        embedder=embedder,
        reranker=reranker,
    )


async def seed_corpus(
    factory: InMemoryUnitOfWorkFactory,
    *,
    bearing_active: bool = True,
) -> None:
    """Store two documents with one passage each."""
    async with factory() as uow:
        bearing = make_knowledge_document(is_active=bearing_active)
        safety = make_knowledge_document(
            document_key="electrical-motor-safety",
            title="Electrical Motor Safety",
            category=DocumentCategory.ELECTRICAL_SAFETY,
            is_active=True,
        )
        await uow.knowledge.add_document(bearing)
        await uow.knowledge.add_document(safety)
        await uow.knowledge.add_chunks(
            [
                make_knowledge_chunk(
                    document_id=bearing.document_id,
                    content=BEARING,
                    section="3. Inspection Steps",
                    page=1,
                    embedding=make_embedding(1.0),
                ),
                make_knowledge_chunk(
                    document_id=safety.document_id,
                    content=SAFETY,
                    page=2,
                    embedding=make_embedding(0.0, 1.0),
                ),
            ]
        )


async def test_returns_the_passages_the_reranker_ordered(
    use_case: SearchMaintenanceKnowledge,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The reranker's order is what the caller receives, not the vector order."""
    await seed_corpus(uow_factory)

    result = await use_case.execute(QUERY)

    assert [match.chunk.content for match in result.matches] == [BEARING, SAFETY]
    assert result.matches[0].score == pytest.approx(0.9)
    assert result.sufficiency.is_sufficient is True


async def test_the_reranker_is_called_once_for_the_whole_candidate_list(
    use_case: SearchMaintenanceKnowledge,
    reranker: StubReranker,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """A cross-encoder call per candidate would be the cost this design avoids."""
    await seed_corpus(uow_factory)

    await use_case.execute(QUERY)

    assert len(reranker.calls) == 1
    assert reranker.calls[0][0] == QUERY
    assert sorted(reranker.calls[0][1]) == sorted([BEARING, SAFETY])


async def test_every_match_carries_a_citable_source(
    use_case: SearchMaintenanceKnowledge,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """PRD section 18's four fields, resolvable from the match alone."""
    await seed_corpus(uow_factory)

    result = await use_case.execute(QUERY)

    citation = result.matches[0].citation
    assert citation.title == "Bearing Inspection SOP"
    assert citation.version == "1.4"
    assert citation.section == "3. Inspection Steps"
    assert citation.page == 1
    assert citation.label == "Bearing Inspection SOP v1.4, section 3. Inspection Steps, page 1"


async def test_the_result_limit_bounds_the_answer(
    use_case: SearchMaintenanceKnowledge,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """An answer is given a few passages to work from, not the whole corpus."""
    await seed_corpus(uow_factory)

    result = await use_case.execute(QUERY, limit=1)

    assert len(result.matches) == 1


async def test_an_empty_corpus_is_insufficient(
    use_case: SearchMaintenanceKnowledge,
) -> None:
    """Nothing stored means the documentation does not cover the question.

    Distinct from an outage: the search succeeded and found no evidence, which
    is what PRD section 19 asks the system to say rather than answering anyway.
    """
    result = await use_case.execute(QUERY)

    assert result.matches == ()
    assert result.sufficiency.is_sufficient is False
    assert result.sufficiency.reason


async def test_a_weak_match_is_insufficient(
    uow_factory: InMemoryUnitOfWorkFactory,
    embedder: StubEmbedder,
) -> None:
    """A nearest neighbour is not evidence, however near it is."""
    use_case = SearchMaintenanceKnowledge(
        unit_of_work_factory=uow_factory,
        embedder=embedder,
        reranker=StubReranker(default=0.05),
        minimum_score=0.5,
    )
    await seed_corpus(uow_factory)

    result = await use_case.execute(QUERY)

    assert result.matches != ()
    assert result.sufficiency.is_sufficient is False
    assert "0.50" in result.sufficiency.reason


async def test_a_withdrawn_document_is_never_searched(
    use_case: SearchMaintenanceKnowledge,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """Withdrawal is what makes AC-009 reachable: unreachable means unusable."""
    await seed_corpus(uow_factory, bearing_active=False)

    result = await use_case.execute(QUERY)

    assert [match.chunk.content for match in result.matches] == [SAFETY]


async def test_the_query_is_what_gets_embedded(
    use_case: SearchMaintenanceKnowledge,
    embedder: StubEmbedder,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The question is embedded, not the passages -- those were embedded at ingest."""
    await seed_corpus(uow_factory)

    await use_case.execute(f"  {QUERY}  ")

    assert embedder.texts_seen == [QUERY]


async def test_a_blank_question_is_refused(use_case: SearchMaintenanceKnowledge) -> None:
    """Searching for nothing would embed an empty string and rank noise."""
    with pytest.raises(DomainValidationError):
        await use_case.execute("   ")


async def test_reranking_can_be_turned_off_for_measurement(
    use_case: SearchMaintenanceKnowledge,
    reranker: StubReranker,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The one variable the evaluation toggles.

    The vector order is returned as it stands and the cross-encoder is never
    called -- which is what makes "reranking contributes X" a measurement rather
    than an assertion.
    """
    await seed_corpus(uow_factory)

    result = await use_case.execute(QUERY, rerank=False)

    assert len(result.matches) == 2
    assert reranker.calls == []
    assert result.matches[0].score == pytest.approx(1.0)
