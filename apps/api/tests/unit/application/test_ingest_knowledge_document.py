"""The document ingest use case."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from api.application.use_cases import IngestKnowledgeDocument, IngestRequest
from api.domain.errors import DocumentContentConflictError, DomainValidationError
from api.domain.value_objects.document_category import DocumentCategory
from api.domain.value_objects.text_line import TextLine
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.fakes import FixedClock, StubEmbedder

BODY = 13.33
SECTION = 17.33


def lines(*texts: str, page: int = 1, font_size: float = BODY) -> list[TextLine]:
    """Build extracted lines at one size."""
    return [TextLine(text=text, page=page, font_size=font_size) for text in texts]


def a_request(**overrides: object) -> IngestRequest:
    """Build an ingest request whose fields a test can vary."""
    fields: dict[str, object] = {
        "document_key": "bearing-inspection-sop",
        "title": "Bearing Inspection SOP",
        "category": DocumentCategory.BEARING_INSPECTION,
        "version": "1.4",
        "source_path": "dummy_pdfs/procedure/Bearing Inspection SOP.pdf",
        "lines": lines("Inspect the bearing housing."),
    }
    fields.update(overrides)
    return IngestRequest(**fields)  # type: ignore[arg-type]


@pytest.fixture
def embedder() -> StubEmbedder:
    """The encoder the use case will call."""
    return StubEmbedder()


@pytest.fixture
def use_case(
    uow_factory: InMemoryUnitOfWorkFactory,
    embedder: StubEmbedder,
) -> IngestKnowledgeDocument:
    """The use case under test, over one in-memory store."""
    return IngestKnowledgeDocument(
        unit_of_work_factory=uow_factory,
        embedder=embedder,
        clock=FixedClock(),
    )


async def test_stores_a_document_and_its_passages(
    use_case: IngestKnowledgeDocument,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """A document arrives as lines and is stored as chunks."""
    result = await use_case.execute(
        a_request(lines=lines("1. Purpose", font_size=SECTION) + lines("Inspect the bearing."))
    )

    async with uow_factory() as uow:
        stored = await uow.knowledge.get_document("bearing-inspection-sop", "1.4")
        chunks = await uow.knowledge.chunks_for(result.document.document_id)

    assert result.chunk_count == 1
    assert stored is not None
    assert stored.page_count == 1
    assert stored.is_active is False
    assert [chunk.content for chunk in chunks] == ["1. Purpose\nInspect the bearing."]


async def test_embeds_every_passage_in_one_call(
    use_case: IngestKnowledgeDocument,
    embedder: StubEmbedder,
) -> None:
    """One call per document, not one per passage.

    The cost of an embedding call on a one-core box is dominated by the forward
    pass, so a document's passages are batched -- and this is the assertion that
    would catch a refactor that embedded them one at a time.
    """
    long_body = lines("Inspect the bearing housing for discoloured grease. " * 20)

    result = await use_case.execute(
        a_request(lines=lines("1. Purpose", font_size=SECTION) + long_body)
    )

    assert result.chunk_count > 1
    assert len(embedder.batches) == 1
    assert len(embedder.batches[0]) == result.chunk_count


async def test_every_stored_passage_records_the_models_identity(
    use_case: IngestKnowledgeDocument,
    embedder: StubEmbedder,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """Retrieval only compares vectors carrying the same model identity."""
    result = await use_case.execute(a_request())

    async with uow_factory() as uow:
        chunks = await uow.knowledge.chunks_for(result.document.document_id)

    assert {chunk.embedding_model for chunk in chunks} == {embedder.model_id}


async def test_re_ingesting_unchanged_content_writes_nothing(
    use_case: IngestKnowledgeDocument,
    embedder: StubEmbedder,
) -> None:
    """The case that makes re-running the corpus safe.

    Nothing is embedded and nothing is written, so a chunking change can be
    re-ingested over the whole corpus without paying for the documents that did
    not change.
    """
    await use_case.execute(a_request())
    embedder.batches.clear()

    second = await use_case.execute(a_request())

    assert second.unchanged is True
    assert second.chunk_count == 1
    assert embedder.batches == []


async def test_a_changed_version_is_refused(use_case: IngestKnowledgeDocument) -> None:
    """A version is a promise that version N is this text.

    Replacing it silently would make every citation recorded against it resolve
    to something its reader never saw.
    """
    await use_case.execute(a_request())

    with pytest.raises(DocumentContentConflictError):
        await use_case.execute(a_request(lines=lines("A different procedure entirely.")))


async def test_allow_replace_rewrites_the_version_and_its_passages(
    use_case: IngestKnowledgeDocument,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The escape hatch for a genuine correction, and it replaces rather than appends."""
    first = await use_case.execute(a_request())
    second = await use_case.execute(
        a_request(lines=lines("Corrected procedure."), allow_replace=True)
    )

    async with uow_factory() as uow:
        chunks = await uow.knowledge.chunks_for(second.document.document_id)

    assert second.replaced is True
    assert [chunk.content for chunk in chunks] == ["Corrected procedure."]
    assert first.document.document_id == second.document.document_id


async def test_a_new_version_does_not_become_active_by_itself(
    use_case: IngestKnowledgeDocument,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """Superseding is an act, so a botched ingest cannot change what is cited."""
    await use_case.execute(a_request(version="1.4", activate=True))
    await use_case.execute(a_request(version="2.0"))

    async with uow_factory() as uow:
        active = await uow.knowledge.get_document("bearing-inspection-sop", "1.4")
        new = await uow.knowledge.get_document("bearing-inspection-sop", "2.0")

    assert active is not None and active.is_active is True
    assert new is not None and new.is_active is False


async def test_activate_supersedes_the_previous_version(
    use_case: IngestKnowledgeDocument,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """One active version per document, whichever order they were ingested in."""
    await use_case.execute(a_request(version="1.4", activate=True))
    await use_case.execute(a_request(version="2.0", activate=True))

    async with uow_factory() as uow:
        old = await uow.knowledge.get_document("bearing-inspection-sop", "1.4")
        new = await uow.knowledge.get_document("bearing-inspection-sop", "2.0")

    assert old is not None and old.is_active is False
    assert new is not None and new.is_active is True


async def test_an_empty_document_is_refused(use_case: IngestKnowledgeDocument) -> None:
    """A document with no text has no pages and nothing to retrieve."""
    with pytest.raises(DomainValidationError):
        await use_case.execute(a_request(lines=[]))


async def test_a_non_synthetic_document_is_refused(use_case: IngestKnowledgeDocument) -> None:
    """PRD section 17 requires the corpus to be demonstration material."""
    with pytest.raises(DomainValidationError):
        await use_case.execute(a_request(is_synthetic=False))


async def test_pages_are_counted_from_the_lines(
    use_case: IngestKnowledgeDocument,
) -> None:
    """Page count is derived at ingest rather than declared by the manifest."""
    two_pages: Sequence[TextLine] = [
        *lines("Page one.", page=1),
        *lines("Page two.", page=2),
    ]

    result = await use_case.execute(a_request(lines=list(two_pages)))

    assert result.document.page_count == 2
    assert result.chunk_count == 2
