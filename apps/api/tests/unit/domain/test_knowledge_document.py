"""The knowledge document and chunk entities."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from api.domain.entities.knowledge_document import (
    EMBEDDING_DIMENSIONS,
    KnowledgeChunk,
    KnowledgeDocument,
)
from api.domain.errors import DomainValidationError
from api.domain.value_objects.document_category import DocumentCategory

INGESTED_AT = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def document(**overrides: object) -> KnowledgeDocument:
    """Build a valid document, with fields a test wants to vary."""
    fields: dict[str, object] = {
        "document_key": "bearing-inspection-sop",
        "title": "Bearing Inspection SOP",
        "category": DocumentCategory.BEARING_INSPECTION,
        "version": "1.4",
        "source_path": "dummy_pdfs/procedure/Bearing Inspection SOP.pdf",
        "content_hash": "0" * 64,
        "page_count": 2,
        "ingested_at": INGESTED_AT,
    }
    fields.update(overrides)
    return KnowledgeDocument.create(**fields)  # type: ignore[arg-type]


def chunk(**overrides: object) -> KnowledgeChunk:
    """Build a valid chunk, with fields a test wants to vary."""
    fields: dict[str, object] = {
        "document_id": "0f8fad5b-d9cb-469f-a165-70867728950e",
        "chunk_index": 0,
        "section": "1. Purpose",
        "page": 1,
        "content": "Inspect the bearing.",
        "embedding": (0.0,) * EMBEDDING_DIMENSIONS,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2@abc123",
    }
    fields.update(overrides)
    return KnowledgeChunk.create(**fields)  # type: ignore[arg-type]


def test_create_assigns_a_identifier() -> None:
    """Documents and chunks carry their own identity, so storage assigns none."""
    first, second = document(), document()

    assert first.document_id
    assert first.document_id != second.document_id


def test_a_new_version_is_inactive_until_it_is_activated() -> None:
    """Activation is an explicit act, never a side effect of ingest."""
    assert document().is_active is False


@pytest.mark.parametrize("field", ["document_key", "title", "version", "content_hash"])
def test_blank_identity_fields_are_refused(field: str) -> None:
    """A document that cannot be cited is not stored."""
    with pytest.raises(DomainValidationError):
        document(**{field: "   "})


def test_a_non_synthetic_document_is_refused() -> None:
    """PRD section 25 makes synthetic documentation a constraint, not a default."""
    with pytest.raises(DomainValidationError):
        document(is_synthetic=False)


def test_a_naive_ingest_time_is_refused() -> None:
    """Every instant in this system is timezone-aware."""
    with pytest.raises(DomainValidationError):
        document(ingested_at=datetime(2026, 9, 26, 12, 0))


@pytest.mark.parametrize("page_count", [0, -1])
def test_a_document_must_have_at_least_one_page(page_count: int) -> None:
    """A document with no pages has nothing to retrieve."""
    with pytest.raises(DomainValidationError):
        document(page_count=page_count)


def test_a_chunk_must_carry_an_embedding_of_the_expected_width() -> None:
    """A vector of the wrong width is the first symptom of a swapped model."""
    with pytest.raises(DomainValidationError):
        chunk(embedding=(0.0,) * (EMBEDDING_DIMENSIONS - 1))


def test_a_chunk_must_cite_a_page_and_content() -> None:
    """Both are shown to the reader, so neither may be empty."""
    with pytest.raises(DomainValidationError):
        chunk(page=0)
    with pytest.raises(DomainValidationError):
        chunk(content="  ")


def test_a_chunk_with_no_section_is_allowed() -> None:
    """Some documents have no headings at all, and that is not an error."""
    assert chunk(section="").section == ""
