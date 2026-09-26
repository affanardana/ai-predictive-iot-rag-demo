"""A maintenance document and the chunks it is retrieved by."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from api.domain.errors import DomainValidationError
from api.domain.timestamps import ensure_aware
from api.domain.value_objects.document_category import DocumentCategory

#: Dimensions of every stored chunk embedding: the output width of
#: `sentence-transformers/all-MiniLM-L6-v2`.
EMBEDDING_DIMENSIONS = 384


@dataclass
class KnowledgeDocument:
    """One version of one maintenance document.

    Mutable because `is_active` changes after ingest: activation is an explicit
    act, and at most one version of a document key may be active at a time.
    """

    document_id: str
    document_key: str
    title: str
    category: DocumentCategory
    version: str
    source_path: str
    content_hash: str
    page_count: int
    ingested_at: datetime
    is_active: bool = False
    is_synthetic: bool = True

    def __post_init__(self) -> None:
        """Validate identity, provenance, and the synthetic-content rule."""
        for name in ("document_id", "document_key", "title", "version", "content_hash"):
            if not getattr(self, name).strip():
                raise DomainValidationError(f"KnowledgeDocument {name} must not be blank.")
        if self.page_count < 1:
            raise DomainValidationError(
                f"KnowledgeDocument page_count must be at least 1, got {self.page_count}."
            )
        if not self.is_synthetic:
            raise DomainValidationError(
                "Maintenance documentation must be synthetic (PRD section 25, constraint 12). "
                "Refusing to store a document that is not demonstration material."
            )
        ensure_aware(self.ingested_at, "ingested_at")

    @classmethod
    def create(
        cls,
        *,
        document_key: str,
        title: str,
        category: DocumentCategory,
        version: str,
        source_path: str,
        content_hash: str,
        page_count: int,
        ingested_at: datetime,
        is_active: bool = False,
        is_synthetic: bool = True,
    ) -> KnowledgeDocument:
        """Build a document with a fresh identifier.

        Identifiers are assigned here rather than by the store, so a repository
        `add` needs no return value.
        """
        return cls(
            document_id=str(uuid4()),
            document_key=document_key,
            title=title,
            category=category,
            version=version,
            source_path=source_path,
            content_hash=content_hash,
            page_count=page_count,
            ingested_at=ingested_at,
            is_active=is_active,
            is_synthetic=is_synthetic,
        )


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """One embedded passage of a document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    section: str
    page: int
    content: str
    embedding: tuple[float, ...]
    embedding_model: str

    def __post_init__(self) -> None:
        """Validate that the chunk is retrievable and citable."""
        if not self.chunk_id.strip():
            raise DomainValidationError("KnowledgeChunk chunk_id must not be blank.")
        if not self.document_id.strip():
            raise DomainValidationError("KnowledgeChunk document_id must not be blank.")
        if self.chunk_index < 0:
            raise DomainValidationError(
                f"KnowledgeChunk chunk_index must not be negative, got {self.chunk_index}."
            )
        if not self.content.strip():
            raise DomainValidationError("KnowledgeChunk content must not be blank.")
        if self.page < 1:
            raise DomainValidationError(f"KnowledgeChunk page must be at least 1, got {self.page}.")
        if len(self.embedding) != EMBEDDING_DIMENSIONS:
            raise DomainValidationError(
                f"KnowledgeChunk embedding has {len(self.embedding)} dimensions, "
                f"expected {EMBEDDING_DIMENSIONS}."
            )
        if not self.embedding_model.strip():
            raise DomainValidationError("KnowledgeChunk embedding_model must not be blank.")

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        chunk_index: int,
        section: str,
        page: int,
        content: str,
        embedding: tuple[float, ...],
        embedding_model: str,
    ) -> KnowledgeChunk:
        """Build a chunk with a fresh identifier."""
        return cls(
            chunk_id=str(uuid4()),
            document_id=document_id,
            chunk_index=chunk_index,
            section=section,
            page=page,
            content=content,
            embedding=embedding,
            embedding_model=embedding_model,
        )
