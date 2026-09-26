"""Persistence models for the maintenance corpus."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from api.domain.entities.knowledge_document import EMBEDDING_DIMENSIONS
from api.domain.value_objects.document_category import DocumentCategory
from api.infrastructure.persistence.sql.base import Base
from api.infrastructure.persistence.sql.models.checks import in_clause

DOCUMENT_ID_MAX_LENGTH = 36
DOCUMENT_KEY_MAX_LENGTH = 64
TITLE_MAX_LENGTH = 128
CATEGORY_MAX_LENGTH = 32
VERSION_MAX_LENGTH = 32
SOURCE_PATH_MAX_LENGTH = 255
CONTENT_HASH_MAX_LENGTH = 64
CHUNK_ID_MAX_LENGTH = 36
SECTION_MAX_LENGTH = 255
EMBEDDING_MODEL_MAX_LENGTH = 96


class KnowledgeDocumentModel(Base):
    """One version of one maintenance document."""

    __tablename__ = "knowledge_documents"

    document_id: Mapped[str] = mapped_column(String(DOCUMENT_ID_MAX_LENGTH), primary_key=True)
    #: Stable identity across versions: the title is a display string and two
    #: documents could share one, whereas `(document_key, version)` is unique.
    document_key: Mapped[str] = mapped_column(String(DOCUMENT_KEY_MAX_LENGTH))
    title: Mapped[str] = mapped_column(String(TITLE_MAX_LENGTH))
    category: Mapped[str] = mapped_column(String(CATEGORY_MAX_LENGTH))
    version: Mapped[str] = mapped_column(String(VERSION_MAX_LENGTH))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True)
    source_path: Mapped[str] = mapped_column(String(SOURCE_PATH_MAX_LENGTH))
    content_hash: Mapped[str] = mapped_column(String(CONTENT_HASH_MAX_LENGTH))
    page_count: Mapped[int] = mapped_column(Integer)
    ingested_at: Mapped[datetime]

    __table_args__ = (
        UniqueConstraint(
            "document_key",
            "version",
            name="uq_knowledge_documents_key_version",
        ),
        CheckConstraint(
            in_clause("category", [category.value for category in DocumentCategory]),
            name="category_valid",
        ),
        CheckConstraint("page_count > 0", name="page_count_positive"),
        CheckConstraint("document_key <> ''", name="document_key_not_blank"),
        CheckConstraint("version <> ''", name="version_not_blank"),
        Index("ix_knowledge_documents_document_key", "document_key"),
        # At most one active version of a key, so "which one is current" has one
        # answer rather than depending on row order. Same shape as
        # `uq_simulation_runs_active_machine`.
        Index(
            "uq_knowledge_documents_active_key",
            "document_key",
            unique=True,
            postgresql_where=text("is_active"),
            sqlite_where=text("is_active"),
        ),
    )


class KnowledgeChunkModel(Base):
    """One embedded passage of a document."""

    __tablename__ = "knowledge_chunks"

    chunk_id: Mapped[str] = mapped_column(String(CHUNK_ID_MAX_LENGTH), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(DOCUMENT_ID_MAX_LENGTH),
        ForeignKey("knowledge_documents.document_id", ondelete="CASCADE"),
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String(SECTION_MAX_LENGTH))
    page: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer)
    #: `vector(384)` on PostgreSQL; JSON on SQLite, which has no vector type.
    #: The SQLite tier then round-trips the vector and exercises every metadata
    #: filter -- only the similarity operator is missing, and one test asserts
    #: that it raises rather than silently returning nothing.
    embedding: Mapped[Sequence[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS).with_variant(JSON(), "sqlite")
    )
    embedding_model: Mapped[str] = mapped_column(String(EMBEDDING_MODEL_MAX_LENGTH))

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_knowledge_chunks_document_index",
        ),
        CheckConstraint("char_count > 0", name="char_count_positive"),
        CheckConstraint("page > 0", name="page_positive"),
        CheckConstraint("chunk_index >= 0", name="chunk_index_non_negative"),
        Index("ix_knowledge_chunks_document_id", "document_id"),
    )
