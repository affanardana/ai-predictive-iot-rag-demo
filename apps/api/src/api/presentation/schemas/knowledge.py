"""Request and response shapes for the maintenance corpus."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from api.domain.value_objects.document_category import DocumentCategory

#: Bounds on how much text one ingest call may carry. A document is chunked and
#: embedded in a single request, so an unbounded body would be an unbounded
#: embedding job on a box with one core.
MAX_LINES_PER_REQUEST = 5000


class TextLineSchema(BaseModel):
    """One line of a parsed document, as the ingest container sends it."""

    text: str = Field(min_length=1, max_length=2000)
    #: One-based, and never null: a citation has to name a page.
    page: int = Field(ge=1)
    #: The size the line was set at, which is the only structure these
    #: documents carry. The chunker decides what counts as a heading.
    font_size: float = Field(gt=0)


class IngestDocumentRequest(BaseModel):
    """A parsed document version, ready to be chunked and stored."""

    document_key: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=128)
    category: DocumentCategory
    version: str = Field(min_length=1, max_length=32)
    source_path: str = Field(default="", max_length=255)
    #: PRD section 17 requires documentation to be identifiable as demonstration
    #: material, and the domain refuses a document that is not.
    is_synthetic: bool = True
    #: Whether this version becomes the retrievable one. Off by default:
    #: superseding is an act rather than a side effect of ingesting.
    activate: bool = False
    #: Whether a version whose text changed may be rewritten in place. Refused
    #: by default, because citations recorded against it would stop resolving to
    #: the text their reader saw.
    allow_replace: bool = False
    lines: list[TextLineSchema] = Field(min_length=1, max_length=MAX_LINES_PER_REQUEST)


class KnowledgeDocumentSchema(BaseModel):
    """One stored version of one document."""

    document_id: str
    document_key: str
    title: str
    category: DocumentCategory
    version: str
    is_active: bool
    is_synthetic: bool
    page_count: int
    ingested_at: datetime


class IngestDocumentResponse(BaseModel):
    """What the ingest did, so a re-run can report it honestly."""

    document: KnowledgeDocumentSchema
    chunk_count: int
    #: True when this exact content was already stored, so nothing was written.
    unchanged: bool
    #: True when an existing version's content was rewritten.
    replaced: bool


class SetActiveVersionRequest(BaseModel):
    """Which version should be retrievable, or none to withdraw the document."""

    version: str | None = Field(default=None, max_length=32)


class CitationSchema(BaseModel):
    """Where a passage came from: PRD section 18's four fields, and a label."""

    document_key: str
    title: str
    version: str
    section: str
    page: int
    #: The four fields rendered for display, so a client does not have to
    #: reassemble them and two clients cannot disagree about the format.
    label: str


class KnowledgeMatchSchema(BaseModel):
    """One retrieved passage."""

    citation: CitationSchema
    content: str
    score: float


class SearchKnowledgeRequest(BaseModel):
    """A maintenance question."""

    query: str = Field(min_length=1, max_length=1000)
    category: DocumentCategory | None = None
    limit: int | None = Field(default=None, ge=1, le=20)
    #: Whether to order the candidates with the cross-encoder. On by default,
    #: because ordering is what the stage is for.
    #:
    #: Turning it off is how the evaluation measures what reranking contributes:
    #: the same questions, the same corpus, one variable. It is deliberately not
    #: a fallback -- an outage still returns 503, because silently ranking
    #: without the model is the failure this flag would otherwise hide.
    rerank: bool = True


class SearchKnowledgeResponse(BaseModel):
    """The passages retrieved, and whether they are enough to answer from.

    `sufficient` is carried rather than left to the client to infer from the
    scores: it is the rule PRD section 19 asks for, and it belongs on the server
    so every client applies the same one.
    """

    matches: list[KnowledgeMatchSchema]
    sufficient: bool
    #: Why the evidence is insufficient, when it is. Empty otherwise.
    reason: str
