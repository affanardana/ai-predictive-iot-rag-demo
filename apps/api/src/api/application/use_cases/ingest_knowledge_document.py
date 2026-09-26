"""Use case: turn a parsed document into stored, embedded passages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from api.domain.entities.knowledge_document import KnowledgeChunk, KnowledgeDocument
from api.domain.errors import (
    DocumentContentConflictError,
    DomainValidationError,
)
from api.domain.ports.clock import Clock
from api.domain.ports.embedder import Embedder
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.services.knowledge_chunking import (
    TextChunk,
    chunk_document,
    content_hash,
    ensure_within_limit,
)
from api.domain.value_objects.document_category import DocumentCategory
from api.domain.value_objects.text_line import TextLine


@dataclass(frozen=True, slots=True)
class IngestRequest:
    """A parsed document, ready to be chunked and stored."""

    document_key: str
    title: str
    category: DocumentCategory
    version: str
    source_path: str
    lines: Sequence[TextLine]
    #: Refused by the entity when false. PRD section 25 makes synthetic
    #: documentation a constraint, and the manifest declares it per document.
    is_synthetic: bool = True
    #: Whether this version should become the retrievable one. Default off:
    #: superseding is an act, not a side effect of ingesting.
    activate: bool = False
    #: Whether a version whose text changed may be rewritten in place. Refused
    #: by default, because a citation recorded against that version would then
    #: resolve to text the reader never saw.
    allow_replace: bool = False


@dataclass(frozen=True, slots=True)
class KnowledgeIngestResult:
    """What the ingest did, which the CLI reports and the tests assert."""

    document: KnowledgeDocument
    chunk_count: int
    #: True when this exact content was already stored, so nothing was written.
    unchanged: bool = False
    #: True when an existing version's content was rewritten.
    replaced: bool = False


@dataclass(frozen=True, slots=True)
class IngestKnowledgeDocument:
    """Chunk a document, embed its passages, and store them.

    Chunking happens here rather than in the parser because it is a rule, and
    rules live behind the port boundary: the parser reports what a page says and
    how large it was set, and this decides what a retrievable passage is.
    """

    unit_of_work_factory: UnitOfWorkFactory
    embedder: Embedder
    clock: Clock

    async def execute(self, request: IngestRequest) -> KnowledgeIngestResult:
        """Store one document version.

        Raises:
            DomainValidationError: if the document carries no text.
            DocumentContentConflictError: if this version exists with different
                content and `allow_replace` was not set.
            RetrievalUnavailableError: if the passages could not be embedded.
            EmbeddingModelMismatchError: if the service is running another model.
        """
        chunks = chunk_document(request.lines)
        if not chunks:
            raise DomainValidationError("A document must carry at least one line of text.")
        ensure_within_limit(chunks)
        digest = content_hash(chunks)

        # A short read, in its own transaction: the choice between writing,
        # refusing, and doing nothing at all is made before the network call,
        # so an unchanged document costs no embedding at all.
        async with self.unit_of_work_factory() as uow:
            existing = await uow.knowledge.get_document(request.document_key, request.version)

        if existing is not None and existing.content_hash == digest:
            return KnowledgeIngestResult(document=existing, chunk_count=len(chunks), unchanged=True)
        if existing is not None and not request.allow_replace:
            raise DocumentContentConflictError(request.document_key, request.version)

        page_count = _page_count(request.lines)
        document = existing or KnowledgeDocument.create(
            document_key=request.document_key,
            title=request.title,
            category=request.category,
            version=request.version,
            source_path=request.source_path,
            content_hash=digest,
            page_count=page_count,
            ingested_at=self.clock.now(),
            is_synthetic=request.is_synthetic,
        )
        # Replacing is a change of content on the same version, so the hash and
        # the page count are rewritten with the passages.
        document.content_hash = digest
        document.page_count = page_count

        # One call for the whole document, and outside the write transaction:
        # embedding is a network call to another service, and holding a database
        # connection open across it would tie up a pooled connection for the
        # duration of someone else's latency.
        embedded = await self._embed(document, chunks)

        async with self.unit_of_work_factory() as uow:
            if existing is None:
                await uow.knowledge.add_document(document)
                await uow.knowledge.add_chunks(embedded)
            else:
                await uow.knowledge.replace_content(document, embedded)
            if request.activate:
                await uow.knowledge.activate(request.document_key, request.version)

        return KnowledgeIngestResult(
            document=document,
            chunk_count=len(embedded),
            replaced=existing is not None,
        )

    async def _embed(
        self,
        document: KnowledgeDocument,
        chunks: Sequence[TextChunk],
    ) -> Sequence[KnowledgeChunk]:
        """Embed every passage of one document in a single call.

        Batched because the cost of a call is dominated by loading the model
        into the batch, and the box has one core: one call for a document is
        one forward pass per batch rather than one per passage.
        """
        vectors = await self.embedder.embed([chunk.content for chunk in chunks])
        model_id = self.embedder.model_id
        return [
            KnowledgeChunk.create(
                document_id=document.document_id,
                chunk_index=index,
                section=chunk.section,
                page=chunk.page,
                content=chunk.content,
                embedding=tuple(vector),
                embedding_model=model_id,
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]


def _page_count(lines: Sequence[TextLine]) -> int:
    """Return how many pages the extracted lines came from."""
    return len({line.page for line in lines})
