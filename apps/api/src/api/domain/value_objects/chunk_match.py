"""A chunk ranked against a query, with the document it came from."""

from __future__ import annotations

from dataclasses import dataclass

from api.domain.entities.knowledge_document import KnowledgeChunk, KnowledgeDocument
from api.domain.value_objects.citation import Citation


@dataclass(frozen=True, slots=True)
class ChunkMatch:
    """One retrieved passage, carrying what a citation needs.

    The document travels with the chunk because the chunk alone cannot be
    cited: PRD section 18 wants the document's title and version, and the chunk
    only knows its `document_id`.
    """

    chunk: KnowledgeChunk
    document: KnowledgeDocument
    score: float

    @property
    def citation(self) -> Citation:
        """Return where this passage came from."""
        return Citation(
            document_key=self.document.document_key,
            title=self.document.title,
            version=self.document.version,
            section=self.chunk.section,
            page=self.chunk.page,
        )
