"""Use case: what the corpus contains."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from api.domain.entities.knowledge_document import KnowledgeDocument
from api.domain.ports.unit_of_work import UnitOfWorkFactory


@dataclass(frozen=True, slots=True)
class ListKnowledgeDocuments:
    """List every ingested document version, newest first.

    Inactive versions are included. They are the record of what a citation may
    still point at, and a document that has been withdrawn is a fact about the
    corpus rather than something to hide from the page that shows it.
    """

    unit_of_work_factory: UnitOfWorkFactory

    async def execute(self, limit: int) -> Sequence[KnowledgeDocument]:
        """Return recent document versions."""
        async with self.unit_of_work_factory() as uow:
            return await uow.knowledge.list_documents(limit)
