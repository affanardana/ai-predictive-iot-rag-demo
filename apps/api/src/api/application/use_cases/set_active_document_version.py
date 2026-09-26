"""Use case: choose which version of a document the Copilot may cite."""

from __future__ import annotations

from dataclasses import dataclass

from api.domain.ports.unit_of_work import UnitOfWorkFactory


@dataclass(frozen=True, slots=True)
class SetActiveDocumentVersion:
    """Make one version retrievable, and the others not.

    Explicit rather than automatic. `version` is a string with no total order,
    so "newest wins" would have to guess; and a botched re-ingest would then
    silently change what the Copilot cites. Superseding is an act.

    Withdrawing is the same operation with no version: the document stays
    readable, and stops being evidence -- which is what AC-009 rests on, since a
    procedure that cannot be retrieved cannot be answered from.
    """

    unit_of_work_factory: UnitOfWorkFactory

    async def execute(self, document_key: str, version: str | None) -> None:
        """Activate one version, or withdraw the document when `version` is None.

        Raises:
            KnowledgeDocumentNotFoundError: if that key and version are not
                stored. Refused rather than treated as a withdrawal, which is
                what the statement would otherwise do.
        """
        async with self.unit_of_work_factory() as uow:
            await uow.knowledge.activate(document_key, version)
