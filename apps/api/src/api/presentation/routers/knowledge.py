"""Maintenance knowledge endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from api.application.use_cases import IngestRequest
from api.presentation.dependencies import (
    IngestKnowledgeDocumentDep,
    IngestTokenDep,
    ListKnowledgeDocumentsDep,
    SearchMaintenanceKnowledgeDep,
    SetActiveDocumentVersionDep,
)
from api.presentation.presenters import (
    to_knowledge_document,
    to_knowledge_document_list,
    to_knowledge_match_list,
    to_text_line,
)
from api.presentation.schemas.knowledge import (
    IngestDocumentRequest,
    IngestDocumentResponse,
    KnowledgeDocumentSchema,
    SearchKnowledgeRequest,
    SearchKnowledgeResponse,
    SetActiveVersionRequest,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

LimitQuery = Annotated[int, Query(ge=1, le=200)]


@router.get(
    "/documents",
    response_model=list[KnowledgeDocumentSchema],
    summary="List the maintenance corpus",
)
async def list_documents(
    use_case: ListKnowledgeDocumentsDep,
    limit: LimitQuery = 50,
) -> list[KnowledgeDocumentSchema]:
    """Return ingested document versions, newest first.

    Inactive versions are included: they are what an existing citation may still
    resolve to, and a withdrawn procedure is a fact about the corpus rather than
    something to hide from the page that shows it.
    """
    return to_knowledge_document_list(await use_case.execute(limit=limit))


@router.post(
    "/documents",
    response_model=IngestDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[IngestTokenDep],
    summary="Ingest a parsed document",
)
async def ingest_document(
    request: IngestDocumentRequest,
    use_case: IngestKnowledgeDocumentDep,
) -> IngestDocumentResponse:
    """Chunk, embed and store one document version.

    **Guarded**, because a caller here can rewrite what the Copilot cites. It is
    machine traffic in the same sense the telemetry endpoints are: the ingest
    container holds the token, and a browser never does.

    The parsing happened in the caller. What arrives is lines with the page they
    came from and the size they were set at, because chunking is a rule and
    rules live here.
    """
    result = await use_case.execute(
        IngestRequest(
            document_key=request.document_key,
            title=request.title,
            category=request.category,
            version=request.version,
            source_path=request.source_path,
            lines=[to_text_line(line) for line in request.lines],
            is_synthetic=request.is_synthetic,
            activate=request.activate,
            allow_replace=request.allow_replace,
        )
    )
    return IngestDocumentResponse(
        document=to_knowledge_document(result.document),
        chunk_count=result.chunk_count,
        unchanged=result.unchanged,
        replaced=result.replaced,
    )


@router.put(
    "/documents/{document_key}/active",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[IngestTokenDep],
    summary="Choose the retrievable version",
)
async def set_active_version(
    document_key: str,
    request: SetActiveVersionRequest,
    use_case: SetActiveDocumentVersionDep,
) -> None:
    """Make one version retrievable, or withdraw the document.

    `{"version": null}` withdraws it: the document stays readable and stops
    being evidence, which is how a procedure is taken out of the corpus. There
    is no delete for the same reason -- a citation recorded against a version
    must keep resolving after it is superseded.

    Guarded, because this decides what may be cited. An unguarded version of it
    would let anyone with the hostname change what the Copilot treats as the
    current procedure.
    """
    await use_case.execute(document_key, request.version)


@router.post(
    "/search",
    response_model=SearchKnowledgeResponse,
    summary="Retrieve maintenance evidence",
)
async def search_knowledge(
    request: SearchKnowledgeRequest,
    use_case: SearchMaintenanceKnowledgeDep,
) -> SearchKnowledgeResponse:
    """Return the passages that best support an answer to a question.

    Unguarded, like the dashboard's reads: it is the Copilot's retrieval tool
    and it writes nothing.

    `sufficient` is the answer to "does the documentation cover this" -- PRD
    section 19 requires the system to say when it does not, rather than answer
    from the nearest passage.
    """
    result = await use_case.execute(
        request.query,
        limit=request.limit,
        category=request.category,
        rerank=request.rerank,
    )
    return SearchKnowledgeResponse(
        matches=to_knowledge_match_list(result.matches),
        sufficient=result.sufficiency.is_sufficient,
        reason=result.sufficiency.reason,
    )
