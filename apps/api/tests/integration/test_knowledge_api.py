"""The maintenance knowledge endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from api.composition import build_in_memory_container
from api.composition.container import Container
from api.infrastructure.config import Settings
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from api.presentation.app import create_app
from tests.support.factories import (
    DEFAULT_EMBEDDING_MODEL,
    make_embedding,
    make_knowledge_chunk,
    make_knowledge_document,
)
from tests.support.fakes import FixedClock, StubEmbedder, StubHealthProbe, StubReranker

INGEST_TOKEN = "s3cret-token"

BEARING = "Inspect the bearing housing for discoloured grease."


@pytest.fixture
def embedder() -> StubEmbedder:
    """An encoder that answers like the real one, in shape."""
    return StubEmbedder(model_id=DEFAULT_EMBEDDING_MODEL)


@pytest.fixture
def reranker() -> StubReranker:
    """A reranker that prefers the bearing passage."""
    return StubReranker(scores={BEARING: 0.9})


def a_client(container: Container) -> AsyncClient:
    """Build an HTTP client bound to a container."""
    return AsyncClient(
        transport=ASGITransport(app=create_app(container)),
        base_url="http://testserver",
    )


@pytest.fixture
async def knowledge_client(
    settings: Settings,
    uow_factory: InMemoryUnitOfWorkFactory,
    clock: FixedClock,
    health_probe: StubHealthProbe,
    embedder: StubEmbedder,
    reranker: StubReranker,
) -> AsyncIterator[AsyncClient]:
    """A client whose container has both retrieval models stubbed."""
    container = build_in_memory_container(
        settings,
        unit_of_work_factory=uow_factory,
        clock=clock,
        health_probe=health_probe,
        embedder=embedder,
        reranker=reranker,
    )
    async with a_client(container) as client:
        yield client


def an_ingest_body(**overrides: object) -> dict:
    """Build an ingest request body whose fields a test can vary."""
    body: dict = {
        "document_key": "bearing-inspection-sop",
        "title": "Bearing Inspection SOP",
        "category": "BEARING_INSPECTION",
        "version": "1.4",
        "source_path": "dummy_pdfs/procedure/Bearing Inspection SOP.pdf",
        "lines": [{"text": BEARING, "page": 1, "font_size": 13.33}],
    }
    body.update(overrides)
    return body


async def test_ingesting_a_document_reports_what_it_stored(
    knowledge_client: AsyncClient,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The response says how many passages were written, and that this was new."""
    response = await knowledge_client.post("/api/v1/knowledge/documents", json=an_ingest_body())

    assert response.status_code == 201
    body = response.json()
    assert body["chunk_count"] == 1
    assert body["unchanged"] is False
    assert body["replaced"] is False
    assert body["document"]["is_active"] is False

    async with uow_factory() as uow:
        chunks = await uow.knowledge.chunks_for(body["document"]["document_id"])

    assert [chunk.content for chunk in chunks] == [BEARING]
    assert chunks[0].embedding_model == DEFAULT_EMBEDDING_MODEL


async def test_re_ingesting_unchanged_content_reports_it(
    knowledge_client: AsyncClient,
) -> None:
    """A re-run of the corpus is idempotent, and says so."""
    first = await knowledge_client.post("/api/v1/knowledge/documents", json=an_ingest_body())
    second = await knowledge_client.post("/api/v1/knowledge/documents", json=an_ingest_body())

    assert second.status_code == 201
    assert second.json()["unchanged"] is True
    assert second.json()["document"]["document_id"] == first.json()["document"]["document_id"]


async def test_a_changed_version_without_replace_is_a_conflict(
    knowledge_client: AsyncClient,
) -> None:
    """409, and a code a client can branch on, rather than a silent rewrite."""
    await knowledge_client.post("/api/v1/knowledge/documents", json=an_ingest_body())

    response = await knowledge_client.post(
        "/api/v1/knowledge/documents",
        json=an_ingest_body(lines=[{"text": "Different text.", "page": 1, "font_size": 13.33}]),
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "document_content_conflict"


async def test_an_unknown_category_is_refused(knowledge_client: AsyncClient) -> None:
    """The category is one of the enum's members, checked at the schema."""
    response = await knowledge_client.post(
        "/api/v1/knowledge/documents", json=an_ingest_body(category="CATASTROPHE")
    )

    assert response.status_code == 422


async def test_a_document_with_no_lines_is_refused(knowledge_client: AsyncClient) -> None:
    """An empty document has no passages, so it is not stored."""
    response = await knowledge_client.post(
        "/api/v1/knowledge/documents", json=an_ingest_body(lines=[])
    )

    assert response.status_code == 422


async def test_ingesting_requires_the_token(
    settings: Settings,
    uow_factory: InMemoryUnitOfWorkFactory,
    clock: FixedClock,
    health_probe: StubHealthProbe,
    embedder: StubEmbedder,
    reranker: StubReranker,
) -> None:
    """A caller who can ingest can change what the Copilot cites.

    The guard is on the route rather than the router because this router also
    serves the dashboard's reads, which have no credential.
    """
    guarded = settings.model_copy(update={"ingest_api_token": INGEST_TOKEN})
    container = build_in_memory_container(
        guarded,
        unit_of_work_factory=uow_factory,
        clock=clock,
        health_probe=health_probe,
        embedder=embedder,
        reranker=reranker,
    )

    async with a_client(container) as client:
        refused = await client.post("/api/v1/knowledge/documents", json=an_ingest_body())
        accepted = await client.post(
            "/api/v1/knowledge/documents",
            json=an_ingest_body(),
            headers={"X-Ingest-Token": INGEST_TOKEN},
        )

    assert refused.status_code == 401
    assert accepted.status_code == 201


async def test_the_corpus_lists_every_version(
    knowledge_client: AsyncClient,
) -> None:
    """Inactive versions are listed: a citation may still resolve to one."""
    await knowledge_client.post("/api/v1/knowledge/documents", json=an_ingest_body())
    await knowledge_client.post(
        "/api/v1/knowledge/documents",
        json=an_ingest_body(version="2.0", activate=True),
    )

    response = await knowledge_client.get("/api/v1/knowledge/documents")

    assert response.status_code == 200
    versions = {document["version"]: document["is_active"] for document in response.json()}
    assert versions == {"1.4": False, "2.0": True}


async def test_activating_a_version_changes_what_is_retrievable(
    knowledge_client: AsyncClient,
) -> None:
    """The write the Copilot's citations depend on."""
    await knowledge_client.post("/api/v1/knowledge/documents", json=an_ingest_body())

    activated = await knowledge_client.put(
        "/api/v1/knowledge/documents/bearing-inspection-sop/active",
        json={"version": "1.4"},
    )
    listing = await knowledge_client.get("/api/v1/knowledge/documents")

    assert activated.status_code == 204
    assert listing.json()[0]["is_active"] is True


async def test_activating_an_unknown_version_is_not_found(
    knowledge_client: AsyncClient,
) -> None:
    """A typo must not withdraw the document, which is what the update would do."""
    await knowledge_client.post("/api/v1/knowledge/documents", json=an_ingest_body())

    response = await knowledge_client.put(
        "/api/v1/knowledge/documents/bearing-inspection-sop/active",
        json={"version": "9.9"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"


async def test_withdrawing_a_document_is_the_same_write(
    knowledge_client: AsyncClient,
) -> None:
    """`{"version": null}` takes a procedure out of the evidence base."""
    await knowledge_client.post("/api/v1/knowledge/documents", json=an_ingest_body(activate=True))

    response = await knowledge_client.put(
        "/api/v1/knowledge/documents/bearing-inspection-sop/active",
        json={"version": None},
    )
    listing = await knowledge_client.get("/api/v1/knowledge/documents")

    assert response.status_code == 204
    assert listing.json()[0]["is_active"] is False


async def test_search_returns_passages_with_their_citations(
    knowledge_client: AsyncClient,
) -> None:
    """PRD section 18's source fields, rendered for display."""
    await knowledge_client.post(
        "/api/v1/knowledge/documents",
        json=an_ingest_body(
            activate=True,
            lines=[
                {"text": "3. Inspection Steps", "page": 1, "font_size": 17.33},
                {"text": BEARING, "page": 1, "font_size": 13.33},
            ],
        ),
    )

    response = await knowledge_client.post(
        "/api/v1/knowledge/search", json={"query": "vibration is rising on M003"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sufficient"] is True
    assert len(body["matches"]) == 1
    citation = body["matches"][0]["citation"]
    assert citation["title"] == "Bearing Inspection SOP"
    assert citation["version"] == "1.4"
    assert citation["section"] == "3. Inspection Steps"
    assert citation["page"] == 1
    assert citation["label"] == "Bearing Inspection SOP v1.4, section 3. Inspection Steps, page 1"


async def test_search_finds_nothing_in_an_empty_corpus(
    knowledge_client: AsyncClient,
) -> None:
    """PRD section 19: say the evidence is not there rather than answer anyway."""
    response = await knowledge_client.post(
        "/api/v1/knowledge/search", json={"query": "what is the torque spec for the gearbox?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["matches"] == []
    assert body["sufficient"] is False
    assert body["reason"]


async def test_search_without_a_retrieval_service_is_a_503(client: AsyncClient) -> None:
    """The default in-memory wiring has no models behind it.

    Reported as an outage rather than as an empty result: "no evidence" is a
    claim about the corpus, and this is a claim about the deployment.
    """
    response = await client.post("/api/v1/knowledge/search", json={"query": "vibration"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "retrieval_unavailable"


async def test_search_refuses_a_blank_question(knowledge_client: AsyncClient) -> None:
    """An empty query embeds to nothing and ranks noise."""
    response = await knowledge_client.post("/api/v1/knowledge/search", json={"query": "   "})

    assert response.status_code == 422


async def test_search_can_be_narrowed_to_a_category(knowledge_client: AsyncClient) -> None:
    """A filter the Copilot uses when it already knows what kind of answer it needs."""
    await knowledge_client.post(
        "/api/v1/knowledge/documents",
        json=an_ingest_body(activate=True),
    )

    matching = await knowledge_client.post(
        "/api/v1/knowledge/search",
        json={"query": "bearing inspection", "category": "BEARING_INSPECTION"},
    )
    other = await knowledge_client.post(
        "/api/v1/knowledge/search",
        json={"query": "bearing inspection", "category": "ELECTRICAL_SAFETY"},
    )

    assert matching.json()["matches"] != []
    assert other.json()["matches"] == []


async def test_a_seeded_corpus_is_searchable_through_the_route(
    knowledge_client: AsyncClient,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """Retrieval over stored passages, without going through ingest."""
    async with uow_factory() as uow:
        document = make_knowledge_document(is_active=True)
        await uow.knowledge.add_document(document)
        await uow.knowledge.add_chunks(
            [
                make_knowledge_chunk(
                    document_id=document.document_id,
                    content=BEARING,
                    embedding=make_embedding(1.0),
                )
            ]
        )

    response = await knowledge_client.post(
        "/api/v1/knowledge/search", json={"query": "what should I inspect?"}
    )

    assert response.status_code == 200
    assert response.json()["matches"][0]["content"] == BEARING
