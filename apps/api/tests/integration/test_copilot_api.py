"""The Copilot route, streamed over ASGI.

The frame protocol is the contract here, and it cannot be generated from
OpenAPI: a `text/event-stream` body has no schema. So this file pins the field
names, and `apps/web/src/api/stream.test.ts` pins the same ones on the client.
Renaming either side fails a test rather than freezing a browser.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from api.composition import build_in_memory_container
from api.composition.container import Container
from api.domain.value_objects.copilot import AnswerVerdict, CopilotTool
from api.domain.value_objects.evidence import EvidenceKind
from api.infrastructure.config import Settings
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from api.presentation.app import create_app
from api.presentation.routers.copilot import DONE_EVENT, ERROR_EVENT, TOKEN_EVENT, TOOL_EVENT
from tests.support.factories import (
    DEFAULT_EMBEDDING_MODEL,
    make_embedding,
    make_knowledge_chunk,
    make_knowledge_document,
    make_machine,
    make_prediction,
    make_reading,
    make_telemetry,
)
from tests.support.fakes import FixedClock, StubChat, StubEmbedder, StubHealthProbe, StubReranker

QUESTION = "Why is M003 becoming risky and what should I inspect according to the SOP?"

BEARING = "Attach an accelerometer to the bearing housing."


def frames_of(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    parsed: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event = next(
            line[len("event: ") :] for line in block.splitlines() if line.startswith("event: ")
        )
        data = next(
            line[len("data: ") :] for line in block.splitlines() if line.startswith("data: ")
        )
        parsed.append((event, json.loads(data)))
    return parsed


@pytest.fixture
async def chat_client(
    settings: Settings,
    uow_factory: InMemoryUnitOfWorkFactory,
    clock: FixedClock,
    health_probe: StubHealthProbe,
) -> AsyncIterator[AsyncClient]:
    """A client over a container with the Copilot's model stubbed.

    The stub answers with a value that is in the seeded evidence, so the
    grounding check passes and the route's success path is what is under test.
    """
    container = build_in_memory_container(
        settings,
        unit_of_work_factory=uow_factory,
        clock=clock,
        health_probe=health_probe,
        embedder=StubEmbedder(model_id=DEFAULT_EMBEDDING_MODEL),
        reranker=StubReranker(scores={BEARING: 0.9}),
        chat_model=StubChat(answer="The failure probability is 0.81 [1]."),
    )
    async with AsyncClient(
        transport=ASGITransport(app=create_app(container)), base_url="http://testserver"
    ) as client:
        yield client


async def seed(factory: InMemoryUnitOfWorkFactory) -> None:
    """A machine with a prediction and a document that answers the question."""
    async with factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.telemetry.add_many_idempotent(
            [make_telemetry(event_id="evt-1", machine_id="M003", reading=make_reading())]
        )
        await uow.predictions.add(make_prediction(machine_id="M003", probability=0.81))
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


def test_the_copilot_contract_is_stable() -> None:
    """The frame names and the enums they carry are a wire contract.

    A `text/event-stream` body cannot be described in OpenAPI, so the types in
    `apps/web/src/api/stream.ts` are hand-written -- the second place the
    "no hand-written client" rule is broken, after the event stream. This test
    is what makes that safe: `the mirrored vocabularies` in
    `apps/web/src/api/stream.test.ts` asserts these same lists, so a value
    renamed on either side fails a suite rather than freezing a page.

    Asserted literally rather than against the enums, because an assertion
    written as `{tool.value for tool in CopilotTool}` passes just as happily
    after somebody renames a tool.
    """
    assert (TOOL_EVENT, TOKEN_EVENT, DONE_EVENT, ERROR_EVENT) == ("tool", "token", "done", "error")
    assert {kind.value for kind in EvidenceKind} == {
        "OBSERVED",
        "PREDICTED",
        "DOCUMENTED",
        "INFERRED",
    }
    assert {verdict.value for verdict in AnswerVerdict} == {"ANSWERED", "REFUSED", "FALLBACK"}
    assert {tool.value for tool in CopilotTool} == {
        "get_machine_current_state",
        "get_machine_telemetry",
        "get_machine_trend",
        "get_machine_prediction",
        "get_machine_incidents",
        "get_machine_history",
        "search_maintenance_knowledge",
    }


async def test_the_stream_reports_tools_then_tokens_then_the_answer(
    chat_client: AsyncClient,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The frame order a streaming client depends on."""
    await seed(uow_factory)

    response = await chat_client.post("/api/v1/copilot/chat", json={"question": QUESTION})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [event for event, _ in frames_of(response.text)]
    assert events[0] == "tool"
    assert "token" in events
    assert events[-1] == "done"
    # Activity first, prose after: that is the whole point of streaming a
    # twenty-second answer.
    assert events.index("tool") < events.index("token") < events.index("done")


async def test_the_final_frame_carries_the_evidence_and_its_sources(
    chat_client: AsyncClient,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """PRD section 18's four fields, on the wire, attached by the system."""
    await seed(uow_factory)

    response = await chat_client.post("/api/v1/copilot/chat", json={"question": QUESTION})
    _, done = frames_of(response.text)[-1]

    assert done["verdict"] == "ANSWERED"
    assert done["answer"] == "The failure probability is 0.81 [1]."
    kinds = {item["kind"] for item in done["evidence"]}
    assert {"OBSERVED", "PREDICTED", "DOCUMENTED"} <= kinds
    documented = [item for item in done["evidence"] if item["kind"] == "DOCUMENTED"]
    assert all(item["source"] for item in documented)
    assert done["citations"][0]["label"].startswith("Bearing Inspection SOP")
    assert done["machine_id"] == "M003"
    assert done["model_id"] == "stub-composer@test"
    assert done["ungrounded"] == []


async def test_a_question_the_corpus_cannot_answer_is_refused_in_band(
    chat_client: AsyncClient,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """A refusal is a 200 with a verdict, because the request was fine.

    The distinction the page renders: a 4xx here would say the question was
    malformed, when what happened is that the documentation does not cover it.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))

    response = await chat_client.post(
        "/api/v1/copilot/chat", json={"question": "What should I inspect on M003 per the SOP?"}
    )

    assert response.status_code == 200
    _, done = frames_of(response.text)[-1]
    assert done["verdict"] == "REFUSED"
    assert done["reason"]
    # No tokens: the model was never asked, so there is no prose to stream.
    assert "token" not in [event for event, _ in frames_of(response.text)]


async def test_a_chat_outage_arrives_as_a_frame_once_the_stream_has_started(
    settings: Settings,
    uow_factory: InMemoryUnitOfWorkFactory,
    clock: FixedClock,
    health_probe: StubHealthProbe,
) -> None:
    """The status is committed as soon as the response begins.

    So a failure mid-answer is an `error` frame carrying the same code the
    envelope would have carried, and the page handles both the same way.
    """
    await seed(uow_factory)
    container = build_in_memory_container(
        settings,
        unit_of_work_factory=uow_factory,
        clock=clock,
        health_probe=health_probe,
        embedder=StubEmbedder(model_id=DEFAULT_EMBEDDING_MODEL),
        reranker=StubReranker(),
        chat_model=StubChat(raise_on_stream=True),
    )

    async with AsyncClient(
        transport=ASGITransport(app=create_app(container)), base_url="http://testserver"
    ) as client:
        response = await client.post("/api/v1/copilot/chat", json={"question": QUESTION})

    assert response.status_code == 200
    event, body = frames_of(response.text)[-1]
    assert event == "error"
    assert body["code"] == "chat_unavailable"
    assert body["status"] == 503


async def test_a_second_question_while_one_is_being_asked_is_a_conflict(
    chat_client: AsyncClient,
) -> None:
    """409 before the stream commits, so it is still a status code.

    One core writes one answer at a time: a second question would halve the
    speed of the first and leave both readers waiting twice as long.
    """
    from tests.support.fakes import StubChat as _StubChat  # noqa: F401 - documents the seam

    # The claim is held by hand because the stub answers instantly; in
    # production the window is the twenty seconds a generation takes.
    container: Container = chat_client._transport.app.state.container  # type: ignore[attr-defined]
    container.ask_copilot.claim()
    try:
        response = await chat_client.post("/api/v1/copilot/chat", json={"question": QUESTION})
    finally:
        container.ask_copilot.release()

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "chat_busy"


async def test_an_over_long_question_is_refused(chat_client: AsyncClient) -> None:
    """The bound is on the prefill, which is half the wait on one core."""
    response = await chat_client.post("/api/v1/copilot/chat", json={"question": "x" * 501})

    assert response.status_code == 422


async def test_the_copilot_is_unavailable_without_a_model(
    client: AsyncClient,
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The default in-memory wiring has no language model behind it.

    Reported as an outage rather than answered from a template: the Copilot's
    claim is that its prose came from evidence, and a placeholder that wrote
    something anyway would be a fabricated answer wearing the product's clothes.

    The machine is seeded first, on purpose. Without it the question names a
    machine that does not exist, and the Copilot refuses — correctly, and before
    it would ever have reached the model.
    """
    await seed(uow_factory)

    response = await client.post("/api/v1/copilot/chat", json={"question": "Why is M003 risky?"})

    assert response.status_code == 200
    event, body = frames_of(response.text)[-1]
    assert event == "error"
    assert body["code"] == "chat_unavailable"
