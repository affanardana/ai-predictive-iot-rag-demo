"""The Copilot use case, with no model and no network.

The assertions that matter are about what the *system* decides: which evidence a
question produces, and — the one this whole design turns on — that a question
the corpus cannot answer never reaches the model at all.
"""

from __future__ import annotations

import pytest

from api.application.use_cases import (
    AskCopilot,
    GetMachineDetail,
    GetPredictionHistory,
    GetTelemetryHistory,
    ListIncidents,
    ListMachines,
    SearchMaintenanceKnowledge,
)
from api.application.use_cases.ask_copilot import INSUFFICIENT_EVIDENCE
from api.domain.value_objects.copilot import AnswerVerdict, CopilotTool, ToolCall
from api.domain.value_objects.evidence import EvidenceKind
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.factories import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NOW,
    make_embedding,
    make_incident,
    make_knowledge_chunk,
    make_knowledge_document,
    make_machine,
    make_prediction,
    make_reading,
    make_telemetry,
)
from tests.support.fakes import FixedClock, StubChat, StubEmbedder, StubReranker

QUESTION = "Why is M003 becoming risky and what should I inspect according to the SOP?"

BEARING = "Attach an accelerometer to the bearing housing."


class RecordingSink:
    """An event sink that keeps what it was told."""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self.tokens: list[str] = []

    @property
    def tools(self) -> list[CopilotTool]:
        """Just the tools, for the tests that only care about the order."""
        return [call.tool for call in self.calls]

    async def tool(self, call: ToolCall) -> None:
        self.calls.append(call)

    async def token(self, text: str) -> None:
        self.tokens.append(text)


def a_copilot(
    factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
    embedder: StubEmbedder | None = None,
    reranker: StubReranker | None = None,
) -> AskCopilot:
    """Build the use case over the in-memory store."""
    clock = FixedClock()
    return AskCopilot(
        list_machines=ListMachines(unit_of_work_factory=factory),
        get_machine_detail=GetMachineDetail(unit_of_work_factory=factory),
        get_telemetry_history=GetTelemetryHistory(unit_of_work_factory=factory, clock=clock),
        get_prediction_history=GetPredictionHistory(unit_of_work_factory=factory),
        list_incidents=ListIncidents(unit_of_work_factory=factory),
        search_knowledge=SearchMaintenanceKnowledge(
            unit_of_work_factory=factory,
            embedder=embedder or StubEmbedder(model_id=DEFAULT_EMBEDDING_MODEL),
            reranker=reranker or StubReranker(scores={BEARING: 0.9}),
        ),
        chat=chat,
    )


async def seed_machine(factory: InMemoryUnitOfWorkFactory, *, with_corpus: bool = True) -> None:
    """Store a machine with telemetry, a prediction, an incident and a document."""
    async with factory() as uow:
        await uow.machines.add(make_machine("M003"))
        readings = [make_reading(vibration=1.4), make_reading(vibration=2.3)]
        for index, reading in enumerate(readings):
            await uow.telemetry.add_many_idempotent(
                [
                    make_telemetry(
                        event_id=f"evt-{index}",
                        machine_id="M003",
                        reading=reading,
                    )
                ]
            )
        await uow.predictions.add(make_prediction(machine_id="M003", probability=0.81))
        await uow.incidents.add(make_incident(machine_id="M003", incident_id="inc-1"))
        if with_corpus:
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


@pytest.fixture
def chat() -> StubChat:
    """The model, stubbed."""
    return StubChat()


async def test_a_risk_question_gathers_every_kind_of_evidence(
    uow_factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
) -> None:
    """`PRD.md` AC-007: state, trend, prediction and documentation together."""
    await seed_machine(uow_factory)

    answer = await a_copilot(uow_factory, chat).execute(QUESTION)

    assert answer.verdict is AnswerVerdict.ANSWERED
    kinds = {item.kind for item in answer.evidence}
    assert kinds == {
        EvidenceKind.OBSERVED,
        EvidenceKind.PREDICTED,
        EvidenceKind.INFERRED,
        EvidenceKind.DOCUMENTED,
    }
    assert answer.citations
    assert answer.citations[0].label.startswith("Bearing Inspection SOP")
    assert answer.model_id == chat.model_id


async def test_the_model_is_not_called_when_the_corpus_cannot_answer(
    uow_factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
) -> None:
    """`PRD.md` section 19, enforced by not asking.

    The question wants a procedure and the corpus has none, so the system
    refuses. A refusal the model never sees is one it cannot talk its way around
    — which is the only version of this rule a 1.5B model cannot defeat.
    """
    await seed_machine(uow_factory, with_corpus=False)

    answer = await a_copilot(uow_factory, chat).execute(
        "What should I inspect on M003 according to the SOP?"
    )

    assert answer.verdict is AnswerVerdict.REFUSED
    assert "insufficient" in answer.answer.casefold()
    assert chat.calls == []


async def test_a_corpus_that_is_present_but_irrelevant_still_refuses(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The case the empty-corpus test does not reach, and the one that shipped.

    A corpus with anything in it always returns *some* nearest neighbour, so a
    check for "were there any matches" never fires and the Copilot answers every
    documentation question from whatever ranked first -- which is exactly what
    `PRD.md` section 19 forbids, and what `evidence_sufficiency.py` exists to
    prevent. The refusal has to be driven by the sufficiency rule, not by
    emptiness.

    Both sentences are asserted, because they answer different questions:
    section 19's own is what the system says, and the rule's is why.
    """
    await seed_machine(uow_factory)
    chat = StubChat()
    # A cross-encoder scores an irrelevant pair below zero; the deployed
    # threshold is 0.0, so this is the torque-spec question's shape.
    copilot = a_copilot(uow_factory, chat, reranker=StubReranker(scores={BEARING: -4.2}))
    sink = RecordingSink()

    answer = await copilot.execute(
        "What is the torque spec for the M003 gearbox output shaft?", sink=sink
    )

    assert answer.verdict is AnswerVerdict.REFUSED
    assert answer.answer == INSUFFICIENT_EVIDENCE
    assert "-4.20" in answer.reason
    assert chat.calls == []
    # The trail says the documentation was consulted and found wanting, rather
    # than saying nothing at all: a tool that was never run and a tool that
    # found nothing are different things.
    knowledge = next(call for call in sink.calls if call.tool is CopilotTool.KNOWLEDGE)
    assert knowledge.evidence_count == 0
    assert "sufficiency" in knowledge.summary


async def test_a_question_about_a_machine_that_is_not_registered_refuses(
    uow_factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
) -> None:
    """A 200 with a reason, not a 404: the request was a question."""
    await seed_machine(uow_factory)

    answer = await a_copilot(uow_factory, chat).execute("What is wrong with M009?")

    assert answer.verdict is AnswerVerdict.REFUSED
    assert answer.reason
    assert chat.calls == []


async def test_an_ungrounded_number_withholds_the_prose(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The evidence was good and the model's answer was not.

    The findings are served in place of the summary, and the offending values are
    named — a caught fabrication that is silently corrected is one nobody sees.
    """
    await seed_machine(uow_factory)
    chat = StubChat(answer="Vibration reached 9.9 mm/s, which is critical.")

    answer = await a_copilot(uow_factory, chat).execute(QUESTION)

    assert answer.verdict is AnswerVerdict.FALLBACK
    assert answer.ungrounded == ("9.9",)
    assert "9.9" not in answer.answer
    assert answer.evidence
    assert "withheld" in answer.reason


async def test_a_grounded_answer_is_kept(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The numbers the evidence actually contained survive."""
    await seed_machine(uow_factory)
    chat = StubChat(answer="The failure probability is 0.81 for M003.")

    answer = await a_copilot(uow_factory, chat).execute(QUESTION)

    assert answer.verdict is AnswerVerdict.ANSWERED
    assert answer.ungrounded == ()
    assert answer.answer == "The failure probability is 0.81 for M003."


async def test_the_prompt_carries_the_evidence_and_the_question(
    uow_factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
) -> None:
    """The prompt is asserted on its contents, not on its prose."""
    await seed_machine(uow_factory)

    await a_copilot(uow_factory, chat).execute(QUESTION)

    prompt = chat.calls[0][-1].content
    assert QUESTION in prompt
    assert BEARING in prompt
    assert "[1]" in prompt
    # The instruction that makes the grounding check possible in the first place.
    assert "Do not state any number that is not in them" in chat.calls[0][0].content
    # And the one it cannot enforce: the first live answer called a measured
    # 1424 rpm "low", which is an invented judgement over a real number. The
    # prompt is the only place that can be forbidden, so it is asserted here.
    assert "Do not describe a value as high, low, rising or falling" in chat.calls[0][0].content


async def test_activity_is_reported_as_it_happens(
    uow_factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
) -> None:
    """The frames a streaming client sees, in the order the tools ran."""
    await seed_machine(uow_factory)
    sink = RecordingSink()

    answer = await a_copilot(uow_factory, chat).execute(QUESTION, sink=sink)

    assert sink.tools == [call.tool for call in answer.tool_calls]
    assert sink.tools[0] is CopilotTool.CURRENT_STATE
    assert sink.tokens
    assert "".join(sink.tokens).strip() == answer.answer


async def test_a_machine_question_does_not_read_the_corpus(
    uow_factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
) -> None:
    """`PRD.md` AC-006, end to end rather than in the planner alone."""
    await seed_machine(uow_factory)

    answer = await a_copilot(uow_factory, chat).execute(
        "What happened to M003 during the last five hours?"
    )

    assert CopilotTool.KNOWLEDGE not in [call.tool for call in answer.tool_calls]
    assert CopilotTool.TELEMETRY in [call.tool for call in answer.tool_calls]
    assert answer.citations == ()


async def test_every_documented_finding_names_its_source(
    uow_factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
) -> None:
    """`Evidence` refuses an unsourced documentation claim, and the wire keeps it."""
    await seed_machine(uow_factory)

    answer = await a_copilot(uow_factory, chat).execute(QUESTION)

    documented = [item for item in answer.evidence if item.kind is EvidenceKind.DOCUMENTED]
    assert documented
    assert all(item.source for item in documented)


async def test_the_answer_carries_the_window_it_used(
    uow_factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
) -> None:
    """A trend without its window is a number without a claim."""
    await seed_machine(uow_factory)

    answer = await a_copilot(uow_factory, chat).execute("What happened to M003 in the last hour?")

    trend_calls = [call for call in answer.tool_calls if call.tool is CopilotTool.TREND]
    assert trend_calls
    assert "1h" in trend_calls[0].summary


async def test_the_trend_tool_reports_movement_and_only_movement(
    uow_factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
) -> None:
    """Two claims per signal that moved, and nothing for the ones that did not.

    Found on the deployed stack: a machine with 240 readings produced **twelve**
    evidence items from this tool -- an endpoint and a fit for all six signals,
    including load, whose fitted change was -0.008 across the whole window. That
    is the deadband's own definition of noise, and eleven more lines of it bury
    the four that matter, on the page and in the prompt alike.

    A tool that found no movement still answers, which is the other half: six
    flat signals and no trend tool at all look identical to a reader, and one of
    them means the machine is steady.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        readings = [make_reading(vibration=1.4), make_reading(vibration=2.3)]
        for index, reading in enumerate(readings):
            await uow.telemetry.add_many_idempotent(
                [make_telemetry(event_id=f"evt-{index}", machine_id="M003", reading=reading)]
            )

    answer = await a_copilot(uow_factory, chat).execute("What happened to M003 in the last hour?")

    trend_call = next(call for call in answer.tool_calls if call.tool is CopilotTool.TREND)
    inferred = [item for item in answer.evidence if item.kind is EvidenceKind.INFERRED]
    # Vibration is the only signal this fixture moves, and the only kind of
    # evidence the trend tool derives. Five flat signals, one INFERRED line.
    assert len(inferred) == 1
    # Two claims for every signal that moved: what was read, and what was
    # derived from it. Not twelve, which is what it produced on the box.
    assert trend_call.evidence_count == 2 * len(inferred)
    assert "1 of 6 signals moved over the 1h window" in trend_call.summary
    assert len(trend_call.summary) < 200


async def test_the_prediction_tool_reports_movement_not_a_repeat(
    uow_factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
) -> None:
    """Two tools, two findings.

    The current-state tool already reports the latest probability, so a
    prediction tool that reports the same number in a different sentence puts
    the same `PREDICTED` line on the page twice and adds nothing. What it can
    say instead is where the number came from -- which is also the closest
    thing to a trend available when a machine's telemetry is older than the
    window, and the question asked is usually *"why is it becoming risky"*.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        for index, probability in enumerate((0.21, 0.44, 0.81)):
            await uow.predictions.add(
                make_prediction(
                    machine_id="M003",
                    prediction_id=f"pred-{index}",
                    probability=probability,
                )
            )

    answer = await a_copilot(uow_factory, chat).execute("Why is M003 risky?")

    predicted = [item.value for item in answer.evidence if item.kind is EvidenceKind.PREDICTED]
    assert len(predicted) == 2
    assert "newest of 3" in predicted[1]
    assert "0.21" in predicted[1]


async def test_no_readings_still_answers_from_what_is_known(
    uow_factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
) -> None:
    """A registered machine with no telemetry is thin, not absent."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.predictions.add(
            make_prediction(machine_id="M003", probability=0.4, predicted_at=DEFAULT_NOW)
        )

    answer = await a_copilot(uow_factory, chat).execute("Why is M003 risky?")

    assert answer.verdict is AnswerVerdict.ANSWERED
    assert any(item.kind is EvidenceKind.OBSERVED for item in answer.evidence)


async def test_an_empty_window_is_reported_rather_than_silently_skipped(
    uow_factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
) -> None:
    """A tool that found nothing still gets a line in the activity trail.

    This is the deployed case rather than a contrived one: a machine whose last
    reading is a day old has nothing in the default five-hour window, so the
    trend the question asks for cannot be produced. Without a line, a reader
    cannot tell whether the Copilot looked and found nothing or never looked --
    and an answer to *"why is it becoming risky"* with no trend in it reads as
    an oversight rather than as an absence of data.

    It carries no evidence, so it cannot change what the answer says. Asserted
    both ways, because that is the property that makes the line cheap.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.predictions.add(
            make_prediction(machine_id="M003", probability=0.4, predicted_at=DEFAULT_NOW)
        )
    sink = RecordingSink()

    answer = await a_copilot(uow_factory, chat).execute("Why is M003 becoming risky?", sink=sink)

    empty = [call for call in sink.calls if call.evidence_count == 0]
    assert {call.tool for call in empty} == {CopilotTool.TELEMETRY, CopilotTool.TREND}
    assert all("5h" in call.summary for call in empty)
    # The answer's own list is the tools that produced findings. A line saying
    # a tool found nothing is progress, not evidence, and must not appear there
    # -- it would otherwise be counted when the refusal is decided.
    assert {call.tool for call in answer.tool_calls} == {
        call.tool for call in sink.calls if call.evidence_count > 0
    }
    assert not any("nothing" in item.value.casefold() for item in answer.evidence)


async def test_the_activity_line_for_the_corpus_names_the_documents(
    uow_factory: InMemoryUnitOfWorkFactory,
    chat: StubChat,
) -> None:
    """The model reads the passages; the trail gets a line.

    Two readers, two renderings. Printing the retrieval summary in the trail
    would put a thousand characters of procedure where one line belongs -- the
    passages are what the *page* shows, in the evidence blocks, with their
    citations.
    """
    await seed_machine(uow_factory)
    sink = RecordingSink()

    answer = await a_copilot(uow_factory, chat).execute(QUESTION, sink=sink)

    # Both renderings of the trail, because there are two and they drifted
    # once already: the frame streamed as the tool finished, and the
    # `tool_calls` on the finished answer, which is what the page renders in
    # the transcript. The first was fixed and the second kept printing the
    # passages, which is how this assertion came to exist.
    streamed = next(call for call in sink.calls if call.tool is CopilotTool.KNOWLEDGE)
    recorded = next(call for call in answer.tool_calls if call.tool is CopilotTool.KNOWLEDGE)
    assert streamed == recorded
    assert BEARING not in recorded.summary
    assert recorded.summary.endswith(".")
    assert len(recorded.summary) < 200
    # And the prompt still carries the passage, or the answer could not cite it.
    assert BEARING in chat.calls[0][-1].content
