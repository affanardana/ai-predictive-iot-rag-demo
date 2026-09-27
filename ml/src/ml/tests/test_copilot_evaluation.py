"""The Copilot scorecard's arithmetic, against a fake API.

The frame parser and the derived rates, no network. The real stack is measured
by running `ml copilot evaluate`, which is a demonstration rather than a test:
it needs a deployed API, a model and several minutes.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ml.copilot import evaluation
from ml.copilot.errors import CopilotEvaluationError

QUESTIONS_PATH = Path("knowledge/eval/copilot-questions.json")


def a_question(**overrides: object) -> evaluation.Question:
    """One question, with whatever the test cares about overridden."""
    fields: dict[str, object] = {
        "id": "q1",
        "text": "Why is M003 risky?",
        "required": frozenset({"get_machine_trend"}),
        "forbidden": frozenset({"search_maintenance_knowledge"}),
        "corpus_covers": True,
    }
    fields.update(overrides)
    return evaluation.Question(**fields)  # type: ignore[arg-type]


def an_answer(**overrides: object) -> evaluation.Answer:
    """One recorded answer."""
    fields: dict[str, object] = {
        "question": a_question(),
        "verdict": evaluation.ANSWERED,
        "tools": ("get_machine_trend",),
        "citation_count": 0,
        "ungrounded": (),
        "latency_seconds": 1.0,
    }
    fields.update(overrides)
    return evaluation.Answer(**fields)  # type: ignore[arg-type]


def test_the_question_set_loads() -> None:
    """The committed file parses, and every question names real fields."""
    questions = evaluation.read_questions(QUESTIONS_PATH)

    assert len(questions) >= 8
    assert any(question.id == "the-negative-control" for question in questions)
    assert any(question.corpus_covers for question in questions)
    assert any(not question.corpus_covers for question in questions)


def test_an_empty_question_set_is_refused(tmp_path: Path) -> None:
    """A scorecard over nothing reports a perfect score.

    Worse than no scorecard, because it looks like evidence.
    """
    empty = tmp_path / "questions.json"
    empty.write_text(json.dumps({"questions": []}), encoding="utf-8")

    with pytest.raises(CopilotEvaluationError, match="no questions"):
        evaluation.read_questions(empty)


def test_the_frames_of_a_stream_are_parsed() -> None:
    """The wire format, pinned by a hand-written stream.

    The API's frame shape is mirrored by hand here, as it is in the browser --
    a `text/event-stream` body has no schema -- so this is the test that fails
    when the server's frame names change.
    """
    stream = [
        'event: tool\ndata: {"tool": "get_machine_trend", "evidence_count": 2}\n\n',
        'event: token\ndata: {"text": "Vibration"}\n\n',
        'event: done\ndata: {"verdict": "ANSWERED", "citations": [{"label": "x"}]}\n\n',
    ]
    # Split so the parser meets frames one line at a time, which is what
    # `iter_lines()` does.
    lines = [line for chunk in stream for line in chunk.splitlines()]

    frames = list(evaluation._frames(lines))

    assert [event for event, _ in frames] == ["tool", "token", "done"]
    assert frames[2][1]["verdict"] == "ANSWERED"


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        # A keepalive comment carries nothing and must not become a frame.
        ([" : keepalive", ""], 0),
        # A frame whose data is not JSON is dropped, not raised: one malformed
        # frame should cost one frame, not the run.
        (["event: token", "data: not json", ""], 0),
    ],
)
def test_the_parser_ignores_what_it_cannot_read(stream: list[str], expected: int) -> None:
    """The parser is deliberately forgiving about one frame."""
    assert len(list(evaluation._frames(stream))) == expected


def test_an_answer_is_read_from_the_done_frame() -> None:
    """The verdict, the tools and the citations all come off the wire."""
    body = {
        "verdict": "ANSWERED",
        "citations": [{"label": "a"}, {"label": "b"}],
        "ungrounded": [],
    }
    stream = (
        'event: tool\ndata: {"tool": "get_machine_trend"}\n\n'
        f"event: done\ndata: {json.dumps(body)}\n\n"
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=stream))
    client = httpx.Client(transport=transport)

    answer = evaluation.ask(client, "http://api", a_question())

    assert answer.verdict == "ANSWERED"
    assert answer.tools == ("get_machine_trend",)
    assert answer.citation_count == 2
    assert answer.error is None
    assert answer.tools_are_correct


def test_a_question_that_does_not_ask_the_corpus_expects_no_refusal() -> None:
    """The harness bug that the first live run found, pinned as a test.

    *"How many incidents has M003 had?"* is answered from the incident record
    and the corpus legitimately does not cover it. Expecting a refusal because
    of the second fact alone reported a correctly-answered question as a PRD
    section 19 failure -- three of them, on the deployment.
    """
    reading = a_question(
        id="incidents",
        corpus_covers=False,
        required=frozenset({"get_machine_incidents"}),
    )

    assert reading.expects_refusal is False


def test_a_documentation_question_the_corpus_lacks_does_expect_one() -> None:
    """The other half: same flag, different required tools, opposite answer."""
    control = a_question(
        id="negative-control",
        corpus_covers=False,
        required=frozenset({"search_maintenance_knowledge"}),
    )

    assert control.expects_refusal is True


def test_a_refusal_is_not_an_error() -> None:
    """The distinction the scorecard rests on.

    A refusal is the system working -- PRD section 19 enforced -- and counting
    it as a failure would make the deployment look broken exactly when it is
    behaving.
    """
    body = {"verdict": "REFUSED", "citations": [], "ungrounded": []}
    stream = f"event: done\ndata: {json.dumps(body)}\n\n"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=stream))

    answer = evaluation.ask(httpx.Client(transport=transport), "http://api", a_question())

    assert answer.verdict == evaluation.REFUSED
    assert answer.error is None


def test_an_outage_is_recorded_rather_than_raised() -> None:
    """A 503 is one of the things this measures."""

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"code": "chat_unavailable"}})

    answer = evaluation.ask(
        httpx.Client(transport=httpx.MockTransport(refuse)), "http://api", a_question()
    )

    assert answer.error is not None
    assert "503" in answer.error
    assert answer.verdict == "ERROR"


def test_it_waits_for_the_deployment_to_be_ready() -> None:
    """The false failures the first honest run produced, pinned.

    A rebuild recreates the inference container, which loads three models in
    about twenty seconds. Asking immediately reported two `chat_unavailable` and
    `retrieval_unavailable` errors -- correctly, in that the requests failed, and
    uselessly, because what failed was the deployment's start-up.
    """
    answers = [
        {"dependencies": [{"name": "inference", "reachable": False}]},
        {"dependencies": [{"name": "inference", "reachable": False}]},
        {"dependencies": [{"name": "inference", "reachable": True}]},
    ]
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=answers.pop(0)),
    )
    client = httpx.Client(transport=transport)

    ready = evaluation.wait_for_ready(client, "http://api", sleep=lambda _: None)

    assert ready is True
    assert answers == []


def test_it_gives_up_rather_than_waiting_for_ever() -> None:
    """A dependency that never arrives is reported, not hung on."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"dependencies": [{"name": "inference", "reachable": False}]}
        )
    )

    ready = evaluation.wait_for_ready(
        httpx.Client(transport=transport), "http://api", timeout_seconds=0.0
    )

    assert ready is False


def test_the_rates_are_computed_over_the_right_denominators() -> None:
    """Each rate has its own population, and mixing them would hide a failure.

    Groundedness is over completed answers; abstention is over the questions
    that *expect* a refusal; citations are over documented answers. A single
    denominator would let a refusal count against grounding, which is the one
    thing a refusal is supposed to be.

    The question that expects a refusal must ask the documentation: that pair of
    conditions is what the first version of this harness got wrong, and it
    reported three correctly-answered questions as failures for it.
    """
    card = evaluation.Scorecard(
        answers=(
            an_answer(question=a_question(corpus_covers=True), verdict=evaluation.ANSWERED),
            an_answer(
                question=a_question(
                    id="q2",
                    corpus_covers=False,
                    required=frozenset({"search_maintenance_knowledge"}),
                ),
                verdict=evaluation.REFUSED,
                tools=("search_maintenance_knowledge",),
            ),
            an_answer(
                question=a_question(id="q3", corpus_covers=False),
                verdict=evaluation.FALLBACK,
                ungrounded=("0.81",),
                tools=(),
            ),
        )
    )

    assert card.grounded_rate == pytest.approx(2 / 3)
    # One question expects a refusal -- the one that asks the documentation and
    # the documentation does not cover -- and it was refused. The third is a
    # question the corpus also does not cover, but it never asked it, so it is
    # not in this denominator at all.
    assert card.abstention_rate == pytest.approx(1.0)
    assert card.verdicts[evaluation.ANSWERED] == 1


def test_the_citation_rate_covers_only_documentation_questions() -> None:
    """The second denominator bug this scorecard has had.

    An answer to *"what is the current temperature"* needs no document, so
    including it reports a citation failure for something that was never meant
    to have a citation. The first live run reported 33% for exactly that reason,
    while every answer that did cite something cited it correctly.
    """
    card = evaluation.Scorecard(
        answers=(
            # Needs no document, and carries none: correct, and must not count.
            an_answer(
                question=a_question(id="temperature", required=frozenset({"get_machine_trend"}))
            ),
            # Asks the corpus, cites it: correct, and is the denominator.
            an_answer(
                question=a_question(
                    id="procedure", required=frozenset({"search_maintenance_knowledge"})
                ),
                citation_count=2,
            ),
        )
    )

    assert card.citation_rate == pytest.approx(1.0)


def test_a_documentation_answer_without_a_source_is_reported() -> None:
    """The other half: same denominator, and it can fail."""
    card = evaluation.Scorecard(
        answers=(
            an_answer(
                question=a_question(
                    id="procedure", required=frozenset({"search_maintenance_knowledge"})
                ),
                citation_count=0,
            ),
        )
    )

    assert card.citation_rate == 0.0


def test_a_forbidden_tool_is_named_in_the_problems() -> None:
    """A rate on its own is not actionable; the offending question is."""
    card = evaluation.Scorecard(
        answers=(an_answer(tools=("get_machine_trend", "search_maintenance_knowledge")),)
    )

    rendered = evaluation.render(card)

    assert "search_maintenance_knowledge" in rendered
    assert "must not reach for" in rendered
    assert card.tool_accuracy == 0.0


def test_an_unanswered_unanswerable_question_is_reported() -> None:
    """The failure PRD section 19 exists to prevent, named."""
    card = evaluation.Scorecard(
        answers=(
            an_answer(
                question=a_question(
                    corpus_covers=False,
                    required=frozenset({"search_maintenance_knowledge"}),
                ),
                verdict=evaluation.ANSWERED,
                tools=("search_maintenance_knowledge",),
            ),
        )
    )

    rendered = evaluation.render(card)

    assert "PRD section 19" in rendered
    assert card.abstention_rate == 0.0
