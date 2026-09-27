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
        "question": "Why is M003 risky?",
        "required": frozenset({"get_machine_trend"}),
        "forbidden": frozenset({"search_maintenance_knowledge"}),
        "answerable": True,
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
    assert any(question.answerable for question in questions)
    assert any(not question.answerable for question in questions)


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


def test_the_rates_are_computed_over_the_right_denominators() -> None:
    """Each rate has its own population, and mixing them would hide a failure.

    Groundedness is over completed answers; abstention is over the
    *unanswerable* ones; citations are over documented answers. A single
    denominator would let a refusal count against grounding, which is the one
    thing a refusal is supposed to be.
    """
    card = evaluation.Scorecard(
        answers=(
            an_answer(question=a_question(answerable=True), verdict=evaluation.ANSWERED),
            an_answer(
                question=a_question(id="q2", answerable=False),
                verdict=evaluation.REFUSED,
                tools=(),
            ),
            an_answer(
                question=a_question(id="q3", answerable=False),
                verdict=evaluation.FALLBACK,
                ungrounded=("0.81",),
                tools=(),
            ),
        )
    )

    assert card.grounded_rate == pytest.approx(2 / 3)
    assert card.abstention_rate == pytest.approx(0.5)
    assert card.verdicts[evaluation.ANSWERED] == 1


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
        answers=(an_answer(question=a_question(answerable=False), verdict=evaluation.ANSWERED),)
    )

    rendered = evaluation.render(card)

    assert "PRD section 19" in rendered
    assert card.abstention_rate == 0.0
