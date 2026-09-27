"""A scorecard for the Copilot, measured against a running stack.

Every number here is mechanical: a verdict, a count, a duration. There is
deliberately **no judge model** -- one would need to be larger than the model
being judged, and it would replace a checkable number with an opinion. What can
be measured without one turns out to be most of what matters:

* whether an unanswerable question was **refused** (AC-009, and the same measure
  `ml knowledge evaluate` reports for retrieval);
* whether an answer was **grounded** -- `ungrounded` is empty, meaning every
  number in the prose appeared in the evidence it was given;
* whether a documented answer came with **citations**;
* and how long a reader waits.

What it cannot measure is whether the prose is *good*, which is the honest gap
ADR 0009 records.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ml.copilot.errors import CopilotEvaluationError

#: Events the API streams, mirrored from `presentation/routers/copilot.py`.
_TOOL_EVENT = "tool"
_TOKEN_EVENT = "token"  # noqa: S105 - an SSE event name, not a secret
_DONE_EVENT = "done"
_ERROR_EVENT = "error"

#: Generous: an answer takes about twenty seconds on the deployment's single
#: core, and the first question after a restart pays the model's load too.
DEFAULT_TIMEOUT_SECONDS = 300.0

#: How many times one question may be re-asked after a rate limit, and the
#: longest it will ever wait for one. A deployment bounding questions to one a
#: minute means eight questions take several minutes; that is the bound working,
#: not a fault, so the evaluation waits rather than reporting it as a failure.
MAX_RATE_LIMIT_WAITS = 8
MAX_WAIT_SECONDS = 120.0

#: Verdicts, as the domain spells them.
ANSWERED = "ANSWERED"
REFUSED = "REFUSED"
FALLBACK = "FALLBACK"


@dataclass(frozen=True, slots=True)
class Question:
    """One question, and what the system owes it."""

    id: str
    text: str
    required: frozenset[str]
    forbidden: frozenset[str]
    #: Whether the maintenance *documentation* covers this question. Not whether
    #: the system can answer it: a question about incidents or telemetry is
    #: answerable from the product's own data and has nothing to do with the
    #: corpus -- which is why the refusal expectation below needs both facts,
    #: not this one alone.
    corpus_covers: bool

    @property
    def expects_refusal(self) -> bool:
        """Whether the system should have refused this question.

        **Both conditions, and the first version used only the second.** A
        refusal is PRD section 19's answer to a question the *documentation*
        cannot support -- so it is only expected when the question actually asks
        the documentation. Three of the set's questions are about what a machine
        has been doing, and the corpus genuinely does not cover them; the system
        answered them from telemetry, incidents and the model, which is the
        product working. The harness called all three failures.
        """
        return "search_maintenance_knowledge" in self.required and not self.corpus_covers


@dataclass(frozen=True, slots=True)
class Answer:
    """What came back for one question."""

    question: Question
    verdict: str
    tools: tuple[str, ...]
    citation_count: int
    ungrounded: tuple[str, ...]
    latency_seconds: float
    #: Set when the request itself failed -- a 429, a 503, a dropped stream.
    #: Distinct from a refusal, which is the system working.
    error: str | None = None
    #: Set only on a 429, carrying the deployment's own countdown. `ask` waits
    #: this long and tries again rather than recording a limit as a failure.
    retry_after: float | None = None

    @property
    def grounded(self) -> bool:
        """Whether every number in the prose was in the evidence."""
        return not self.ungrounded

    @property
    def selected_forbidden(self) -> frozenset[str]:
        """Tools the plan should not have reached for."""
        return self.question.forbidden & set(self.tools)

    @property
    def tools_are_correct(self) -> bool:
        """Whether every required tool ran and no forbidden one did.

        Required rather than exact: a question can legitimately admit more
        sources than one, so what is checked is that nothing needed is missing
        and nothing excluded has crept in.
        """
        return self.question.required <= set(self.tools) and not self.selected_forbidden


@dataclass(frozen=True, slots=True)
class Scorecard:
    """Every answer, and the numbers derived from them."""

    answers: tuple[Answer, ...]

    @property
    def verdicts(self) -> Counter[str]:
        """How often each verdict came back."""
        return Counter(answer.verdict for answer in self.answers if answer.error is None)

    @property
    def errors(self) -> tuple[Answer, ...]:
        """Questions the stack could not answer for infrastructural reasons."""
        return tuple(answer for answer in self.answers if answer.error is not None)

    @property
    def answerable(self) -> tuple[Answer, ...]:
        """Answers to questions that should have been answered."""
        return tuple(a for a in self.answers if not a.question.expects_refusal and a.error is None)

    @property
    def unanswerable(self) -> tuple[Answer, ...]:
        """Answers to questions that should have been refused."""
        return tuple(a for a in self.answers if a.question.expects_refusal and a.error is None)

    @property
    def grounded_rate(self) -> float:
        """Share of completed answers whose numbers were all in the evidence."""
        completed = [a for a in self.answers if a.error is None]
        if not completed:
            return 0.0
        return sum(1.0 for a in completed if a.grounded) / len(completed)

    @property
    def abstention_rate(self) -> float:
        """Share of unanswerable questions the system refused.

        AC-009's measure, and the one that decides whether PRD section 19 is
        enforced by the system or merely by the model's good manners.
        """
        if not self.unanswerable:
            return 0.0
        refused = sum(1.0 for a in self.unanswerable if a.verdict == REFUSED)
        return refused / len(self.unanswerable)

    @property
    def citation_rate(self) -> float:
        """Share of documented answers that carried at least one source."""
        documented = [a for a in self.answerable if a.verdict in {ANSWERED, FALLBACK}]
        if not documented:
            return 0.0
        return sum(1.0 for a in documented if a.citation_count > 0) / len(documented)

    @property
    def medians(self) -> dict[str, float]:
        """Latency in seconds, over the questions that completed."""
        completed = [a.latency_seconds for a in self.answers if a.error is None]
        if not completed:
            return {}
        return {
            "median": statistics.median(completed),
            "slowest": max(completed),
            "fastest": min(completed),
        }

    @property
    def tool_accuracy(self) -> float:
        """Share of questions whose tool selection was right."""
        completed = [a for a in self.answers if a.error is None]
        if not completed:
            return 0.0
        return sum(1.0 for a in completed if a.tools_are_correct) / len(completed)


def read_questions(path: Path) -> tuple[Question, ...]:
    """Read the committed question set."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CopilotEvaluationError(f"Could not read '{path}': {exc}") from exc

    questions = tuple(
        Question(
            id=entry["id"],
            text=entry["question"],
            required=frozenset(entry.get("required_tools", ())),
            forbidden=frozenset(entry.get("forbidden_tools", ())),
            corpus_covers=bool(entry.get("corpus_covers", False)),
        )
        for entry in manifest.get("questions", ())
    )
    if not questions:
        raise CopilotEvaluationError(
            f"'{path}' contains no questions. A scorecard over nothing reports a "
            "perfect score, which is worse than no scorecard."
        )
    return questions


def ask(client: httpx.Client, api_url: str, question: Question) -> Answer:
    """Ask one question and record what came back, waiting out a rate limit.

    **The waiting is deliberate.** A running deployment bounds the Copilot to
    about one question a minute, and this asks eight in a row: without this the
    third onwards would be refused and the scorecard would report the
    deployment's own limits as failures. The API publishes `Retry-After` for
    exactly this, so the evaluation reads it and waits rather than hammering --
    which also means a run against the deployed stack takes several minutes, and
    says so on stderr while it waits.

    Never raises on a failed request: an outage is one of the things this
    measures, and a scorecard that stopped at the first error would report a
    broken deployment as a crash.
    """
    for attempt in range(MAX_RATE_LIMIT_WAITS + 1):
        # Each attempt times itself, so a question that waited 57 seconds to be
        # asked is not reported as a 57-second answer.
        answer = _ask_once(client, api_url, question)
        if answer.retry_after is None:
            return answer
        if attempt == MAX_RATE_LIMIT_WAITS:
            return answer
        wait = min(answer.retry_after, MAX_WAIT_SECONDS)
        print(
            f"  rate limited; waiting {wait:.0f}s for {question.id}",
            file=sys.stderr,
        )
        time.sleep(wait)
    return answer  # pragma: no cover - the loop returns


def _ask_once(client: httpx.Client, api_url: str, question: Question) -> Answer:
    """Ask one question once, recording whatever came back."""
    started = time.perf_counter()
    tools: list[str] = []
    done: dict[str, Any] = {}
    error: str | None = None
    retry_after: float | None = None

    try:
        with client.stream(
            "POST",
            f"{api_url.rstrip('/')}/api/v1/copilot/chat",
            json={"question": question.text},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        ) as response:
            if response.status_code == 429:
                response.read()
                retry_after = float(response.headers.get("retry-after", 60))
                error = f"HTTP 429: {response.text[:200]}"
            elif response.status_code >= 400:
                response.read()
                error = f"HTTP {response.status_code}: {response.text[:200]}"
            else:
                for event, body in _frames(response.iter_lines()):
                    if event == _TOOL_EVENT:
                        tools.append(str(body.get("tool", "")))
                    elif event == _DONE_EVENT:
                        done = body
                    elif event == _ERROR_EVENT:
                        error = f"{body.get('code')}: {body.get('message')}"
    except httpx.HTTPError as exc:
        error = f"{type(exc).__name__}: {exc}"

    # Measured from this attempt's start, so the wait above is excluded: a
    # latency that included the harness pacing itself would report the
    # deployment's rate limit rather than its speed.
    latency = time.perf_counter() - started
    if error is not None:
        return Answer(
            question=question,
            verdict="ERROR",
            tools=tuple(tools),
            citation_count=0,
            ungrounded=(),
            latency_seconds=latency,
            error=error,
            retry_after=retry_after,
        )

    return Answer(
        question=question,
        verdict=str(done.get("verdict", "ERROR")),
        tools=tuple(tools),
        citation_count=len(done.get("citations", ())),
        ungrounded=tuple(done.get("ungrounded", ())),
        latency_seconds=latency,
        error=None if done else "the stream ended without a verdict",
        retry_after=None,
    )


def _frames(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (event, payload) pairs from a server-sent event stream.

    A small parser of its own rather than a shared one: the API's frames are
    three fields wide, this is the only Python consumer, and the browser's
    parser is TypeScript in another package.
    """
    event: str | None = None
    data: list[str] = []
    for line in lines:
        if line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            data.append(line[len("data: ") :])
        elif not line.strip():
            if event and data:
                # A frame this cannot parse is dropped rather than raised: a
                # scorecard is a measurement, and one malformed frame should
                # cost one question rather than the whole run.
                with suppress(json.JSONDecodeError):
                    yield event, json.loads("\n".join(data))
            event, data = None, []


def measure(client: httpx.Client, api_url: str, questions: tuple[Question, ...]) -> Scorecard:
    """Ask every question, in order, and collect the answers."""
    return Scorecard(answers=tuple(ask(client, api_url, question) for question in questions))


def as_payload(card: Scorecard) -> dict[str, Any]:
    """Render the scorecard as JSON, for `--out`."""
    return {
        "verdicts": dict(card.verdicts),
        "grounded_rate": round(card.grounded_rate, 4),
        "abstention_rate": round(card.abstention_rate, 4),
        "citation_rate": round(card.citation_rate, 4),
        "tool_accuracy": round(card.tool_accuracy, 4),
        "latency": {key: round(value, 3) for key, value in card.medians.items()},
        "answers": [
            {
                "id": answer.question.id,
                "verdict": answer.verdict,
                "tools": list(answer.tools),
                "citations": answer.citation_count,
                "ungrounded": list(answer.ungrounded),
                "latency_seconds": round(answer.latency_seconds, 3),
                "error": answer.error,
            }
            for answer in card.answers
        ],
    }


def render(card: Scorecard) -> str:
    """Render the scorecard as a table, with the failures underneath."""
    rows = [
        ("questions", str(len(card.answers))),
        ("answered / refused / fallback", _verdict_counts(card)),
        ("grounded answers", _percent(card.grounded_rate)),
        ("refused when unanswerable", _percent(card.abstention_rate)),
        ("documented answers with a source", _percent(card.citation_rate)),
        ("tool selection correct", _percent(card.tool_accuracy)),
        ("errors", str(len(card.errors))),
    ]
    if card.medians:
        rows.append(
            (
                "latency (median / fastest / slowest)",
                f"{card.medians['median']:.1f}s / {card.medians['fastest']:.1f}s / "
                f"{card.medians['slowest']:.1f}s",
            )
        )

    width = max(len(label) for label, _ in rows)
    lines = ["Copilot scorecard", ""]
    lines.extend(f"{label:<{width}}  {value}" for label, value in rows)

    problems = _problems(card)
    if problems:
        lines.append("")
        lines.append("What to look at")
        lines.extend(f"  {problem}" for problem in problems)

    return "\n".join(lines)


def _verdict_counts(card: Scorecard) -> str:
    """Counts in a fixed order, so two runs are compared by eye."""
    counts = card.verdicts
    return " / ".join(str(counts.get(verdict, 0)) for verdict in (ANSWERED, REFUSED, FALLBACK))


def _percent(value: float) -> str:
    """Render a rate, with the raw fraction beside it when it is not clean."""
    return f"{value:.0%}"


def _problems(card: Scorecard) -> list[str]:
    """Name each failure with the question that caused it.

    A rate on its own is not actionable: "75% grounded" says nothing about
    which quarter was not, and the offending number is the thing to look at.
    """
    problems: list[str] = []
    for answer in card.errors:
        problems.append(f"{answer.question.id}: {answer.error}")
    for answer in card.answers:
        if answer.error is not None:
            continue
        if not answer.grounded:
            problems.append(
                f"{answer.question.id}: stated {list(answer.ungrounded)}, which is in "
                "none of the evidence it was given"
            )
        if answer.selected_forbidden:
            problems.append(
                f"{answer.question.id}: selected {sorted(answer.selected_forbidden)}, "
                "which this question must not reach for"
            )
        missing = answer.question.required - set(answer.tools)
        if missing:
            problems.append(f"{answer.question.id}: never consulted {sorted(missing)}")
        if answer.question.expects_refusal and answer.verdict != REFUSED:
            problems.append(
                f"{answer.question.id}: answered a question the documentation cannot "
                f"support ({answer.verdict}) -- PRD section 19 wants a refusal"
            )
    return problems
