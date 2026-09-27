"""The planner, scored against a committed question set.

`MASTERPLAN.md` section 6 asks for a Copilot that *"dynamically determines which
information sources are required"*, and ADR 0009 answers that with a pure
function rather than a model call. That trade is only defensible if the function
is measured -- otherwise the plan swapped a 50%-accurate model for an
unmeasured table, which is not obviously better.

**This file is the offline half of `ml copilot evaluate`, and it lives here
rather than there for a layering reason**: `ml` is a separate package that does
not depend on `api`, and its image does not contain it, so a CLI there cannot
import `plan_question`. The question set is shared -- `knowledge/eval/
copilot-questions.json` -- and the online half scores a running stack.

Every failure is reported with the tool it violated, because "3 of 8 failed" is
not a thing anyone can act on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from api.domain.services.copilot_plan import plan_question
from api.domain.value_objects.copilot import CopilotTool

QUESTIONS_PATH = (
    Path(__file__).resolve().parents[5] / "knowledge" / "eval" / "copilot-questions.json"
)


@dataclass(frozen=True, slots=True)
class Expectation:
    """One question, and what the planner owes it."""

    id: str
    question: str
    required: frozenset[str]
    forbidden: frozenset[str]
    note: str


def load_expectations() -> tuple[Expectation, ...]:
    """Read the committed question set."""
    manifest = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return tuple(
        Expectation(
            id=entry["id"],
            question=entry["question"],
            required=frozenset(entry["required_tools"]),
            forbidden=frozenset(entry["forbidden_tools"]),
            note=entry["note"],
        )
        for entry in manifest["questions"]
    )


EXPECTATIONS = load_expectations()


def selected(question: str) -> set[str]:
    """Return the tool names the planner chose for `question`."""
    return {tool.value for tool in plan_question(question).tools}


def test_the_question_set_is_not_empty() -> None:
    """A scoring harness over nothing reports a perfect score.

    Asserted because it is the failure mode of every evaluation: the file moves,
    the loader finds no questions, and the suite goes green having measured
    nothing.
    """
    assert len(EXPECTATIONS) >= 8


@pytest.mark.parametrize("expectation", EXPECTATIONS, ids=lambda e: e.id)
def test_the_planner_selects_what_the_question_needs(expectation: Expectation) -> None:
    """Required tools are present and forbidden ones are absent.

    Required rather than exact, because a question can legitimately admit more
    sources than one: what must hold is that nothing needed is missing and
    nothing excluded has crept in.
    """
    chosen = selected(expectation.question)

    missing = expectation.required - chosen
    unwanted = expectation.forbidden & chosen

    assert not missing and not unwanted, (
        f"{expectation.id}: {expectation.question!r}\n"
        f"  missing: {sorted(missing)}\n"
        f"  forbidden but selected: {sorted(unwanted)}\n"
        f"  selected: {sorted(chosen)}\n"
        f"  why it matters: {expectation.note}"
    )


def test_every_named_tool_is_one_the_system_has() -> None:
    """A typo in the question set would otherwise read as a planner failure.

    The set is data, so nothing type-checks it; this is what stops a renamed
    tool from turning into a mysterious assertion error about a missing source.
    """
    known = {tool.value for tool in CopilotTool}
    named = {
        tool
        for expectation in EXPECTATIONS
        for tool in expectation.required | expectation.forbidden
    }

    assert named <= known, f"the question set names tools that do not exist: {named - known}"


def test_the_demonstration_question_is_covered() -> None:
    """`PRD.md` AC-007 is the question the whole phase is measured on.

    Asserted separately from the table so that removing it -- or renaming its
    id -- fails loudly rather than quietly shrinking the set.
    """
    ids = {expectation.id for expectation in EXPECTATIONS}

    assert "ac-007-the-demonstration-question" in ids
    assert "the-negative-control" in ids


def test_the_two_halves_of_the_negative_example_are_both_here() -> None:
    """The planner's real test is the pair, not either half.

    A question about what a machine has been doing must not read the manuals,
    and a question about what to do must. One without the other can be satisfied
    by a planner that always searches the corpus, or never does.
    """
    reading = next(
        e for e in EXPECTATIONS if e.id == "telemetry-must-not-reach-for-the-documentation"
    )
    procedure = next(e for e in EXPECTATIONS if e.id == "a-procedure-question-needs-the-corpus")

    assert "search_maintenance_knowledge" in reading.forbidden
    assert "search_maintenance_knowledge" in procedure.required
