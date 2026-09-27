"""Which sources a question needs.

`MASTERPLAN.md` section 6 names two worked examples, and the negative half of
each is the assertion that matters: a question about what a machine has been
doing must not reach for the documentation, and a question about what to inspect
must. The rest of this file is the vocabulary those two are built from.
"""

from __future__ import annotations

import pytest

from api.domain.errors import DomainValidationError
from api.domain.services.copilot_plan import plan_question
from api.domain.value_objects.copilot import CopilotTool
from api.domain.value_objects.time_window import TimeWindow


def tools_for(question: str) -> set[CopilotTool]:
    """Return the tools a question selects."""
    return set(plan_question(question).tools)


def test_a_telemetry_question_does_not_search_the_corpus() -> None:
    """`MASTERPLAN.md` section 6's first example, and its point.

    *"What happened to M003 during the last five hours?"* is answered from
    telemetry. Retrieving documentation for it would be the failure `PRD.md`
    AC-006 names: relying on RAG as the telemetry source.
    """
    selected = tools_for("What happened to M003 during the last five hours?")

    assert CopilotTool.KNOWLEDGE not in selected
    assert CopilotTool.TELEMETRY in selected
    assert CopilotTool.TREND in selected
    assert CopilotTool.CURRENT_STATE in selected


def test_a_procedure_question_searches_the_corpus() -> None:
    """The second example: what to inspect is a documentation question."""
    selected = tools_for(
        "M003 has increasing vibration. What should I inspect according to the SOP?"
    )

    assert CopilotTool.KNOWLEDGE in selected
    assert CopilotTool.CURRENT_STATE in selected


def test_a_risk_question_asks_the_model() -> None:
    """`PRD.md` AC-007's question, which needs the prediction as well."""
    selected = tools_for("Why is M003 becoming risky?")

    assert CopilotTool.PREDICTION in selected
    assert CopilotTool.TREND in selected
    assert CopilotTool.CURRENT_STATE in selected


def test_a_question_with_no_machine_reads_the_corpus() -> None:
    """The corpus is the only source that can answer it."""
    plan = plan_question("How do I lubricate a motor bearing?")

    assert plan.machine_id is None
    assert plan.tools == (CopilotTool.KNOWLEDGE,)


def test_a_machine_is_read_from_inside_the_sentence() -> None:
    """Identifiers arrive in prose, not as a field."""
    machine_id = plan_question("what about M003?").machine_id

    assert machine_id is not None
    assert machine_id.value == "M003"


@pytest.mark.parametrize(
    ("question", "window"),
    [
        ("what happened in the last hour?", TimeWindow.ONE_HOUR),
        ("what happened in the last five hours?", TimeWindow.FIVE_HOURS),
        ("how has M003 been today?", TimeWindow.ONE_DAY),
        ("summarise the last week for M003", TimeWindow.SEVEN_DAYS),
        ("how has M003 been over the last 30 days?", TimeWindow.THIRTY_DAYS),
    ],
)
def test_a_named_window_is_read(question: str, window: TimeWindow) -> None:
    """Windows are extracted from phrases, and the phrases are tested.

    Handing this to the model is what the design rejects: a wrong tool selection
    produces a visibly thin answer, whereas a wrong window produces a confident
    answer about the wrong five hours.
    """
    assert plan_question(question).window is window


def test_an_unnamed_window_gets_the_demonstration_default() -> None:
    """`PRD.md` section 15's window, which the demo is built around."""
    assert plan_question("what about M003?").window is TimeWindow.FIVE_HOURS


def test_a_blank_question_is_refused() -> None:
    """A question with no words has no sources to plan."""
    with pytest.raises(DomainValidationError):
        plan_question("   ")


def test_every_tool_is_named_at_most_once() -> None:
    """The plan is also the sequence of activity frames, so order matters."""
    plan = plan_question("why is M003 risky and what should I inspect per the manual?")

    assert len(plan.tools) == len(set(plan.tools))
    assert plan.tools[-1] is CopilotTool.KNOWLEDGE


def test_a_question_naming_a_machine_never_plans_documentation_alone() -> None:
    """A documented claim about a machine is still grounded in that machine.

    `CopilotPlan` refuses to hold a machine no tool would read, which is the
    value object enforcing the same thing one level down.
    """
    selected = tools_for("M003 vibration SOP procedure")

    assert CopilotTool.KNOWLEDGE in selected
    assert any(tool in selected for tool in (CopilotTool.CURRENT_STATE, CopilotTool.TELEMETRY))
