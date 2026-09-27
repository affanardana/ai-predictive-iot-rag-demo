"""What the Copilot asks for, and what a tool hands back."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from api.domain.value_objects.evidence import Evidence
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.time_window import TimeWindow


class CopilotTool(StrEnum):
    """The tools `MASTERPLAN.md` section 6 names, by those names.

    The values are the names a reader of that section would look for, and the
    strand of it that matters is the negative one: a question about what a
    machine has been doing must not reach for the documentation, and a question
    about what to inspect must.
    """

    CURRENT_STATE = "get_machine_current_state"
    TELEMETRY = "get_machine_telemetry"
    TREND = "get_machine_trend"
    PREDICTION = "get_machine_prediction"
    INCIDENTS = "get_machine_incidents"
    HISTORY = "get_machine_history"
    KNOWLEDGE = "search_maintenance_knowledge"


@dataclass(frozen=True, slots=True)
class CopilotPlan:
    """Which sources a question needs, decided before anything is fetched.

    Ordered, and deduplicated: cheap context first, retrieval last, because the
    list is also the sequence of tool-activity frames the browser sees.
    """

    question: str
    tools: tuple[CopilotTool, ...]
    machine_id: MachineId | None
    window: TimeWindow = TimeWindow.FIVE_HOURS
    #: What to search the corpus for. The question itself unless the planner
    #: narrowed it, which keeps the retrieval honest: a rewritten query is a
    #: second place the meaning can go wrong.
    search_query: str | None = None

    def __post_init__(self) -> None:
        """Validate that the plan names something to do."""
        if not self.question.strip():
            raise ValueError("CopilotPlan question must not be blank.")
        if not self.tools:
            raise ValueError("CopilotPlan must name at least one tool.")
        if self.machine_id is not None and not (MACHINE_TOOLS & set(self.tools)):
            raise ValueError(
                "CopilotPlan names a machine but no tool that reads one, so the "
                "identifier would never be used."
            )


#: The tools that read a specific machine. `search_maintenance_knowledge` is
#: the one that does not: it reads the corpus, and a question about a procedure
#: needs no machine at all.
MACHINE_TOOLS: frozenset[CopilotTool] = frozenset(
    {
        CopilotTool.CURRENT_STATE,
        CopilotTool.TELEMETRY,
        CopilotTool.TREND,
        CopilotTool.PREDICTION,
        CopilotTool.INCIDENTS,
        CopilotTool.HISTORY,
    }
)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What one tool found.

    `summary` is the line the model reads; `evidence` is the same finding with
    its provenance attached, which is what the reader sees. The split is
    deliberate — a citation must never depend on the model having reproduced
    it — and it is why a tool returns both rather than one rendered string.
    """

    tool: CopilotTool
    summary: str
    evidence: tuple[Evidence[str], ...] = ()

    def __post_init__(self) -> None:
        """Validate that the result says something."""
        if not self.summary.strip():
            raise ValueError(f"ToolResult for {self.tool} must carry a summary.")


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool's execution, as the browser sees it in the activity trail."""

    tool: CopilotTool
    summary: str
    evidence_count: int


#: What a tool hands the composer. Aliased because the generic would otherwise
#: appear in every signature in the application layer.
#:
#: Values are rendered by the tool that produces them, at the precision it
#: chooses, because the grounding check compares what the model wrote against
#: what it was given: a number rounded on the way in would make a faithful
#: answer look invented, and one rounded on the way out would hide a real
#: fabrication.
ToolEvidence = Evidence[str]
