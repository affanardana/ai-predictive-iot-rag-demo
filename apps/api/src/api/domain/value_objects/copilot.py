"""What the Copilot asks for, and what a tool hands back."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from api.domain.value_objects.citation import Citation
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
    #: The documents this result came from. Carried here because a documented
    #: finding's source is a rendered label, and recovering the four citation
    #: fields from it would put the citation format in a second place.
    citations: tuple[Citation, ...] = ()
    #: What the activity trail shows, when it is not `summary`.
    #:
    #: The retrieval tool is why this exists: its summary is the passages
    #: themselves, because the model has to read them, and printing that in a
    #: one-line trail is a paragraph where a line belongs. Only set it when the
    #: two genuinely differ.
    display: str = ""

    def __post_init__(self) -> None:
        """Validate that the result says something."""
        if not self.summary.strip():
            raise ValueError(f"ToolResult for {self.tool} must carry a summary.")

    @property
    def shown(self) -> str:
        """The line the reader sees for this tool in the activity trail."""
        return self.display or self.summary

    @property
    def call(self) -> ToolCall:
        """This result as one line of the activity trail.

        Built here rather than at each call site. There are two -- the frame
        streamed as the tool finishes, and the `tool_calls` on the finished
        answer -- and they are the same line. Building it twice is not
        hypothetical: the retrieval summary reached the page through the second
        one after the first had been fixed.
        """
        return ToolCall(tool=self.tool, summary=self.shown, evidence_count=len(self.evidence))


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool's execution, as the browser sees it in the activity trail."""

    tool: CopilotTool
    summary: str
    evidence_count: int


class AnswerVerdict(StrEnum):
    """What the Copilot did with a question.

    Three states rather than two, and the third is the one that makes the
    grounding check visible: `FALLBACK` means the evidence was good and the
    model's prose was not, so the reader is shown the evidence instead of an
    answer that failed verification.
    """

    ANSWERED = "ANSWERED"
    #: The evidence does not support an answer. The model was never called --
    #: `PRD.md` section 19 is enforced by not asking, which is the only version
    #: of the rule a small model cannot talk its way around.
    REFUSED = "REFUSED"
    #: The evidence was sufficient and the model's answer did not survive the
    #: grounding check, so the deterministic rendering is served in its place.
    FALLBACK = "FALLBACK"


#: What a tool hands the composer. Aliased because the generic would otherwise
#: appear in every signature in the application layer.
#:
#: Values are rendered by the tool that produces them, at the precision it
#: chooses, because the grounding check compares what the model wrote against
#: what it was given: a number rounded on the way in would make a faithful
#: answer look invented, and one rounded on the way out would hide a real
#: fabrication.
ToolEvidence = Evidence[str]
