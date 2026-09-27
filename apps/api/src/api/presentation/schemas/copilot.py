"""Request and stream shapes for the Copilot."""

from __future__ import annotations

from pydantic import BaseModel, Field

from api.domain.value_objects.copilot import AnswerVerdict, CopilotTool
from api.domain.value_objects.evidence import EvidenceKind
from api.presentation.schemas.knowledge import CitationSchema

#: Long enough for a question with a machine and a metric in it, short enough
#: that the prefill -- which is half the wait on one core -- is bounded by the
#: schema rather than by luck.
MAX_QUESTION_LENGTH = 500


class ConversationTurn(BaseModel):
    """One earlier turn, as the browser remembers it."""

    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=2000)


class AskRequest(BaseModel):
    """A maintenance question."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    #: Sent by the page so a follow-up reads as one. The server does not store
    #: it: conversation history is the page's, and a stored transcript would
    #: need a table, a session and a retention rule that nothing asks for.
    history: list[ConversationTurn] = Field(default_factory=list)


class EvidenceSchema(BaseModel):
    """One finding, with the provenance PRD section 16 requires."""

    kind: EvidenceKind
    text: str
    #: Set exactly when the kind is DOCUMENTED -- `Evidence.__post_init__`
    #: enforces that in the domain, so this is a rendering of a guarantee
    #: rather than a convention the schema asserts.
    source: str | None = None


class ToolCallSchema(BaseModel):
    """One tool's execution, for the activity trail."""

    tool: CopilotTool
    summary: str
    evidence_count: int


class CopilotAnswerSchema(BaseModel):
    """The finished answer and everything behind it."""

    question: str
    verdict: AnswerVerdict
    answer: str
    machine_id: str | None = None
    evidence: list[EvidenceSchema]
    citations: list[CitationSchema]
    tool_calls: list[ToolCallSchema]
    #: Why the answer is a refusal, when it is one.
    reason: str = ""
    #: Numbers the model stated that are in none of the evidence it was given.
    #: Non-empty is what turns an answer into a fallback, and it is reported
    #: rather than quietly corrected.
    ungrounded: list[str] = Field(default_factory=list)
    model_id: str | None = None
