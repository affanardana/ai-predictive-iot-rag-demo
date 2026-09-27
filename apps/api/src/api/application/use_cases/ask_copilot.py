"""Use case: answer a question about a machine, from evidence.

The order of operations is the design. Plan, gather, decide whether an answer is
possible, and only then write one — so that the refusal PRD section 19 requires
is a decision the system makes rather than a request it makes of a language
model. Everything the model is given has already been checked; everything it
produces is checked again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from api.application.use_cases.get_machine_detail import GetMachineDetail
from api.application.use_cases.get_prediction_history import GetPredictionHistory
from api.application.use_cases.get_telemetry_history import GetTelemetryHistory
from api.application.use_cases.list_incidents import ListIncidents
from api.application.use_cases.list_machines import ListMachines
from api.application.use_cases.search_maintenance_knowledge import SearchMaintenanceKnowledge
from api.domain.errors import ChatBusyError, MachineNotFoundError
from api.domain.ports.chat import ChatMessage, ChatModel, ChatRole
from api.domain.read_models import TelemetrySeries
from api.domain.services.copilot_plan import plan_question
from api.domain.services.grounding import ungrounded_numbers
from api.domain.services.trend import DEADBAND_RATIO, TrendDirection, summarise
from api.domain.value_objects.citation import Citation
from api.domain.value_objects.copilot import (
    AnswerVerdict,
    CopilotPlan,
    CopilotTool,
    ToolCall,
    ToolResult,
)
from api.domain.value_objects.evidence import Evidence
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.time_window import TimeWindow

#: How many passages the answer may draw on. Fewer than the retrieval layer's own
#: five, because this text goes into a prompt that a one-core box must prefill.
DOCUMENTED_LIMIT = 3

#: The longest answer worth waiting for. A demonstration answer is three
#: sentences; every extra token is a second of a reader's attention.
MAX_ANSWER_TOKENS = 180

#: PRD section 19's sentence, used verbatim. The requirement names it as the
#: example of what the system should say, and a refusal written by the system
#: cannot be paraphrased into a procedure.
INSUFFICIENT_EVIDENCE = (
    "The available documentation is insufficient to support that recommendation."
)

NOTHING_RECORDED = "There is nothing recorded for this machine to answer from."

UNKNOWN_MACHINE = "That machine is not registered, so there is nothing to look up."


class EventSink(Protocol):
    """Where the use case reports progress as it happens.

    The route implements this with SSE frames; tests implement it by recording;
    the default does nothing. Keeping the sink out of the return value is what
    lets the same use case serve a streaming client and a test that just wants
    the finished answer.
    """

    async def tool(self, call: ToolCall) -> None:
        """Report that a tool has run."""
        ...

    async def token(self, text: str) -> None:
        """Report a piece of the answer as it is written."""
        ...


class _NoEvents:
    """The sink used when the caller is not watching."""

    async def tool(self, call: ToolCall) -> None:
        """Discard."""
        del call

    async def token(self, text: str) -> None:
        """Discard."""
        del text


@dataclass(frozen=True, slots=True)
class CopilotAnswer:
    """A question, the evidence found for it, and what was made of that."""

    question: str
    verdict: AnswerVerdict
    answer: str
    machine_id: MachineId | None = None
    evidence: tuple[Evidence[str], ...] = ()
    citations: tuple[Citation, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    #: Why the answer is a refusal, when it is one.
    reason: str = ""
    #: Numbers the model stated that are in no evidence it was given. Empty for
    #: a verified answer; non-empty is what forces `FALLBACK`.
    ungrounded: tuple[str, ...] = ()
    #: Which model wrote the prose, or None when none was called.
    model_id: str | None = None


@dataclass(frozen=True, slots=True)
class AskCopilot:
    """Answer a maintenance question from the system's own evidence.

    Every dependency here is an existing use case: the Copilot invents no reads
    of its own, so it cannot reach data the rest of the product does not already
    account for, and a change to what a machine's state *is* reaches it for free.
    """

    list_machines: ListMachines
    get_machine_detail: GetMachineDetail
    get_telemetry_history: GetTelemetryHistory
    get_prediction_history: GetPredictionHistory
    list_incidents: ListIncidents
    search_knowledge: SearchMaintenanceKnowledge
    chat: ChatModel
    max_tokens: int = MAX_ANSWER_TOKENS

    #: How many questions may be answered at once, from
    #: `COPILOT_MAX_CONCURRENT_QUESTIONS` by way of `Settings`. One, and the
    #: count is the whole bound: a generation burns the single core for about
    #: twenty seconds, so a second question would halve the speed of the first
    #: and leave both readers waiting twice as long.
    max_concurrent: int = 1

    #: `compare=False` because it is a resource rather than part of what makes
    #: one use case equal another.
    _in_flight: int = field(default=0, compare=False)

    def claim(self) -> None:
        """Take the single answer slot, or refuse.

        Called by the route *before* it commits to a streaming response. That
        ordering is the reason this is a method rather than something `execute`
        does for itself: once a 200 has begun, a refusal can only be a frame,
        and a busy Copilot should be a 409 like any other conflict.
        """
        if self._in_flight >= self.max_concurrent:
            raise ChatBusyError
        object.__setattr__(self, "_in_flight", self._in_flight + 1)

    def release(self) -> None:
        """Give the slot back, whatever happened."""
        object.__setattr__(self, "_in_flight", max(0, self._in_flight - 1))

    async def execute(
        self,
        question: str,
        *,
        sink: EventSink | None = None,
        history: Sequence[ChatMessage] = (),
    ) -> CopilotAnswer:
        """Answer `question`, or refuse to.

        Raises:
            DomainValidationError: if the question is blank.
            ChatUnavailableError: if the model could not write an answer. The
                caller may still have been shown the evidence by then.
        """
        events = sink or _NoEvents()
        plan = plan_question(question)

        found: list[ToolResult] = []
        for tool in plan.tools:
            try:
                result = await self._call(tool, plan)
            except MachineNotFoundError:
                # Nothing else can succeed, and it is a refusal rather than an
                # answer: "that machine is not registered" is a fact about the
                # request, and handing it to the model to expand on is how a
                # question about a typo becomes a paragraph about a machine that
                # does not exist.
                missing = ToolCall(tool=tool, summary=UNKNOWN_MACHINE, evidence_count=0)
                await events.tool(missing)
                return CopilotAnswer(
                    question=plan.question,
                    verdict=AnswerVerdict.REFUSED,
                    answer=UNKNOWN_MACHINE,
                    machine_id=plan.machine_id,
                    tool_calls=(*_calls_of(found), missing),
                    reason=UNKNOWN_MACHINE,
                )
            if result is None:
                # A tool that found nothing is still worth reporting, for the
                # two that read a window. Without it, an answer written from a
                # stale machine's single reading gives no sign that the trend
                # was looked for and missing -- and "why is it *becoming* risky"
                # answered without a trend reads as an oversight rather than as
                # an absence of data. It contributes no evidence, so it cannot
                # change what the answer says.
                nothing = _nothing_found(tool, plan)
                if nothing is not None:
                    await events.tool(nothing)
                continue
            found.append(result)
            await events.tool(result.call)

        refusal = self._refusal(plan, found)
        if refusal is not None:
            return CopilotAnswer(
                question=plan.question,
                verdict=AnswerVerdict.REFUSED,
                answer=refusal,
                machine_id=plan.machine_id,
                evidence=_evidence_of(found),
                citations=_citations_of(found),
                tool_calls=_calls_of(found),
                reason=refusal,
            )

        findings = tuple(result.summary for result in found)
        prose = ""
        async for chunk in self.chat.stream(
            _messages(plan, findings, history), max_tokens=self.max_tokens
        ):
            prose += chunk
            await events.token(chunk)

        ungrounded = ungrounded_numbers(prose, findings)
        if ungrounded:
            # The evidence was good and the prose was not. Serving the evidence
            # rather than the prose is the whole reason the check exists, and
            # saying so is what keeps a caught fabrication visible.
            return CopilotAnswer(
                question=plan.question,
                verdict=AnswerVerdict.FALLBACK,
                answer=_render(found),
                machine_id=plan.machine_id,
                evidence=_evidence_of(found),
                citations=_citations_of(found),
                tool_calls=_calls_of(found),
                reason=(
                    "The assistant's summary was withheld: it stated values that are "
                    "not in the evidence. The findings are shown instead."
                ),
                ungrounded=ungrounded,
                model_id=self.chat.model_id,
            )

        return CopilotAnswer(
            question=plan.question,
            verdict=AnswerVerdict.ANSWERED,
            answer=prose.strip(),
            machine_id=plan.machine_id,
            evidence=_evidence_of(found),
            citations=_citations_of(found),
            tool_calls=_calls_of(found),
            model_id=self.chat.model_id,
        )

    async def _call(self, tool: CopilotTool, plan: CopilotPlan) -> ToolResult | None:
        """Dispatch one tool against the use case that owns the data."""
        machine = plan.machine_id
        if tool is CopilotTool.CURRENT_STATE:
            return await self._current_state(machine)
        if tool is CopilotTool.TELEMETRY:
            return await self._telemetry(machine, plan.window)
        if tool is CopilotTool.TREND:
            return await self._trend(machine, plan.window)
        if tool is CopilotTool.PREDICTION:
            return await self._prediction(machine)
        if tool is CopilotTool.INCIDENTS:
            return await self._incidents(machine)
        if tool is CopilotTool.HISTORY:
            return await self._history(machine)
        return await self._knowledge(plan)

    async def _current_state(self, machine_id: MachineId | None) -> ToolResult | None:
        """What the machine is doing now."""
        if machine_id is None:
            return None
        detail = await self.get_machine_detail.execute(machine_id)
        summary = detail.summary
        reading = summary.latest_reading
        observed = (
            f"{machine_id} last reported at {reading.recorded_at:%Y-%m-%d %H:%M} with "
            f"vibration {reading.reading.vibration:g} mm/s, temperature "
            f"{reading.reading.temperature:g}C, rpm {reading.reading.rpm:g}."
            if reading is not None
            else f"{machine_id} has never reported a reading."
        )
        findings = [Evidence.observed(observed)]
        context = [observed]
        if summary.latest_prediction is not None:
            prediction = summary.latest_prediction
            predicted = (
                f"The model put the 60-minute failure probability at "
                f"{prediction.probability.value:g} ({prediction.risk_level.value})."
            )
            findings.append(Evidence.predicted(predicted))
            context.append(predicted)
        return ToolResult(
            tool=CopilotTool.CURRENT_STATE,
            summary=" ".join(context),
            evidence=tuple(findings),
        )

    async def _telemetry(
        self, machine_id: MachineId | None, window: TimeWindow
    ) -> ToolResult | None:
        """What the machine has been reporting, summarised rather than dumped.

        `PRD.md` section 15 is explicit that the LLM receives summarized
        analytical information rather than hundreds of raw records; the same
        bound is what keeps the prompt inside a one-core box's patience.
        """
        if machine_id is None:
            return None
        series = await self.get_telemetry_history.execute(machine_id, window)
        if not series.points:
            return None
        first, last = series.points[0], series.points[-1]
        statement = (
            f"{machine_id} has {len(series.points)} points over the {window.value} window "
            f"({_resolution(series)}). Vibration moved from {first.reading.vibration:g} to "
            f"{last.reading.vibration:g} mm/s, temperature from "
            f"{first.reading.temperature:g} to {last.reading.temperature:g}C."
        )
        return ToolResult(
            tool=CopilotTool.TELEMETRY,
            summary=statement,
            evidence=(Evidence.observed(statement),),
        )

    async def _trend(self, machine_id: MachineId | None, window: TimeWindow) -> ToolResult | None:
        """Which way the signals are going, with the fit stated separately."""
        if machine_id is None:
            return None
        series = await self.get_telemetry_history.execute(machine_id, window)
        trends = summarise(series)
        if not trends:
            return None
        # Only the signals that moved. A trend is a report of movement, and a
        # fitted change of -0.008 across a window is not one -- it is the
        # deadband's own definition of noise, and eleven more lines of it would
        # bury the four that matter on the page and in the prompt alike.
        moved = [trend for trend in trends if trend.direction is not TrendDirection.FLAT]
        resolution = trends[0].resolution
        if moved:
            # Both claims per signal, each labelled: what was read, and what was
            # derived from it. A reader can check the first against the charts.
            evidence = tuple(
                item
                for trend in moved
                for item in (
                    Evidence.observed(
                        f"{machine_id} {trend.endpoints} over the {window.value} window."
                    ),
                    Evidence.inferred(f"{machine_id} {trend.fit}."),
                )
            )
            summary = "; ".join(trend.summary for trend in moved)
        else:
            # Said rather than omitted. Six flat signals and no trend tool at
            # all look the same to a reader, and one of them means the machine
            # is steady.
            evidence = (
                Evidence.inferred(
                    f"No signal on {machine_id} moved by more than "
                    f"{DEADBAND_RATIO:.0%} of its own value over the {window.value} window."
                ),
            )
            summary = "no signal moved beyond the deadband"
        return ToolResult(
            tool=CopilotTool.TREND,
            summary=f"{machine_id} over the {window.value} window ({resolution}): {summary}",
            # The window is in the trail line as well as in the summary: a
            # trail that says "vibration rising" without saying over what is a
            # claim the reader cannot check, and the trail is what they see.
            display=(
                f"{len(moved)} of {len(trends)} signals moved over the {window.value} window: "
                + ", ".join(f"{trend.signal} {trend.direction.value.lower()}" for trend in moved)
                if moved
                else f"no signal moved over the {window.value} window"
            ),
            evidence=evidence,
        )

    async def _prediction(self, machine_id: MachineId | None) -> ToolResult | None:
        """What the model says, and how it has moved."""
        if machine_id is None:
            return None
        history = await self.get_prediction_history.execute(machine_id)
        if not history:
            return None
        latest = history[0]
        statement = (
            f"The latest prediction for {machine_id} is a "
            f"{latest.probability.value:g} failure probability within "
            f"{int(latest.horizon.total_seconds() // 60)} minutes, band "
            f"{latest.risk_level.value}, "
            f"recorded {latest.predicted_at:%Y-%m-%d %H:%M} by model "
            f"{latest.model_version}."
        )
        if len(history) > 1:
            # The movement, which is the one thing this tool can say that the
            # current-state tool cannot -- and the closest thing to a trend when
            # a machine's telemetry is older than the window. Without it the two
            # tools report the same number in two sentences, and the reader sees
            # the same `PREDICTED` line twice.
            earliest = history[-1]
            statement += (
                f" It is the newest of {len(history)} on record; the earliest, "
                f"{earliest.predicted_at:%Y-%m-%d %H:%M}, was "
                f"{earliest.probability.value:g}."
            )
        return ToolResult(
            tool=CopilotTool.PREDICTION,
            summary=statement,
            evidence=(Evidence.predicted(statement),),
        )

    async def _incidents(self, machine_id: MachineId | None) -> ToolResult | None:
        """What has already been raised."""
        if machine_id is None:
            return None
        incidents = await self.list_incidents.execute(machine_id=machine_id)
        if not incidents:
            statement = f"{machine_id} has no incidents on record."
            return ToolResult(
                tool=CopilotTool.INCIDENTS,
                summary=statement,
                evidence=(Evidence.observed(statement),),
            )
        latest = incidents[0]
        statement = (
            f"{machine_id} has {len(incidents)} incidents on record; the most recent is "
            f"{latest.severity.value} severity, {latest.status.value}, detected "
            f"{latest.detected_at:%Y-%m-%d %H:%M}."
        )
        return ToolResult(
            tool=CopilotTool.INCIDENTS,
            summary=statement,
            evidence=(Evidence.observed(statement),),
        )

    async def _history(self, machine_id: MachineId | None) -> ToolResult | None:
        """A compressed timeline: when it was registered, and what followed.

        The weakest of the seven, and it says so by being a summary rather than
        a new read: everything in it comes from the incident and prediction
        histories the other tools already fetch.
        """
        if machine_id is None:
            return None
        detail = await self.get_machine_detail.execute(machine_id)
        incidents = await self.list_incidents.execute(machine_id=machine_id)
        history = await self.get_prediction_history.execute(machine_id)
        statement = (
            f"{machine_id} was registered {detail.summary.machine.registered_at:%Y-%m-%d}; "
            f"since then {len(history)} predictions and {len(incidents)} incidents have "
            f"been recorded."
        )
        return ToolResult(
            tool=CopilotTool.HISTORY,
            summary=statement,
            evidence=(Evidence.observed(statement),),
        )

    async def _knowledge(self, plan: CopilotPlan) -> ToolResult | None:
        """What the documentation says."""
        query = plan.search_query or plan.question
        result = await self.search_knowledge.execute(query, limit=DOCUMENTED_LIMIT)
        if not result.matches:
            return None
        evidence = tuple(
            Evidence.documented(match.chunk.content, source=match.citation.label)
            for match in result.matches
        )
        # Two renderings, because there are two readers. The model gets the
        # passages, since it cannot cite what it has not read; the activity
        # trail gets the documents, because a trail that prints a thousand
        # characters of procedure is not a trail.
        summary = " ".join(
            f"{match.citation.label}: {match.chunk.content}" for match in result.matches
        )
        titles = ", ".join(dict.fromkeys(match.citation.title for match in result.matches))
        return ToolResult(
            tool=CopilotTool.KNOWLEDGE,
            summary=summary,
            display=f"{len(result.matches)} passages from {titles}.",
            evidence=evidence,
            citations=tuple(match.citation for match in result.matches),
        )

    def _refusal(self, plan: CopilotPlan, found: Sequence[ToolResult]) -> str | None:
        """Return why no answer is possible, or None when one is.

        Decided on the evidence, before the model is asked anything. This is
        where `PRD.md` section 19 lives: the rule is that the system must not
        invent a procedure, and the only enforcement a small model cannot defeat
        is never asking it to.
        """
        if CopilotTool.KNOWLEDGE in plan.tools:
            knowledge = next(
                (result for result in found if result.tool is CopilotTool.KNOWLEDGE),
                None,
            )
            if knowledge is None:
                # The question asked for a procedure and the corpus had nothing
                # that cleared the sufficiency threshold.
                return INSUFFICIENT_EVIDENCE
        machine_findings = [result for result in found if result.tool is not CopilotTool.KNOWLEDGE]
        if not machine_findings and plan.machine_id is None:
            return NOTHING_RECORDED
        if not found:
            return NOTHING_RECORDED
        return None


def _nothing_found(tool: CopilotTool, plan: CopilotPlan) -> ToolCall | None:
    """The activity line for a window tool that came back empty, or None.

    Only the two that read a window, and deliberately. The corpus is the other
    tool that can find nothing, and it does not need a line here: a question
    whose documentation was insufficient is *refused*, in prose, with PRD
    section 19's own sentence. These two are the ones that vanish quietly.
    """
    if tool not in _WINDOW_TOOLS:
        return None
    return ToolCall(
        tool=tool,
        summary=f"Nothing recorded for {plan.machine_id} in the {plan.window.value} window.",
        evidence_count=0,
    )


#: The tools whose emptiness is worth a line in the trail.
_WINDOW_TOOLS = frozenset({CopilotTool.TELEMETRY, CopilotTool.TREND})


def _messages(
    plan: CopilotPlan,
    findings: Sequence[str],
    history: Sequence[ChatMessage],
) -> tuple[ChatMessage, ...]:
    """Build what the model is given: instructions, evidence, question."""
    system = (
        "You are a maintenance assistant for an industrial motor fleet. "
        "Write three sentences at most.\n"
        "Use only the findings below. Do not state any number that is not in them. "
        "Do not describe a procedure that is not in a DOCUMENTED finding. "
        "Do not state that a failure will happen or that a machine is safe. "
        # Added after the first live answer described a measured 1424 rpm as
        # "low" -- a judgement no finding made, and one the grounding check
        # cannot catch because the number itself was real. A 1.5B model follows
        # an explicit prohibition far better than an implied one.
        "Do not describe a value as high, low, rising or falling unless a finding does. "
        "If the findings do not answer the question, say so plainly."
    )
    numbered = "\n".join(f"[{index}] {finding}" for index, finding in enumerate(findings, start=1))
    user = f"Findings:\n{numbered}\n\nQuestion: {plan.question}\nAnswer:"
    return (
        ChatMessage(role=ChatRole.SYSTEM, content=system),
        *history,
        ChatMessage(role=ChatRole.USER, content=user),
    )


def _render(found: Sequence[ToolResult]) -> str:
    """Render the evidence as prose-free findings, with no model involved.

    The `FALLBACK` answer, and the reason the phase degrades rather than fails:
    everything a verified answer would have contained except the sentence.
    """
    lines = ["The findings are listed below. The assistant's summary was withheld."]
    lines.extend(f"- {result.summary}" for result in found)
    return "\n".join(lines)


def _evidence_of(found: Sequence[ToolResult]) -> tuple[Evidence[str], ...]:
    """Every finding, in the order the tools produced them."""
    return tuple(item for result in found for item in result.evidence)


def _citations_of(found: Sequence[ToolResult]) -> tuple[Citation, ...]:
    """The documents that contributed, deduplicated by identity.

    Carried by the tool rather than recovered from the evidence: a documented
    finding's source is a rendered label, and parsing a label back into its four
    fields would be a second place the citation format lives.
    """
    seen: dict[tuple[str, str], Citation] = {}
    for result in found:
        for citation in result.citations:
            seen[(citation.document_key, citation.version)] = citation
    return tuple(seen.values())


def _calls_of(found: Sequence[ToolResult]) -> tuple[ToolCall, ...]:
    """The activity trail, in the order the tools ran."""
    return tuple(result.call for result in found)


def _resolution(series: TelemetrySeries) -> str:
    """Describe a series' resolution, for the telemetry summary.

    Stated rather than implied, for the reason every other response states it: a
    mean of five readings and a reading are different claims.
    """
    bucket = series.resolution.bucket
    if bucket is None:
        return "each stored measurement"
    minutes = int(bucket.total_seconds() // 60) or 1
    return f"mean of {minutes}-minute buckets"
