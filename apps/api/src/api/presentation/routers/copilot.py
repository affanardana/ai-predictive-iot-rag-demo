"""The Copilot endpoint.

One route, and it answers `text/event-stream`. What it streams is not the answer
alone: the tool activity frames arrive first, as each source is read, so a
reader watches the evidence being gathered before the prose starts. On a box
where an answer takes about twenty seconds, that is the difference between a
demonstration and a page that looks broken.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.application.use_cases import AskCopilot
from api.domain.errors import DomainError
from api.domain.ports.chat import ChatMessage, ChatRole
from api.domain.value_objects.copilot import ToolCall
from api.observability.metrics import (
    ToolOutcome,
    Verdict,
    record_copilot_run,
    record_copilot_tool,
)
from api.presentation.dependencies import AskCopilotDep, CopilotBudgetDep
from api.presentation.errors.error_catalog import resolve
from api.presentation.presenters import to_copilot_answer, to_tool_call
from api.presentation.schemas.copilot import AskRequest

router = APIRouter(prefix="/copilot", tags=["copilot"])

_SEPARATOR = "\n\n"

#: Frame names. The browser switches on these, and they are mirrored by hand in
#: `apps/web/src/api/stream.ts` — a `text/event-stream` body cannot be described
#: in OpenAPI, so it cannot be generated, and a test on each side pins them.
TOOL_EVENT = "tool"
TOKEN_EVENT = "token"  # noqa: S105 - an SSE event name, not a secret
DONE_EVENT = "done"
ERROR_EVENT = "error"


@router.post(
    "/chat",
    # First, so an over-budget caller is refused before the single answer slot
    # is even considered: a 429 and a 409 are different conversations, and the
    # budget is the cheaper one to answer.
    dependencies=[CopilotBudgetDep],
    summary="Ask the Copilot a question",
)
async def chat(request: AskRequest, use_case: AskCopilotDep) -> StreamingResponse:
    """Answer a maintenance question, streaming the work as it happens.

    **Unguarded**, like the other browser-facing routes, and bounded instead —
    see ADR 0009. The bound that matters runs first: the single answer slot is
    claimed *before* this returns a response, because a 409 that arrives as a
    frame inside a 200 is no longer a status code a client can act on.

    The failure modes are split deliberately. A refusal (the evidence does not
    support an answer) and a fallback (the prose failed its grounding check) are
    200s carrying a verdict, because the request was fine and the system is
    answering honestly. A chat outage or a busy Copilot are real statuses,
    because those are about the deployment rather than about the question.
    """
    use_case.claim()
    history = tuple(
        ChatMessage(role=ChatRole(turn.role), content=turn.content) for turn in request.history
    )
    return StreamingResponse(
        _frames(use_case, request.question, history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
        },
    )


async def _frames(
    use_case: AskCopilot,
    question: str,
    history: tuple[ChatMessage, ...],
) -> AsyncIterator[str]:
    """Run the Copilot and emit frames as it reports progress.

    The use case is awaited in a task while this generator drains a queue,
    because the two have to make progress together: the sink is what the use
    case calls, and only this generator can yield. Cancelling the task in the
    `finally` is what stops a browser that navigated away from leaving a
    twenty-second generation running on a one-core box.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    class Sink:
        """Writes the use case's progress into the queue as frames."""

        async def tool(self, call: ToolCall) -> None:
            """Report that a tool has run."""
            await queue.put(_frame(TOOL_EVENT, to_tool_call(call).model_dump(mode="json")))

        async def token(self, text: str) -> None:
            """Report a piece of the answer."""
            await queue.put(_frame(TOKEN_EVENT, {"text": text}))

    async def run() -> None:
        try:
            answer = await use_case.execute(question, sink=Sink(), history=history)
            # Recorded here rather than inside the use case: the verdict and the
            # tool trail are on the answer the presentation layer already holds,
            # and a metric is not something `api.application` should know about.
            record_copilot_run(verdict=Verdict.of(answer.verdict.value))
            for call in answer.tool_calls:
                record_copilot_tool(
                    tool=call.tool.value,
                    outcome=ToolOutcome.FOUND if call.evidence_count else ToolOutcome.EMPTY,
                )
            await queue.put(_frame(DONE_EVENT, to_copilot_answer(answer).model_dump(mode="json")))
        except DomainError as exc:
            # An outage is not a verdict the Copilot reached, but it is what a
            # reader experiences: a question that produced no answer. Recorded
            # as its own value so a rising `error` count is visible rather than
            # hidden among the refusals.
            record_copilot_run(verdict=Verdict.ERROR)
            # The response is already a 200 by now, so a failure has to be a
            # frame. It carries the same code the error catalog would have put
            # in the envelope, so a client handles both the same way.
            await queue.put(_frame(ERROR_EVENT, _error_body(exc)))
        finally:
            use_case.release()
            await queue.put(None)

    task = asyncio.create_task(run())
    try:
        while (frame := await queue.get()) is not None:
            yield frame
    finally:
        if not task.done():
            task.cancel()


def _frame(event: str, body: dict[str, object]) -> str:
    """Render one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(body, ensure_ascii=False)}{_SEPARATOR}"


def _error_body(exc: DomainError) -> dict[str, object]:
    """Render a failure as the envelope's fields, for a channel with no status."""
    mapping = resolve(exc)
    return {"code": mapping.code, "status": mapping.status_code, "message": str(exc)}
