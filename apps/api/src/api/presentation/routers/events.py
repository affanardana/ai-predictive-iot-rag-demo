"""The dashboard's live update stream."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.domain.ports.events import MachineEvent, StreamItem, StreamTick
from api.presentation.dependencies import EventSubscriberDep

router = APIRouter(prefix="/events", tags=["events"])

#: How long a browser should wait before reconnecting. Explicit rather than
#: left to the browser's default, so the recovery interval is a decision.
RECONNECT_DELAY_MS = 3000

#: Frame separator. Two newlines end an event; one separates its fields.
_SEPARATOR = "\n\n"


@router.get("", summary="Stream fleet changes as they happen")
async def stream_events(subscriber: EventSubscriberDep) -> StreamingResponse:
    """Push a hint whenever a machine changes.

    **The payload is a hint, not the data.** Each frame names a machine and what
    changed about it; the client responds by refetching the REST endpoint that
    owns that representation. So the stream never becomes a second source of
    truth, a missed frame is repaired by the next one, and the JSON here has no
    schema to drift from the ones the rest of the API serves.

    The frame shape is deliberately mirrored by hand in the frontend
    (`apps/web/src/realtime/types.ts`), because a `text/event-stream` body
    cannot be described in OpenAPI and so cannot be generated from it.
    `apps/api/tests/integration/test_events_api.py` pins the field names and
    every `EventKind` value, so renaming either here fails the Python suite
    rather than silently freezing the browser.

    Unguarded, and it has to be: `EventSource` cannot set request headers, so no
    `X-Ingest-Token` can ever be presented to this route.
    """
    return StreamingResponse(
        _frames(subscriber),
        media_type="text/event-stream",
        headers={
            # No intermediary may cache or buffer this. `X-Accel-Buffering` is
            # for nginx-alikes; Caddy flushes `text/event-stream` without being
            # asked.
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
        },
    )


async def _frames(subscriber: EventSubscriberDep) -> AsyncIterator[str]:
    """Translate the stream into SSE frames.

    A tick becomes a comment -- `: keepalive` -- which is valid SSE that
    carries no event, so it can never reach an `onmessage` handler and provoke
    a refetch that nothing asked for.
    """
    yield f"retry: {RECONNECT_DELAY_MS}{_SEPARATOR}"

    async for item in subscriber.subscribe():
        yield _format(item)


def _format(item: StreamItem) -> str:
    """Render one stream item as a frame."""
    if isinstance(item, StreamTick):
        return f": keepalive{_SEPARATOR}"

    return f"event: change\ndata: {_payload(item)}{_SEPARATOR}"


def _payload(event: MachineEvent) -> str:
    """Serialise the hint.

    Compact separators because every byte here is sent once per open browser
    per change, and the payload is read by a program rather than a person.
    """
    return json.dumps(
        {"kind": event.kind.value, "machine_id": event.machine_id.value},
        separators=(",", ":"),
    )
