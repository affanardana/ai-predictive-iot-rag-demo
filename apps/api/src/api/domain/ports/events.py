"""Change-notification port, used by the dashboard's live update stream.

**What travels here is a hint, not the data.** A `MachineEvent` says which
machine changed and how; it never carries the reading, the prediction or the
incident itself. A client that receives one refetches the REST endpoint that
owns that representation.

That one decision buys four things, and each of them would otherwise be a
separate piece of machinery:

* There is no second wire format. The presentation layer stays the only place
  that turns a domain object into JSON, so a streamed payload cannot drift
  from the schema served at `/api/v1/machines`.
* A dropped event is survivable, because the next one corrects it. That is
  what makes a *bounded* queue and a drop-on-overflow policy legal rather than
  lossy -- see `InProcessEventBroadcaster`.
* Reconnecting needs no replay buffer and no sequence tracking, since the
  recovery for a gap and the recovery for a fresh connection are the same
  action.
* The non-functional requirement that historical queries not ship large
  datasets to the frontend holds by construction: this stream ships none.

The cost is one REST round trip per event, which the client coalesces.

**Deviation from the convention in `api.domain.ports`:** every other port
method is `async def`. `subscribe` is not, because it returns an async
*generator* -- awaiting it would produce an iterator rather than a value, and
`async for event in await port.subscribe()` reads worse than the plain form.
The convention's intent, that no port method performs blocking I/O, holds.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from api.domain.value_objects.machine_id import MachineId


class EventKind(StrEnum):
    """What changed about a machine.

    One member per event the PRD's realtime section names, plus the two
    transitions later phases needed. A client switches on this exhaustively, so
    adding a member is a visible, deliberate change rather than a payload nobody
    handles.

    `SIMULATION_STATE_CHANGED` completes the PRD's list rather than extending it:
    §21 names "simulation state changes" among the events the dashboard must
    receive, and until Phase 8 this enum did not have one -- while its own
    docstring claimed otherwise.
    """

    TELEMETRY_RECEIVED = "TELEMETRY_RECEIVED"
    PREDICTION_RECORDED = "PREDICTION_RECORDED"
    INCIDENT_RAISED = "INCIDENT_RAISED"
    INCIDENT_STATUS_CHANGED = "INCIDENT_STATUS_CHANGED"
    SIMULATION_STATE_CHANGED = "SIMULATION_STATE_CHANGED"


@dataclass(frozen=True, slots=True)
class MachineEvent:
    """A change to one machine, announced to whoever is listening.

    Deliberately carries no timestamp. A consumer's response is to refetch, and
    what it refetches carries real `recorded_at` and `predicted_at` values --
    so a time on the hint would be a second, less accurate answer to a question
    already answered, and would drag a `Clock` into every use case that
    publishes.
    """

    kind: EventKind
    machine_id: MachineId


@dataclass(frozen=True, slots=True)
class StreamTick:
    """A keepalive: the stream is alive and nothing has changed.

    Not an event, and deliberately a distinct type rather than a `None` -- a
    consumer that forgot to handle it would fail loudly instead of trying to
    read a machine out of nothing.

    It belongs on the port rather than being invented by the endpoint for two
    reasons. The broadcaster already runs one timer for every subscriber, where
    a per-connection timer would be one task per open tab on a single-core box.
    And the obvious per-connection spelling -- `asyncio.wait_for(queue.get(),
    timeout=...)` -- has a window in which a real event can be discarded: the
    item is popped after the await resumes, so a timeout landing in that
    instant loses it, rarely and unreproducibly. Pushing ticks removes the
    question.
    """


#: What a subscription yields: changes, and keepalives between them.
StreamItem = MachineEvent | StreamTick


class EventPublisher(Protocol):
    """Announces changes. Implemented by infrastructure.

    Implementations **must not raise**. A publisher sits on the write path of
    ingest and scoring, and a subscriber that has stopped reading is not a
    reason for a telemetry batch to fail -- it is a reason to forget that
    subscriber's backlog and let it resynchronise.
    """

    async def publish(self, event: MachineEvent) -> None:
        """Announce one change, returning as soon as it is queued."""
        ...


class EventSubscriber(Protocol):
    """Streams changes to one listener."""

    def subscribe(self) -> AsyncIterator[StreamItem]:
        """Yield changes and keepalives until the caller stops iterating.

        Cleanup runs when the generator is closed, so a client that disconnects
        releases its slot without the caller having to say so. That is the
        property the whole design leans on: nothing else unsubscribes.
        """
        ...


class NullEventPublisher:
    """Discards every event.

    The default for wiring with no stream behind it -- the in-memory test
    container, and any deployment that has not turned the dashboard on. A
    publisher that fails loudly would be wrong here: having no subscribers is
    the ordinary state of a working system, not an error.
    """

    async def publish(self, event: MachineEvent) -> None:
        """Discard the event."""
        del event
