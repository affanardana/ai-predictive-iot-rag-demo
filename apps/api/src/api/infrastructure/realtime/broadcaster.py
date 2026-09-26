"""In-process fan-out for the dashboard's live update stream.

**This is correct only because the API runs as a single process.** Fan-out
happens in memory, so an event published on one worker reaches only the
subscribers attached to that worker. `infra/compose/Dockerfile.api` starts
uvicorn with no `--workers` flag, and that is load-bearing rather than
incidental -- a second worker would not fail, it would quietly deliver each
change to an arbitrary subset of connected browsers, which is the kind of bug
that is blamed on the browser.

`apps/api/tests/unit/infrastructure/test_deployment_shape.py` parses that
Dockerfile and fails if a worker count ever appears, and
`docs/architecture.md` records the constraint. A cross-process bus would mean
PostgreSQL `LISTEN`/`NOTIFY` or Redis, neither of which this deployment needs:
the API is the only writer, so every event originates in this process anyway.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from itertools import count

from api.domain.ports.events import MachineEvent, StreamItem, StreamTick

#: How many changes one subscriber may fall behind before its backlog is
#: dropped. Generous, because the cost of a large queue is a few kilobytes and
#: the cost of dropping is a client that misses a change until its next poll.
DEFAULT_QUEUE_SIZE = 64

#: Seconds between keepalives. Well under the idle timeout of the proxies this
#: stream passes through, and infrequent enough to be invisible in a log.
DEFAULT_HEARTBEAT_SECONDS = 15.0


@dataclass
class _Subscriber:
    """One open stream."""

    queue: asyncio.Queue[StreamItem]


class InProcessEventBroadcaster:
    """Fans changes out to every open subscription in this process.

    Satisfies both `EventPublisher` and `EventSubscriber`, which are separate
    protocols so that a use case structurally cannot subscribe and a router
    cannot publish. This class is the one place they meet, and it is only ever
    reached through the composition root.
    """

    def __init__(
        self,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    ) -> None:
        # Both injectable so tests can use a two-slot queue and a heartbeat of
        # milliseconds. A test that has to `sleep` to observe a timer is a test
        # that will be deleted the first time it is flaky.
        self._queue_size = queue_size
        self._heartbeat_seconds = heartbeat_seconds
        self._subscribers: dict[int, _Subscriber] = {}
        self._tokens = count()
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def publish(self, event: MachineEvent) -> None:
        """Queue `event` for every subscriber, and never block on one.

        `put_nowait` throughout, so a browser tab that has stopped reading
        cannot slow down `POST /api/v1/telemetry`. That is the whole reason the
        payload is a hint: an event is only safe to drop because the client
        corrects itself from the REST endpoint rather than accumulating state
        from the stream.

        An overflowing subscriber has its backlog **cleared**, not trimmed by
        one. Keeping the oldest of a full queue would deliver a burst of stale
        hints and provoke a refetch storm; keeping only the newest tells the
        client the current truth once.

        Dropping is safe without any acknowledgement or replay, and the reason
        is worth stating because it is not obvious: a client's response to *any*
        event is to invalidate the whole fleet list, not just the machine named.
        So one delivered event repairs every dropped one, and nothing needs to
        notice that a gap occurred. That, in turn, is why this method can
        promise not to fail and not to block.
        """
        for subscriber in list(self._subscribers.values()):
            if subscriber.queue.full():
                self._discard_backlog(subscriber)
            subscriber.queue.put_nowait(event)

    def subscribe(self) -> AsyncGenerator[StreamItem, None]:
        """Return the stream for one subscriber, cleaned up when it is closed.

        An async *generator*, so the `finally` runs when the caller stops
        iterating -- which is what a disconnected client does to it. There is
        no other unsubscribe path, and that is deliberate: a subscription whose
        lifetime is a `finally` cannot be leaked by an error path.

        Typed as the generator rather than the `AsyncIterator` the port
        declares, so a caller holding the concrete class can `aclose()` it. The
        port stays on the narrower type because a consumer has no business
        closing someone else's stream.
        """
        return self._stream()

    async def _stream(self) -> AsyncGenerator[StreamItem, None]:
        """Yield this subscriber's items until it stops being read."""
        token = next(self._tokens)
        subscriber = _Subscriber(queue=asyncio.Queue(maxsize=self._queue_size))
        self._subscribers[token] = subscriber
        self._ensure_heartbeat()
        try:
            while True:
                yield await subscriber.queue.get()
        finally:
            # Synchronous, and so not cancellable part-way through. An `await`
            # here could itself be cancelled and leave the entry behind, which
            # would leak a queue per disconnected browser.
            del self._subscribers[token]

    @property
    def subscriber_count(self) -> int:
        """Return how many streams are open. Exists for tests and diagnostics."""
        return len(self._subscribers)

    async def aclose(self) -> None:
        """Stop the keepalive timer. Called from the container's shutdown."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

    def _discard_backlog(self, subscriber: _Subscriber) -> None:
        """Empty a full queue so the newest event has somewhere to go."""
        while not subscriber.queue.empty():
            subscriber.queue.get_nowait()

    def _ensure_heartbeat(self) -> None:
        """Start the keepalive timer on the first subscriber.

        Lazy because the broadcaster is built at import time of the container,
        which may be before a running event loop exists -- and because a
        process with no dashboard open should not hold a timer.
        """
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat())

    async def _heartbeat(self) -> None:
        """Push a tick to every subscriber, forever.

        Ticks are pushed rather than awaited for per connection. The stream
        passes through Caddy and the open internet, where an idle connection is
        eventually reaped; a comment frame every few seconds keeps it alive
        without the endpoint holding a timer of its own.
        """
        tick = StreamTick()
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            for subscriber in list(self._subscribers.values()):
                if subscriber.queue.full():
                    continue
                subscriber.queue.put_nowait(tick)
