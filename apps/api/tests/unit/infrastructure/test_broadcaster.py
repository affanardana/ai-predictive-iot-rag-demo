"""The in-process event broadcaster."""

from __future__ import annotations

import asyncio
import contextlib

from api.domain.ports.events import EventKind, MachineEvent, StreamTick
from api.domain.value_objects.machine_id import MachineId
from api.infrastructure.realtime import InProcessEventBroadcaster


def _event(
    machine_id: str = "M003", kind: EventKind = EventKind.TELEMETRY_RECEIVED
) -> MachineEvent:
    return MachineEvent(kind=kind, machine_id=MachineId(machine_id))


async def _started(broadcaster: InProcessEventBroadcaster, count: int = 1) -> None:
    """Yield control until `count` subscriptions are registered.

    A subscription exists only once its generator has been iterated, so a test
    that publishes immediately after `subscribe()` would race. Yielding rather
    than sleeping keeps this deterministic; the bound turns a hang into a
    failure that names itself.
    """
    for _ in range(200):
        if broadcaster.subscriber_count == count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"Expected {count} subscribers, saw {broadcaster.subscriber_count}.")


async def test_a_subscriber_receives_a_published_event() -> None:
    """The basic contract: what is published is what is streamed."""
    broadcaster = InProcessEventBroadcaster()
    stream = broadcaster.subscribe()
    pending = asyncio.ensure_future(anext(stream))
    await _started(broadcaster)

    await broadcaster.publish(_event())

    try:
        received = await asyncio.wait_for(pending, timeout=1.0)
    finally:
        await stream.aclose()
    assert isinstance(received, MachineEvent)
    assert received.machine_id == MachineId("M003")


async def test_every_subscriber_receives_the_event() -> None:
    """Fan-out reaches all open streams, not just the first.

    Two browser tabs must show the same fleet.
    """
    broadcaster = InProcessEventBroadcaster()
    streams = [broadcaster.subscribe() for _ in range(2)]
    pending = [asyncio.ensure_future(anext(stream)) for stream in streams]
    await _started(broadcaster, count=2)

    await broadcaster.publish(_event())

    try:
        received = await asyncio.wait_for(asyncio.gather(*pending), timeout=1.0)
    finally:
        for stream in streams:
            await stream.aclose()
    assert all(isinstance(item, MachineEvent) for item in received)
    assert [item.machine_id for item in received if isinstance(item, MachineEvent)] == [
        MachineId("M003")
    ] * 2


async def test_publishing_without_subscribers_does_nothing() -> None:
    """A write must succeed when nobody is watching.

    A dashboard with no tabs open is the normal state, and a publisher that
    raised here would turn `POST /api/v1/telemetry` into a 500 whenever the
    stream happened to be unused.
    """
    broadcaster = InProcessEventBroadcaster()

    await broadcaster.publish(_event())


async def test_a_slow_subscriber_does_not_block_the_publisher() -> None:
    """Publishing completes even when a subscriber never reads.

    This is the guarantee that keeps a wedged browser tab off the ingest path.
    A publisher that awaited room in a full queue would make a telemetry batch
    wait on somebody's stalled connection.
    """
    broadcaster = InProcessEventBroadcaster(queue_size=4)
    stream = broadcaster.subscribe()
    pending = asyncio.ensure_future(anext(stream))
    await _started(broadcaster)

    try:
        # Far more than the queue holds, and nothing is ever read from it.
        await asyncio.wait_for(
            asyncio.gather(*(broadcaster.publish(_event()) for _ in range(50))),
            timeout=1.0,
        )
    finally:
        pending.cancel()
        await stream.aclose()


async def test_a_burst_does_not_grow_the_backlog() -> None:
    """Fifty rapid changes leave a bounded number queued, not fifty.

    The bound is the contract. Without it a subscriber that falls behind
    accumulates every hint it missed, and when it finally reads them the client
    issues one REST request per event -- a refetch storm aimed at a single-core
    server, triggered by the very backlog that was meant to protect it.

    `publish` is awaited here without the event loop ever running between
    calls, because a coroutine with no `await` of its own runs to completion.
    That makes the fill deterministic rather than a race.
    """
    queue_size = 2
    broadcaster = InProcessEventBroadcaster(queue_size=queue_size)
    stream = broadcaster.subscribe()
    pending = asyncio.ensure_future(anext(stream))
    await _started(broadcaster)

    try:
        for _ in range(50):
            await broadcaster.publish(_event())

        readable = [await asyncio.wait_for(pending, timeout=1.0)]
        readable.append(await asyncio.wait_for(anext(stream), timeout=1.0))
    finally:
        await stream.aclose()

    assert len(readable) == queue_size


async def test_a_cancelled_read_releases_the_subscription() -> None:
    """A client that disconnects mid-stream leaves nothing behind.

    Cancelling the read is not a test convenience: it is precisely what
    happens in production. Starlette cancels the streaming task when the socket
    closes, and that throws into the generator's `await`. The `finally` there
    is the only unsubscribe path in the class, so this is what stands between a
    long-running server and a leaked queue for every browser tab that ever
    closed -- a leak that would be invisible until the process ran out of
    memory days later.
    """
    broadcaster = InProcessEventBroadcaster()
    stream = broadcaster.subscribe()
    pending = asyncio.ensure_future(anext(stream))
    await _started(broadcaster)
    assert broadcaster.subscriber_count == 1

    pending.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await pending

    assert broadcaster.subscriber_count == 0


async def test_a_closed_generator_releases_the_subscription() -> None:
    """The graceful path releases it too, not only cancellation.

    Once the stream has yielded an item it is suspended at the `yield`, so
    closing it throws `GeneratorExit` there rather than into a pending read.
    Both routes out have to clean up, because which one runs depends on where
    the client happened to be when it went away.
    """
    broadcaster = InProcessEventBroadcaster()
    stream = broadcaster.subscribe()
    first = asyncio.ensure_future(anext(stream))
    await _started(broadcaster)

    await broadcaster.publish(_event())
    await asyncio.wait_for(first, timeout=1.0)
    assert broadcaster.subscriber_count == 1

    await stream.aclose()

    assert broadcaster.subscriber_count == 0


async def test_a_quiet_stream_emits_keepalives() -> None:
    """An idle stream still produces traffic.

    The stream passes through Caddy and the open internet, where an idle
    connection is eventually reaped. A tick every few seconds keeps it alive,
    and a tick is not an event -- a client that mistook one for a change would
    refetch on a timer it never asked for.
    """
    broadcaster = InProcessEventBroadcaster(heartbeat_seconds=0.01)
    stream = broadcaster.subscribe()

    try:
        item = await asyncio.wait_for(anext(stream), timeout=1.0)
    finally:
        await stream.aclose()

    assert isinstance(item, StreamTick)
