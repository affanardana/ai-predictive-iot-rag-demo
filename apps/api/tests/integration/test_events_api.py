"""The server-sent event stream.

Driven against the ASGI application directly rather than through
`httpx.AsyncClient`. `ASGITransport` collects a complete response body before
returning, and an event stream by definition never completes -- so a test
written the ordinary way does not fail, it hangs. Talking to the app's `send`
callable is the only way to observe frames as they are produced.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import MutableMapping
from typing import Any

import pytest
from fastapi import FastAPI

from api.composition.container import Container
from api.domain.ports.events import EventKind
from api.domain.value_objects.machine_id import MachineId
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from api.presentation.app import create_app
from tests.support.factories import DEFAULT_NOW, make_machine, make_telemetry

#: A failure deadline, not a synchronisation device. A test reading from a
#: stream can wait forever, and a hang says nothing about which frame was
#: missing.
READ_TIMEOUT_SECONDS = 5.0

STREAM_PATH = "/api/v1/events"


@pytest.fixture
def app(container: Container) -> FastAPI:
    """The application, wired to the shared in-memory container.

    Built here rather than taken from the `client` fixture because this module
    drives the ASGI callable itself. The container is the same object the other
    fixtures use, which is what lets a test publish through a use case and read
    the result off the stream.
    """
    return create_app(container)


class OpenStream:
    """One in-flight event stream, read frame by frame."""

    def __init__(self, task: asyncio.Task[None], queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._task = task
        self._queue = queue
        self._buffer = ""

    async def _pull(self) -> None:
        """Wait for the next ASGI message and buffer its body."""
        message = await asyncio.wait_for(self._queue.get(), timeout=READ_TIMEOUT_SECONDS)
        if message["type"] == "http.response.body":
            self._buffer += message["body"].decode()

    async def next_frame(self) -> str:
        """Return the next complete frame.

        Frames are separated by a blank line, so a frame is only returned once
        both newlines have arrived -- a partial read would let a test assert
        against half a payload and pass.
        """
        while "\n\n" not in self._buffer:
            await self._pull()
        frame, _, self._buffer = self._buffer.partition("\n\n")
        return frame

    async def next_change(self) -> dict[str, Any]:
        """Return the payload of the next change frame, skipping keepalives.

        Also the assertion that a keepalive is not an event: one arriving here
        is skipped rather than mistaken for a change, which is why the format
        uses a comment rather than an unnamed event.
        """
        while True:
            frame = await self.next_frame()
            for line in frame.splitlines():
                if line.startswith("data: "):
                    decoded: dict[str, Any] = json.loads(line.removeprefix("data: "))
                    return decoded

    async def close(self) -> None:
        """Disconnect, as a browser closing a tab does."""
        self._task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await self._task


async def _open(app: FastAPI) -> tuple[OpenStream, dict[str, Any]]:
    """Open the stream, returning it alongside the response-start message."""
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    started: dict[str, Any] = {}

    async def receive() -> dict[str, Any]:
        # Never yields, so the request never reports a disconnect and the
        # stream stays open until the test cancels it. That is what keeps the
        # test in control of the stream's lifetime rather than the transport's.
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(message: MutableMapping[str, Any]) -> None:
        # `MutableMapping` because that is what the ASGI callable is declared
        # to accept; copied into a plain dict so the queue has one concrete
        # type to hand back.
        if message["type"] == "http.response.start":
            started.update(message)
        queue.put_nowait(dict(message))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": STREAM_PATH,
        "raw_path": STREAM_PATH.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"accept", b"text/event-stream"),
            # A dashboard origin, so the assertions below cover the streaming
            # case. `EventSource` sends no custom headers, so this request is
            # CORS-simple and gets no preflight -- but the browser still drops
            # the response without `Access-Control-Allow-Origin`.
            (b"origin", b"https://pdm-dashboard.vercel.app"),
        ],
        "client": ("127.0.0.1", 45678),
        "server": ("testserver", 80),
    }

    task = asyncio.create_task(app(scope, receive, send))
    stream = OpenStream(task, queue)
    # The start message is queued before any body, so this cannot deadlock.
    await asyncio.wait_for(queue.get(), timeout=READ_TIMEOUT_SECONDS)
    return stream, started


async def _register_and_publish(container: Container, event_id: str = "evt-1") -> None:
    """Write one reading through the use case the ingest endpoint calls."""
    await container.ingest_telemetry.execute(
        [make_telemetry(event_id=event_id, machine_id="M003", recorded_at=DEFAULT_NOW)]
    )


async def test_opens_an_event_stream(app: FastAPI, uow_factory: InMemoryUnitOfWorkFactory) -> None:
    """The endpoint answers as an event stream and states its retry delay.

    It also opens with no `X-Ingest-Token`, which is a constraint rather than a
    choice: `EventSource` cannot set request headers, so a guarded stream would
    be unreachable from a browser rather than merely inconvenient.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))

    stream, started = await _open(app)
    try:
        assert started["status"] == 200
        headers = dict(started["headers"])
        assert headers[b"content-type"] == b"text/event-stream; charset=utf-8"
        # A proxy that buffers this would leave the dashboard frozen with no
        # server-side error to explain it.
        assert headers[b"x-accel-buffering"] == b"no"
        # Set on `http.response.start`, which is the same message the middleware
        # decorates -- so a stream is covered by the same CORS rules as any
        # other response rather than being a special case.
        assert headers[b"access-control-allow-origin"] == b"*"

        first = await stream.next_frame()
    finally:
        await stream.close()

    assert first == "retry: 3000"


async def test_a_write_reaches_a_connected_client(
    app: FastAPI, container: Container, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """The whole point: a write appears on a stream opened before it.

    This is the assertion that fails if the publishing use cases and the
    streaming endpoint are ever wired to two different broadcasters -- a
    mistake that is invisible everywhere else, because each half works
    perfectly in isolation.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))

    stream, _ = await _open(app)
    try:
        await stream.next_frame()
        await _register_and_publish(container)
        change = await stream.next_change()
    finally:
        await stream.close()

    assert change == {"kind": "TELEMETRY_RECEIVED", "machine_id": "M003"}


async def test_a_redelivered_batch_announces_nothing(
    app: FastAPI, container: Container, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """A batch that inserts nothing produces no event.

    n8n redelivers on retry, so this is the ordinary path rather than an edge
    case. Announcing a no-op would make every open dashboard refetch for a
    change that did not happen.

    A duplicate is followed by a genuine write, and the genuine one must be the
    first change to arrive -- the deadline is what makes the absence an
    assertion rather than a silence.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))

    await _register_and_publish(container, event_id="evt-1")

    stream, _ = await _open(app)
    try:
        await stream.next_frame()

        await _register_and_publish(container, event_id="evt-1")  # the duplicate
        await _register_and_publish(container, event_id="evt-2")

        change = await stream.next_change()
    finally:
        await stream.close()

    # `next_change` returns the first change frame, so reaching this line at
    # all means the duplicate produced none.
    assert change["machine_id"] == "M003"


async def test_a_disconnected_client_releases_its_subscription(
    app: FastAPI, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Closing the stream leaves nothing registered.

    The subscriber count is the leak detector. Without the generator's
    `finally` this would climb by one for every browser tab that ever closed,
    invisibly, until the process ran out of memory days later.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))

    subscribers = app.state.container.event_subscriber
    baseline = subscribers.subscriber_count

    stream, _ = await _open(app)
    await stream.next_frame()
    assert subscribers.subscriber_count == baseline + 1

    await stream.close()

    assert subscribers.subscriber_count == baseline


def test_every_event_kind_is_stable() -> None:
    """The enum's values are a wire contract, so they are asserted literally.

    A `text/event-stream` body cannot be described in OpenAPI, so the TypeScript
    type in `apps/web/src/realtime/types.ts` is hand-written -- the one place
    the "no hand-written client" rule is broken. This test is what makes that
    safe: changing a value here fails the Python suite rather than silently
    freezing the browser.

    Writing the fifth member in was the deliberate edit this test exists to
    force. `SIMULATION_STATE_CHANGED` completes `PRD.md` §21's list rather than
    extending it -- "simulation state changes" was named there from the start,
    and the enum's docstring claimed to have one member per named event while
    having four of five.
    """
    assert {kind.value for kind in EventKind} == {
        "TELEMETRY_RECEIVED",
        "PREDICTION_RECORDED",
        "INCIDENT_RAISED",
        "INCIDENT_STATUS_CHANGED",
        "SIMULATION_STATE_CHANGED",
    }


async def test_the_hint_names_a_machine_without_carrying_its_data(
    app: FastAPI, container: Container, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """The payload is a hint, and the absence of telemetry is the design.

    If a reading ever appears in this frame the stream has become a second
    source of truth: the client would be building state from it rather than
    refetching, the caching rules would no longer hold, and a dropped frame
    would start to matter. Asserted as an exact key set for that reason.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))

    stream, _ = await _open(app)
    try:
        await stream.next_frame()
        await _register_and_publish(container)
        change = await stream.next_change()
    finally:
        await stream.close()

    assert set(change) == {"kind", "machine_id"}
    assert MachineId(str(change["machine_id"])) == MachineId("M003")
