"""MQTT sink: telemetry published to a broker.

The `paho` import is deliberately inside `connect_paho` rather than at module
level. `simulator.infrastructure.sinks` re-exports every sink and
`simulator.cli` imports that package, so a top-level import would make the CLI
-- and the test that imports it -- unimportable in the default environment,
where the `mqtt` extra is not installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from simulator.domain.state import GroundTruthState, TelemetrySample
from simulator.infrastructure.sinks.jsonl import json_default

if TYPE_CHECKING:
    from paho.mqtt.client import Client as PahoClient
    from paho.mqtt.client import MQTTMessageInfo

#: At-least-once. The API's primary key on `event_id` absorbs a duplicate for
#: free, which is a better trade than losing a reading to a broker hiccup.
DEFAULT_QOS = 1

KEEPALIVE_SECONDS = 60

#: How long `close` waits for the last messages to be acknowledged.
#:
#: `publish` only queues to paho's background thread, so at the end of a run
#: there are messages in flight that the socket has not carried yet. Calling
#: `disconnect` immediately discards them -- with QoS 1 that is silent, because
#: the publisher already has a success code for every one of them.
PUBLISH_DRAIN_TIMEOUT_SECONDS = 10.0


class PublishResult(Protocol):
    """What a publish hands back: enough to tell success from failure."""

    #: Zero means the message was queued. Anything else means it was not.
    rc: int


class Publisher(Protocol):
    """The slice of an MQTT client this sink needs.

    Declared structurally so a test can pass a fake and the real client can be
    passed without an adapter. The sink owns the connection it is given, which
    is why `close` is here rather than on the caller.
    """

    def publish(self, topic: str, payload: bytes, qos: int) -> PublishResult:
        """Queue `payload` for `topic`."""
        ...

    def close(self) -> int:
        """Finish sending and disconnect.

        Returns how many messages could not be confirmed delivered. A transport
        that silently discards what it was handed is the failure mode this whole
        module exists to make visible, so the count is part of the contract
        rather than a detail of one implementation.
        """
        ...


@dataclass(frozen=True, slots=True)
class BrokerSettings:
    """Where the broker is, and how to reach it."""

    host: str
    port: int
    client_id: str
    topic_prefix: str
    tls: bool = True
    username: str | None = None
    password: str | None = None


class MqttTelemetrySink:
    """Publishes each observation to `{prefix}/{machine_id}/telemetry`.

    No buffering, unlike the file sinks: a stream that batches is not a stream,
    and a buffered message is one that has not arrived.

    Publish failures are counted rather than raised. A broker hiccup midway
    through a run should not abort a 720-reading demonstration, but it also must
    not pass unnoticed -- the count is reported in the run summary, which is
    what makes an ingestion failure diagnosable.
    """

    def __init__(self, publisher: Publisher, topic_prefix: str) -> None:
        self._publisher = publisher
        self._topic_prefix = topic_prefix.strip("/")
        self.failures = 0

    @property
    def topic_prefix(self) -> str:
        """The prefix every topic is built from."""
        return self._topic_prefix

    def topic_for(self, machine_id: str) -> str:
        """Return the topic one machine's telemetry is published to."""
        return f"{self._topic_prefix}/{machine_id}/telemetry"

    def write(self, sample: TelemetrySample) -> None:
        """Publish one observation."""
        # `json_default` is reused from the JSON Lines sink so the payload is
        # exactly the line that sink would write: one serialisation of
        # `as_row()`, and the timestamp ISO-formatted the same way.
        payload = json.dumps(sample.as_row(), default=json_default).encode("utf-8")
        if self._publisher.publish(self.topic_for(sample.machine_id), payload, DEFAULT_QOS).rc != 0:
            self.failures += 1

    def close(self) -> None:
        """Flush, disconnect, and fold anything undelivered into the failures.

        Draining is the publisher's job; deciding that an undelivered message
        counts as a failure is the sink's, because `failures` is the number the
        run summary prints.
        """
        self.failures += self._publisher.close()


class DiscardingGroundTruthSink:
    """Drops the ground-truth half of a run.

    Ground truth is the answer key: `failure_imminent`, the degradation
    channels, the scenario label. Publishing it to a broker would put the label
    on the same wire as the readings it is supposed to be predicted from, which
    is the leak `services/simulator/README.md` rules out.

    Dataset mode still writes it, because a dataset needs its labels. A live
    stream does not.
    """

    def write(self, state: GroundTruthState) -> None:
        """Discard one ground-truth record."""
        del state

    def close(self) -> None:
        """Nothing to close."""


def connect_paho(settings: BrokerSettings) -> Publisher:
    """Connect a real MQTT client, with the network loop running.

    Imports `paho` here rather than at module level; see this module's
    docstring. Raises `ModuleNotFoundError` with a usable message when the extra
    is not installed, because the alternative is an `ImportError` naming a
    module the caller never mentioned.
    """
    try:
        import paho.mqtt.client as mqtt
        from paho.mqtt.enums import CallbackAPIVersion
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the extra
        raise ModuleNotFoundError(
            "The MQTT sink needs paho-mqtt, which is an optional extra. "
            "Install it with: uv sync --all-packages --extra mqtt"
        ) from exc

    client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=settings.client_id)
    if settings.username:
        client.username_pw_set(settings.username, settings.password)
    if settings.tls:
        # No arguments: verify against the system CA store, which is what a
        # broker with a publicly trusted certificate needs. Nothing has to be
        # shipped with the repository.
        client.tls_set()

    client.connect(settings.host, settings.port, keepalive=KEEPALIVE_SECONDS)
    # A background thread, so a paced synchronous loop can keep publishing
    # without servicing the socket itself.
    client.loop_start()
    return _PahoPublisher(client)


class _PahoPublisher:
    """Adapts a paho client to the `Publisher` Protocol."""

    def __init__(self, client: PahoClient) -> None:
        self._client = client
        #: Messages handed to paho's background thread and not yet acknowledged.
        #: Kept so `close` can wait for them rather than cutting them off.
        self._in_flight: list[MQTTMessageInfo] = []

    def publish(self, topic: str, payload: bytes, qos: int) -> PublishResult:
        """Queue a message, returning paho's result code."""
        info = self._client.publish(topic, payload, qos)
        if info.rc == 0:
            self._in_flight.append(info)
        return _MessageInfo(info.rc)

    def close(self) -> int:
        """Wait for in-flight messages, disconnect, and report what was lost.

        The wait is the point. `publish` returns as soon as the message is
        queued, so a run that finishes quickly still has its last readings in
        the queue; disconnecting first would drop exactly the readings nearest
        the failure the demonstration is about, and report no failure while
        doing it.
        """
        undelivered = 0
        for info in self._in_flight:
            try:
                info.wait_for_publish(timeout=PUBLISH_DRAIN_TIMEOUT_SECONDS)
            except (RuntimeError, ValueError):
                # Not acknowledged within the drain window, or never queued.
                # Counted rather than raised: the run is over, and this is the
                # number that says how much of it did not leave.
                undelivered += 1
        self._in_flight.clear()

        self._client.disconnect()
        self._client.loop_stop()
        return undelivered


@dataclass(slots=True)
class _MessageInfo:
    """A `PublishResult` carrying nothing but the code.

    Not frozen, because `PublishResult` declares `rc` as an attribute and a
    read-only one does not satisfy it -- a Protocol promising a field has to be
    satisfiable by something that has a field.
    """

    rc: int
