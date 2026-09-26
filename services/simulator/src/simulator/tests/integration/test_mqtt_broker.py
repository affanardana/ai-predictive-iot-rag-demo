"""The MQTT sink against a real broker.

Opt-in, and the only test that exercises `connect_paho` at all. Everything else
in the suite passes the sink a fake publisher, which means TLS setup, the client
id, the QoS handshake, and paho's own serialisation are unverified without this
— and those are precisely the parts a fake cannot stand in for.

Selected with `-m mqtt` and pointed at a broker with `MQTT_BROKER_URL`:

    uv sync --locked --all-packages --extra mqtt
    $env:MQTT_BROKER_URL = "mqtts://broker.emqx.io:8883"
    uv run pytest -m mqtt

The extra belongs to the `simulator` member rather than the virtual root, so
`uv run --extra mqtt` fails with "Extra `mqtt` is not defined"; it has to be
synced in first.

EMQX's public broker is the default subject of the demonstration, needs no
credentials, and carries a publicly trusted certificate — so this runs against
the real thing rather than a local stand-in. Its topics are world-readable, which
is why each run uses a unique prefix: without one, a concurrent run of this same
file could observe the other's messages and pass for the wrong reason.

**This file earned its place before it was ever green.** Written while paho was
uninstalled, it failed intermittently once run — and the cause was not the test
but the sink: `close()` disconnected while QoS 1 messages were still queued, so
the last readings of every run were discarded and `failures` reported zero. Two
of the first five runs lost messages. The failure rate against real
infrastructure is what made it visible; no amount of faking would have.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import pytest

from simulator.domain.engine import MachineSimulator
from simulator.domain.scenario import Scenario
from simulator.domain.session import SimulationSession
from simulator.domain.state import TelemetrySample
from simulator.infrastructure.sinks.jsonl import json_default
from simulator.infrastructure.sinks.mqtt import (
    BrokerSettings,
    MqttTelemetrySink,
    connect_paho,
)
from simulator.tests.conftest import FIXED_START

if TYPE_CHECKING:
    # Type-only, because the default tier collects this module without the
    # `mqtt` extra installed. The runtime import lives inside `subscribed`.
    from paho.mqtt.client import Client, MQTTMessage

pytestmark = pytest.mark.mqtt

BROKER_URL_VARIABLE = "MQTT_BROKER_URL"

#: How long to wait for a message to make a round trip through a public broker.
#: Generous: this is someone else's infrastructure over the open internet, and a
#: flaky timeout would be indistinguishable from a real regression.
ARRIVAL_TIMEOUT_SECONDS = 30.0

CONNECT_TIMEOUT_SECONDS = 20.0


def broker_url() -> str:
    """The configured broker, or a failure explaining what to set.

    `pytest.fail` rather than `skip`, matching the `postgres` tier: reaching here
    means `-m mqtt` was selected explicitly, so a skip would report green while
    having verified nothing.
    """
    url = os.environ.get(BROKER_URL_VARIABLE)
    if not url:
        pytest.fail(
            f"{BROKER_URL_VARIABLE} is not set. Point it at a broker, e.g. "
            "mqtts://broker.emqx.io:8883 (TLS) or mqtt://localhost:1883 (plain)."
        )
    return url


def broker_settings(role: str, topic_prefix: str) -> BrokerSettings:
    """Parse a broker URL into the settings the sink takes.

    The scheme chooses TLS, so `mqtts://` and `mqtt://` are the only difference
    between a secured broker and a plaintext one.

    The client id carries the run's random suffix. A constant id would collide
    with a session left over from a previous run of this file on a shared
    broker, and the broker resolves a collision by evicting one of the two —
    which arrives at this test as messages going missing.
    """
    parsed = urlparse(broker_url())
    secure = parsed.scheme == "mqtts"
    if parsed.scheme not in {"mqtt", "mqtts"}:
        pytest.fail(
            f"{BROKER_URL_VARIABLE} must start with mqtt:// or mqtts://, not {parsed.scheme}"
        )

    return BrokerSettings(
        host=parsed.hostname or "",
        port=parsed.port or (8883 if secure else 1883),
        tls=secure,
        username=parsed.username,
        password=parsed.password,
        topic_prefix=topic_prefix,
        client_id=f"{role}-{topic_prefix.rsplit('-', 1)[-1]}",
    )


def a_sample(index: int = 0) -> TelemetrySample:
    """A reading from a real run."""
    session = SimulationSession.create(
        machine_ids=["M003"],
        scenario=Scenario.BEARING_DEGRADATION,
        seed=20_260_923,
        duration=timedelta(minutes=5),
        started_at=FIXED_START,
    )
    simulator = MachineSimulator(session, session.machines[0])
    return simulator.tick(index).telemetry


class Received:
    """One message as it arrived, with the topic it arrived on."""

    def __init__(self, topic: str, payload: bytes) -> None:
        self.topic = topic
        self.payload = payload

    def json(self) -> dict[str, object]:
        """The payload decoded, as the API's ingest schema would receive it."""
        decoded: dict[str, object] = json.loads(self.payload)
        return decoded

    def number(self, key: str) -> float:
        """One field, checked to be the number the wire promised.

        JSON is untyped, so this asserts before converting rather than letting
        a string reach `approx` and produce a confusing comparison failure.
        """
        value = self.json()[key]
        assert isinstance(value, (int, float)), f"{key} arrived as {type(value).__name__}"
        return float(value)


@contextmanager
def subscribed(settings: BrokerSettings, topic_filter: str) -> Iterator[list[Received]]:
    """Subscribe with a second, independent client and collect what arrives.

    A separate connection on purpose. Using the sink's own client to observe its
    own publishes would prove only that paho can talk to itself.
    """
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion

    received: list[Received] = []
    subscribed_event = threading.Event()

    client = mqtt.Client(
        CallbackAPIVersion.VERSION2,
        client_id=f"{settings.client_id}-observer-{uuid.uuid4().hex[:6]}",
    )
    if settings.tls:
        client.tls_set()
    if settings.username:
        client.username_pw_set(settings.username, settings.password)

    def on_connect(
        client_: Client,
        userdata: object,
        flags: object,
        reason_code: object,
        properties: object,
    ) -> None:
        # Sent here, but not yet in effect. The broker routes nothing to a
        # subscription until it has processed this, so readiness is signalled
        # from `on_subscribe` rather than from here.
        client_.subscribe(topic_filter, qos=1)

    def on_subscribe(
        client_: Client,
        userdata: object,
        mid: int,
        reason_codes: object,
        properties: object,
    ) -> None:
        # The SUBACK. Only now will the broker deliver anything published to
        # this topic, which is what makes the publisher starting afterwards
        # safe.
        subscribed_event.set()

    def on_message(client_: Client, userdata: object, message: MQTTMessage) -> None:
        received.append(Received(message.topic, message.payload))

    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message

    client.connect(settings.host, settings.port, keepalive=60)
    client.loop_start()
    try:
        if not subscribed_event.wait(timeout=CONNECT_TIMEOUT_SECONDS):
            pytest.fail(f"the observer never subscribed to {topic_filter} on {settings.host}")
        yield received
    finally:
        client.disconnect()
        client.loop_stop()


def wait_for(received: list[Received], count: int) -> None:
    """Block until `count` messages have arrived, or fail saying how many did."""
    deadline = time.monotonic() + ARRIVAL_TIMEOUT_SECONDS
    while len(received) < count and time.monotonic() < deadline:
        time.sleep(0.05)
    if len(received) < count:
        pytest.fail(f"expected {count} messages and {len(received)} arrived within the timeout")


@contextmanager
def a_publisher(settings: BrokerSettings) -> Iterator[MqttTelemetrySink]:
    """The real sink, over a real paho client, closed afterwards."""
    publisher = connect_paho(settings)
    try:
        yield MqttTelemetrySink(publisher, settings.topic_prefix)
    finally:
        publisher.close()


@pytest.fixture
def unique_prefix() -> str:
    """A topic tree nobody else is using.

    The broker is public. Without this, a second run of this file could observe
    the first run's messages and pass without its own publish ever arriving.
    """
    return f"pdm/test-{uuid.uuid4().hex[:10]}"


def test_a_published_reading_arrives_unchanged(unique_prefix: str) -> None:
    """The whole wire path: sink → paho → TLS → broker → subscriber.

    The payload is compared key by key rather than by bytes, because the
    question is whether the broker preserved the message rather than whether
    `json.dumps` is deterministic — it is not, for float formatting across
    versions.
    """
    settings = broker_settings("roundtrip", unique_prefix)
    sample = a_sample()

    with subscribed(settings, f"{unique_prefix}/+/telemetry") as received:
        with a_publisher(settings) as sink:
            sink.write(sample)
        wait_for(received, 1)

    arrived = received[0]
    expected = json.loads(json.dumps(sample.as_row(), default=json_default))

    assert arrived.json().keys() == expected.keys()
    assert arrived.json()["event_id"] == sample.event_id
    assert arrived.json()["machine_id"] == sample.machine_id
    assert arrived.json()["session_id"] == sample.session_id
    assert arrived.number("temperature") == pytest.approx(sample.reading.temperature)
    assert arrived.number("vibration") == pytest.approx(sample.reading.vibration)


def test_it_lands_on_the_topic_the_filter_expects(unique_prefix: str) -> None:
    """The machine id sits in the wildcard position, not somewhere else.

    A topic one level out would still publish, still deliver to *some*
    subscription, and never reach the workflow's filter — which is the failure
    that looks like a working simulator and an empty database.
    """
    settings = broker_settings("topic", unique_prefix)

    with subscribed(settings, f"{unique_prefix}/+/telemetry") as received:
        with a_publisher(settings) as sink:
            sink.write(a_sample())
        wait_for(received, 1)

    assert received[0].topic == f"{unique_prefix}/M003/telemetry"


def test_several_readings_all_arrive(unique_prefix: str) -> None:
    """QoS 1 delivers every message, not a sample of them.

    Not an ordering assertion. MQTT preserves order for one publisher on one
    topic, but this test is about whether a run's readings survive the trip at
    all, and ordering is checked without a broker elsewhere.
    """
    settings = broker_settings("batch", unique_prefix)
    samples = [a_sample(index) for index in range(5)]

    with subscribed(settings, f"{unique_prefix}/+/telemetry") as received:
        with a_publisher(settings) as sink:
            for sample in samples:
                sink.write(sample)
        wait_for(received, len(samples))

    arrived = {item.json()["event_id"] for item in received}
    assert arrived == {sample.event_id for sample in samples}


def test_nothing_is_lost_when_a_sink_reports_no_failure(unique_prefix: str) -> None:
    """The failure counter agrees with what actually happened.

    `failures` is reported to the operator as the run's ingestion health, so it
    has to mean "every publish was accepted", not merely "nothing raised".
    """
    settings = broker_settings("counting", unique_prefix)
    samples = [a_sample(index) for index in range(3)]

    with subscribed(settings, f"{unique_prefix}/+/telemetry") as received:
        with a_publisher(settings) as sink:
            for sample in samples:
                sink.write(sample)
        wait_for(received, len(samples))
        assert sink.failures == 0
