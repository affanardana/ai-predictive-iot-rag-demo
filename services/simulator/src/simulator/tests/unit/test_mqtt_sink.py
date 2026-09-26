"""The MQTT sink, against a fake publisher.

No broker here and none needed: the sink's job is to serialise an observation
and address it, which is testable without a network. What a broker adds --
delivery guarantees, reconnection, TLS -- is `connect_paho`'s and the opt-in
`mqtt` tier's to cover.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta

from simulator.domain.engine import MachineSimulator
from simulator.domain.scenario import Scenario
from simulator.domain.session import SimulationSession
from simulator.domain.state import GroundTruthState, TelemetrySample
from simulator.infrastructure.sinks.mqtt import (
    DEFAULT_QOS,
    DiscardingGroundTruthSink,
    MqttTelemetrySink,
    PublishResult,
)
from simulator.tests.conftest import FIXED_START


@dataclass
class FakeMessageInfo:
    """Stands in for paho's `MQTTMessageInfo`."""

    rc: int = 0


@dataclass
class FakePublisher:
    """Records what was published, and can be told to fail."""

    rc: int = 0
    #: How many the fake claims were still unacknowledged when it closed.
    undelivered: int = 0
    published: list[tuple[str, bytes, int]] = field(default_factory=list)
    closed: bool = False

    def publish(self, topic: str, payload: bytes, qos: int) -> PublishResult:
        self.published.append((topic, payload, qos))
        return FakeMessageInfo(rc=self.rc)

    def close(self) -> int:
        self.closed = True
        return self.undelivered


def a_sample(scenario: Scenario = Scenario.NORMAL) -> TelemetrySample:
    """One observation from a real run, rather than a hand-built stand-in."""
    return _a_simulator(scenario).tick(0).telemetry


def a_ground_truth() -> GroundTruthState:
    """The hidden half of the same tick."""
    return _a_simulator(Scenario.BEARING_DEGRADATION).tick(0).ground_truth


def _a_simulator(scenario: Scenario) -> MachineSimulator:
    session = SimulationSession.create(
        machine_ids=["M003"],
        scenario=scenario,
        seed=2026,
        duration=timedelta(minutes=2),
        started_at=FIXED_START,
    )
    return MachineSimulator(session, session.machines[0])


def test_a_sample_is_published_to_the_machines_topic() -> None:
    publisher = FakePublisher()
    sink = MqttTelemetrySink(publisher, topic_prefix="pdm/demo")

    sink.write(a_sample())

    topic, _, _ = publisher.published[0]
    assert topic == "pdm/demo/M003/telemetry"


def test_surrounding_slashes_in_the_prefix_are_tolerated() -> None:
    """A prefix typed with a trailing slash must not produce `//` in the topic.

    The topic is the one thing the simulator and the n8n subscription have to
    agree on exactly, so a cosmetic difference is a pipeline that silently
    delivers nothing.
    """
    sink = MqttTelemetrySink(FakePublisher(), topic_prefix="/pdm/demo/")

    assert sink.topic_for("M003") == "pdm/demo/M003/telemetry"


def test_the_payload_is_the_flat_row_the_api_ingests() -> None:
    """The same serialisation the JSON Lines sink writes.

    Both go through `as_row()` and the same `json_default`, so this asserts the
    payload is a JSON object holding the ten expected keys -- not a string
    wrapping one, and not the nested shape the API's *responses* use.
    """
    publisher = FakePublisher()
    sink = MqttTelemetrySink(publisher, topic_prefix="pdm/demo")
    sample = a_sample()

    sink.write(sample)

    _, payload, _ = publisher.published[0]
    decoded = json.loads(payload)

    assert set(decoded) == {
        "event_id",
        "machine_id",
        "recorded_at",
        "session_id",
        "temperature",
        "vibration",
        "rpm",
        "current",
        "load",
        "voltage",
    }
    assert decoded["event_id"] == sample.event_id
    assert decoded["session_id"] == sample.session_id
    assert decoded["recorded_at"] == sample.recorded_at.isoformat()
    assert decoded["temperature"] == sample.reading.temperature


def test_publication_is_at_least_once() -> None:
    """QoS 1, and the reason is the primary key on the other side.

    A duplicate costs one ignored `ON CONFLICT`; a lost reading is a hole in a
    window the model needs to be contiguous. The cheaper mistake is the
    duplicate.
    """
    publisher = FakePublisher()

    MqttTelemetrySink(publisher, topic_prefix="pdm/demo").write(a_sample())

    assert publisher.published[0][2] == DEFAULT_QOS == 1


def test_a_failed_publish_is_counted_rather_than_raised() -> None:
    """A broker hiccup must not abort a 720-reading run.

    It must not pass unnoticed either: the count reaches the run summary, which
    is what makes an ingestion failure diagnosable rather than a gap someone
    finds later in the charts.
    """
    publisher = FakePublisher(rc=4)
    sink = MqttTelemetrySink(publisher, topic_prefix="pdm/demo")

    sink.write(a_sample())
    sink.write(a_sample())

    assert sink.failures == 2
    assert len(publisher.published) == 2, "a failed publish is still attempted"


def test_closing_the_sink_closes_the_publisher() -> None:
    """The network thread has to be released, or the process will not exit."""
    publisher = FakePublisher()
    sink = MqttTelemetrySink(publisher, topic_prefix="pdm/demo")

    sink.close()

    assert publisher.closed


def test_a_message_lost_on_close_is_counted_as_a_failure() -> None:
    """The tail of a run must not vanish quietly.

    `publish` returns as soon as a message is queued, so a run that finishes
    quickly still has its last readings in flight when the sink closes. Those
    are the readings nearest the failure the demonstration is about, and they
    used to be discarded by `disconnect` while `failures` reported zero.

    This was found against a real broker, not here — the fake cannot reproduce
    paho's queue, only the accounting that surfaces it.
    """
    publisher = FakePublisher(undelivered=3)
    sink = MqttTelemetrySink(publisher, topic_prefix="pdm/demo")

    sink.write(a_sample())
    sink.close()

    assert sink.failures == 3


def test_the_ground_truth_half_is_not_published() -> None:
    """`DiscardingGroundTruthSink` drops rather than forwards.

    Ground truth is the answer key -- `failure_imminent`, the scenario label.
    Putting it on the same wire as the readings it is meant to be predicted
    from is the leak the simulator's design exists to prevent, so this sink
    deliberately does nothing. There is no publisher to capture, which is the
    point: it holds no transport at all.
    """
    sink = DiscardingGroundTruthSink()

    sink.write(a_ground_truth())
    sink.close()

    assert not hasattr(sink, "_publisher")
