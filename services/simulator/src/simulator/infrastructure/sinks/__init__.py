"""Output destinations for generated data.

The `TelemetrySink` and `GroundTruthSink` interfaces are re-exported here for
convenience, but they are defined in `simulator.domain.ports` — an implementation
dependency flowing back into the domain would invert the layering.
"""

from simulator.domain.ports import GroundTruthSink, TelemetrySink
from simulator.infrastructure.sinks.base import DEFAULT_FLUSH_THRESHOLD, BufferedSink
from simulator.infrastructure.sinks.console import (
    ConsoleGroundTruthSink,
    ConsoleTelemetrySink,
)
from simulator.infrastructure.sinks.jsonl import (
    JsonLinesGroundTruthSink,
    JsonLinesTelemetrySink,
)
from simulator.infrastructure.sinks.mqtt import (
    BrokerSettings,
    DiscardingGroundTruthSink,
    MqttTelemetrySink,
    Publisher,
    connect_paho,
)
from simulator.infrastructure.sinks.parquet import (
    GROUND_TRUTH_SCHEMA,
    TELEMETRY_SCHEMA,
    ParquetGroundTruthSink,
    ParquetTelemetrySink,
)

__all__ = [
    "DEFAULT_FLUSH_THRESHOLD",
    "GROUND_TRUTH_SCHEMA",
    "TELEMETRY_SCHEMA",
    "BrokerSettings",
    "BufferedSink",
    "ConsoleGroundTruthSink",
    "ConsoleTelemetrySink",
    "DiscardingGroundTruthSink",
    "GroundTruthSink",
    "JsonLinesGroundTruthSink",
    "JsonLinesTelemetrySink",
    "MqttTelemetrySink",
    "ParquetGroundTruthSink",
    "ParquetTelemetrySink",
    "Publisher",
    "TelemetrySink",
    "connect_paho",
]
