"""Output destinations for generated data.

The `TelemetrySink` and `GroundTruthSink` interfaces are re-exported here for
convenience, but they are defined in `simulator.domain.ports` — an implementation
dependency flowing back into the domain would invert the layering.

**The Parquet sinks are imported on demand, not eagerly.** Importing any
submodule executes its parent package, so `from
simulator.infrastructure.sinks.mqtt import MqttTelemetrySink` — which the
control container does — would otherwise pull `pyarrow` in through this file.
PyArrow is around 40 MB of wheel and a slow import, and the control container
publishes to a broker and never writes a file. The eager import would mean
either a container carrying a columnar library it cannot use, or an
`ImportError` at startup.

The same trick the `mqtt` extra already relies on for `paho`, applied to a
module rather than a symbol. `tests/unit/test_lazy_sinks.py` asserts it holds,
in a subprocess, because `sys.modules` inside a running test is polluted by
whatever ran first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    # For type checkers only, which resolve names statically and never execute
    # this branch. Named here so `__getattr__` below can stay untyped.
    from simulator.infrastructure.sinks.parquet import (
        GROUND_TRUTH_SCHEMA,
        TELEMETRY_SCHEMA,
        ParquetGroundTruthSink,
        ParquetTelemetrySink,
    )

#: Names served from `simulator.infrastructure.sinks.parquet`, resolved on first
#: access by `__getattr__`. PEP 562.
_DEFERRED = frozenset(
    {
        "GROUND_TRUTH_SCHEMA",
        "TELEMETRY_SCHEMA",
        "ParquetGroundTruthSink",
        "ParquetTelemetrySink",
    }
)


def __getattr__(name: str) -> object:
    """Import a Parquet name the first time it is asked for.

    Declared `-> object` rather than `-> Any`: the linter rejects a bare `Any`
    annotation, and `object` is honest anyway -- what comes back is a class or a
    schema constant, and the caller narrows it by importing it by name.

    Raises:
        AttributeError: for a name this module does not export, which is what
            makes `from ... import x` fail the way a caller expects rather than
            returning something surprising.
    """
    if name in _DEFERRED:
        from simulator.infrastructure.sinks import parquet

        return getattr(parquet, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
