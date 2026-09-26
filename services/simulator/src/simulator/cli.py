"""Command line entry point.

Composition happens here: this is the only module that names concrete sinks and
decides where generated data goes. The application layer receives them already
built.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO

from simulator.application import run_session, stream_session
from simulator.domain.demo import (
    DEMO_DURATION,
    DEMO_MACHINE_ID,
    DEMO_SCENARIO,
    DEMO_SEED,
)
from simulator.domain.errors import SimulationError
from simulator.domain.ports import GroundTruthSink, TelemetrySink
from simulator.domain.scenario import Scenario, profile_for
from simulator.domain.session import SimulationSession
from simulator.domain.timestamps import utc_now
from simulator.infrastructure.sinks import (
    BrokerSettings,
    ConsoleGroundTruthSink,
    ConsoleTelemetrySink,
    DiscardingGroundTruthSink,
    JsonLinesGroundTruthSink,
    JsonLinesTelemetrySink,
    MqttTelemetrySink,
    ParquetGroundTruthSink,
    ParquetTelemetrySink,
    connect_paho,
)

DEFAULT_MACHINE_COUNT = 3
DEFAULT_DATASET_MINUTES = 60
DEFAULT_REALTIME_MINUTES = 120
DEFAULT_TICK_SECONDS = 0.2

DATASET_OUTPUT_DIR = Path("data/raw")
REALTIME_OUTPUT_DIR = Path("data/stream")

#: EMQX's public broker needs no credentials and carries a publicly trusted
#: certificate, so the defaults are a working pipeline rather than a stub.
DEFAULT_MQTT_HOST = "broker.emqx.io"
DEFAULT_MQTT_PORT = 8883

#: The topic tree telemetry is published under. On a shared public broker this
#: prefix is the only isolation there is -- anyone may subscribe to it or
#: publish into it -- so it is configuration rather than a constant, and n8n
#: subscribes to the same value. `infra/n8n/README.md` says so.
DEFAULT_MQTT_TOPIC_PREFIX = "pdm/demo"

#: Overwritten by `--demo`, so the demonstrated sequence is always the same one.
_FALLBACK_MACHINE_ID = "M001"
_FALLBACK_SEED = 1

EXIT_INTERRUPTED = 130
EXIT_USAGE_ERROR = 2


def _parse_instant(value: str) -> datetime:
    """Parse an ISO 8601 instant, insisting on a timezone.

    A naive value would be rejected by the domain a moment later, but with a
    message about a field name rather than about the flag the caller typed.
    """
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not an ISO 8601 instant, e.g. 2026-09-23T12:00:00+00:00."
        ) from exc
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"'{value}' has no timezone. Add an offset, e.g. 2026-09-23T12:00:00+00:00."
        )
    return moment


_STARTED_AT_HELP = (
    "Pin the instant the run starts from, so its output is byte-identical "
    "between runs. Defaults to now. PRD section 12 needs this for automated "
    "tests, screenshots, and reviewer reproduction."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="simulator",
        description="Generate temporally coherent motor telemetry.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    dataset = subcommands.add_parser(
        "dataset",
        help="Generate a historical dataset as files.",
    )
    dataset.add_argument("--machines", type=int, default=DEFAULT_MACHINE_COUNT)
    dataset.add_argument("--scenario", default=Scenario.BEARING_DEGRADATION.value)
    dataset.add_argument("--minutes", type=int, default=DEFAULT_DATASET_MINUTES)
    dataset.add_argument("--seed", type=int, default=_FALLBACK_SEED)
    dataset.add_argument("--out", type=Path, default=DATASET_OUTPUT_DIR)
    dataset.add_argument("--started-at", type=_parse_instant, default=None, help=_STARTED_AT_HELP)
    dataset.add_argument(
        "--format",
        choices=("parquet", "jsonl"),
        default="parquet",
        help=(
            "Parquet for volume; jsonl to read the output by eye. Both write "
            "telemetry and ground truth to separate files."
        ),
    )

    realtime = subcommands.add_parser(
        "realtime",
        help="Emit telemetry continuously, at a wall-clock pace.",
    )
    realtime.add_argument("--machine", default=None)
    realtime.add_argument("--scenario", default=None)
    realtime.add_argument("--seed", type=int, default=None)
    realtime.add_argument("--minutes", type=int, default=None)
    realtime.add_argument("--tick-seconds", type=float, default=DEFAULT_TICK_SECONDS)
    realtime.add_argument("--sink", choices=("console", "jsonl", "mqtt"), default="console")
    realtime.add_argument("--out", type=Path, default=REALTIME_OUTPUT_DIR)
    realtime.add_argument("--started-at", type=_parse_instant, default=None, help=_STARTED_AT_HELP)
    realtime.add_argument(
        "--session-id",
        default=None,
        help=(
            "Name this run. Derived from the scenario and seed by default, so "
            "re-running with the same seed republishes the same event ids and "
            "the pipeline stores nothing new -- which is idempotency working, "
            "not a fault. Pass a new id to ingest the series a second time."
        ),
    )
    realtime.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Use the canonical PRD section 12 demonstration: M003, bearing "
            "degradation, and a fixed seed, so the sequence is always the same. "
            "Any value given explicitly still wins."
        ),
    )

    subcommands.add_parser("scenarios", help="List the available scenarios.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    try:
        return _dispatch(args)
    except SimulationError as exc:
        # A bad scenario name or an impossible configuration is a usage problem,
        # not a crash. One line and the conventional exit code, rather than a
        # traceback that buries the actual message.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR


def _dispatch(args: argparse.Namespace) -> int:
    """Route to the requested subcommand."""
    if args.command == "dataset":
        return _run_dataset(args)
    if args.command == "realtime":
        return _run_realtime(args)
    return _list_scenarios()


def _run_dataset(args: argparse.Namespace) -> int:
    """Generate a bounded dataset and write it to files."""
    session = SimulationSession.create(
        machine_ids=[f"M{index:03d}" for index in range(1, args.machines + 1)],
        scenario=Scenario.from_name(args.scenario),
        seed=args.seed,
        duration=timedelta(minutes=args.minutes),
        started_at=args.started_at or utc_now(),
    )
    telemetry_sink, ground_truth_sink = _file_sinks(args.out, args.format)

    summary = run_session(session, telemetry_sink, ground_truth_sink)

    _report(
        [
            ("session", summary.session_id),
            ("scenario", session.scenario.value),
            ("machines", str(summary.machines)),
            ("ticks per machine", str(summary.ticks_per_machine)),
            ("observations", str(summary.total_samples)),
            ("written to", str(args.out)),
        ]
    )
    return 0


def _run_realtime(args: argparse.Namespace) -> int:
    """Stream telemetry at a wall-clock pace."""
    machine_id, scenario, seed, duration = _realtime_options(args)

    session = SimulationSession.create(
        machine_ids=[machine_id],
        scenario=scenario,
        seed=seed,
        duration=duration,
        started_at=args.started_at or utc_now(),
        session_id=args.session_id,
    )
    telemetry_sink, ground_truth_sink = _realtime_sinks(args, session)

    print(
        f"streaming {session.session_id}  scenario={scenario.value}  "
        f"machine={machine_id}  ticks={session.tick_count}  "
        f"tick_seconds={args.tick_seconds}",
        file=sys.stderr,
    )

    try:
        summary = stream_session(
            session,
            telemetry_sink,
            ground_truth_sink,
            tick_seconds=args.tick_seconds,
        )
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_INTERRUPTED

    # Both sinks are closed by `stream_session`'s `finally`, so the counts here
    # are final. Publish failures are reported even at zero: a silent zero and
    # a missing line look the same, and only one of them means "nothing was
    # dropped".
    rows = [("observations", str(summary.total_samples))]
    if isinstance(telemetry_sink, MqttTelemetrySink):
        rows.append(("published to", f"{telemetry_sink.topic_prefix}/{{machine}}/telemetry"))
        rows.append(("publish failures", str(telemetry_sink.failures)))
    _report(rows, stream=sys.stderr)
    return 0


def _list_scenarios() -> int:
    """Print the available scenarios and the mechanisms each one drives."""
    header = f"{'scenario':<22}{'bearing':>9}{'thermal':>9}{'load':>7}"
    print(header)
    print("-" * len(header))
    for scenario in Scenario:
        weights = profile_for(scenario)
        print(
            f"{scenario.value:<22}"
            f"{weights.bearing_weight:>9.2f}"
            f"{weights.thermal_weight:>9.2f}"
            f"{weights.load_weight:>7.2f}"
        )
    return 0


def _realtime_options(args: argparse.Namespace) -> tuple[str, Scenario, int, timedelta]:
    """Resolve the demonstration defaults against anything given explicitly.

    Resolved rather than defaulted in the parser, because `--demo` has to fill
    in only the values the caller left alone: passing `--machine M004 --demo`
    should demonstrate M004, not silently override it back to M003.
    """
    machine_id = args.machine or (DEMO_MACHINE_ID if args.demo else _FALLBACK_MACHINE_ID)
    scenario = (
        Scenario.from_name(args.scenario)
        if args.scenario
        else (DEMO_SCENARIO if args.demo else Scenario.BEARING_DEGRADATION)
    )
    seed = args.seed if args.seed is not None else (DEMO_SEED if args.demo else _FALLBACK_SEED)
    duration = timedelta(
        minutes=args.minutes
        if args.minutes is not None
        else (int(DEMO_DURATION.total_seconds() // 60) if args.demo else DEFAULT_REALTIME_MINUTES)
    )
    return machine_id, scenario, seed, duration


def _file_sinks(directory: Path, file_format: str) -> tuple[TelemetrySink, GroundTruthSink]:
    """Build a pair of file sinks, whichever format was asked for."""
    if file_format == "jsonl":
        return (
            JsonLinesTelemetrySink(directory / "telemetry.jsonl"),
            JsonLinesGroundTruthSink(directory / "ground_truth.jsonl"),
        )
    return (
        ParquetTelemetrySink(directory / "telemetry.parquet"),
        ParquetGroundTruthSink(directory / "ground_truth.parquet"),
    )


def _realtime_sinks(
    args: argparse.Namespace, session: SimulationSession
) -> tuple[TelemetrySink, GroundTruthSink]:
    """Build the sinks a realtime run writes to."""
    if args.sink == "console":
        return ConsoleTelemetrySink(), ConsoleGroundTruthSink()
    if args.sink == "mqtt":
        settings = _broker_settings(session.session_id)
        # Ground truth is not published; see `DiscardingGroundTruthSink`.
        return (
            MqttTelemetrySink(connect_paho(settings), settings.topic_prefix),
            DiscardingGroundTruthSink(),
        )
    return _file_sinks(args.out, "jsonl")


def _broker_settings(session_id: str) -> BrokerSettings:
    """Read broker settings from the environment.

    From the environment rather than the command line because one of them is a
    password: a credential in `argv` is a credential in the shell history and in
    every process listing. Everything else follows the same route for
    consistency, and the defaults are EMQX's public broker.
    """
    return BrokerSettings(
        host=os.environ.get("MQTT_HOST", DEFAULT_MQTT_HOST),
        port=int(os.environ.get("MQTT_PORT", DEFAULT_MQTT_PORT)),
        tls=os.environ.get("MQTT_TLS", "true").strip().lower() not in {"0", "false", "no"},
        username=os.environ.get("MQTT_USERNAME") or None,
        password=os.environ.get("MQTT_PASSWORD") or None,
        topic_prefix=os.environ.get("MQTT_TOPIC_PREFIX", DEFAULT_MQTT_TOPIC_PREFIX),
        # Derived from the session, so two runs cannot collide. paho's default
        # is random, and a duplicate client id has the broker evict one of the
        # two connections rather than refusing the second.
        client_id=f"pdm-sim-{session_id}",
    )


def _report(rows: Sequence[tuple[str, str]], stream: TextIO | None = None) -> None:
    """Print an aligned summary."""
    label_width = max(len(label) for label, _ in rows)
    target = stream if stream is not None else sys.stdout
    for label, value in rows:
        print(f"{label:<{label_width}}  {value}", file=target)


__all__ = ["build_parser", "main"]
