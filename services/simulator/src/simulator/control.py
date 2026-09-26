"""The simulator as a service: an HTTP surface the API starts runs through.

**A composition point, peer to `cli.py` and not beneath it.** The layering
contract puts it on the same rung, which is what stops either from importing the
other -- and that is the correct shape rather than an inconvenience: both exist
to construct the same graph of application and infrastructure, and neither is
the other's dependency. They share `infrastructure.broker` because that lives a
rung below both.

**Nothing imports this module.** FastAPI is the `control` extra, so a
module-level import anywhere else -- `cli.py`, a test, `sinks/__init__.py` --
would break the default offline tier, which installs no extras. It is reached
only by running it:

    python -m simulator.control
"""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn
from fastapi import FastAPI

from simulator.application.run_registry import RunRegistry
from simulator.domain.ports import GroundTruthSink, RunReporter, TelemetrySink
from simulator.domain.session import SimulationSession
from simulator.infrastructure.broker import broker_settings_from_env
from simulator.infrastructure.control.server import create_app
from simulator.infrastructure.reporting import HttpRunReporter
from simulator.infrastructure.sinks import (
    DiscardingGroundTruthSink,
    MqttTelemetrySink,
    connect_paho,
)

#: Reached from the API container over the Compose bridge, never from the
#: internet: the service publishes no host port, and the API is its only client.
#: The control surface has no authentication, which is only defensible because
#: of that -- a published port would be an open "start a run" endpoint.
#:
#: The all-interfaces bind is what `S104` warns about, and it is correct here
#: for the reason immediately above. It is also asserted rather than trusted:
#: `tests/unit/test_deployment_shape.py` fails if the compose service ever gains
#: a `ports:` entry or loses its resource limits, because either would turn this
#: line from a private detail into a public exposure.
DEFAULT_HOST = "0.0.0.0"  # noqa: S104
DEFAULT_PORT = 8002


def build_sinks(session: SimulationSession) -> tuple[TelemetrySink, GroundTruthSink]:
    """Build the sinks one run publishes to.

    The same pairing `realtime --sink mqtt` uses, and for the same reason:
    ground truth labels the training data and must never reach the pipeline, so
    it is discarded at the only point where both channels exist together.
    """
    settings = broker_settings_from_env(session.session_id)
    return (
        MqttTelemetrySink(connect_paho(settings), settings.topic_prefix),
        DiscardingGroundTruthSink(),
    )


def build_reporter() -> RunReporter | None:
    """Build the reporter that tells the API what runs are doing.

    Absent when `PDM_API_BASE_URL` is unset, and that is a supported shape
    rather than a misconfiguration: running this service standalone -- to watch
    it publish to a broker by hand -- should not require an API to talk to. The
    API's own view then goes stale instead, which is the honest symptom.
    """
    base_url = os.environ.get("PDM_API_BASE_URL")
    if not base_url:
        return None
    return HttpRunReporter(base_url=base_url, token=os.environ.get("PDM_INGEST_TOKEN"))


def build_app() -> FastAPI:
    """Build the control application, wired to a fresh registry."""
    return create_app(RunRegistry(sink_factory=build_sinks, reporter=build_reporter()))


def main(argv: list[str] | None = None) -> int:
    """Run the control surface until interrupted."""
    parser = argparse.ArgumentParser(
        prog="simulator.control",
        description="Serve the HTTP control surface the API starts runs through.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    print(f"control surface on http://{args.host}:{args.port}", file=sys.stderr)
    # No `--workers`. The run registry is in-process state, so a second worker
    # would answer `GET /runs` with an empty list and accept a second run for a
    # machine already running in the first -- the same silent-subset failure the
    # API's broadcaster has, with a worse symptom.
    uvicorn.run(build_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
