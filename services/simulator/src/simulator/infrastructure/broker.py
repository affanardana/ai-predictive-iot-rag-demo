"""Broker configuration, read from the environment.

Here rather than in `cli.py` because there are now two composition points that
need it -- the command line and the control surface -- and they sit on the same
rung of the import-linter layers contract, so neither may import the other.
Infrastructure is below both, which is where a thing they share belongs.

From the environment rather than the command line because one of the values is a
password: a credential in `argv` is a credential in the shell history and in
every process listing. Everything else follows the same route for consistency,
and the defaults are EMQX's public broker.
"""

from __future__ import annotations

import os

from simulator.infrastructure.sinks import BrokerSettings

#: EMQX's public broker needs no credentials and carries a publicly trusted
#: certificate, so the defaults are a working pipeline rather than a stub.
DEFAULT_MQTT_HOST = "broker.emqx.io"
DEFAULT_MQTT_PORT = 8883

#: The topic tree telemetry is published under. On a shared public broker this
#: prefix is the only isolation there is -- anyone may subscribe to it or publish
#: into it -- so it is configuration rather than a constant, and n8n subscribes
#: to the same value. `infra/n8n/README.md` says so, and
#: `infra/n8n/telemetry-ingest.json` subscribes to `pdm/demo/+/telemetry`;
#: a run published under any other prefix is accepted by the broker and silently
#: ignored by the orchestrator, with nothing logged anywhere.
DEFAULT_MQTT_TOPIC_PREFIX = "pdm/demo"


def broker_settings_from_env(session_id: str) -> BrokerSettings:
    """Build broker settings for one run."""
    return BrokerSettings(
        host=os.environ.get("MQTT_HOST", DEFAULT_MQTT_HOST),
        port=int(os.environ.get("MQTT_PORT", DEFAULT_MQTT_PORT)),
        tls=os.environ.get("MQTT_TLS", "true").strip().lower() not in {"0", "false", "no"},
        username=os.environ.get("MQTT_USERNAME") or None,
        password=os.environ.get("MQTT_PASSWORD") or None,
        topic_prefix=os.environ.get("MQTT_TOPIC_PREFIX", DEFAULT_MQTT_TOPIC_PREFIX),
        # Derived from the session, so two runs cannot collide. paho's default
        # is random, and a duplicate client id has the broker evict one of the
        # two connections rather than refusing the second -- which, with several
        # machines running at once, would be a run that stops for no visible
        # reason.
        client_id=f"pdm-sim-{session_id}",
    )
