"""The n8n workflow points at the things it claims to point at.

`infra/n8n/telemetry-ingest.json` is committed configuration that nothing
imports, so nothing would notice if it drifted: a renamed route, a changed
header, a topic prefix edited on one side only. Each of those produces a
pipeline that delivers nothing and reports no error -- the worst failure shape
there is.

This is the same technique the Colab notebook's thinness check uses: the
artifact is not code, so a test parses it. The import is confined to tests;
neither service depends on the other at runtime.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import APIRouter
from fastapi.routing import APIRoute

from api.presentation.app import API_V1_PREFIX
from api.presentation.dependencies import require_ingest_token
from api.presentation.routers.machines import router as machines_router
from api.presentation.routers.telemetry import router as telemetry_router
from simulator.cli import DEFAULT_MQTT_TOPIC_PREFIX
from simulator.infrastructure.sinks.mqtt import MqttTelemetrySink, PublishResult


def workflow_path() -> Path:
    """Find the workflow by walking up, so the test survives being moved."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "infra" / "n8n" / "telemetry-ingest.json"
        if candidate.exists():
            return candidate
    raise AssertionError("infra/n8n/telemetry-ingest.json was not found above this test.")


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    """The exported workflow, parsed."""
    parsed: dict[str, Any] = json.loads(workflow_path().read_text(encoding="utf-8"))
    return parsed


def nodes_of_type(workflow: dict[str, Any], node_type: str) -> list[dict[str, Any]]:
    """Every node of one n8n type."""
    return [node for node in workflow["nodes"] if node["type"] == node_type]


def parameter(workflow: dict[str, Any], node_name: str, key: str) -> Any:  # noqa: ANN401
    """One parameter of one named node.

    `Any` because this reads parsed JSON, whose shape no annotation could
    honestly narrow. Every caller asserts the type it expects.
    """
    (node,) = [node for node in workflow["nodes"] if node["name"] == node_name]
    return node["parameters"][key]


def routes_of(router: APIRouter) -> set[str]:
    """Every path a router serves, including the API version prefix.

    `route.path` already carries the router's own prefix -- FastAPI bakes it in
    when the route is declared -- so adding `router.prefix` here would produce
    `/api/v1/telemetry/telemetry`.

    Filtered to `APIRoute` because `router.routes` is typed as the base class,
    which has no `path`; the mounted sub-applications a router can also hold do
    not serve an API path.
    """
    return {API_V1_PREFIX + route.path for route in router.routes if isinstance(route, APIRoute)}


def test_the_subscription_covers_where_the_simulator_publishes(workflow: dict[str, Any]) -> None:
    """The topic filter and the published topic have to agree exactly.

    MQTT has no error for "nobody is listening". A prefix edited on one side and
    not the other gives a broker that accepts every message and delivers it to
    no one, and the symptom is an empty database with no failing component to
    point at.
    """
    subscription = parameter(workflow, "Telemetry topic", "topics")

    assert subscription == f"{DEFAULT_MQTT_TOPIC_PREFIX}/+/telemetry"


def test_the_subscription_matches_a_real_published_topic() -> None:
    """Not just the pattern -- an actual topic the sink produces.

    The wildcard could match the filter and still leave the machine id in the
    wrong position, which a pattern comparison alone would not catch. The
    prefix itself contains a slash, so the split is from the right.
    """

    class _NeverPublishes:
        """Satisfies `Publisher` without ever reaching a broker."""

        def publish(self, topic: str, payload: bytes, qos: int) -> PublishResult:
            raise AssertionError("this test must not publish")

        def close(self) -> int:
            return 0

    published = MqttTelemetrySink(_NeverPublishes(), DEFAULT_MQTT_TOPIC_PREFIX).topic_for("M003")
    head, machine, suffix = published.rsplit("/", 2)

    assert head == DEFAULT_MQTT_TOPIC_PREFIX
    assert machine == "M003"
    assert suffix == "telemetry"
    assert f"{head}/+/telemetry" == f"{DEFAULT_MQTT_TOPIC_PREFIX}/+/telemetry"


def test_the_ingest_node_posts_to_a_route_that_exists(workflow: dict[str, Any]) -> None:
    """Asserted against the router, not against a restated string."""
    url = parameter(workflow, "Ingest", "url")

    assert f"{API_V1_PREFIX}/telemetry" in url
    assert f"{API_V1_PREFIX}/telemetry" in routes_of(telemetry_router)


def test_the_scoring_node_posts_to_a_route_that_exists(workflow: dict[str, Any]) -> None:
    url = parameter(workflow, "Score", "url")

    assert f"{API_V1_PREFIX}/machines/" in url
    assert f"{API_V1_PREFIX}/machines/{{machine_id}}/predictions" in routes_of(machines_router)


def test_the_token_header_is_the_one_the_api_reads(workflow: dict[str, Any]) -> None:
    """Read out of the dependency's signature rather than restated.

    The header name exists in exactly one place in the codebase -- a parameter
    name on `require_ingest_token`, which FastAPI turns into a header. Renaming
    it there must break this, or the workflow would keep sending a header
    nothing checks and every write would be a 401.
    """
    parameter_names = inspect.signature(require_ingest_token).parameters
    (header_parameter,) = [name for name in parameter_names if name != "container"]
    expected = header_parameter.replace("_", "-")

    for node_name in ("Ingest", "Score"):
        sent = parameter(workflow, node_name, "headerParameters")["parameters"]
        names = {header["name"].lower() for header in sent}
        assert expected.lower() in names, f"{node_name} does not send {expected}"


def test_the_workflow_carries_no_credentials(workflow: dict[str, Any]) -> None:
    """Every credential is a reference or an expression, never a value.

    The file is committed. A token written into it would be a token in git, and
    the mistake would not surface until someone read the history.
    """
    serialised = json.dumps(workflow)

    for node in workflow["nodes"]:
        for header in node["parameters"].get("headerParameters", {}).get("parameters", []):
            assert header["value"].startswith("={{"), f"{node['name']} inlines a header value"

    assert "PDM_INGEST_TOKEN" in serialised, "the token expression should be present"
    assert "n8n_api_" not in serialised
    assert "password" not in serialised.lower()


def test_the_workflow_reaches_every_node_from_the_trigger(workflow: dict[str, Any]) -> None:
    """No orphaned nodes.

    An n8n node that is in the file but not wired into the chain still imports,
    still appears on the canvas, and never runs -- which looks like a workflow
    that is missing a step rather than one that has a disconnected one.
    """
    connections = workflow["connections"]
    reachable = {"Telemetry topic"}
    frontier = ["Telemetry topic"]

    while frontier:
        for source in frontier:
            frontier = []
            for branch in connections.get(source, {}).get("main", []):
                for link in branch:
                    if link["node"] not in reachable:
                        reachable.add(link["node"])
                        frontier.append(link["node"])

    assert reachable == {node["name"] for node in workflow["nodes"]}


def test_the_trigger_is_the_only_entry_point(workflow: dict[str, Any]) -> None:
    """Nothing arrives at a node with no inbound connection."""
    has_inbound = {
        link["node"]
        for connection in workflow["connections"].values()
        for branch in connection.get("main", [])
        for link in branch
    }

    assert set(workflow["connections"]) - has_inbound == {"Telemetry topic"}
