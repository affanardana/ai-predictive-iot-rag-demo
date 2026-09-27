"""The metrics this system exposes, and the helpers that record them.

`PRD.md` section 24 names six failures the system must let a human diagnose:
telemetry ingestion, inference, workflow, Copilot tool, retrieval, and LLM. Each
one has a series here, and the mapping is exact rather than approximate -- where
it is not, that is said out loud below rather than left for a reader to discover
during an incident.

**Every label value comes from a closed set defined in this file.** No metric
carries a machine id, an IP address, a path or a question. That is one rule, it
is checkable by a test, and it is what stops a well-meaning `machine_id` label
from turning a scraped endpoint into a memory leak with a per-customer series
for every request. The route label is a *template* -- `/api/v1/machines/{id}` --
which is why the middleware reads it off the matched route rather than off the
path it was asked for.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    ProcessCollector,
    generate_latest,
)

#: This deployment's own registry, rather than the library's default.
#:
#: `prometheus_client` registers a garbage-collection collector into the default
#: registry at import, which would add dozens of series to every scrape for no
#: diagnostic value on a box whose memory question is answered by RSS. Owning
#: the registry also makes `render()` unambiguous.
REGISTRY = CollectorRegistry(auto_describe=True)

#: Resident memory, CPU and the start time -- one line to install, and the
#: series that matters most here. This host has 3.9 GB and no swap, so the
#: failure mode is an OOM kill part-way through an answer, and `process_resident_
#: memory_bytes` is how that is seen coming rather than deduced afterwards.
ProcessCollector(registry=REGISTRY)

#: Latency buckets, and they are not the defaults. The library's top out at 10
#: seconds, which would put every Copilot answer -- about twenty seconds on this
#: hardware, and longer when the model is loading -- into `+Inf`, hiding the one
#: latency the product's design is built around.
_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0)


class Dependency(StrEnum):
    """The services this API calls, and the database under it."""

    PREDICTION = "prediction"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    CHAT = "chat"
    SIMULATOR = "simulator"
    DATABASE = "database"


class Outcome(StrEnum):
    """How a dependency call ended.

    `INVALID_RESPONSE` is separate from `UNAVAILABLE` because they are different
    incidents: one is a service that is down, the other is a service that is up
    and answering something unusable. Collapsing them would make a version
    mismatch look like an outage, and the two have opposite first responses.
    """

    OK = "ok"
    UNAVAILABLE = "unavailable"
    INVALID_RESPONSE = "invalid_response"
    #: The service answered with a *different* model than the configured one --
    #: only the embedder can produce this, and it means stored vectors and query
    #: vectors are about to be incomparable.
    MISMATCH = "mismatch"


class Verdict(StrEnum):
    """What the Copilot did with a question. `ERROR` is ours, not the domain's."""

    ANSWERED = "answered"
    REFUSED = "refused"
    FALLBACK = "fallback"
    ERROR = "error"

    @classmethod
    def of(cls, verdict: str) -> Verdict:
        """Map a domain verdict onto this closed set.

        A mapping rather than a cast from the domain's value: `AnswerVerdict` is
        free to grow a fourth member, and `Verdict(value)` would turn that into
        a `ValueError` raised while recording a metric -- an answer failing
        because a dashboard did not recognise it. An unknown verdict lands on
        `error` instead: visible on the graph, and harmless.
        """
        known = {
            "answered": cls.ANSWERED,
            "refused": cls.REFUSED,
            "fallback": cls.FALLBACK,
        }
        return known.get(verdict.casefold(), cls.ERROR)


class ToolOutcome(StrEnum):
    """What a Copilot tool found."""

    FOUND = "found"
    EMPTY = "empty"


class IngestOutcome(StrEnum):
    """What happened to a telemetry batch.

    `DUPLICATE` is the one worth having: the route answers 201 for a batch it
    stored nothing from, because the events were already there. A status-code
    counter cannot see that, and a pipeline that is quietly re-sending every
    message looks identical to one that is delivering them.
    """

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"


_http_requests = Counter(
    "pdm_http_requests_total",
    "Requests by route template, method and status.",
    ("route", "method", "status"),
    registry=REGISTRY,
)

_http_duration = Histogram(
    "pdm_http_request_duration_seconds",
    "Request duration, by route template.",
    ("route", "method"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

_http_in_flight = Gauge(
    "pdm_http_requests_in_flight",
    "Requests currently being served.",
    registry=REGISTRY,
)

_dependency_calls = Counter(
    "pdm_dependency_calls_total",
    "Calls to a dependency by outcome.",
    ("dependency", "outcome"),
    registry=REGISTRY,
)

_dependency_duration = Histogram(
    "pdm_dependency_duration_seconds",
    "Dependency call duration.",
    ("dependency",),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

_database_health = Gauge(
    "pdm_database_health",
    "1 when the readiness probe's last check succeeded, 0 when it did not.",
    registry=REGISTRY,
)

_copilot_runs = Counter(
    "pdm_copilot_runs_total",
    "Copilot answers by verdict.",
    ("verdict",),
    registry=REGISTRY,
)

_copilot_tools = Counter(
    "pdm_copilot_tool_calls_total",
    "Copilot tool calls by what they found.",
    ("tool", "outcome"),
    registry=REGISTRY,
)

_telemetry_records = Counter(
    "pdm_telemetry_records_total",
    "Telemetry events by what the ingest did with them.",
    ("outcome",),
    registry=REGISTRY,
)

_event_subscribers = Gauge(
    "pdm_event_stream_subscribers",
    "Open event-stream subscribers.",
    registry=REGISTRY,
)

_build_info = Info(
    "pdm_build",
    "Which build and environment this process is.",
    registry=REGISTRY,
)


def record_request(*, route: str, method: str, status: int, duration_seconds: float | None) -> None:
    """Record one finished request.

    `duration_seconds` is `None` for a long-lived stream. An event stream is
    open until the browser goes away, so timing it would put a single sample in
    the top bucket every time and tell nobody anything -- the count and the
    in-flight gauge are the useful facts about it.
    """
    _http_requests.labels(route=route, method=method, status=str(status)).inc()
    if duration_seconds is not None:
        _http_duration.labels(route=route, method=method).observe(duration_seconds)


def record_request_started() -> None:
    """Count a request as in flight."""
    _http_in_flight.inc()


def record_request_finished() -> None:
    """Count a request as no longer in flight."""
    _http_in_flight.dec()


def record_dependency_call(*, dependency: Dependency, outcome: Outcome) -> None:
    """Record how one call to a dependency ended."""
    _dependency_calls.labels(dependency=dependency.value, outcome=outcome.value).inc()


def record_dependency_duration(*, dependency: Dependency, seconds: float) -> None:
    """Record how long one call to a dependency took."""
    _dependency_duration.labels(dependency=dependency.value).observe(seconds)


@contextmanager
def observe_dependency(dependency: Dependency) -> Iterator[None]:
    """Time one dependency call, recording success unless the body raised.

    The *outcome* of a failure is recorded by the caller, in its own `except`
    branch, because only that layer can tell an outage from an unusable answer
    from a model mismatch -- three incidents with different first responses, and
    a generic "failed" here would collapse them. What this centralises is the
    timing, so that adding a call site cannot quietly skip it.
    """
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        # Duration still recorded: a call that timed out took time, and that is
        # the sample an operator wants when asking why requests are slow.
        record_dependency_duration(dependency=dependency, seconds=time.perf_counter() - started)
        raise
    record_dependency_duration(dependency=dependency, seconds=time.perf_counter() - started)
    record_dependency_call(dependency=dependency, outcome=Outcome.OK)


def record_database_health(*, healthy: bool) -> None:
    """Record the readiness probe's last verdict."""
    _database_health.set(1.0 if healthy else 0.0)


def record_copilot_run(*, verdict: Verdict) -> None:
    """Record what the Copilot did with a question."""
    _copilot_runs.labels(verdict=verdict.value).inc()


def record_copilot_tool(*, tool: str, outcome: ToolOutcome) -> None:
    """Record one tool call and whether it found anything."""
    _copilot_tools.labels(tool=tool, outcome=outcome.value).inc()


def record_telemetry(*, outcome: IngestOutcome, count: int) -> None:
    """Record a telemetry batch, or the part of one that was stored."""
    if count:
        _telemetry_records.labels(outcome=outcome.value).inc(count)


def bind_event_subscribers(count: Callable[[], int]) -> None:
    """Expose the broadcaster's subscriber count as a gauge.

    Bound rather than recorded: the count is borrowed state that changes without
    anyone calling in, and a gauge set on every event would be a write per
    telemetry message on a box with one core. `set_function` costs a call per
    scrape, which is a scheduled handful per minute.
    """
    _event_subscribers.set_function(count)


def record_build(*, version: str, app_env: str) -> None:
    """Record which build this process is, once at startup."""
    _build_info.info({"version": version, "app_env": app_env})


def render() -> bytes:
    """Render the registry in the Prometheus exposition format."""
    return generate_latest(REGISTRY)


def sample(name: str, **labels: str) -> float:
    """Return one series' current value, or 0.0 if it does not exist yet.

    For tests and for a shell. A counter that has never been incremented has no
    series at all rather than a zero, which is correct in the exposition format
    and awkward to assert against.
    """
    value = REGISTRY.get_sample_value(name, labels or None)
    return 0.0 if value is None else value
