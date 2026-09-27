"""Application configuration.

Configuration lives outside the source: values come from the environment,
optionally via a local `.env` file that is gitignored. Unknown keys in `.env`
are rejected rather than ignored, so a misspelled variable fails at startup
instead of silently taking a default.
"""

from __future__ import annotations

import logging
from typing import Literal, Self
from urllib.parse import urlparse

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from api.domain.value_objects.risk_thresholds import (
    DEFAULT_CRITICAL_THRESHOLD,
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_WARNING_THRESHOLD,
    RiskThresholds,
)

#: Supabase's transaction pooler port. Supabase documents that ORMs relying on
#: server-side prepared statements break behind it, and the resulting failures
#: surface as confusing driver errors rather than anything pointing at the
#: port. Rejecting it here turns that into one clear message at startup.
SUPABASE_TRANSACTION_POOLER_PORT = 6543
SUPABASE_POOLER_HOST_SUFFIX = ".pooler.supabase.com"


class Settings(BaseSettings):
    """Runtime configuration, read from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        case_sensitive=False,
    )

    app_env: Literal["local", "ci", "production"] = "local"
    log_level: str = "INFO"

    #: Operational database. Required -- there is deliberately no default, so a
    #: missing value fails loudly instead of connecting somewhere unexpected.
    database_url: str

    #: Optional. Only the `postgres`-marked tests use it; when unset they skip
    #: and the suite runs entirely on in-memory SQLite.
    test_database_url: str | None = None

    #: Where the model inference service lives. The API calls it over HTTP so
    #: that torch stays out of this process and out of its container image.
    inference_service_url: str = "http://localhost:8001"

    #: Where the simulator service lives. Over HTTP for the same reason as
    #: inference: a simulation run blocks a thread for its whole wall-clock
    #: duration, which is not something to do inside a request-serving process.
    #:
    #: The default is `localhost` for a developer running both, and the same
    #: trap inference has applies verbatim: inside a container, `localhost` is
    #: the API itself, and the failure looks like a cold service rather than a
    #: misconfiguration. `compose.yaml` sets the service name.
    simulation_service_url: str = "http://localhost:8002"

    #: How many runs may be active at once across the fleet.
    #:
    #: A ceiling rather than a preference. `POST /api/v1/simulations` is
    #: reachable by anyone who finds the hostname, and each run costs CPU on a
    #: box that already shares one core between the API, the orchestrator and
    #: the model service. Three is the point at which a fleet-wide demonstration
    #: still works and the box does not visibly suffer.
    simulation_max_concurrent_runs: int = 3

    #: How long a run may go without reporting before it is treated as gone.
    #: The simulator reports every few seconds, so this is several missed
    #: heartbeats -- long enough that a busy box does not mark live runs dead,
    #: short enough that a container which died is noticed.
    simulation_heartbeat_timeout_seconds: float = 45.0

    #: Shared secret the orchestrator presents on the write endpoints. Required
    #: outside `local`; see `_validate_consistency`. This is deliberately not a
    #: user authentication scheme -- Phase 11 owns that -- it is one machine
    #: proving to another that it is the expected caller.
    ingest_api_token: str | None = None

    #: The embedding model the corpus was embedded with, name and revision.
    #: Retrieval only considers chunks carrying this string, and the embedding
    #: service is refused if it reports running anything else -- vectors from two
    #: models are not comparable, and mixing them returns plausible nonsense
    #: with nothing logged.
    knowledge_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2@main"

    #: The Copilot's language model, as the inference service names it. It
    #: travels into every answer as provenance, so a reader can see which model
    #: wrote the prose -- the evidence is attached by the system and unaffected
    #: by a model swap, and naming the model is what lets the two be judged
    #: separately. `compose.yaml` sets it from the same value the inference
    #: container is given, so the two cannot drift.
    copilot_model: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

    #: Which commit this image was built from, put there by the Dockerfile's
    #: `SOURCE_COMMIT` build argument. `unknown` means the build did not pass
    #: one -- which is a fact worth reporting rather than hiding, because the
    #: question it answers ("is the box running what I pushed?") is asked
    #: during an incident, and a plausible-looking wrong answer is worse than
    #: an obvious gap.
    source_commit: str = "unknown"

    #: Credential for `GET /metrics`, in the same sense `ingest_api_token`
    #: guards writes: one machine proving to another that it is the expected
    #: caller. Separate from the ingest token because the two protect different
    #: surfaces, and a leak of one should not widen the other.
    #:
    #: **Optional.** Unset means the endpoint is public, which is what local
    #: work and a hosted scraper that can only be given a URL both need. The
    #: exposition carries route templates, statuses and timings -- no machine
    #: ids, no addresses, no question text -- so public is a defensible default
    #: rather than an oversight.
    metrics_token: str | None = None

    #: Addresses whose `X-Forwarded-For` is believed when working out who a
    #: request is charged to.
    #:
    #: **A deployment fact, not a constant**, and the default is the whole
    #: story of why: the API publishes only to loopback, but Docker forwards
    #: through the bridge, so the peer this process sees is the gateway
    #: (`172.17.0.1`) rather than `127.0.0.1`. Uvicorn's own default --
    #: `127.0.0.1,::1`, which `FORWARDED_ALLOW_IPS` in `compose.yaml` overrides
    #: from this same value -- does not include it, so a deployment that trusted
    #: only loopback would believe no forwarding header at all and charge every
    #: visitor on earth to one bucket. `172.16.0.0/12` is Docker's default
    #: address pool; a host that overrides `default-address-pools` needs this.
    rate_limit_trusted_proxies: str = "127.0.0.1,::1,172.16.0.0/12"

    #: Per-address rate ceilings, in requests per minute, for the routes that
    #: cost real resources. See `presentation/rate_limit.py` for what these do
    #: and do not guarantee.
    #:
    #: **Sized to clear the demonstration, not to shape it.** A single dashboard
    #: tab polls roughly ten times a minute idle, and about fifty with a live
    #: run; two tabs plus a run is the worst honest case at around two hundred.
    #: A limiter that fires during the canonical demonstration has cost more
    #: than it saved, so the global ceiling is thirty times the resting rate and
    #: exists to stop a scraper rather than a person. The route ceilings are the
    #: ones with teeth.
    #:
    #: The bursts are module constants rather than settings, because a burst is
    #: a shape decision and the rate is a property of the host.
    rate_limit_global_per_minute: int = 600
    #: One sustained question a minute, with a burst of two so a follow-up is
    #: immediate. The Copilot's job here is **fairness, not protection**: the
    #: single answer slot already bounds the box at about three answers a
    #: minute, and what it does not do is stop one visitor holding that slot so
    #: everyone else gets `409 chat_busy` forever.
    rate_limit_copilot_per_minute: int = 1
    #: Three concurrent runs, one per machine and a sixty-minute floor already
    #: bound the CPU. What is unbounded is a start/reset loop.
    rate_limit_simulation_per_minute: int = 1
    #: Cheap next to the others and not free: an embedding pass and a
    #: cross-encoder pass per call, on one core. A person typing queries does
    #: two to four a minute.
    rate_limit_search_per_minute: int = 6
    #: How many caller-and-budget buckets to keep before evicting the least
    #: recently used. Bounds the limiter's own memory, which is the point: an
    #: unbounded map keyed by address is a memory leak with a security story.
    rate_limit_max_tracked_clients: int = 4096

    #: How many questions the Copilot may be answering at once. One, because a
    #: generation burns the single core for about twenty seconds: a second
    #: question would halve the speed of the first and leave both readers
    #: waiting twice as long.
    copilot_max_concurrent_questions: int = 1

    #: Lowest rerank score that counts as evidence, below which the answer is
    #: "the documentation does not cover this" (PRD section 19).
    #:
    #: 0.0 here is the *uncalibrated* default, not a safe one: the reranker is a
    #: cross-encoder whose logits centre near zero, so this stands in the middle
    #: of its range. Measured against the deployed corpus, answerable questions
    #: score a median of +4.26 and unanswerable ones -8.17, with the worst
    #: answerable at -2.48 and the best unanswerable at -4.20 — so the value
    #: belongs in that gap. `env.template` carries it and explains how to
    #: re-derive it; `ml knowledge evaluate` prints the two medians.
    knowledge_minimum_score: float = 0.0

    risk_warning_threshold: float = DEFAULT_WARNING_THRESHOLD
    risk_high_threshold: float = DEFAULT_HIGH_THRESHOLD
    risk_critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD

    @field_validator("metrics_token", mode="before")
    @classmethod
    def _blank_metrics_token_is_unset(cls, value: object) -> object:
        """Treat an empty `METRICS_TOKEN` as "not configured".

        Compose passes the variable explicitly so the deployment shows what it
        sets, and `${METRICS_TOKEN:-}` arrives as an empty string rather than as
        absent. Without this, that empty string is a *blank secret*, which the
        token rules reject -- so leaving the guard off would stop the API from
        starting, and the fix would look like "delete the variable" rather than
        "leave it empty".
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Reject a log level that `logging` would not recognise."""
        level = value.upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError(
                f"Unknown log level '{value}'. "
                "Expected one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
            )
        return level

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        """Validate the database URLs and risk thresholds together."""
        _reject_driverless_postgres_url(self.database_url, "DATABASE_URL")
        _reject_transaction_pooler(self.database_url, "DATABASE_URL")

        if self.test_database_url:
            _reject_driverless_postgres_url(self.test_database_url, "TEST_DATABASE_URL")
            _reject_transaction_pooler(self.test_database_url, "TEST_DATABASE_URL")

        # Constructing the value object validates ordering and bounds, so a
        # contradictory threshold configuration fails at startup.
        _ = self.risk_thresholds

        # Required in *every* environment except local, rather than in
        # production alone. `app_env` defaults to "local", so gating on
        # `is_production` would mean a deployment that forgot to set APP_ENV
        # failed open -- running unauthenticated while looking configured.
        if self.app_env != "local" and not self.ingest_api_token:
            raise ValueError(
                "INGEST_API_TOKEN is required when APP_ENV is not 'local'. The "
                "write endpoints are reachable from the internet and are the "
                "only way telemetry and machines enter the database, so an "
                "unauthenticated deployment lets anyone write readings."
            )

        _validate_token(self.ingest_api_token, "INGEST_API_TOKEN")
        _validate_token(self.metrics_token, "METRICS_TOKEN")
        return self

    @property
    def risk_thresholds(self) -> RiskThresholds:
        """Build the risk band configuration from these settings."""
        return RiskThresholds(
            warning=self.risk_warning_threshold,
            high=self.risk_high_threshold,
            critical=self.risk_critical_threshold,
        )

    @property
    def is_production(self) -> bool:
        """Whether this process is running in the production environment."""
        return self.app_env == "production"


def _validate_token(token: str | None, variable_name: str) -> None:
    """Reject a shared secret that could never be presented.

    One function for both tokens so the rules cannot drift apart: whatever makes
    `INGEST_API_TOKEN` unusable makes `METRICS_TOKEN` unusable in exactly the
    same way, and a second copy of these two checks is a second place for one of
    them to be forgotten.

    Raises:
        ValueError: if the token is blank or not ASCII.
    """
    if token is None:
        return
    # A blank secret is not a secret: `compare_digest("", "")` is true, so an
    # empty token would authenticate a request carrying no header at all.
    if not token.strip():
        raise ValueError(f"{variable_name} must not be blank.")
    # HTTP header values are byte strings. A token outside ASCII could never be
    # presented by any client, so it would refuse every request -- a
    # misconfiguration best caught here rather than diagnosed from a wall of 401s.
    if not token.isascii():
        raise ValueError(
            f"{variable_name} must be ASCII. Header values are byte strings and "
            "cannot carry a non-ASCII character, so this token could never be "
            "presented."
        )


def _reject_driverless_postgres_url(url: str, variable_name: str) -> None:
    """Fail if `url` uses a bare PostgreSQL scheme.

    SQLAlchemy resolves ``postgresql://`` to **psycopg2**, which this project
    does not install -- it uses psycopg 3, whose dialect is
    ``postgresql+psycopg://``. A bare scheme therefore fails with
    ``ModuleNotFoundError: No module named 'psycopg2'``, an error that points at
    a missing package rather than at the URL that caused it.

    Supabase's dashboard hands out connection strings without a driver, so
    pasting one verbatim is the obvious thing to do and lands exactly here.
    Note the check is on the scheme only: ``postgresql+psycopg://`` does not
    start with ``postgresql://``, so a correctly-specified URL passes through.
    """
    if url.lower().startswith(("postgresql://", "postgres://")):
        raise ValueError(
            f"{variable_name} uses a bare 'postgresql://' scheme, which SQLAlchemy "
            "resolves to psycopg2 -- a driver this project does not install. "
            "Use 'postgresql+psycopg://' for psycopg 3. Supabase shows the URL "
            "without a driver, so the '+psycopg' has to be added by hand."
        )


def _reject_transaction_pooler(url: str, variable_name: str) -> None:
    """Fail if `url` points at Supabase's transaction pooler."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if (
        host.endswith(SUPABASE_POOLER_HOST_SUFFIX)
        and parsed.port == SUPABASE_TRANSACTION_POOLER_PORT
    ):
        raise ValueError(
            f"{variable_name} uses Supabase's transaction pooler (port "
            f"{SUPABASE_TRANSACTION_POOLER_PORT}). That pooler does not support "
            "server-side prepared statements, which SQLAlchemy relies on, so "
            "queries fail intermittently. Use the session pooler on port 5432 "
            "instead -- see .env.example."
        )
