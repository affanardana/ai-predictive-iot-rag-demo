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

    risk_warning_threshold: float = DEFAULT_WARNING_THRESHOLD
    risk_high_threshold: float = DEFAULT_HIGH_THRESHOLD
    risk_critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD

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

        if self.ingest_api_token is not None:
            # A blank secret is not a secret: `compare_digest("", "")` is true,
            # so an empty token would authenticate a request carrying no header
            # at all.
            if not self.ingest_api_token.strip():
                raise ValueError("INGEST_API_TOKEN must not be blank.")
            # HTTP header values are byte strings. A token outside ASCII could
            # never be presented by any client, so it would refuse every
            # request -- a misconfiguration best caught here rather than
            # diagnosed from a wall of 401s.
            if not self.ingest_api_token.isascii():
                raise ValueError(
                    "INGEST_API_TOKEN must be ASCII. Header values are byte "
                    "strings and cannot carry a non-ASCII character, so this "
                    "token could never be presented."
                )
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
