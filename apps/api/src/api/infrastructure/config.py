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
