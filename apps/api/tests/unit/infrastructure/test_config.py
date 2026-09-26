"""Configuration loading and validation.

The suite's autouse fixture pins `DATABASE_URL`, `APP_ENV`, and `LOG_LEVEL` in
the environment, and environment variables outrank `.env`. Tests here therefore
exercise the real loading path rather than a bypassed one, and a developer's
local `.env` cannot change the outcome.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.domain.value_objects.risk_level import RiskLevel
from api.infrastructure.config import SUPABASE_TRANSACTION_POOLER_PORT, Settings

#: A session-pooler URL, the supported Supabase connection mode.
SESSION_POOLER_URL = (
    "postgresql+psycopg://postgres.abcdefghijklmno:secret"
    "@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres?sslmode=require"
)

#: The transaction pooler, which breaks ORMs relying on prepared statements.
TRANSACTION_POOLER_URL = (
    "postgresql+psycopg://postgres.abcdefghijklmno:secret"
    "@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres?sslmode=require"
)


def test_applies_documented_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional settings take their documented defaults.

    Constructed with `.env` excluded and the relevant variables cleared, so this
    asserts the defaults declared in code rather than whatever the developer's
    local `.env` happens to contain. Before `.env` was excluded it passed by
    coincidence, which is not the same thing as passing.

    `TEST_DATABASE_URL` is cleared here rather than by the suite fixture, which
    deliberately leaves it alone -- clearing it globally would disable the whole
    `postgres` tier.
    """
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)

    # database_url is required, so it has to be supplied for anything to validate.
    settings = Settings(_env_file=None, database_url=SESSION_POOLER_URL)

    assert settings.app_env == "local"
    assert settings.log_level == "INFO"
    assert settings.test_database_url is None
    assert not settings.is_production


def test_database_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting the database URL fails rather than defaulting somewhere.

    A default would let the application connect to an unexpected database
    instead of refusing to start. `.env` is excluded so this exercises the code
    rather than the presence of a local file -- otherwise the test silently
    stops testing anything the moment someone creates one.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None)

    assert "database_url" in str(caught.value)


def test_accepts_the_supabase_session_pooler() -> None:
    """The session pooler is the supported connection mode."""
    assert Settings(database_url=SESSION_POOLER_URL).database_url == SESSION_POOLER_URL


def test_rejects_the_supabase_transaction_pooler() -> None:
    """Port 6543 is refused with an explanatory message.

    Supabase documents that the transaction pooler does not support server-side
    prepared statements, which SQLAlchemy relies on. Left unchecked, the symptom
    is intermittent driver errors that say nothing about the port, so the
    misconfiguration is caught at startup instead.
    """
    with pytest.raises(ValidationError) as caught:
        Settings(database_url=TRANSACTION_POOLER_URL)

    message = str(caught.value)
    assert str(SUPABASE_TRANSACTION_POOLER_PORT) in message
    assert "session pooler" in message
    assert "5432" in message


def test_rejects_the_transaction_pooler_in_the_test_url_too() -> None:
    """The same guard applies to the test database."""
    with pytest.raises(ValidationError):
        Settings(
            database_url=SESSION_POOLER_URL,
            test_database_url=TRANSACTION_POOLER_URL,
        )


def test_accepts_a_non_supabase_database_on_any_port() -> None:
    """The guard is specific to Supabase's pooler host.

    A local or CI PostgreSQL on 6543 is none of this check's business, so the
    rule is scoped by hostname rather than by port alone.
    """
    url = "postgresql+psycopg://user:pass@localhost:6543/app"

    assert Settings(database_url=url).database_url.endswith(":6543/app")


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://user:pass@host:5432/db",
        "postgres://user:pass@host:5432/db",
    ],
)
def test_rejects_a_driverless_postgres_url(url: str) -> None:
    """A bare PostgreSQL scheme is refused, with the fix in the message.

    SQLAlchemy resolves it to psycopg2, which this project does not install, so
    the failure would otherwise surface as "No module named 'psycopg2'" -- an
    error pointing at a missing package rather than the URL that caused it.
    Supabase hands out URLs in exactly this form, so it is the obvious thing to
    paste.
    """
    with pytest.raises(ValidationError) as caught:
        Settings(database_url=url)

    assert "postgresql+psycopg" in str(caught.value)


def test_rejects_a_driverless_url_in_the_test_variable_too() -> None:
    """The same guard applies to the test database."""
    with pytest.raises(ValidationError):
        Settings(
            database_url=SESSION_POOLER_URL,
            test_database_url="postgresql://user:pass@host:5432/db",
        )


@pytest.mark.parametrize(
    "url",
    [SESSION_POOLER_URL, "postgresql+asyncpg://user:pass@host:5432/db"],
)
def test_accepts_a_url_that_names_its_driver(url: str) -> None:
    """An explicit driver passes the scheme check untouched.

    `postgresql+psycopg://` does not start with `postgresql://`, which is why
    this is a scheme check rather than a substring search -- a substring check
    would reject the correct URL as well as the wrong one.
    """
    assert Settings(database_url=url).database_url == url


def test_reads_values_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables override defaults."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    # Required alongside APP_ENV=production; see the ingest-token tests below.
    monkeypatch.setenv("INGEST_API_TOKEN", "s3cret")

    settings = Settings()

    assert settings.app_env == "production"
    assert settings.is_production
    assert settings.log_level == "DEBUG", "log level should be normalised to uppercase"


def test_requires_an_ingest_token_outside_local() -> None:
    """A deployment without a token refuses to start.

    The write endpoints are the only way telemetry and machines enter the
    database and are reachable from the internet, so starting without a
    credential would serve them to anyone.
    """
    with pytest.raises(ValidationError, match="INGEST_API_TOKEN"):
        Settings(app_env="production")

    with pytest.raises(ValidationError, match="INGEST_API_TOKEN"):
        Settings(app_env="ci")


def test_local_needs_no_ingest_token() -> None:
    """Local development and the offline test tier run unauthenticated.

    Gating on anything broader would require every developer and every CI job
    to hold a shared secret to run a test suite that never leaves the machine.
    """
    assert Settings(app_env="local").ingest_api_token is None


def test_the_ingest_token_defaults_to_unset() -> None:
    """A missing token is absent rather than empty.

    The dependency distinguishes `None` (nothing to check) from a configured
    value, so an empty string would silently become a required credential that
    no caller could ever satisfy.
    """
    assert Settings().ingest_api_token is None


def test_rejects_an_unknown_log_level() -> None:
    """A typo in LOG_LEVEL fails at startup rather than being ignored."""
    with pytest.raises(ValidationError):
        Settings(log_level="verbose")


def test_rejects_an_unknown_environment() -> None:
    """Only the recognised environments are accepted."""
    with pytest.raises(ValidationError):
        Settings(app_env="staging")


def test_risk_thresholds_come_from_configuration() -> None:
    """The risk bands are configurable, as PRD section 9 requires."""
    settings = Settings(
        risk_warning_threshold=0.1,
        risk_high_threshold=0.5,
        risk_critical_threshold=0.9,
    )

    thresholds = settings.risk_thresholds

    assert thresholds.warning == 0.1
    assert thresholds.high == 0.5
    assert thresholds.critical == 0.9


def test_rejects_unordered_risk_thresholds() -> None:
    """Contradictory thresholds fail at startup.

    The check lives in the value object; this confirms the settings model
    actually constructs it rather than storing three unchecked floats.
    """
    with pytest.raises(ValidationError):
        Settings(
            risk_warning_threshold=0.9,
            risk_high_threshold=0.5,
            risk_critical_threshold=0.1,
        )


def test_default_thresholds_match_the_prd_table() -> None:
    """The shipped defaults are the PRD section 9 boundaries."""
    thresholds = Settings(_env_file=None, database_url=SESSION_POOLER_URL).risk_thresholds

    assert thresholds.warning == 0.30
    assert thresholds.high == 0.60
    assert thresholds.critical == 0.80


def test_every_risk_level_is_reachable() -> None:
    """The configured bands leave no level unreachable.

    Guards against a configuration that would make a band impossible, which
    would present as a classifier bug rather than a settings mistake.
    """
    thresholds = Settings(_env_file=None, database_url=SESSION_POOLER_URL).risk_thresholds

    assert 0.0 < thresholds.warning < thresholds.high < thresholds.critical <= 1.0
    assert len(list(RiskLevel)) == 4
