"""Structured logging and secret redaction."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, cast

import pytest

from api.infrastructure.logging import (
    REDACTED,
    JsonFormatter,
    configure_logging,
    is_sensitive_key,
    redact_mapping,
    redact_text,
)
from api.request_context import reset_request_id, set_request_id

DATABASE_URL = (
    "postgresql+psycopg://postgres:supersecret@aws-0-region.pooler.supabase.com:5432/postgres"
)


def _format(record: logging.LogRecord) -> dict[str, Any]:
    """Render a record through the JSON formatter and parse it back.

    Typed with `Any` values because parsed JSON is genuinely dynamic; each
    assertion below narrows what it needs.
    """
    return cast("dict[str, Any]", json.loads(JsonFormatter().format(record)))


def _record(message: str, **extra: object) -> logging.LogRecord:
    """Build a log record with the given extras."""
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_emits_a_single_json_object() -> None:
    """Output is machine-readable, one object per line."""
    payload = _format(_record("something happened"))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["message"] == "something happened"
    assert "timestamp" in payload


def test_includes_structured_extras() -> None:
    """Values passed via `extra` are preserved as context."""
    payload = _format(_record("request", method="GET", status_code=200))

    assert payload["context"] == {"method": "GET", "status_code": 200}


@pytest.mark.parametrize(
    "message",
    [
        f"connecting to {DATABASE_URL}",
        "failed: postgresql://user:pass@host:5432/db",
        "redis://default:token@cache:6379/0 unavailable",
    ],
)
def test_redacts_connection_strings_in_messages(message: str) -> None:
    """A connection string in a message never reaches the log.

    Driver exceptions routinely embed the DSN, which contains the password, so
    redaction happens in the formatter rather than relying on call sites.
    """
    payload = _format(_record(message))

    assert REDACTED in payload["message"]
    assert "supersecret" not in json.dumps(payload)
    assert "pass@host" not in json.dumps(payload)


@pytest.mark.parametrize(
    "key",
    ["password", "db_password", "API_KEY", "auth_token", "database_url", "authorization"],
)
def test_recognises_sensitive_keys(key: str) -> None:
    """Sensitive-looking attribute names are detected."""
    assert is_sensitive_key(key)


@pytest.mark.parametrize("key", ["machine_id", "duration_ms", "status_code"])
def test_leaves_ordinary_keys_alone(key: str) -> None:
    """Ordinary context keys are not mangled."""
    assert not is_sensitive_key(key)


def test_redacts_sensitive_extras_wholesale() -> None:
    """A sensitive key is replaced entirely, whatever its value."""
    redacted = redact_mapping({"database_url": DATABASE_URL, "machine_id": "M003"})

    assert redacted["database_url"] == REDACTED
    assert redacted["machine_id"] == "M003"


def test_redacts_connection_strings_inside_ordinary_extras() -> None:
    """A DSN hiding inside a harmless key is still caught."""
    detail = redact_mapping({"detail": f"could not reach {DATABASE_URL}"})["detail"]

    # `redact_mapping` is typed over `object` because its values genuinely are
    # arbitrary, so the narrowing belongs at the assertion rather than in the
    # signature.
    assert isinstance(detail, str)
    assert REDACTED in detail
    assert "supersecret" not in detail


def test_redacts_exception_text() -> None:
    """A traceback's own text is redacted before being serialised."""
    try:
        raise RuntimeError(f"connection failed for {DATABASE_URL}")
    except RuntimeError:
        record = _record("boom", exc_info=sys.exc_info())

    payload = _format(record)

    assert REDACTED in payload["exception"]
    assert "supersecret" not in json.dumps(payload)


def test_includes_the_request_id_when_set() -> None:
    """Log lines emitted during a request carry its correlation id."""
    token = set_request_id("trace-123")
    try:
        payload = _format(_record("handled"))
    finally:
        reset_request_id(token)

    assert payload["request_id"] == "trace-123"


def test_omits_the_request_id_outside_a_request() -> None:
    """Lines logged outside a request are not tagged with a stale id."""
    assert "request_id" not in _format(_record("startup"))


def test_configure_logging_is_idempotent() -> None:
    """Calling it twice replaces handlers rather than doubling every line."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level

    try:
        configure_logging("INFO")
        configure_logging("WARNING")

        assert len(root.handlers) == 1
        assert root.level == logging.WARNING
    finally:
        root.handlers.clear()
        root.handlers.extend(original_handlers)
        root.setLevel(original_level)


def test_redact_text_leaves_plain_text_untouched() -> None:
    """Redaction does not alter text with nothing to hide."""
    assert redact_text("machine M003 vibration 2.3") == "machine M003 vibration 2.3"
