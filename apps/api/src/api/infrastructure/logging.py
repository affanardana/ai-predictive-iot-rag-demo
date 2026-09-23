"""Structured logging with secret redaction.

Emits one JSON object per line so logs are machine-readable in the cloud.

Redaction is applied to both the message and any structured extra, because the
coding standards forbid logging credentials and driver errors routinely embed
connection strings. Redacting at the formatter means every call site is covered
without each one having to remember.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from api.request_context import current_request_id

REDACTED = "***REDACTED***"

#: Substrings that mark a log-record attribute as sensitive, matched
#: case-insensitively against the key name.
SENSITIVE_KEY_FRAGMENTS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "database_url",
    "connection_string",
    "dsn",
)

#: Connection strings appearing inside free text, e.g. inside an exception
#: message. Matches `scheme[+driver]://...` up to the next whitespace.
_DSN_IN_TEXT = re.compile(
    r"(?i)\b(?:postgres(?:ql)?|mysql|mariadb|redis|amqp|mongodb|mssql)"
    r"(?:\+\w+)?://\S+"
)

#: Attributes present on every `LogRecord`; anything else was passed via
#: `extra=` and belongs in the output.
_RESERVED_RECORD_ATTRIBUTES = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "stacklevel",
        "taskName",
        "thread",
        "threadName",
    }
)


def redact_text(value: str) -> str:
    """Replace any connection string embedded in `value`."""
    return _DSN_IN_TEXT.sub(REDACTED, value)


def is_sensitive_key(key: str) -> bool:
    """Whether an attribute name looks like it holds a secret."""
    lowered = key.lower()
    return any(fragment in lowered for fragment in SENSITIVE_KEY_FRAGMENTS)


def redact_mapping(values: Mapping[str, object]) -> dict[str, object]:
    """Redact sensitive keys and embedded connection strings in a mapping.

    Typed over `object` rather than `Any`: the values genuinely are arbitrary,
    and `object` says that without switching off type checking for everything
    that touches them.
    """
    redacted: dict[str, object] = {}
    for key, value in values.items():
        if is_sensitive_key(key):
            redacted[key] = REDACTED
        elif isinstance(value, str):
            redacted[key] = redact_text(value)
        else:
            redacted[key] = value
    return redacted


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the record serialised as JSON."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_text(record.getMessage()),
        }

        request_id = current_request_id()
        if request_id is not None:
            payload["request_id"] = request_id

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_ATTRIBUTES and not key.startswith("_")
        }
        if extras:
            payload["context"] = redact_mapping(extras)

        if record.exc_info is not None:
            payload["exception"] = redact_text(self.formatException(record.exc_info))

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger.

    Idempotent: replaces handlers rather than appending, so calling it twice
    (for example from both the app factory and a script) does not double every
    line.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
