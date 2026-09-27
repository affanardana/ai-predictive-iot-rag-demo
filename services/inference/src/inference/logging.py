"""One JSON object per line, so this service's logs are readable beside the API's.

**Deliberately a near-copy of `api/infrastructure/logging.py` rather than a
shared package**, and the reason is worth stating because the duplication is
real. Three services ship as three images with three `PYTHONPATH`s; the API
image does not contain `inference` and the inference image does not contain
`api`. Sharing thirty lines would mean a fourth package that both images install
to hold one formatter -- a module whose only purpose is to avoid a copy, and a
new thing to keep in step when the images change. The copy is the smaller lie.

What is *not* copied is the redaction. That exists in the API because driver
errors embed connection strings and the API is the only service holding a DSN.
This one speaks HTTP to nobody and holds no credentials, so a redaction pass
here would be a ritual rather than a defence. If that ever changes -- if this
service is given a database, or a hosted model key -- the redaction comes with
it, and this paragraph is the note to whoever adds it.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

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


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the record serialised as JSON."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_ATTRIBUTES and not key.startswith("_")
        }
        if extras:
            payload["context"] = extras

        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger.

    Idempotent: replaces handlers rather than appending, so calling it from both
    the app factory and a script does not double every line.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
