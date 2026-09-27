"""One JSON object per line, so this service's logs match the API's.

**Deliberately a near-copy of `api/infrastructure/logging.py` and
`inference/logging.py`**, and the trigger for extracting a shared package is
written down here rather than left to taste: a fourth service, or the first time
one copy needs a fix the others do not. Until then, three images with three
`PYTHONPATH`s would each have to install a package whose entire content is a
formatter, and `docker compose logs` would still be three formats if the copies
drifted.

The redaction the API's copy carries is absent on purpose: that exists because
driver errors embed connection strings, and this service holds no DSN. It does
carry the broker credentials -- `PDM_INGEST_TOKEN` and the MQTT password -- so
nothing here logs a settings object, and the docstrings on `broker.py` and
`reporting.py` say so at the two places that could.
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
    """Install the JSON formatter on the root logger, uvicorn's included.

    Idempotent, and called before `uvicorn.run(..., log_config=None)` -- passing
    `None` is what stops uvicorn installing its own configuration over this one,
    which it would otherwise do *after* this module is imported, leaving
    `uvicorn.access` and everything this service logs in two different formats.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Empty rather than absent: uvicorn's default configuration gives these
    # `propagate = False` and their own handlers, so clearing the root logger
    # alone leaves the access line and any request traceback formatted
    # differently from every line around them.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
