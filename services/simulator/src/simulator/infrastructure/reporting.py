"""Tells the API what a run is doing.

Uses `urllib.request` from the standard library rather than `httpx`, because this
service already installs two optional extras and a third HTTP client for one
POST would be a dependency bought for nothing. The API's own container
healthchecks use the same module.

**A reporting failure is logged, never raised.** A reporter runs inside the run
loop, and a demonstration that aborts because a status update failed would be a
worse failure than the one being reported: the run would stop producing
telemetry, which is the thing the dashboard is actually watching.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime

from simulator.domain.ports import RunView

logger = logging.getLogger(__name__)

#: Short, because this is a status ping on the run loop's critical path. A slow
#: API should cost a late update, not a stalled simulation.
DEFAULT_TIMEOUT_SECONDS = 5.0

#: The header the API's write endpoints require. The same secret n8n presents,
#: under the same name, set from the same variable in `compose.yaml` -- so a
#: mismatch is structurally impossible rather than a thing to remember.
#:
#: The linter reads `TOKEN` in the name and supposes a credential. The value is
#: a header *name*; the secret travels in the header, from the environment, and
#: is never written here.
INGEST_TOKEN_HEADER = "X-Ingest-Token"  # noqa: S105 - a header name, not a secret


def _default(value: object) -> str:
    """Serialise a datetime the way the API parses it."""
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Cannot serialise {type(value).__name__}.")


class HttpRunReporter:
    """Posts run state to the API."""

    def __init__(
        self,
        base_url: str,
        token: str | None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._endpoint = f"{base_url.rstrip('/')}/api/v1/simulations"
        self._token = token
        self._timeout = timeout

    def report(self, run: RunView) -> None:
        """Send one run's state, and swallow anything that goes wrong.

        The status is sent as a `PATCH` to the run's own URL, so the API needs no
        separate "which run is this" lookup and a report for an unknown run is a
        404 rather than a silently created row.
        """
        payload = json.dumps(asdict(run), default=_default).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - the URL is configuration, not input
            f"{self._endpoint}/{run.session_id}",
            data=payload,
            method="PATCH",
            headers={"Content-Type": "application/json", INGEST_TOKEN_HEADER: self._token or ""},
        )
        try:
            urllib.request.urlopen(request, timeout=self._timeout)  # noqa: S310 - as above
        except (urllib.error.URLError, OSError, ValueError) as error:
            # Deliberately broad and deliberately swallowed: the run must not
            # care. `stale` in the API's own view is what surfaces a reporter
            # that has stopped working.
            logger.warning("Could not report run %s: %s", run.session_id, error)

    def close(self) -> None:
        """Nothing to release; the connection is per request."""
