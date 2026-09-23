"""Request-scoped middleware."""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.request_context import REQUEST_ID_HEADER, reset_request_id, set_request_id

logger = logging.getLogger(__name__)


class RequestContextMiddleware:
    """Assigns a request id and emits one structured access log per request.

    Written against the raw ASGI interface rather than `BaseHTTPMiddleware`,
    which buffers responses and interferes with streaming. The realtime endpoint
    planned for a later phase needs to stream, so that door is left open now.

    An inbound `x-request-id` is honoured so a trace can span the client, this
    API, and any future upstream caller; otherwise one is generated.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI call."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _inbound_request_id(scope) or str(uuid4())
        token = set_request_id(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            logger.info(
                "http.request",
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            reset_request_id(token)


def _inbound_request_id(scope: Scope) -> str | None:
    """Return the caller-supplied request id, if any."""
    return Headers(scope=scope).get(REQUEST_ID_HEADER)
