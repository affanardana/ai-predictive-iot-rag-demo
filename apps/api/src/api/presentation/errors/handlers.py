"""FastAPI exception handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.domain.errors import DomainError
from api.presentation.errors.authentication import AuthenticationError
from api.presentation.errors.error_catalog import resolve
from api.presentation.errors.rate_limit import RateLimitError
from api.presentation.schemas.errors import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)

#: Statuses at or above this are the server's fault and worth a stack trace.
_SERVER_ERROR_THRESHOLD = 500

#: Where the client's wait is published. Named here because two places have to
#: agree on it: this handler sets it, and the CORS configuration has to *expose*
#: it, or a cross-origin browser cannot read it -- the same silent gap the
#: request-id header needed exposing for.
RETRY_AFTER_HEADER = "Retry-After"


def register_error_handlers(app: FastAPI) -> None:
    """Attach the error handlers to `app`."""
    app.add_exception_handler(DomainError, _handle_domain_error)
    app.add_exception_handler(AuthenticationError, _handle_authentication_error)
    app.add_exception_handler(RateLimitError, _handle_rate_limit_exceeded)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(Exception, _handle_unexpected_error)


def _envelope(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Build a JSON error response from the shared envelope."""
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details))
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=headers,
    )


async def _handle_domain_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate a domain failure into its mapped HTTP response."""
    error = cast(DomainError, exc)
    mapping = resolve(error)

    if mapping.status_code >= _SERVER_ERROR_THRESHOLD:
        logger.error(
            "http.domain_error",
            extra={"error_code": mapping.code, "error_type": type(error).__name__},
        )

    return _envelope(mapping.status_code, mapping.code, str(error))


async def _handle_authentication_error(request: Request, exc: Exception) -> JSONResponse:
    """Render a missing or wrong credential as a 401 in the shared envelope.

    Deliberately indistinguishable from one another in the response, and logged
    at warning rather than error: a rejected token is an expected event on an
    endpoint exposed to the internet, not a fault in this process.
    """
    logger.warning("http.authentication_failed", extra={"path": request.url.path})
    return _envelope(401, "unauthorized", "A valid ingest token is required.")


def rate_limited_response(limited: RateLimitError, *, path: str) -> JSONResponse:
    """Render an exhausted budget as a 429 carrying its own countdown.

    A function rather than only a handler, because there are two places a rate
    limit is enforced and they sit on opposite sides of Starlette's
    `ExceptionMiddleware`. The per-route budgets are dependencies, inside it, and
    their error reaches the handler. The global ceiling is a middleware, outside
    it, where a raised exception never reaches any registered handler and would
    surface as a `500 internal_error` -- so it renders its own response through
    this function. One implementation, so the two cannot drift into answering
    differently.

    Logged at warning, like a rejected token: a caller that asks too often is an
    expected event on an endpoint open to the internet, not a fault here. The
    address is deliberately **not** logged -- it is the key the limiter uses, and
    putting visitor addresses into a public demonstration's logs on a schedule
    is a worse habit than the request it would help diagnose.
    """
    logger.warning(
        "http.rate_limited",
        extra={
            "path": path,
            "limit": limited.limit,
            "retry_after_seconds": limited.retry_after_seconds,
        },
    )
    return _envelope(
        429,
        "rate_limited",
        str(limited),
        {"limit": limited.limit, "retry_after_seconds": str(limited.retry_after_seconds)},
        # Delta-seconds rather than an HTTP date: it is the form browsers, and
        # therefore the dashboard, understand without parsing a clock.
        headers={RETRY_AFTER_HEADER: str(limited.retry_after_seconds)},
    )


async def _handle_rate_limit_exceeded(request: Request, exc: Exception) -> JSONResponse:
    """Render a rate limit raised from inside the exception middleware."""
    return rate_limited_response(cast(RateLimitError, exc), path=request.url.path)


async def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate a request validation failure into a 422."""
    validation_error = cast(RequestValidationError, exc)
    details = {
        ".".join(str(part) for part in error["loc"]): error["msg"]
        for error in validation_error.errors()
    }
    return _envelope(
        422,
        "request_validation_failed",
        "The request could not be validated.",
        details,
    )


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for anything not classified above.

    Returns a deliberately vague message. An unexpected exception may carry a
    connection string or internal detail, so the specifics are logged (through
    the redacting formatter) rather than returned.
    """
    logger.exception(
        "http.unhandled_error",
        extra={"error_type": type(exc).__name__},
    )
    return _envelope(500, "internal_error", "An unexpected error occurred.")
