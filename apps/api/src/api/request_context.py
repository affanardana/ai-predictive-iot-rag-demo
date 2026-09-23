"""Request correlation context.

The request identifier is written by the presentation layer's middleware and
read by the infrastructure layer's log formatter. Both need it, neither owns it,
and neither may import the other — so it lives here, as a leaf module that
depends on nothing else inside `api`.

This is the only genuinely shared mutable state in the system. It is a
`ContextVar` rather than a module global so that concurrent requests cannot see
each other's identifier, and it is deliberately tiny: anything larger belongs in
a layer.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

#: Header carrying the correlation id in both directions.
REQUEST_ID_HEADER = "x-request-id"

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str) -> Token[str | None]:
    """Bind `request_id` to the current execution context.

    Returns a token that must be passed to `reset_request_id` when the request
    finishes, so the binding does not leak into whatever runs next on the same
    task.
    """
    return _request_id_var.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the binding that was in effect before `set_request_id`."""
    _request_id_var.reset(token)


def current_request_id() -> str | None:
    """Return the identifier bound to the current context, if any.

    Returns `None` outside a request, which is the normal case for startup,
    shutdown, and batch work.
    """
    return _request_id_var.get()
