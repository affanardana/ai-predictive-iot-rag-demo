"""What gets logged when something goes wrong.

`PRD.md` section 24 lists six failures the system must let a human diagnose, and
a log line is the only place several of them are visible at all. Nothing in this
repository asserted *what* was logged until this file: the formatter had tests,
and the call sites had none -- which is the half that carries the information.

The assertions are about fields rather than prose. A message is for a person and
may be reworded; `error_type`, `cause` and `exception` are what a grep during an
incident actually uses.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.composition.container import Container
from api.domain.errors import ChatUnavailableError
from api.infrastructure.chat.http_chat import HttpChat
from api.presentation.app import create_app


def records_named(caplog: pytest.LogCaptureFixture, name: str) -> list[logging.LogRecord]:
    """Return the captured records with this message name."""
    return [record for record in caplog.records if record.getMessage() == name]


async def test_an_unexpected_error_is_logged_with_its_traceback(
    container: Container,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The catch-all path, which is the one an operator reads first.

    A 500 with no traceback is a support ticket; the traceback is the whole of
    what makes it diagnosable. Reached through a route that raises, because
    nothing in the application does -- which is the point of a catch-all.
    """
    app: FastAPI = create_app(container)

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("kaput")

    with caplog.at_level(logging.ERROR):
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/boom")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"

    logged = records_named(caplog, "http.unhandled_error")
    assert logged, "an unhandled error was returned without being logged"
    record = logged[0]
    assert record.exc_info is not None, "the traceback is the point of this line"
    assert getattr(record, "error_type", None) == "RuntimeError"


async def test_a_chat_failure_logs_the_cause_and_not_only_its_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PRD section 24's "LLM failures", which is this line and nothing else.

    `type(exc).__name__` alone says "ConnectError" without saying to what, from
    where, or why -- and that is what the four adapters logged before Phase 11.
    """

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    chat = HttpChat(
        base_url="http://inference:8001",
        client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)),
        model_id="test-model.gguf",
    )

    with caplog.at_level(logging.WARNING), pytest.raises(ChatUnavailableError):
        async for _ in chat.stream((), max_tokens=1):
            pass

    logged = records_named(caplog, "infrastructure.chat.call_failed")
    assert logged, "a chat outage was raised without being logged"
    record = logged[0]
    assert getattr(record, "error_type", None) == "ConnectError"
    assert "connection refused" in str(getattr(record, "cause", ""))
    assert record.exc_info is not None


async def test_a_rate_limited_request_is_logged_at_warning(
    container: Container,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Which budget was spent, and never the address that spent it.

    A rejected caller is an expected event on a route open to the internet, so
    it is a warning rather than an error -- and the address is left out on
    purpose: it is the limiter's key, and putting visitor addresses into a
    public demonstration's logs on a schedule is a habit worth not having.
    """
    app: FastAPI = create_app(container)

    @app.get("/budgeted")
    async def budgeted() -> dict[str, str]:
        return {"ok": "yes"}

    with caplog.at_level(logging.WARNING):
        async with AsyncClient(
            transport=ASGITransport(app=app, client=("172.17.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            for _ in range(200):
                response = await client.get("/budgeted", headers={"X-Forwarded-For": "203.0.113.9"})
                if response.status_code == 429:
                    break

    assert response.status_code == 429
    logged = records_named(caplog, "http.rate_limited")
    assert logged, "a refused caller was not logged"
    record = logged[0]
    assert getattr(record, "limit", None) == "global"
    assert "203.0.113.9" not in str(record.__dict__)
