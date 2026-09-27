"""The chat adapter, against a mock transport.

What is checked is the streaming translation — newline-delimited JSON in, an
async iterator of text out — and the failures. A stub `ChatModel` cannot
exercise either, which is the whole reason this file exists.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import httpx
import pytest

from api.domain.errors import ChatUnavailableError
from api.domain.ports.chat import ChatMessage, ChatRole
from api.infrastructure.chat import HttpChat

MODEL = "qwen2.5-1.5b-instruct-q4_k_m.gguf"


def a_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def a_chat(client: httpx.AsyncClient) -> HttpChat:
    return HttpChat(base_url="http://model", client=client, model_id=MODEL)


def lines(*pieces: str) -> bytes:
    """Render a newline-delimited response body."""
    return "".join(json.dumps({"text": piece}) + "\n" for piece in pieces).encode()


def asking() -> Sequence[ChatMessage]:
    """One question, as the use case would send it."""
    return (
        ChatMessage(role=ChatRole.SYSTEM, content="Use only these findings."),
        ChatMessage(role=ChatRole.USER, content="What should I inspect?"),
    )


async def collect(chat: HttpChat, *, max_tokens: int = 160) -> list[str]:
    """Read a whole answer."""
    return [piece async for piece in chat.stream(asking(), max_tokens=max_tokens)]


async def test_an_answer_arrives_piece_by_piece() -> None:
    """Each line is one piece of the answer, in order."""
    chat = a_chat(
        a_client(lambda request: httpx.Response(200, content=lines("Vibration ", "is ", "rising.")))
    )

    assert await collect(chat) == ["Vibration ", "is ", "rising."]


async def test_the_request_carries_the_turns_and_the_bound() -> None:
    """Roles and contents go as the service's schema declares them."""
    seen: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=lines("ok"))

    await collect(a_chat(a_client(capture)), max_tokens=120)

    assert seen == [
        {
            "messages": [
                {"role": "system", "content": "Use only these findings."},
                {"role": "user", "content": "What should I inspect?"},
            ],
            "max_tokens": 120,
        }
    ]


async def test_blank_lines_are_ignored() -> None:
    """A stream may carry separators that are not pieces of an answer."""
    chat = a_chat(a_client(lambda request: httpx.Response(200, content=b'{"text": "hi"}\n\n')))

    assert await collect(chat) == ["hi"]


async def test_a_refusal_names_the_services_own_reason() -> None:
    """The 503 for an unconfigured model carries the variable to set."""

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, json={"detail": "No chat model is configured. Set INFERENCE_CHAT_MODEL."}
        )

    with pytest.raises(ChatUnavailableError, match="INFERENCE_CHAT_MODEL"):
        await collect(a_chat(a_client(refuse)))


async def test_an_unreachable_service_is_a_chat_outage() -> None:
    """A refused connection is the same class of failure as a refusal."""

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ChatUnavailableError):
        await collect(a_chat(a_client(fail)))


async def test_a_malformed_line_is_refused_rather_than_skipped() -> None:
    """A gap in the middle of an answer is worse than an outage that says so."""
    body = json.dumps({"text": "Vibration"}).encode() + b"\nnot json at all\n"

    chat = a_chat(a_client(lambda request: httpx.Response(200, content=body)))

    with pytest.raises(ChatUnavailableError, match="not a piece of an answer"):
        await collect(chat)


async def test_a_line_without_text_is_refused() -> None:
    """The field is the contract; a body without it is not an answer."""
    chat = a_chat(a_client(lambda request: httpx.Response(200, content=b'{"delta": "x"}\n')))

    with pytest.raises(ChatUnavailableError):
        await collect(chat)


async def test_the_model_identity_is_reported_for_provenance() -> None:
    """Every answer names the model that wrote it."""
    assert a_chat(a_client(lambda request: httpx.Response(200))).model_id == MODEL
