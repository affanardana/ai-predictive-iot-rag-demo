"""Talks to the inference service's chat endpoint.

The answer arrives as newline-delimited JSON and leaves here as an async
iterator of text, which is the whole translation this adapter performs. It
deliberately does *not* translate newline-delimited JSON into server-sent
events: the presentation layer owns the browser's frame format, and a second
place that knows about `event:` lines would be a second place for them to
disagree.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence

import httpx

from api.domain.errors import ChatUnavailableError
from api.domain.ports.chat import ChatMessage

logger = logging.getLogger(__name__)

#: Long, because this call writes prose on one core: the measured answer takes
#: about twenty seconds, and a cold start adds the model load. A timeout shorter
#: than the work would turn a slow machine into an outage.
DEFAULT_TIMEOUT_SECONDS = 180.0


class HttpChat:
    """A `ChatModel` backed by the inference service."""

    def __init__(
        self,
        *,
        base_url: str,
        client: httpx.AsyncClient,
        model_id: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._model_id = model_id
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        """Identity of the model, as configured for this deployment.

        Configured rather than asked for, because it travels into every answer
        as provenance and a round trip to fetch it would fail exactly when the
        answer does. Compose names it explicitly so the two services agree.
        """
        return self._model_id

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield the answer as it is written.

        Raises:
            ChatUnavailableError: if the service could not be reached, refused
                the request, or sent something that is not a chunk of an answer.
        """
        payload = {
            "messages": [
                {"role": message.role.value, "content": message.content} for message in messages
            ],
            "max_tokens": max_tokens,
        }
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/chat",
                json=payload,
                timeout=self._timeout,
            ) as response:
                if response.status_code >= 400:
                    # The body has to be read before it can be reported: on a
                    # streaming response httpx has not consumed it yet.
                    await response.aread()
                    raise ChatUnavailableError(_message(response))
                async for line in response.aiter_lines():
                    text = _text_of(line)
                    if text:
                        yield text
        except httpx.HTTPError as exc:
            logger.warning("Chat call failed: %s", type(exc).__name__)
            raise ChatUnavailableError(
                "The model service could not be reached while writing an answer."
            ) from exc


def _text_of(line: str) -> str:
    """Return the text of one streamed line, or "" for a blank one.

    Raises:
        ChatUnavailableError: if a line is not the JSON object the service
            promises. Refused rather than skipped: a chunk that cannot be read
            is a gap in the middle of an answer, and an answer with a hole in it
            is worse than an outage that says so.
    """
    if not line.strip():
        return ""
    try:
        body = json.loads(line)
        return str(body["text"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ChatUnavailableError(
            "The model service returned something that is not a piece of an answer."
        ) from exc


def _message(response: httpx.Response) -> str:
    """Return the service's own explanation, when it sent one."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail) if detail else "The model service could not write an answer."
