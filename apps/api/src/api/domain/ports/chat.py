"""The Copilot's language model.

The Copilot is given evidence and asked to describe it. It is given no tools and
no way to fetch anything, so this port is deliberately narrow: messages in,
text out, streamed. A port that could call functions would put the choice of
evidence back inside the model, which is the decision
`domain/services/copilot_plan.py` exists to keep out of it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from api.domain.errors import DomainValidationError


class ChatRole(StrEnum):
    """Who said it."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn of the conversation handed to the model."""

    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        """Refuse an empty turn: it is a bug in the caller, not a message."""
        if not self.content.strip():
            raise DomainValidationError(f"A {self.role} message must not be blank.")


class ChatModel(Protocol):
    """Writes an answer from the evidence it is given."""

    @property
    def model_id(self) -> str:
        """Identity of the loaded model, carried into every answer.

        The reader is told which model wrote the prose. The *evidence* is
        unaffected by a model swap — it is attached by the system — and naming
        the model is what lets the two be judged separately.
        """
        ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield the answer's text as it is written.

        Chunks rather than one string because the answer takes tens of seconds
        on a one-core box, and a stream is the difference between a
        demonstration and a page that looks broken.

        Raises:
            ChatUnavailableError: the model service could not be reached, or
                refused the request.
        """
        ...
