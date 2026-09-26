"""Embedding port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Turns text into the vectors that chunks are retrieved by."""

    @property
    def model_id(self) -> str:
        """Identifier of the model in use, stored alongside every vector.

        Written to each chunk so a later search can tell whether the stored
        vectors came from the model now answering.
        """
        ...

    async def embed(self, texts: Sequence[str]) -> Sequence[tuple[float, ...]]:
        """Embed a batch of texts, returning one vector per text in order.

        Raises:
            RetrievalUnavailableError: the service could not be reached, or
                refused the request.
            EmbeddingModelMismatchError: it answered with a different model than
                the one these vectors will be compared against.
        """
        ...
