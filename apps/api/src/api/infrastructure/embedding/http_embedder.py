"""Talks to the inference service's embedding endpoint.

Every failure -- a refused connection, a timeout, a body that is not a batch of
vectors -- becomes `RetrievalUnavailableError`. A response carrying vectors from
a *different* model is refused separately, because that is not an outage: it is a
deployment that would rank the corpus by vectors that are not comparable to it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

from api.domain.entities.knowledge_document import EMBEDDING_DIMENSIONS
from api.domain.errors import EmbeddingModelMismatchError, RetrievalUnavailableError

logger = logging.getLogger(__name__)

#: Generous, because the first request after a cold start loads a model.
DEFAULT_TIMEOUT_SECONDS = 30.0


class HttpEmbedder:
    """An `Embedder` backed by the inference service."""

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
        """The model every stored vector is attributed to."""
        return self._model_id

    async def embed(self, texts: Sequence[str]) -> Sequence[tuple[float, ...]]:
        """Embed a batch of texts, one vector each, in order.

        Raises:
            RetrievalUnavailableError: the service could not be reached, or
                answered with something that is not one vector per text.
            EmbeddingModelMismatchError: it answered with a different model than
                the one this corpus was embedded with.
        """
        if not texts:
            return ()

        try:
            response = await self._client.post(
                f"{self._base_url}/embed",
                json={"texts": list(texts)},
                timeout=self._timeout,
            )
            response.raise_for_status()
            body = response.json()
            served = str(body["model"])
            vectors = [tuple(float(value) for value in vector) for vector in body["embeddings"]]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Embedding call failed: %s", type(exc).__name__)
            raise RetrievalUnavailableError(
                "The embedding service could not be reached or returned an unusable response."
            ) from exc

        # Outside the try, so a mismatch is not reported as an outage.
        if served != self._model_id:
            raise EmbeddingModelMismatchError(expected=self._model_id, actual=served)
        if len(vectors) != len(texts):
            raise RetrievalUnavailableError(
                f"The embedding service returned {len(vectors)} vectors for {len(texts)} texts, "
                "so they cannot be matched to the passages they belong to."
            )
        for vector in vectors:
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise RetrievalUnavailableError(
                    f"The embedding service returned a {len(vector)}-dimension vector, "
                    f"but the corpus is stored at {EMBEDDING_DIMENSIONS}."
                )
        return vectors
