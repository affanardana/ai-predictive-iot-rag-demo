"""Sentence embeddings for retrieval.

The encoder is a bi-encoder: it embeds a passage and a query independently, which
is what makes the search a vector comparison rather than a model call per stored
chunk. Reranking is the module next door.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from inference.settings import Settings

#: Width of `all-MiniLM-L6-v2`'s output, and therefore of the `vector(384)`
#: column the corpus is stored in.
EMBEDDING_DIMENSIONS = 384

#: Texts per forward pass. Small, because the box has one core and the batches
#: here are a document's chunks rather than a training set.
BATCH_SIZE = 16


class Embedder(Protocol):
    """What the app needs from an encoder, so tests can supply their own."""

    @property
    def model_id(self) -> str:
        """Identity of the loaded model, echoed to callers that store vectors."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text, in the order they were given."""
        ...


class SentenceEmbedder:
    """A sentence-transformer, loaded once at startup.

    The import lives in the constructor rather than at module scope: nothing
    else in this service needs `sentence_transformers`, and importing it costs
    seconds that only a deployment that embeds anything should pay.
    """

    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            settings.embed_model,
            revision=settings.embed_revision,
            cache_folder=str(settings.model_cache) if settings.model_cache else None,
            device="cpu",
        )
        self._model_id = settings.embed_model_id

    @property
    def model_id(self) -> str:
        """Identity of the loaded model, name and revision."""
        return self._model_id

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, in order.

        Vectors are normalized, so a dot product is the cosine similarity the
        corpus is ranked by, and the stored values are comparable across calls.

        Truncation is off, deliberately. The library truncates at the model's
        window by default and says nothing, which would embed a chunk without
        its tail -- so text longer than the window raises instead.
        """
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            processing_kwargs={"text": {"truncation": False}},
        )
        return [[float(value) for value in vector] for vector in vectors]
