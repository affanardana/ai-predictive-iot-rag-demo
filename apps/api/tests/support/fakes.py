"""Test doubles for domain ports.

Hand-written rather than produced by `unittest.mock`. A `Mock` accepts any
attribute and any call, so a renamed port method would let tests keep passing
while the real code fails at runtime; these implement the same Protocols and so
are type-checked against them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from api.domain.ports.events import MachineEvent
from api.domain.ports.health import HealthStatus
from api.domain.ports.reranker import RerankResult
from tests.support.factories import DEFAULT_NOW, make_embedding


class FixedClock:
    """A clock pinned to a chosen instant.

    Injecting time rather than patching `datetime` keeps time-dependent
    behaviour deterministic without a global monkeypatch, and makes `advance`
    read explicitly in the tests that need it.
    """

    def __init__(self, now: datetime | None = None) -> None:
        self._now = now or DEFAULT_NOW

    def now(self) -> datetime:
        """Return the pinned instant."""
        return self._now

    def advance(self, delta: timedelta) -> None:
        """Move the clock forward."""
        self._now += delta

    def set(self, moment: datetime) -> None:
        """Move the clock to a specific instant."""
        self._now = moment


class StubHealthProbe:
    """A health probe with a fixed verdict."""

    def __init__(self, healthy: bool = True, detail: str = "ok") -> None:
        self._healthy = healthy
        self._detail = detail
        self.check_count = 0

    async def check(self) -> HealthStatus:
        """Return the configured verdict, counting how often it was asked."""
        self.check_count += 1
        return HealthStatus(healthy=self._healthy, detail=self._detail)


class RecordingEventPublisher:
    """An event publisher that keeps what it was given.

    A list rather than a count, because the assertions that matter are about
    *which* changes were announced and in what order -- an ingest batch naming
    two machines, or a scoring that raises an incident and so publishes two
    events rather than one.
    """

    def __init__(self) -> None:
        self.events: list[MachineEvent] = []

    async def publish(self, event: MachineEvent) -> None:
        """Record the event."""
        self.events.append(event)

    @property
    def kinds(self) -> list[str]:
        """Return the recorded event kinds, for readable assertions."""
        return [event.kind.value for event in self.events]


class StubEmbedder:
    """An encoder that returns a chosen vector per text, and records its calls.

    It records *batches* rather than individual texts, because how many calls
    were made is what the ingest path is asserted on: one call per document,
    not one per chunk.
    """

    def __init__(
        self,
        vectors: Mapping[str, tuple[float, ...]] | None = None,
        default: tuple[float, ...] | None = None,
        model_id: str = "stub-encoder@test",
    ) -> None:
        self._vectors = dict(vectors or {})
        self._default = default or make_embedding(1.0)
        self._model_id = model_id
        self.batches: list[list[str]] = []

    @property
    def model_id(self) -> str:
        """Identity stored with every vector this produces."""
        return self._model_id

    async def embed(self, texts: Sequence[str]) -> Sequence[tuple[float, ...]]:
        """Return one vector per text, recording the batch."""
        self.batches.append(list(texts))
        return [self._vectors.get(text, self._default) for text in texts]

    @property
    def texts_seen(self) -> list[str]:
        """Every text this was asked to embed, across all calls."""
        return [text for batch in self.batches for text in batch]


class StubReranker:
    """A reranker that scores by a table, and records what it was asked to rank."""

    def __init__(
        self,
        scores: Mapping[str, float] | None = None,
        default: float = 0.0,
        model_id: str = "stub-reranker@test",
    ) -> None:
        self._scores = dict(scores or {})
        self._default = default
        self._model_id = model_id
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def model_id(self) -> str:
        """Identity of the model, reported by the health endpoint."""
        return self._model_id

    def _score(self, document: str) -> float:
        """Return the configured score for one passage."""
        return self._scores.get(document, self._default)

    async def rank(
        self,
        query: str,
        documents: Sequence[str],
        limit: int,
    ) -> Sequence[RerankResult]:
        """Score every passage, best first, at most `limit` of them."""
        self.calls.append((query, list(documents)))
        ranked = sorted(
            range(len(documents)), key=lambda index: (-self._score(documents[index]), index)
        )
        return [
            RerankResult(index=index, score=self._score(documents[index]))
            for index in ranked[:limit]
        ]

    @property
    def documents_ranked(self) -> list[str]:
        """Every passage this was asked to score, across all calls."""
        return [document for _, documents in self.calls for document in documents]
