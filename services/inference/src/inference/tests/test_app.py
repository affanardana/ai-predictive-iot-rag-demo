"""The HTTP contract, against a fake scorer.

Marked `torch` because importing the app pulls the scorer, which pulls torch.
The service is the torch service, so that is where its tests belong.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

torch = pytest.importorskip("torch")

from inference.app import create_app  # noqa: E402
from inference.reranker import RerankResult  # noqa: E402
from inference.scorer import InsufficientHistoryError, Prediction  # noqa: E402
from inference.tests.conftest import a_window  # noqa: E402

pytestmark = pytest.mark.torch

WINDOW = 60


class FakeScorer:
    """A scorer that returns what it is told, and records what it was asked."""

    def __init__(self, probability: float = 0.42, refuse: bool = False) -> None:
        self.probability = probability
        self.refuse = refuse
        self.seen: list[list[list[float]]] = []

    @property
    def model_version(self) -> str:
        return "run-test"

    @property
    def window(self) -> int:
        return WINDOW

    def score(self, readings):
        self.seen.append([list(row) for row in readings])
        if self.refuse:
            raise InsufficientHistoryError("not enough history")
        return Prediction(failure_probability=self.probability, model_version=self.model_version)


class FakeEmbedder:
    """An encoder whose vectors say which position the text arrived in."""

    def __init__(self, dimensions: int = 384, refuse: bool = False) -> None:
        self.dimensions = dimensions
        self.refuse = refuse
        self.seen: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return "fake-encoder@test"

    def embed(self, texts):
        self.seen.append(list(texts))
        if self.refuse:
            raise RuntimeError("input is longer than the model can read")
        return [[float(index)] + [0.0] * (self.dimensions - 1) for index, _ in enumerate(texts)]


class FakeReranker:
    """A reranker that scores by position, so ordering is visible."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, list[str]]] = []

    @property
    def model_id(self) -> str:
        return "fake-cross-encoder@test"

    def rank(self, query, documents, limit):
        self.seen.append((query, list(documents)))
        # Reversed, so a test can tell the reranker's order from the input's.
        ranked = list(reversed(range(len(documents))))
        return [RerankResult(index=index, score=float(index)) for index in ranked[:limit]]


def an_app(
    scorer: FakeScorer | None = None,
    embedder: FakeEmbedder | None = None,
    reranker: FakeReranker | None = None,
) -> TestClient:
    """Build a client over an app with every model stubbed.

    All three must be injected: any left out is loaded from disk at startup,
    which in a test means downloading weights. A `TestClient` rather than the
    app, because entering it is what runs the lifespan that loads them.
    """
    return TestClient(
        create_app(
            scorer or FakeScorer(),
            embedder or FakeEmbedder(),
            reranker or FakeReranker(),
        )
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A client whose lifespan has run.

    Entering the context is what runs startup, and startup is what puts the
    models on `app.state`. Returning a bare `TestClient` gives a working-looking
    client that 500s on every request.
    """
    with an_app() as running:
        yield running


def test_health_reports_the_loaded_models() -> None:
    with an_app() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "run-test",
        "window": WINDOW,
        "embed_model": "fake-encoder@test",
        "rerank_model": "fake-cross-encoder@test",
    }


def test_embedding_returns_one_vector_per_text_in_order(client: TestClient) -> None:
    """Order is the contract: the caller maps vectors back onto its own chunks.

    The fake's vectors encode the position they were asked for, so a service
    that reordered a batch would be caught rather than merely suspected.
    """
    response = client.post("/embed", json={"texts": ["first", "second", "third"]})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "fake-encoder@test"
    assert body["dimensions"] == 384
    assert [vector[0] for vector in body["embeddings"]] == [0.0, 1.0, 2.0]


def test_embedding_an_empty_batch_is_an_empty_answer(client: TestClient) -> None:
    """The caller decides what an empty document means, not the service."""
    response = client.post("/embed", json={"texts": []})

    assert response.status_code == 200
    assert response.json()["embeddings"] == []


def test_text_longer_than_the_model_can_read_is_refused() -> None:
    """422 rather than a vector embedded without its tail.

    The library truncates silently by default, which would store a chunk that
    cannot be found by the words at its end -- so truncation is off and the
    failure surfaces here.
    """
    with an_app(embedder=FakeEmbedder(refuse=True)) as client:
        response = client.post("/embed", json={"texts": ["x" * 4000]})

    assert response.status_code == 422
    assert "longer" in response.text


def test_the_embedding_model_is_echoed_for_the_caller_to_store(client: TestClient) -> None:
    """The API refuses to compare vectors from two models, and needs the name."""
    body = client.post("/embed", json={"texts": ["anything"]}).json()

    assert body["model"]


def test_reranking_returns_the_candidates_in_its_own_order(client: TestClient) -> None:
    """The reranker's order is what the caller uses, indices included.

    Indices rather than the passages themselves: the caller maps them back onto
    its own candidates, so nothing here needs to know what a chunk is.
    """
    response = client.post(
        "/rerank",
        json={"query": "vibration is rising", "documents": ["a", "b", "c"], "limit": 2},
    )

    assert response.status_code == 200
    assert response.json()["results"] == [{"index": 2, "score": 2.0}, {"index": 1, "score": 1.0}]


def test_reranking_no_candidates_is_an_empty_answer(client: TestClient) -> None:
    """Nothing to order is not an error."""
    response = client.post("/rerank", json={"query": "anything", "documents": []})

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_reranking_refuses_a_limit_below_one(client: TestClient) -> None:
    """A limit of zero would silently return nothing, which reads as no match."""
    response = client.post("/rerank", json={"query": "anything", "documents": ["a"], "limit": 0})

    assert response.status_code == 422


def test_a_full_window_is_scored(client: TestClient) -> None:
    response = client.post("/predict", json={"machine_id": "M003", "readings": a_window()})

    assert response.status_code == 200
    assert response.json() == {"failure_probability": 0.42, "model_version": "run-test"}


def test_the_response_carries_no_risk_level(client: TestClient) -> None:
    """The mapping is the product's, and it must not drift into the model.

    `PRD.md` section 9 puts the bands in the application, and the API applies
    them. A risk level arriving from here would be the same decision made twice,
    in two places that could disagree.
    """
    body = client.post("/predict", json={"machine_id": "M003", "readings": a_window()}).json()

    assert set(body) == {"failure_probability", "model_version"}
    assert "risk_level" not in body


def test_a_short_window_is_refused_before_the_model_sees_it() -> None:
    """422 at the schema, naming how many readings arrived."""
    with an_app() as client:
        response = client.post(
            "/predict", json={"machine_id": "M003", "readings": a_window(count=59)}
        )

    assert response.status_code == 422
    assert "60" in response.text
    assert "59" in response.text


def test_a_scorer_refusal_surfaces_as_422() -> None:
    """Defence in depth: the schema checks the count, the scorer checks again."""
    with an_app(FakeScorer(refuse=True)) as client:
        response = client.post("/predict", json={"machine_id": "M003", "readings": a_window()})

    assert response.status_code == 422
    assert "history" in response.text


def test_readings_arrive_in_the_models_column_order() -> None:
    """A transposed temperature would be silent and would poison every score."""
    from ml.dataset.features import FEATURE_COLUMNS

    scorer = FakeScorer()
    with an_app(scorer) as client:
        client.post(
            "/predict",
            json={
                "machine_id": "M003",
                "readings": [
                    {name: float(index) for index, name in enumerate(FEATURE_COLUMNS)}
                    for _ in range(WINDOW)
                ],
            },
        )

    assert scorer.seen[0][0] == [float(index) for index in range(len(FEATURE_COLUMNS))]


def test_more_than_a_window_is_passed_through_untruncated() -> None:
    """The caller may send more history than needed.

    The app does not trim it — `Scorer.score` takes the most recent hour — so a
    fake sees all of it. Asserting the trim here would be asserting it twice, in
    the wrong place.
    """
    scorer = FakeScorer()
    with an_app(scorer) as client:
        response = client.post(
            "/predict", json={"machine_id": "M003", "readings": a_window(count=90)}
        )

    assert response.status_code == 200
    assert len(scorer.seen[0]) == 90


def test_a_machine_id_is_optional() -> None:
    """The service scores readings; it does not need to know whose they are.

    The API omits the field — its port takes readings, not an identifier — so
    requiring it here made every real call fail. An end-to-end test caught it;
    nothing that faked one side of the boundary could have.
    """
    with an_app() as client:
        response = client.post("/predict", json={"readings": a_window()})

    assert response.status_code == 200


def test_an_over_long_machine_id_is_still_refused() -> None:
    """Optional is not unbounded: the column it would be stored in has a width."""
    with an_app() as client:
        response = client.post("/predict", json={"machine_id": "M" * 17, "readings": a_window()})

    assert response.status_code == 422
