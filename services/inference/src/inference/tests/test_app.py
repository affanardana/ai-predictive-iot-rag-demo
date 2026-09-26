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


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A client whose lifespan has run.

    Entering the context is what runs startup, and startup is what puts the
    scorer on `app.state`. Returning a bare `TestClient` gives a working-looking
    client that 500s on every request.
    """
    with TestClient(create_app(FakeScorer())) as running:
        yield running


def test_health_reports_the_loaded_model() -> None:
    with TestClient(create_app(FakeScorer())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "run-test", "window": WINDOW}


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
    with TestClient(create_app(FakeScorer())) as client:
        response = client.post(
            "/predict", json={"machine_id": "M003", "readings": a_window(count=59)}
        )

    assert response.status_code == 422
    assert "60" in response.text
    assert "59" in response.text


def test_a_scorer_refusal_surfaces_as_422() -> None:
    """Defence in depth: the schema checks the count, the scorer checks again."""
    with TestClient(create_app(FakeScorer(refuse=True))) as client:
        response = client.post("/predict", json={"machine_id": "M003", "readings": a_window()})

    assert response.status_code == 422
    assert "history" in response.text


def test_readings_arrive_in_the_models_column_order() -> None:
    """A transposed temperature would be silent and would poison every score."""
    from ml.dataset.features import FEATURE_COLUMNS

    scorer = FakeScorer()
    with TestClient(create_app(scorer)) as client:
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
    with TestClient(create_app(scorer)) as client:
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
    with TestClient(create_app(FakeScorer())) as client:
        response = client.post("/predict", json={"readings": a_window()})

    assert response.status_code == 200


def test_an_over_long_machine_id_is_still_refused() -> None:
    """Optional is not unbounded: the column it would be stored in has a width."""
    with TestClient(create_app(FakeScorer())) as client:
        response = client.post("/predict", json={"machine_id": "M" * 17, "readings": a_window()})

    assert response.status_code == 422
