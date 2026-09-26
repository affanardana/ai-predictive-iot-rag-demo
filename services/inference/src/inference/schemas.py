"""The wire contract.

Named fields rather than positional lists, because a list of six floats has no
way to say which is which and the failure mode is a silently transposed
temperature. The API has its own equivalent schema; they are duplicated
deliberately, because this is a service boundary and neither side should be able
to change the other's shape by accident.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ml.dataset.features import FEATURE_COLUMNS
from ml.dataset.windows import WINDOW_MINUTES


class Reading(BaseModel):
    """One minute of telemetry."""

    temperature: float
    vibration: float
    rpm: float
    current: float
    load: float
    voltage: float

    def as_row(self) -> list[float]:
        """Return the signals in the model's own column order."""
        return [float(getattr(self, name)) for name in FEATURE_COLUMNS]


class PredictRequest(BaseModel):
    """A window of recent telemetry for one machine."""

    #: Optional, and unused for scoring. The service reads a window of signals
    #: and nothing else; the identifier is carried for tracing a request back to
    #: a machine when one is available, which is why the API omits it.
    machine_id: str = Field(default="", max_length=16)
    readings: list[Reading]

    @field_validator("readings")
    @classmethod
    def _must_fill_a_window(cls, value: list[Reading]) -> list[Reading]:
        """Refuse a short window rather than padding it.

        Padding would invent readings, and scoring a short window would feed the
        model an input shape it was never trained on. Neither is recoverable
        downstream, so it is refused here where the caller can see why.
        """
        if len(value) < WINDOW_MINUTES:
            raise ValueError(
                f"a prediction needs {WINDOW_MINUTES} minutes of history, {len(value)} arrived"
            )
        return value


class PredictResponse(BaseModel):
    """What the model produced.

    A probability and the version that produced it. Deliberately no risk level:
    mapping probability to an application risk band is the product's decision
    (`PRD.md` section 9) and it belongs to the API's domain, not here.
    """

    failure_probability: float = Field(ge=0.0, le=1.0)
    model_version: str


class EmbedRequest(BaseModel):
    """Texts to embed, in the order the caller wants them back."""

    texts: list[str] = Field(default_factory=list)


class EmbedResponse(BaseModel):
    """The vectors, and the model that produced them.

    `model` is echoed rather than assumed: the caller stores it with every
    vector and refuses to compare vectors that came from a different one, so a
    deployment that swapped models is caught at the first ingest rather than
    showing up as poor ranking later.
    """

    model: str
    dimensions: int
    embeddings: list[list[float]]


class RerankRequest(BaseModel):
    """A query and the candidate passages to order against it."""

    query: str
    documents: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=100)


class RerankHit(BaseModel):
    """One passage's score, and its position in the request."""

    index: int = Field(ge=0)
    score: float


class RerankResponse(BaseModel):
    """The candidates that scored highest, best first."""

    results: list[RerankHit]
