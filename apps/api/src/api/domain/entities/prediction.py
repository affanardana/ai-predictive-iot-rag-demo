"""Prediction entity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

from api.domain.timestamps import ensure_aware
from api.domain.value_objects.failure_probability import FailureProbability
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.risk_level import RiskLevel

#: How far ahead a prediction looks. Fixed at 60 minutes by the product
#: constraints; modelled as data so a second horizon is a configuration
#: change rather than a schema change.
DEFAULT_PREDICTION_HORIZON = timedelta(minutes=60)

#: How many consecutive readings a prediction needs. PRD section 25.9 fixes the
#: window at 60 minutes, the same as the horizon, so it is derived rather than
#: restated -- two independent sixties is one place for them to drift apart.
#:
#: The coincidence is a product decision, not a consequence: 25.8 and 25.9 fix
#: the horizon and the window separately, and nothing forces them to stay
#: equal. This lives in the domain rather than in the request schema because it
#: is a rule about the problem, and the API now reads stored telemetry to
#: satisfy it rather than validating a caller's payload against it.
PREDICTION_WINDOW_READINGS = int(DEFAULT_PREDICTION_HORIZON.total_seconds() // 60)


@dataclass(frozen=True, slots=True)
class Prediction:
    """A model output estimating near-term failure risk for one machine.

    `risk_level` is stored rather than recomputed on read. The mapping from
    probability to risk is configurable, so recomputing would silently relabel
    historical predictions whenever the thresholds changed; storing it keeps
    the record a faithful account of what the system said at the time.
    """

    machine_id: MachineId
    predicted_at: datetime
    probability: FailureProbability
    risk_level: RiskLevel
    model_version: str
    prediction_id: str
    horizon: timedelta = DEFAULT_PREDICTION_HORIZON

    def __post_init__(self) -> None:
        """Validate the timestamp, horizon, and model version."""
        ensure_aware(self.predicted_at, "predicted_at")
        if self.horizon <= timedelta(0):
            raise ValueError("Prediction horizon must be positive.")
        if not self.model_version.strip():
            raise ValueError("Prediction model_version must not be blank.")
        if not self.prediction_id.strip():
            raise ValueError("Prediction prediction_id must not be blank.")

    @classmethod
    def create(
        cls,
        machine_id: MachineId,
        predicted_at: datetime,
        probability: FailureProbability,
        risk_level: RiskLevel,
        model_version: str,
        horizon: timedelta = DEFAULT_PREDICTION_HORIZON,
    ) -> Prediction:
        """Build a prediction, assigning a fresh identifier."""
        return cls(
            machine_id=machine_id,
            predicted_at=predicted_at,
            probability=probability,
            risk_level=risk_level,
            model_version=model_version,
            prediction_id=str(uuid4()),
            horizon=horizon,
        )
