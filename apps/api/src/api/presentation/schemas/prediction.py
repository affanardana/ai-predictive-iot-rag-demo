"""Prediction response schema."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from api.domain.value_objects.risk_level import RiskLevel


class PredictionSchema(BaseModel):
    """A model prediction, as stored at the time it was made."""

    model_config = ConfigDict(frozen=True)

    prediction_id: str
    machine_id: str
    predicted_at: datetime
    failure_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Model output, not a certainty. See the API's safety notice.",
    )
    risk_level: RiskLevel = Field(
        description="Application band derived from the probability at prediction time.",
    )
    horizon_seconds: int = Field(description="How far ahead the prediction looks.")
    model_version: str
