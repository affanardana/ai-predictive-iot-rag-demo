"""Persistence model for predictions."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from api.domain.value_objects.risk_level import RiskLevel
from api.infrastructure.persistence.sql.base import Base
from api.infrastructure.persistence.sql.models.checks import in_clause

PREDICTION_ID_MAX_LENGTH = 36
MODEL_VERSION_MAX_LENGTH = 64
RISK_LEVEL_MAX_LENGTH = 16


class PredictionModel(Base):
    """A persisted model prediction."""

    __tablename__ = "predictions"

    prediction_id: Mapped[str] = mapped_column(String(PREDICTION_ID_MAX_LENGTH), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(16),
        ForeignKey("machines.machine_id", ondelete="CASCADE"),
    )
    predicted_at: Mapped[datetime]
    failure_probability: Mapped[float]
    risk_level: Mapped[str] = mapped_column(String(RISK_LEVEL_MAX_LENGTH))
    horizon_seconds: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(MODEL_VERSION_MAX_LENGTH))

    __table_args__ = (
        CheckConstraint(
            "failure_probability >= 0.0 AND failure_probability <= 1.0",
            name="failure_probability_range",
        ),
        CheckConstraint(
            in_clause("risk_level", [level.value for level in RiskLevel]),
            name="risk_level_valid",
        ),
        CheckConstraint("horizon_seconds > 0", name="horizon_positive"),
        Index("ix_predictions_machine_id_predicted_at", "machine_id", "predicted_at"),
    )
