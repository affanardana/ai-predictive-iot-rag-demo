"""Persistence model for incidents."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.incident_type import IncidentType
from api.domain.value_objects.risk_level import IncidentSeverity
from api.infrastructure.persistence.sql.base import Base
from api.infrastructure.persistence.sql.models.checks import in_clause

INCIDENT_ID_MAX_LENGTH = 36
SEVERITY_MAX_LENGTH = 16
STATUS_MAX_LENGTH = 16
INCIDENT_TYPE_MAX_LENGTH = 32


class IncidentModel(Base):
    """A maintenance incident."""

    __tablename__ = "incidents"

    incident_id: Mapped[str] = mapped_column(String(INCIDENT_ID_MAX_LENGTH), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(16),
        ForeignKey("machines.machine_id", ondelete="CASCADE"),
    )
    incident_type: Mapped[str] = mapped_column(String(INCIDENT_TYPE_MAX_LENGTH))
    severity: Mapped[str] = mapped_column(String(SEVERITY_MAX_LENGTH))
    failure_probability: Mapped[float]
    detected_at: Mapped[datetime]
    status: Mapped[str] = mapped_column(String(STATUS_MAX_LENGTH))
    prediction_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("predictions.prediction_id", ondelete="SET NULL"),
        default=None,
    )

    __table_args__ = (
        CheckConstraint(
            "failure_probability >= 0.0 AND failure_probability <= 1.0",
            name="failure_probability_range",
        ),
        CheckConstraint(
            in_clause("severity", [level.value for level in IncidentSeverity]),
            name="severity_valid",
        ),
        CheckConstraint(
            in_clause("status", [status.value for status in IncidentStatus]),
            name="status_valid",
        ),
        CheckConstraint(
            in_clause("incident_type", [kind.value for kind in IncidentType]),
            name="incident_type_valid",
        ),
        Index("ix_incidents_machine_id_detected_at", "machine_id", "detected_at"),
        Index("ix_incidents_status_detected_at", "status", "detected_at"),
    )
