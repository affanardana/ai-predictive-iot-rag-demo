"""Persistence model for telemetry."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from api.infrastructure.persistence.sql.base import Base

EVENT_ID_MAX_LENGTH = 64
SESSION_ID_MAX_LENGTH = 64


class TelemetryModel(Base):
    """One telemetry sample.

    `event_id` is the primary key rather than a synthetic surrogate. The
    transport may redeliver a message, and the non-functional requirements
    forbid that from creating duplicate rows -- making the event identifier the
    key means the uniqueness guarantee is enforced by the schema itself, so no
    application code path can bypass it.
    """

    __tablename__ = "telemetry"

    event_id: Mapped[str] = mapped_column(String(EVENT_ID_MAX_LENGTH), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(16),
        ForeignKey("machines.machine_id", ondelete="CASCADE"),
    )
    recorded_at: Mapped[datetime]
    temperature: Mapped[float]
    vibration: Mapped[float]
    rpm: Mapped[float]
    current: Mapped[float]
    load: Mapped[float]
    voltage: Mapped[float]
    simulation_session_id: Mapped[str | None] = mapped_column(
        String(SESSION_ID_MAX_LENGTH), default=None
    )

    __table_args__ = (
        # Supports the windowed history queries, which always filter by machine
        # and range over time.
        Index("ix_telemetry_machine_id_recorded_at", "machine_id", "recorded_at"),
    )
