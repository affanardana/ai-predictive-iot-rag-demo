"""Persistence model for simulation runs."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from api.domain.value_objects.run_status import ACTIVE_RUN_STATUSES, RunStatus
from api.domain.value_objects.simulation_scenario import SimulationScenario
from api.infrastructure.persistence.sql.base import Base
from api.infrastructure.persistence.sql.models.checks import in_clause

SESSION_ID_MAX_LENGTH = 32
SCENARIO_MAX_LENGTH = 32
STATUS_MAX_LENGTH = 16
DETAIL_MAX_LENGTH = 255


class SimulationRunModel(Base):
    """A controlled simulation run.

    The run's *configuration* is stored, not just its identity, and that is what
    makes recovery possible: a container that restarted mid-demonstration can be
    re-issued from `completed_ticks` without asking anyone, because the engine
    is a pure function of the tick index.
    """

    __tablename__ = "simulation_runs"

    session_id: Mapped[str] = mapped_column(String(SESSION_ID_MAX_LENGTH), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(16),
        ForeignKey("machines.machine_id", ondelete="CASCADE"),
    )
    scenario: Mapped[str] = mapped_column(String(SCENARIO_MAX_LENGTH))
    seed: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime]
    sample_interval_seconds: Mapped[int] = mapped_column(Integer)
    duration_seconds: Mapped[int] = mapped_column(Integer)
    #: Wall-clock pace. Not part of `SimulationSession` -- `stream_session`
    #: takes it as an argument -- but stored all the same, because it is the one
    #: piece of the plan the API cannot recompute when it re-issues a run.
    tick_seconds: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(STATUS_MAX_LENGTH))
    created_at: Mapped[datetime]
    completed_ticks: Mapped[int] = mapped_column(Integer, default=0)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(default=None)
    finished_at: Mapped[datetime | None] = mapped_column(default=None)
    detail: Mapped[str | None] = mapped_column(String(DETAIL_MAX_LENGTH), default=None)
    resume_count: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        CheckConstraint("sample_interval_seconds > 0", name="sample_interval_positive"),
        CheckConstraint("duration_seconds > 0", name="duration_positive"),
        CheckConstraint("tick_seconds >= 0.0", name="tick_seconds_non_negative"),
        CheckConstraint("completed_ticks >= 0", name="completed_ticks_non_negative"),
        CheckConstraint("resume_count >= 0", name="resume_count_non_negative"),
        CheckConstraint(
            in_clause("status", [status.value for status in RunStatus]),
            name="status_valid",
        ),
        CheckConstraint(
            in_clause("scenario", [scenario.value for scenario in SimulationScenario]),
            name="scenario_valid",
        ),
        Index("ix_simulation_runs_machine_id_created_at", "machine_id", "created_at"),
        Index("ix_simulation_runs_status_created_at", "status", "created_at"),
        # At most one active run per machine, enforced by the schema rather than
        # by a check-then-insert in a use case.
        #
        # Two runs on one machine would interleave two scenarios' readings on the
        # same charts, and the risk band they produced would describe neither --
        # with nothing logged to explain it. `RecordPrediction` accepts that race
        # for incident suppression because a duplicate incident is merely noisy;
        # this one corrupts the demonstration, and both supported dialects have
        # partial indexes, so it can be made impossible instead.
        #
        # The statuses come from the domain, so a run added to
        # `ACTIVE_RUN_STATUSES` cannot leave this constraint behind.
        Index(
            "uq_simulation_runs_active_machine",
            "machine_id",
            unique=True,
            postgresql_where=text(
                in_clause("status", sorted(s.value for s in ACTIVE_RUN_STATUSES))
            ),
            sqlite_where=text(in_clause("status", sorted(s.value for s in ACTIVE_RUN_STATUSES))),
        ),
    )
