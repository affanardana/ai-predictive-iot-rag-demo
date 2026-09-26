"""Simulation run schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from api.application.use_cases.start_simulation import MINIMUM_DURATION_MINUTES
from api.domain.value_objects.run_status import RunStatus
from api.domain.value_objects.simulation_scenario import SimulationScenario


class StartSimulationRequest(BaseModel):
    """A run to begin.

    The four fields `PRD.md` §11 names -- machine, scenario, duration, and
    seed/demo mode -- and nothing else. Playback speed is deliberately not among
    them: the spec does not list it, and a caller setting a fast pace on a
    one-core box is the failure mode the resource limits exist to bound.
    """

    model_config = ConfigDict(frozen=True)

    machine_id: str = Field(min_length=1, max_length=16)
    scenario: SimulationScenario | None = Field(
        default=None,
        description="Required unless `demo` is set, which supplies the demonstration's own.",
    )
    demo: bool = Field(
        default=False,
        description=(
            "Use the canonical PRD section 12 demonstration: bearing degradation "
            "and a fixed seed, so the sequence is always the same one. Supplies "
            "only what the caller left unset -- an explicit scenario or seed "
            "still wins."
        ),
    )
    seed: int | None = Field(default=None, description="Omit for a fixed default seed.")
    duration_minutes: int = Field(
        default=240,
        ge=MINIMUM_DURATION_MINUTES,
        le=1440,
        description=(
            f"Simulated time to cover, not wall-clock time. At least "
            f"{MINIMUM_DURATION_MINUTES} minutes, because the model needs that many "
            "readings before it will predict anything -- a shorter run produces no "
            "risk band and no incident, and looks identical to a broken pipeline."
        ),
    )


class ReportSimulationStateRequest(BaseModel):
    """A state report from the simulator service."""

    model_config = ConfigDict(frozen=True)

    status: RunStatus
    completed_ticks: int | None = Field(default=None, ge=0)


class SimulationRunSchema(BaseModel):
    """A simulation run, as the dashboard sees it."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    machine_id: str
    scenario: SimulationScenario
    seed: int
    status: RunStatus
    duration_seconds: int = Field(description="Simulated time the run covers.")
    sample_interval_seconds: int = Field(description="Simulated time between readings.")
    tick_count: int = Field(description="How many readings the run produces.")
    completed_ticks: int = Field(ge=0)
    started_at: datetime
    created_at: datetime
    finished_at: datetime | None = None
    detail: str | None = Field(
        default=None,
        description="Why a run failed or was stopped without the simulator's confirmation.",
    )
    is_active: bool = Field(description="Whether the run is still going.")
    is_stale: bool = Field(
        description=(
            "Whether an active run has stopped reporting, which means the simulator "
            "service is gone or unreachable. Derived by the API, never by a client: "
            "it needs a clock and a timeout, and a browser has neither."
        ),
    )
