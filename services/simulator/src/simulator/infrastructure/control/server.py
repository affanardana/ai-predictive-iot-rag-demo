"""HTTP control surface, driven by the API.

The API is the source of truth for what runs exist; this surface executes them.
That division is deliberate and is what the endpoints below are shaped around:
there is no way to *list* runs across restarts, because a run this process has
never heard of is not this process's to report. The API holds the record and
asks here only about runs it believes are live.

Pushing rather than polling, on one point: **stop**. A simulator that polled the
API for work would learn to stop on its next poll, which means either a request
per tick or a run that keeps publishing after the operator pressed the button.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from simulator.domain.errors import RunAlreadyActiveError, SimulationValidationError
from simulator.domain.ports import RunControl, RunState, RunView
from simulator.domain.scenario import Scenario
from simulator.domain.session import DEFAULT_SAMPLE_INTERVAL, SimulationSession


class StartRunRequest(BaseModel):
    """Everything needed to reproduce a run, and nothing else.

    `started_at` and `session_id` come from the caller rather than being minted
    here, because both are persisted by the API and a resumed run has to
    reproduce the first attempt exactly: `recorded_at` is
    `started_at + sample_interval x index`, so a new `started_at` would move
    every timestamp, and a new `session_id` would mint new `event_id`s and
    duplicate every reading.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(min_length=1, max_length=32)
    machine_id: str = Field(min_length=1, max_length=16)
    scenario: str
    seed: int
    duration_minutes: int = Field(gt=0)
    started_at: datetime
    sample_interval_seconds: int = Field(
        default=int(DEFAULT_SAMPLE_INTERVAL.total_seconds()),
        gt=0,
    )


class RunSchema(BaseModel):
    """One run's state, as this process sees it."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    machine_id: str
    state: RunState
    completed_ticks: int = Field(ge=0)
    total_ticks: int = Field(ge=0)
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None


def _to_schema(run: RunView) -> RunSchema:
    """Translate a run into its wire shape."""
    return RunSchema(
        session_id=run.session_id,
        machine_id=run.machine_id,
        state=run.state,
        completed_ticks=run.completed_ticks,
        total_ticks=run.total_ticks,
        started_at=run.started_at,
        finished_at=run.finished_at,
        error=run.error,
    )


def create_app(registry: RunControl) -> FastAPI:
    """Build the control application around an already-built registry.

    `RunControl` rather than the registry itself, and not by preference: the
    registry lives in the application layer and this module is in
    infrastructure, which the layering contract forbids an import between. The
    protocol sits in the domain precisely so this file can be written at all.
    The wiring -- which sink, which pace -- lives in `simulator.control`, a
    composition point, which is the module permitted to construct a sink.
    """
    app = FastAPI(
        title="Simulator control",
        description="Starts and stops simulated runs on behalf of the API.",
        docs_url=None,
        redoc_url=None,
    )

    @app.post("/runs", response_model=RunSchema, status_code=status.HTTP_201_CREATED)
    async def start_run(request: StartRunRequest) -> RunSchema:
        """Begin a run.

        The session is built here rather than accepted ready-made, so the
        validation the domain already performs -- non-blank identifiers, a
        positive duration, at least one machine -- is applied to a caller's
        request rather than trusted.
        """
        try:
            session = SimulationSession.create(
                machine_ids=[request.machine_id],
                scenario=Scenario.from_name(request.scenario),
                seed=request.seed,
                duration=timedelta(minutes=request.duration_minutes),
                started_at=request.started_at,
                sample_interval=timedelta(seconds=request.sample_interval_seconds),
                session_id=request.session_id,
            )
        except SimulationValidationError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error

        try:
            return _to_schema(registry.start(session))
        except RunAlreadyActiveError as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error

    @app.get("/runs", response_model=list[RunSchema])
    async def list_runs() -> list[RunSchema]:
        """Return the runs this process is executing or has executed."""
        return [_to_schema(run) for run in registry.list()]

    @app.get("/runs/{session_id}", response_model=RunSchema)
    async def get_run(session_id: str) -> RunSchema:
        """Return one run, or 404 if this process has never seen it."""
        run = registry.get(session_id)
        if run is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No run '{session_id}'.")
        return _to_schema(run)

    @app.delete("/runs/{session_id}", response_model=RunSchema)
    async def stop_run(session_id: str) -> RunSchema:
        """Ask a run to end.

        Returns as soon as the request is recorded, not when the run has
        finished stopping: the loop notices at its next tick boundary, and a
        handler that waited would hold the connection open for up to a tick.

        Idempotent. Stopping a run that has already stopped is a success, so a
        double-click or a retry is not an error the caller has to interpret.
        """
        run = registry.stop(session_id)
        if run is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No run '{session_id}'.")
        return _to_schema(run)

    return app
