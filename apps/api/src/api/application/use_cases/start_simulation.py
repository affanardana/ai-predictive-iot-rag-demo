"""Use case: begin a controlled simulation run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from api.domain.entities.prediction import PREDICTION_WINDOW_READINGS
from api.domain.entities.simulation_run import SimulationRun, new_session_id
from api.domain.errors import (
    DomainValidationError,
    MachineNotFoundError,
    SimulationAlreadyRunningError,
    SimulationUnavailableError,
    TooManySimulationsError,
)
from api.domain.ports.clock import Clock
from api.domain.ports.events import EventKind, EventPublisher, MachineEvent
from api.domain.ports.simulation import SimulationController, SimulationPlan
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.simulation_scenario import (
    DEMO_SCENARIO,
    DEMO_SEED,
    SimulationScenario,
)

#: The shortest run that can show anything. A prediction needs this many
#: consecutive readings, and a run that produces fewer can never be scored -- so
#: it can produce no risk band and no incident, which is the opposite of what
#: `MASTERPLAN.md` asks a demonstration to show. Refused up front, because a run
#: that produces nothing looks identical to a broken pipeline.
MINIMUM_DURATION_MINUTES = PREDICTION_WINDOW_READINGS

#: Long enough to watch a machine degrade, and the pace the runbook demonstrates.
DEFAULT_DURATION_MINUTES = 240

#: Wall-clock seconds between readings. A 240-minute session at a one-minute
#: sample interval is 240 readings, so four minutes of watching replays four
#: hours of degradation.
DEFAULT_TICK_SECONDS = 1.0

#: One reading per minute, matching the dataset definition.
DEFAULT_SAMPLE_INTERVAL = timedelta(minutes=1)

#: Used when a caller asks for neither demo mode nor a seed of its own. Fixed
#: rather than random, because `MASTERPLAN.md` §3.5 asks that the same inputs
#: reproduce the same output -- and a run whose values changed every time would
#: make a reviewer's second look at the same scenario inexplicably different.
#: A caller wanting variation passes an explicit seed, which the form offers.
DEFAULT_SEED = 0


@dataclass(frozen=True, slots=True)
class StartSimulation:
    """Create a run, hand it to the simulator, and record what happened.

    The order matters and is the opposite of the obvious one. The row is written
    **before** the simulator is asked, so that a service which accepts the run
    and then fails to answer cannot leave a run executing with no record of it.
    A failure to reach the service is recorded against the row rather than
    discarding it, because the row is the diagnostic.
    """

    unit_of_work_factory: UnitOfWorkFactory
    controller: SimulationController
    clock: Clock
    events: EventPublisher
    max_concurrent_runs: int

    async def execute(
        self,
        machine_id: MachineId,
        scenario: SimulationScenario | None = None,
        *,
        demo: bool = False,
        seed: int | None = None,
        duration_minutes: int | None = None,
        tick_seconds: float | None = None,
    ) -> SimulationRun:
        """Start a run.

        `demo` supplies the canonical demonstration's defaults, per PRD §12, and
        fills in only what the caller left alone -- so `demo=True` with an
        explicit machine demonstrates that machine, and an explicit seed still
        wins over the demo's.

        Raises:
            MachineNotFoundError: if the machine is not registered.
            SimulationAlreadyRunningError: if it already has an active run.
            TooManySimulationsError: if the fleet is at its concurrency ceiling.
            SimulationUnavailableError: if the simulator could not be reached.
        """
        resolved_scenario = scenario if scenario is not None else (DEMO_SCENARIO if demo else None)
        if resolved_scenario is None:
            raise DomainValidationError("A scenario is required unless demo mode is requested.")

        run = self._build(
            machine_id=machine_id,
            scenario=resolved_scenario,
            seed=seed if seed is not None else (DEMO_SEED if demo else DEFAULT_SEED),
            duration_minutes=duration_minutes
            if duration_minutes is not None
            else DEFAULT_DURATION_MINUTES,
            tick_seconds=tick_seconds if tick_seconds is not None else DEFAULT_TICK_SECONDS,
        )
        plan = self._plan_for(run)

        async with self.unit_of_work_factory() as uow:
            if await uow.machines.get(machine_id) is None:
                raise MachineNotFoundError(str(machine_id))

            active = await uow.simulations.active_for_machine(machine_id)
            if active is not None:
                raise SimulationAlreadyRunningError(str(machine_id), active.session_id)

            if await uow.simulations.count_active() >= self.max_concurrent_runs:
                raise TooManySimulationsError(self.max_concurrent_runs)

            await uow.simulations.add(run)

        # Outside the transaction, and before anything claims the run is going.
        # Holding a database connection open across a network call to another
        # service would tie up a pooled connection for the duration of someone
        # else's latency -- the same reasoning `RecordPrediction` gives for
        # calling the model outside its transaction.
        try:
            await self.controller.start(plan)
        except SimulationUnavailableError as error:
            await self._mark_failed(run, str(error))
            raise

        async with self.unit_of_work_factory() as uow:
            stored = await uow.simulations.get(run.session_id)
            if stored is not None:
                stored.mark_running()
                await uow.simulations.update(stored)

        # After the commit, so no subscriber is told about a run that rolled
        # back. A run starting changes no fleet number -- the readings that
        # follow arrive as telemetry and move those -- so the fleet list is
        # deliberately not announced here.
        await self.events.publish(
            MachineEvent(kind=EventKind.SIMULATION_STATE_CHANGED, machine_id=machine_id)
        )
        return run

    async def _mark_failed(self, run: SimulationRun, detail: str) -> None:
        """Record that the run could not be handed to the simulator."""
        async with self.unit_of_work_factory() as uow:
            stored = await uow.simulations.get(run.session_id)
            if stored is not None:
                stored.mark_failed(detail)
                await uow.simulations.update(stored)

    def _build(
        self,
        *,
        machine_id: MachineId,
        scenario: SimulationScenario,
        seed: int,
        duration_minutes: int,
        tick_seconds: float,
    ) -> SimulationRun:
        """Build the run, refusing a duration that cannot show anything."""
        if duration_minutes < MINIMUM_DURATION_MINUTES:
            raise DomainValidationError(
                f"A run must cover at least {MINIMUM_DURATION_MINUTES} minutes of "
                f"simulated time and this one covers {duration_minutes}. The model "
                f"needs {PREDICTION_WINDOW_READINGS} readings before it will predict "
                "anything, so a shorter run produces no risk band and no incident."
            )

        now = self.clock.now()
        return SimulationRun(
            # Minted here rather than derived from the scenario and seed, so two
            # demonstrations of the same thing do not share event ids -- which
            # would make the second one's readings all land as duplicates.
            session_id=new_session_id(scenario),
            machine_id=machine_id,
            scenario=scenario,
            seed=seed,
            started_at=now,
            sample_interval=DEFAULT_SAMPLE_INTERVAL,
            duration=timedelta(minutes=duration_minutes),
            created_at=now,
            tick_seconds=tick_seconds,
        )

    def _plan_for(self, run: SimulationRun) -> SimulationPlan:
        """Describe the run to the simulator."""
        return SimulationPlan(
            session_id=run.session_id,
            machine_id=run.machine_id,
            scenario=run.scenario,
            seed=run.seed,
            started_at=run.started_at,
            duration=run.duration,
            sample_interval=run.sample_interval,
            tick_seconds=run.tick_seconds,
        )
