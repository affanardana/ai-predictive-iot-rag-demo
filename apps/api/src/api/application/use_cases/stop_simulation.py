"""Use case: end a simulation run."""

from __future__ import annotations

from dataclasses import dataclass

from api.domain.entities.simulation_run import SimulationRun
from api.domain.errors import SimulationRunNotFoundError, SimulationUnavailableError
from api.domain.ports.clock import Clock
from api.domain.ports.events import EventKind, EventPublisher, MachineEvent
from api.domain.ports.simulation import SimulationController
from api.domain.ports.unit_of_work import UnitOfWorkFactory


@dataclass(frozen=True, slots=True)
class StopSimulation:
    """Stop a run, whatever the simulator has to say about it.

    **The run is always terminalised**, and that is the design rather than a
    convenience. Stop is an operator's escape hatch; one that reported "I could
    not stop it" because the container was unreachable would leave a run stuck
    at RUNNING with no way out, and the only remaining recourse would be editing
    the database. So a failure to reach the simulator is recorded in the run's
    detail and the run is stopped anyway -- which is also the truth, because a
    run whose container is gone is not running.
    """

    unit_of_work_factory: UnitOfWorkFactory
    controller: SimulationController
    clock: Clock
    events: EventPublisher

    async def execute(self, session_id: str) -> SimulationRun:
        """End a run.

        Idempotent: stopping a run that has already stopped returns it
        unchanged. A double-click, or a client retrying a request whose response
        it never saw, must not read as a fault.

        Raises:
            SimulationRunNotFoundError: if no run carries this identifier.
        """
        async with self.unit_of_work_factory() as uow:
            run = await uow.simulations.get(session_id)
            if run is None:
                raise SimulationRunNotFoundError(session_id)

            if run.is_terminal:
                return run

        detail: str | None = None
        try:
            await self.controller.stop(session_id)
        except SimulationUnavailableError as error:
            # Recorded, not raised. See the class docstring.
            detail = f"The simulator could not be reached to stop this run: {error}"

        async with self.unit_of_work_factory() as uow:
            stored = await uow.simulations.get(session_id)
            if stored is None:
                raise SimulationRunNotFoundError(session_id)
            if stored.is_terminal:
                # The run finished on its own between the two transactions --
                # a real race, and the run's own outcome is the more accurate
                # one. Reported as it ended rather than overwritten.
                return stored
            stored.mark_stopped(self.clock.now(), detail)
            await uow.simulations.update(stored)

        await self.events.publish(
            MachineEvent(kind=EventKind.SIMULATION_STATE_CHANGED, machine_id=stored.machine_id)
        )
        return stored
