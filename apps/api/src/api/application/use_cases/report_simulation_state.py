"""Use case: accept a state report from the simulator service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from api.domain.entities.simulation_run import SimulationRun
from api.domain.errors import InvalidRunTransitionError, SimulationRunNotFoundError
from api.domain.ports.clock import Clock
from api.domain.ports.events import EventKind, EventPublisher, MachineEvent
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.value_objects.run_status import RunStatus


@dataclass(frozen=True, slots=True)
class ReportSimulationState:
    """Record what the simulator says a run is doing.

    This exists because the API's realtime broadcaster is **in-process**. Nothing
    outside the API can put an event on the dashboard's stream, so a run that
    finished in another container has to tell the API rather than publish --
    otherwise a finished run stays displayed as running until something happens
    to ask.

    **Progress reports do not publish events.** Only a change of status does.
    The simulator reports every few seconds; publishing on each would make every
    open dashboard refetch on that cadence, which on a one-core box is a
    self-inflicted load spike in service of a progress bar. The bar polls.
    """

    unit_of_work_factory: UnitOfWorkFactory
    clock: Clock
    events: EventPublisher

    async def execute(
        self,
        session_id: str,
        status: RunStatus,
        completed_ticks: int | None = None,
    ) -> SimulationRun:
        """Apply one report.

        Raises:
            SimulationRunNotFoundError: if the API has no record of this run.
                Reported rather than ignored: it means the simulator is running
                something the API does not know about, which is worth knowing.
            InvalidRunTransitionError: if the report contradicts the lifecycle,
                e.g. a completion arriving for a run already stopped.
        """
        async with self.unit_of_work_factory() as uow:
            run = await uow.simulations.get(session_id)
            if run is None:
                raise SimulationRunNotFoundError(session_id)

            previous = run.status
            now = self.clock.now()

            if completed_ticks is not None:
                run.record_progress(completed_ticks, now)

            if status is not previous:
                self._apply(run, status, now)

            await uow.simulations.update(run)

        # Published only when the status actually moved, so a run reporting
        # progress every few seconds is silent on the wire.
        if status is not previous:
            await self.events.publish(
                MachineEvent(
                    kind=EventKind.SIMULATION_STATE_CHANGED,
                    machine_id=run.machine_id,
                )
            )
        return run

    def _apply(self, run: SimulationRun, status: RunStatus, now: datetime) -> None:
        """Move the run to `status`, or raise if the lifecycle forbids it."""
        if status is RunStatus.RUNNING:
            if run.status is RunStatus.FAILED:
                run.resume()
            else:
                run.mark_running()
        elif status is RunStatus.COMPLETED:
            run.mark_completed(now)
        elif status is RunStatus.STOPPED:
            run.mark_stopped(now, "Reported stopped by the simulator.")
        elif status is RunStatus.FAILED:
            run.mark_failed("Reported failed by the simulator.")
        else:
            # `PENDING` is never a report: it is the state before the simulator
            # has been asked. A service claiming it would be describing a run it
            # has not accepted.
            raise InvalidRunTransitionError(current=run.status.value, requested=status.value)
