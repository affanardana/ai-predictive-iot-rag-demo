"""Use case: clear a simulation run so the machine is free for another."""

from __future__ import annotations

from dataclasses import dataclass

from api.domain.errors import SimulationRunActiveError, SimulationRunNotFoundError
from api.domain.ports.events import EventKind, EventPublisher, MachineEvent
from api.domain.ports.unit_of_work import UnitOfWorkFactory


@dataclass(frozen=True, slots=True)
class ResetSimulation:
    """Forget a run. Not its telemetry.

    `PRD.md` §20.5 lists reset beside start and stop, and the obvious reading --
    rewind to the first tick and replay -- is **actively broken here**. A
    session's event ids are `{session_id}-{machine}-{index}`, so replaying one
    re-mints identifiers the database already holds and every reading lands as a
    duplicate. The demo would publish hundreds of messages and store none.

    So reset removes the run record, and only that.

    **It cannot remove the readings, and that is a structural limit rather than
    caution.** Deleting stored telemetry is a destructive write, which would
    have to be guarded by the ingest token -- and a browser cannot hold the
    token, which is precisely why the dashboard's routes are unguarded (ADR
    0007). A destructive reset would therefore be unreachable from the button
    that is supposed to offer it.

    The consequence, which belongs in the runbook and on the page: a second run
    **appends** to a machine's charts rather than replacing them. The windows
    hide it in practice -- `recorded_at` is wall-clock and the charts are
    windowed -- so an earlier run falls out of the 1-hour view on its own.
    """

    unit_of_work_factory: UnitOfWorkFactory
    events: EventPublisher

    async def execute(self, session_id: str) -> None:
        """Remove a run.

        Raises:
            SimulationRunNotFoundError: if no run carries this identifier.
            SimulationRunActiveError: if it is still going. Stopping and
                resetting are separate on purpose: a caller that meant to end a
                run should not discover it also discarded the record of it.
        """
        async with self.unit_of_work_factory() as uow:
            run = await uow.simulations.get(session_id)
            if run is None:
                raise SimulationRunNotFoundError(session_id)
            if run.is_active:
                raise SimulationRunActiveError(session_id)

            machine_id = run.machine_id
            await uow.simulations.delete(session_id)

        await self.events.publish(
            MachineEvent(kind=EventKind.SIMULATION_STATE_CHANGED, machine_id=machine_id)
        )
