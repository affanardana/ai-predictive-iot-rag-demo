"""Use case: consolidate one machine's operational state."""

from __future__ import annotations

from dataclasses import dataclass

from api.application.read_models import MachineDetail
from api.application.summaries import summarise_machines
from api.domain.errors import MachineNotFoundError
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.value_objects.machine_id import MachineId

#: How many past incidents the detail response shows inline. The full history
#: remains available from the machine incidents endpoint.
DEFAULT_INCIDENT_LIMIT = 20


@dataclass(frozen=True, slots=True)
class GetMachineDetail:
    """Return the current state, latest prediction, and recent incidents."""

    unit_of_work_factory: UnitOfWorkFactory
    incident_limit: int = DEFAULT_INCIDENT_LIMIT

    async def execute(self, machine_id: MachineId) -> MachineDetail:
        """Return the consolidated view for one machine.

        Raises:
            MachineNotFoundError: if the machine is not registered.
        """
        async with self.unit_of_work_factory() as uow:
            machine = await uow.machines.get(machine_id)
            if machine is None:
                raise MachineNotFoundError(str(machine_id))

            # One machine, but through the same bulk path the fleet list uses,
            # so the detail page cannot describe a machine differently from the
            # row that linked to it.
            (summary,) = await summarise_machines(uow, [machine])
            incidents = await uow.incidents.list_for_machine(machine_id, limit=self.incident_limit)
            return MachineDetail(summary=summary, recent_incidents=incidents)
