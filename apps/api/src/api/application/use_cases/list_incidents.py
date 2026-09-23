"""Use case: list incidents, for the whole fleet or for one machine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from api.domain.entities.incident import Incident
from api.domain.errors import MachineNotFoundError
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.risk_level import IncidentSeverity

DEFAULT_INCIDENT_LIST_LIMIT = 100


@dataclass(frozen=True, slots=True)
class ListIncidents:
    """Return incidents, optionally narrowed to one machine.

    A single use case covers both the fleet-wide incident page and the
    per-machine list because they differ only by a filter. Splitting them
    would duplicate the machine-existence check and the ordering rule.
    """

    unit_of_work_factory: UnitOfWorkFactory
    limit: int = DEFAULT_INCIDENT_LIST_LIMIT

    async def execute(
        self,
        *,
        machine_id: MachineId | None = None,
        status: IncidentStatus | None = None,
        severity: IncidentSeverity | None = None,
    ) -> Sequence[Incident]:
        """Return incidents, most recent first.

        Raises:
            MachineNotFoundError: if a machine filter is supplied and that
                machine is not registered. Without this check a typo would
                silently return an empty list rather than a 404.
        """
        async with self.unit_of_work_factory() as uow:
            if machine_id is not None:
                machine = await uow.machines.get(machine_id)
                if machine is None:
                    raise MachineNotFoundError(str(machine_id))
                return await uow.incidents.list_for_machine(machine_id, limit=self.limit)

            return await uow.incidents.list_filtered(
                status=status,
                severity=severity,
                limit=self.limit,
            )
