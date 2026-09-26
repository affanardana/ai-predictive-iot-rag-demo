"""Use case: list the fleet with each machine's current condition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from api.application.read_models import MachineSummary
from api.application.summaries import summarise_machines
from api.domain.ports.unit_of_work import UnitOfWorkFactory


@dataclass(frozen=True, slots=True)
class ListMachines:
    """Return every registered machine with its latest observed state.

    Four queries in total, whatever the fleet size: the registry, then all
    latest readings, predictions, and open incident counts in one query each.
    This was O(3n) until the dashboard needed to poll it.
    """

    unit_of_work_factory: UnitOfWorkFactory

    async def execute(self) -> Sequence[MachineSummary]:
        """Return a summary for every registered machine."""
        async with self.unit_of_work_factory() as uow:
            machines = await uow.machines.list_all()
            return await summarise_machines(uow, machines)
