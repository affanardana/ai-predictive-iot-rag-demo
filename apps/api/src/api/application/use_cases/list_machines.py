"""Use case: list the fleet with each machine's current condition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from api.application.read_models import MachineSummary
from api.application.summaries import summarise_machine
from api.domain.ports.unit_of_work import UnitOfWorkFactory


@dataclass(frozen=True, slots=True)
class ListMachines:
    """Return every registered machine with its latest observed state.

    KNOWN LIMITATION: assembling each summary costs three additional reads, so
    the endpoint issues O(3n) queries for n machines. That is acceptable at the
    current fleet scale, and the coding standards explicitly discourage
    premature optimisation. The fix when the fleet grows is a bulk read port
    returning all latest readings, predictions, and open incident counts in
    three queries regardless of n -- worth doing before the Phase 7 dashboard,
    which polls this endpoint.
    """

    unit_of_work_factory: UnitOfWorkFactory

    async def execute(self) -> Sequence[MachineSummary]:
        """Return a summary for every registered machine."""
        async with self.unit_of_work_factory() as uow:
            machines = await uow.machines.list_all()
            return [await summarise_machine(uow, machine) for machine in machines]
