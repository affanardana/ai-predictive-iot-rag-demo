"""Use case: retrieve how a machine's predicted failure risk has changed."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from api.domain.entities.prediction import Prediction
from api.domain.errors import MachineNotFoundError
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.value_objects.machine_id import MachineId

#: How many past predictions the detail view charts by default.
DEFAULT_HISTORY_LIMIT = 200


@dataclass(frozen=True, slots=True)
class GetPredictionHistory:
    """Return recent predictions for a machine, most recent first."""

    unit_of_work_factory: UnitOfWorkFactory
    limit: int = DEFAULT_HISTORY_LIMIT

    async def execute(self, machine_id: MachineId) -> Sequence[Prediction]:
        """Return the prediction history for one machine.

        Raises:
            MachineNotFoundError: if the machine is not registered.
        """
        async with self.unit_of_work_factory() as uow:
            machine = await uow.machines.get(machine_id)
            if machine is None:
                raise MachineNotFoundError(str(machine_id))

            return await uow.predictions.history_for(machine_id, limit=self.limit)
