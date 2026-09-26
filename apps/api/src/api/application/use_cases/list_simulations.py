"""Use case: report what runs exist and how they are doing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

from api.application.read_models import SimulationRunView
from api.domain.errors import MachineNotFoundError
from api.domain.ports.clock import Clock
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.value_objects.machine_id import MachineId

#: How many runs a listing returns when the caller does not say. The same
#: bounded-list approach the incident and prediction endpoints take -- and the
#: same limitation, that a caller wanting page two has no way to ask.
DEFAULT_SIMULATION_LIMIT = 50

#: How long a run may go without a report before it is treated as gone.
#: Three missed heartbeats at the simulator's five-second cadence, with room for
#: a slow one. Too short and a busy box reports live runs as dead; too long and
#: a container that died leaves the dashboard showing a run that is not
#: happening.
HEARTBEAT_TIMEOUT_SECONDS = 45


@dataclass(frozen=True, slots=True)
class ListSimulations:
    """Return runs, with the two derived facts a client cannot work out itself.

    `is_active` and `is_stale` are computed here rather than in a presenter
    because both need the clock, and the presentation layer has none. A client
    must never derive staleness: it would need to know the timeout, and it would
    be reading a different clock from the one the runs were recorded against.
    """

    unit_of_work_factory: UnitOfWorkFactory
    clock: Clock
    heartbeat_timeout_seconds: float = HEARTBEAT_TIMEOUT_SECONDS

    async def execute(
        self,
        machine_id: MachineId | None = None,
        limit: int = DEFAULT_SIMULATION_LIMIT,
    ) -> Sequence[SimulationRunView]:
        """Return runs, newest first, optionally scoped to one machine.

        Raises:
            MachineNotFoundError: if `machine_id` names a machine that is not
                registered. Checked rather than answered with an empty list,
                matching every other machine-scoped read: an unknown machine and
                a machine with no runs are different situations, and only one of
                them means the caller mistyped something.
        """
        async with self.unit_of_work_factory() as uow:
            if machine_id is None:
                runs = await uow.simulations.list_recent(limit)
            else:
                if await uow.machines.get(machine_id) is None:
                    raise MachineNotFoundError(str(machine_id))
                runs = await uow.simulations.list_for_machine(machine_id, limit)

        now = self.clock.now()
        timeout = timedelta(seconds=self.heartbeat_timeout_seconds)
        return [
            SimulationRunView(
                run=run,
                is_active=run.is_active,
                is_stale=run.is_stale(now, timeout),
            )
            for run in runs
        ]
