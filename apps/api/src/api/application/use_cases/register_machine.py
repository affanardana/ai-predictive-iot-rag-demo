"""Use case: add a machine to the registry."""

from __future__ import annotations

from dataclasses import dataclass

from api.domain.entities.machine import Machine
from api.domain.ports.clock import Clock
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.value_objects.machine_id import MachineId


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    """A machine, and whether this call is what registered it."""

    machine: Machine
    created: bool


@dataclass(frozen=True, slots=True)
class RegisterMachine:
    """Register a machine, or return the one already registered.

    Idempotent, so a producer can declare its fleet on every start without
    tracking what it has already declared. `created` is what lets the route
    answer 201 or 200 honestly instead of guessing which happened.

    Check-then-act, so two concurrent registrations of the same new machine
    race: the SQL adapter raises on the second and the in-memory one overwrites.
    Both leave one machine registered, which is the outcome that matters; a
    unique-violation from the database is the honest way to lose that race.
    """

    unit_of_work_factory: UnitOfWorkFactory
    clock: Clock

    async def execute(self, machine_id: MachineId, name: str) -> RegistrationResult:
        """Register `machine_id` under `name`, or return what is already there."""
        async with self.unit_of_work_factory() as uow:
            existing = await uow.machines.get(machine_id)
            if existing is not None:
                return RegistrationResult(machine=existing, created=False)

            # No `create` factory: unlike `Prediction` and `Incident`, whose
            # identifiers the domain generates, a machine's identifier is the
            # natural key its caller supplies.
            machine = Machine(id=machine_id, name=name, registered_at=self.clock.now())
            await uow.machines.add(machine)

        return RegistrationResult(machine=machine, created=True)
