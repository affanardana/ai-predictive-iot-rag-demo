"""Use case: accept telemetry from the transport and persist it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from api.domain.entities.telemetry import TelemetryRecord
from api.domain.errors import InvalidTelemetryError, MachineNotFoundError
from api.domain.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
from api.domain.value_objects.machine_id import MachineId

#: How many readings one request may carry. A bound rather than a policy: the
#: transport delivers a message at a time, and this exists so a malformed or
#: hostile caller cannot make the API allocate an unbounded list.
MAX_INGEST_BATCH = 500


@dataclass(frozen=True, slots=True)
class IngestResult:
    """What became of one batch of telemetry."""

    accepted: int
    duplicates: int
    ready: tuple[MachineId, ...]


@dataclass(frozen=True, slots=True)
class IngestTelemetry:
    """Persist a batch of readings, then report which machines are scoreable.

    Idempotent by construction. `event_id` is the table's primary key, so a
    redelivered message cannot create a second row, and a batch already seen
    reports `accepted=0` rather than failing -- a retry is an expected event on
    an at-least-once transport, not an error.

    `ready` is what the orchestrator uses to decide when to ask for a
    prediction. Reporting it here rather than leaving it to be discovered keeps
    the readiness rule in one place.
    """

    unit_of_work_factory: UnitOfWorkFactory
    window_readings: int
    max_batch_size: int = MAX_INGEST_BATCH

    async def execute(self, records: Sequence[TelemetryRecord]) -> IngestResult:
        """Store `records`, returning counts and the newly scoreable machines.

        Raises:
            InvalidTelemetryError: if the batch is empty or oversized.
            MachineNotFoundError: if it names a machine that is not registered.
        """
        if not records:
            raise InvalidTelemetryError("A telemetry batch must contain at least one record.")
        if len(records) > self.max_batch_size:
            raise InvalidTelemetryError(
                f"A telemetry batch may hold at most {self.max_batch_size} records "
                f"and this one held {len(records)}."
            )

        # One transaction for the insert and the readiness counts, so `ready`
        # describes the rows this call actually committed. Computing it after
        # the commit would double the queries and let another execution change
        # the answer in between.
        async with self.unit_of_work_factory() as uow:
            await self._require_machines(uow, records)
            accepted = await uow.telemetry.add_many_idempotent(records)

            # A redelivery cannot have changed any count, so there is nothing
            # to recompute. This is the retry path, which the spec cares about
            # most, and it costs one statement instead of one per machine.
            ready = await self._ready_machines(uow, records) if accepted else ()

        return IngestResult(
            accepted=accepted,
            duplicates=len(records) - accepted,
            ready=ready,
        )

    async def _require_machines(self, uow: UnitOfWork, records: Sequence[TelemetryRecord]) -> None:
        """Refuse the batch if any machine it names is unregistered.

        The foreign key would catch this as well, but as an integrity error
        surfacing as a 500. Naming the machine is the difference between an
        operator fixing a typo and an operator reading a stack trace.
        """
        for machine_id in _distinct_machines(records):
            if await uow.machines.get(machine_id) is None:
                raise MachineNotFoundError(str(machine_id))

    async def _ready_machines(
        self, uow: UnitOfWork, records: Sequence[TelemetryRecord]
    ) -> tuple[MachineId, ...]:
        """Return the machines in this batch holding a full prediction window.

        Sorted rather than left in encounter order, so the response is a
        function of the data and not of the order messages happened to arrive.
        """
        ready = [
            machine_id
            for machine_id in _distinct_machines(records)
            if await uow.telemetry.count_for(machine_id) >= self.window_readings
        ]
        return tuple(sorted(ready, key=lambda machine_id: machine_id.value))


def _distinct_machines(records: Sequence[TelemetryRecord]) -> list[MachineId]:
    """Return each machine named by `records`, once, in first-seen order.

    `dict.fromkeys` rather than a set: the machines are checked in the order
    the batch names them, so the error a caller sees is deterministic.
    """
    return list(dict.fromkeys(record.machine_id for record in records))
