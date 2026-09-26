"""Use case: move an incident to a new lifecycle status."""

from __future__ import annotations

from dataclasses import dataclass

from api.domain.entities.incident import Incident
from api.domain.errors import IncidentNotFoundError
from api.domain.ports.events import EventKind, EventPublisher, MachineEvent
from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.value_objects.incident_status import IncidentStatus


@dataclass(frozen=True, slots=True)
class UpdateIncidentStatus:
    """Apply an operator's decision to an incident.

    The lifecycle rules are not restated here. `Incident.transition_to` owns
    them through the table in `incident_status`, and this use case only loads,
    asks, and persists -- so there is exactly one place a transition can be
    legal or illegal, and it is the place the domain tests already cover.

    This is what makes the suppression rule survivable. `RecordPrediction`
    raises an incident only when a machine has none open, so before this
    existed an incident was permanent: every later excursion on that machine
    was suppressed by the first one, forever. Resolving or dismissing it is
    what lets the next one through.
    """

    unit_of_work_factory: UnitOfWorkFactory
    events: EventPublisher

    async def execute(self, incident_id: str, requested: IncidentStatus) -> Incident:
        """Move one incident to `requested`.

        Raises:
            IncidentNotFoundError: if no incident carries this identifier.
            InvalidIncidentTransitionError: if the lifecycle forbids the move.
        """
        async with self.unit_of_work_factory() as uow:
            incident = await uow.incidents.get(incident_id)
            if incident is None:
                raise IncidentNotFoundError(incident_id)

            # Validated before the write, so an illegal request never opens a
            # transaction it is only going to abandon. The repository would
            # raise on a missing row anyway; failing first means the error a
            # caller sees is always about the transition rather than
            # sometimes about the row.
            incident.transition_to(requested)
            await uow.incidents.update(incident)

        # After the commit, so no subscriber is told about a status change that
        # rolled back.
        #
        # Announced as a change to the *machine* rather than to the incident,
        # because resolving or dismissing one drops it out of
        # `open_incident_count` -- a number on the fleet page, not only on the
        # incident page. Acknowledging changes no count (is_open covers OPEN
        # and ACKNOWLEDGED alike), so that case refetches slightly more than it
        # had to. One event kind for all three transitions is worth that.
        await self.events.publish(
            MachineEvent(kind=EventKind.INCIDENT_STATUS_CHANGED, machine_id=incident.machine_id)
        )
        return incident
