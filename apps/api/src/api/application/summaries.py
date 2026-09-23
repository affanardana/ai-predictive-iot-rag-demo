"""Machine summary assembly.

Shared by the fleet list and the machine detail use cases so both describe a
machine's condition the same way. If these diverged, the list and the detail
page could disagree about the same machine.
"""

from __future__ import annotations

from api.application.read_models import MachineSummary
from api.domain.entities.machine import Machine
from api.domain.ports.unit_of_work import UnitOfWork

#: Upper bound on incidents scanned when counting the open ones for a summary.
#: A machine with more than this many incidents has a problem the count would
#: understate, so the limit is deliberately generous rather than tight.
OPEN_INCIDENT_SCAN_LIMIT = 500


async def summarise_machine(uow: UnitOfWork, machine: Machine) -> MachineSummary:
    """Assemble the latest observed and predicted state for one machine.

    Either the reading or the prediction may be missing: a machine that is
    registered but silent has neither, and one that has stopped reporting
    keeps its last known values so an operator can see how stale it is.
    """
    latest_reading = await uow.telemetry.latest_for(machine.id)
    latest_prediction = await uow.predictions.latest_for(machine.id)
    incidents = await uow.incidents.list_for_machine(machine.id, limit=OPEN_INCIDENT_SCAN_LIMIT)
    open_count = sum(1 for incident in incidents if incident.is_open)

    return MachineSummary(
        machine=machine,
        latest_reading=latest_reading,
        latest_prediction=latest_prediction,
        risk_level=latest_prediction.risk_level if latest_prediction else None,
        open_incident_count=open_count,
    )
