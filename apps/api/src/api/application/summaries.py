"""Machine summary assembly.

Shared by the fleet list and the machine detail use cases so both describe a
machine's condition the same way. If these diverged, the list and the detail
page could disagree about the same machine.
"""

from __future__ import annotations

from collections.abc import Sequence

from api.application.read_models import MachineSummary
from api.domain.entities.machine import Machine
from api.domain.entities.prediction import Prediction
from api.domain.entities.telemetry import TelemetryRecord
from api.domain.ports.unit_of_work import UnitOfWork

#: Upper bound on incidents scanned when counting the open ones per machine.
#: Only `RecordPrediction`'s suppression check uses this now -- summaries count
#: in SQL -- and it is deliberately generous rather than tight: a machine with
#: more than this many incidents has a problem one duplicate would not worsen.
OPEN_INCIDENT_SCAN_LIMIT = 500


def summarise_from(
    machine: Machine,
    latest_reading: TelemetryRecord | None,
    latest_prediction: Prediction | None,
    open_incident_count: int,
) -> MachineSummary:
    """Assemble a summary from parts that have already been read.

    The only place a `MachineSummary` is constructed. Both readers go through
    here, so the list and the detail page cannot come to describe the same
    machine differently.

    Either the reading or the prediction may be missing: a machine that is
    registered but silent has neither, and one that has stopped reporting keeps
    its last known values so an operator can see how stale it is.
    """
    return MachineSummary(
        machine=machine,
        latest_reading=latest_reading,
        latest_prediction=latest_prediction,
        risk_level=latest_prediction.risk_level if latest_prediction else None,
        open_incident_count=open_incident_count,
    )


async def summarise_machines(
    uow: UnitOfWork,
    machines: Sequence[Machine],
) -> Sequence[MachineSummary]:
    """Summarise every machine in `machines` with three reads in total.

    Not three reads *each*. Assembling summaries one machine at a time is what
    made `GET /api/v1/machines` O(3n) queries, and the fleet view now polls
    that endpoint on every telemetry event.

    A machine with no telemetry, no prediction, or no open incidents is simply
    absent from the corresponding mapping, which is why each lookup supplies
    its own default rather than being indexed.
    """
    if not machines:
        return []

    machine_ids = [machine.id for machine in machines]
    readings = await uow.telemetry.latest_for_many(machine_ids)
    predictions = await uow.predictions.latest_for_many(machine_ids)
    open_counts = await uow.incidents.open_counts_by_machine(machine_ids)

    return [
        summarise_from(
            machine,
            latest_reading=readings.get(machine.id),
            latest_prediction=predictions.get(machine.id),
            open_incident_count=open_counts.get(machine.id, 0),
        )
        for machine in machines
    ]
