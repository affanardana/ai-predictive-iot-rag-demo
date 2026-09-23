"""Incident listing use case."""

from __future__ import annotations

from datetime import timedelta

import pytest

from api.application.use_cases import ListIncidents
from api.domain.errors import MachineNotFoundError
from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.risk_level import IncidentSeverity
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.factories import DEFAULT_NOW, make_incident, make_machine


async def _seed(uow_factory: InMemoryUnitOfWorkFactory) -> None:
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M001"))
        await uow.machines.add(make_machine("M002"))
        await uow.incidents.add(
            make_incident(
                machine_id="M001",
                incident_id="inc-critical",
                detected_at=DEFAULT_NOW,
                severity=IncidentSeverity.CRITICAL,
                probability=0.9,
            )
        )
        high = make_incident(
            machine_id="M001",
            incident_id="inc-high",
            detected_at=DEFAULT_NOW - timedelta(hours=1),
            severity=IncidentSeverity.HIGH,
            probability=0.7,
        )
        high.acknowledge()
        await uow.incidents.add(high)
        await uow.incidents.add(
            make_incident(
                machine_id="M002",
                incident_id="inc-other",
                detected_at=DEFAULT_NOW - timedelta(hours=2),
            )
        )


async def test_lists_incidents_across_the_fleet(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """Without a machine filter, every incident is returned."""
    await _seed(uow_factory)

    incidents = await ListIncidents(unit_of_work_factory=uow_factory).execute()

    assert [incident.incident_id for incident in incidents] == [
        "inc-critical",
        "inc-high",
        "inc-other",
    ]


async def test_filters_by_machine(uow_factory: InMemoryUnitOfWorkFactory) -> None:
    """A machine filter narrows the list to that machine."""
    await _seed(uow_factory)

    incidents = await ListIncidents(unit_of_work_factory=uow_factory).execute(
        machine_id=MachineId("M001")
    )

    assert [incident.incident_id for incident in incidents] == ["inc-critical", "inc-high"]


async def test_raises_for_an_unknown_machine_filter(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """Filtering by an unregistered machine is a 404, not an empty list.

    Without this check a typo in the machine identifier would look like "this
    machine has no incidents", which is a different and misleading answer.
    """
    await _seed(uow_factory)

    with pytest.raises(MachineNotFoundError):
        await ListIncidents(unit_of_work_factory=uow_factory).execute(machine_id=MachineId("M999"))


async def test_filters_by_status(uow_factory: InMemoryUnitOfWorkFactory) -> None:
    """Status filtering selects only matching incidents."""
    await _seed(uow_factory)

    incidents = await ListIncidents(unit_of_work_factory=uow_factory).execute(
        status=IncidentStatus.ACKNOWLEDGED
    )

    assert [incident.incident_id for incident in incidents] == ["inc-high"]


async def test_filters_by_severity(uow_factory: InMemoryUnitOfWorkFactory) -> None:
    """Severity filtering selects only matching incidents."""
    await _seed(uow_factory)

    incidents = await ListIncidents(unit_of_work_factory=uow_factory).execute(
        severity=IncidentSeverity.CRITICAL
    )

    assert [incident.incident_id for incident in incidents] == ["inc-critical"]


async def test_combines_filters_conjunctively(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """Both filters must match for an incident to be returned."""
    await _seed(uow_factory)

    incidents = await ListIncidents(unit_of_work_factory=uow_factory).execute(
        status=IncidentStatus.OPEN,
        severity=IncidentSeverity.CRITICAL,
    )

    assert [incident.incident_id for incident in incidents] == ["inc-critical"]
