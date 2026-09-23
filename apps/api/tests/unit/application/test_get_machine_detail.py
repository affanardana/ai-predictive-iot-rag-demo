"""Machine detail use case."""

from __future__ import annotations

import pytest

from api.application.use_cases import GetMachineDetail
from api.domain.errors import MachineNotFoundError
from api.domain.value_objects.machine_id import MachineId
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.factories import make_incident, make_machine, make_prediction


async def test_raises_for_an_unknown_machine(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """An unregistered machine is a 404, not an empty view."""
    use_case = GetMachineDetail(unit_of_work_factory=uow_factory)

    with pytest.raises(MachineNotFoundError) as caught:
        await use_case.execute(MachineId("M999"))

    assert caught.value.machine_id == "M999"


async def test_returns_current_state_with_recent_incidents(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The detail view consolidates state and history for one machine."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003", name="Compressor motor"))
        await uow.predictions.add(make_prediction(machine_id="M003", probability=0.81))
        await uow.incidents.add(
            make_incident(machine_id="M003", incident_id="inc-1", prediction_id="pred-1")
        )

    detail = await GetMachineDetail(unit_of_work_factory=uow_factory).execute(MachineId("M003"))

    assert detail.summary.machine.name == "Compressor motor"
    assert detail.summary.latest_prediction is not None
    assert detail.summary.latest_prediction.probability.value == pytest.approx(0.81)
    assert [incident.incident_id for incident in detail.recent_incidents] == ["inc-1"]


async def test_excludes_other_machines_incidents(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """Only this machine's incidents appear in its detail view."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M001"))
        await uow.machines.add(make_machine("M002"))
        await uow.incidents.add(make_incident(machine_id="M002", incident_id="other"))
        await uow.incidents.add(make_incident(machine_id="M001", incident_id="mine"))

    detail = await GetMachineDetail(unit_of_work_factory=uow_factory).execute(MachineId("M001"))

    assert [incident.incident_id for incident in detail.recent_incidents] == ["mine"]


async def test_respects_the_incident_limit(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The inline incident list is capped."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M001"))
        for index in range(5):
            await uow.incidents.add(make_incident(incident_id=f"inc-{index}"))

    detail = await GetMachineDetail(unit_of_work_factory=uow_factory, incident_limit=2).execute(
        MachineId("M001")
    )

    assert len(detail.recent_incidents) == 2
