"""Fleet listing use case."""

from __future__ import annotations

from datetime import timedelta

from api.application.use_cases import ListMachines
from api.domain.value_objects.risk_level import RiskLevel
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.factories import (
    DEFAULT_NOW,
    make_incident,
    make_machine,
    make_prediction,
    make_reading,
    make_telemetry,
)


async def test_returns_empty_when_no_machines_are_registered(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """An empty fleet is a valid result, not an error.

    This is the expected state at the end of Phase 1: machines arrive with the
    simulator in Phase 2.
    """
    result = await ListMachines(unit_of_work_factory=uow_factory).execute()

    assert result == []


async def test_summarises_each_machine_with_its_latest_state(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """A summary carries the newest reading and prediction, not the oldest."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M001"))
        await uow.telemetry.add_many_idempotent(
            [
                make_telemetry(event_id="old", recorded_at=DEFAULT_NOW - timedelta(hours=2)),
                make_telemetry(
                    event_id="new",
                    recorded_at=DEFAULT_NOW,
                    reading=make_reading(vibration=2.3),
                ),
            ]
        )
        await uow.predictions.add(make_prediction(predicted_at=DEFAULT_NOW, probability=0.85))

    summaries = await ListMachines(unit_of_work_factory=uow_factory).execute()

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.machine.id.value == "M001"
    assert summary.latest_reading is not None
    assert summary.latest_reading.event_id == "new"
    assert summary.latest_reading.reading.vibration == 2.3
    assert summary.risk_level is RiskLevel.CRITICAL
    assert summary.is_reporting


async def test_machine_without_telemetry_is_not_reporting(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """A registered but silent machine has neither reading nor risk.

    Reported as `None` rather than zeros, so the UI can distinguish "no data"
    from "measured zero".
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M002"))

    [summary] = await ListMachines(unit_of_work_factory=uow_factory).execute()

    assert summary.latest_reading is None
    assert summary.latest_prediction is None
    assert summary.risk_level is None
    assert not summary.is_reporting
    assert summary.open_incident_count == 0


async def test_counts_only_open_incidents(uow_factory: InMemoryUnitOfWorkFactory) -> None:
    """Resolved and dismissed incidents are excluded from the open count."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M001"))

        open_incident = make_incident(incident_id="open")
        acknowledged = make_incident(incident_id="ack")
        acknowledged.acknowledge()
        resolved = make_incident(incident_id="resolved")
        resolved.resolve()
        dismissed = make_incident(incident_id="dismissed")
        dismissed.dismiss()

        for incident in (open_incident, acknowledged, resolved, dismissed):
            await uow.incidents.add(incident)

    [summary] = await ListMachines(unit_of_work_factory=uow_factory).execute()

    assert summary.open_incident_count == 2


async def test_returns_every_machine_ordered_by_identifier(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The fleet is listed deterministically."""
    async with uow_factory() as uow:
        for machine_id in ("M003", "M001", "M002"):
            await uow.machines.add(make_machine(machine_id))

    summaries = await ListMachines(unit_of_work_factory=uow_factory).execute()

    assert [summary.machine.id.value for summary in summaries] == ["M001", "M002", "M003"]


async def test_incidents_for_other_machines_are_not_counted(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """A summary counts only its own machine's incidents."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M001"))
        await uow.machines.add(make_machine("M002"))
        await uow.incidents.add(make_incident(machine_id="M002", incident_id="other"))

    summaries = {
        summary.machine.id.value: summary
        for summary in await ListMachines(unit_of_work_factory=uow_factory).execute()
    }

    assert summaries["M001"].open_incident_count == 0
    assert summaries["M002"].open_incident_count == 1
