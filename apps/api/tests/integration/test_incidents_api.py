"""Fleet-wide incident endpoints."""

from __future__ import annotations

from datetime import timedelta

from httpx import AsyncClient

from api.domain.value_objects.risk_level import IncidentSeverity
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.factories import DEFAULT_NOW, make_incident, make_machine


async def _seed(uow_factory: InMemoryUnitOfWorkFactory) -> None:
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M001"))
        await uow.incidents.add(
            make_incident(
                machine_id="M001",
                incident_id="inc-critical",
                detected_at=DEFAULT_NOW,
                severity=IncidentSeverity.CRITICAL,
                probability=0.9,
            )
        )
        acknowledged = make_incident(
            machine_id="M001",
            incident_id="inc-ack",
            detected_at=DEFAULT_NOW - timedelta(hours=1),
            severity=IncidentSeverity.LOW,
        )
        acknowledged.acknowledge()
        await uow.incidents.add(acknowledged)


async def test_lists_incidents_across_the_fleet(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Every incident is returned, most recent first."""
    await _seed(uow_factory)

    response = await client.get("/api/v1/incidents")

    assert response.status_code == 200
    assert [incident["incident_id"] for incident in response.json()] == [
        "inc-critical",
        "inc-ack",
    ]


async def test_filters_by_status(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Status filtering narrows the result."""
    await _seed(uow_factory)

    response = await client.get("/api/v1/incidents?status=ACKNOWLEDGED")

    assert response.status_code == 200
    assert [incident["incident_id"] for incident in response.json()] == ["inc-ack"]


async def test_filters_by_severity(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Severity filtering narrows the result."""
    await _seed(uow_factory)

    response = await client.get("/api/v1/incidents?severity=CRITICAL")

    assert response.status_code == 200
    assert [incident["incident_id"] for incident in response.json()] == ["inc-critical"]


async def test_rejects_an_unknown_status(client: AsyncClient) -> None:
    """An unknown status is a 422, not an empty list.

    Silently returning nothing would look like "there are no such incidents",
    which is a different and misleading answer to a malformed question.
    """
    response = await client.get("/api/v1/incidents?status=ESCALATED")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_failed"


async def test_incident_payload_carries_the_documented_fields(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """The incident shape matches PRD section 10."""
    await _seed(uow_factory)

    [incident] = (await client.get("/api/v1/incidents?severity=CRITICAL")).json()

    assert set(incident) == {
        "incident_id",
        "machine_id",
        "incident_type",
        "severity",
        "failure_probability",
        "detected_at",
        "status",
        "prediction_id",
    }
