"""Changing an incident's lifecycle status over HTTP."""

from __future__ import annotations

from httpx import AsyncClient

from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.factories import make_incident, make_machine


async def _seed(uow_factory: InMemoryUnitOfWorkFactory, incident_id: str = "inc-1") -> None:
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M001"))
        await uow.incidents.add(make_incident(machine_id="M001", incident_id=incident_id))


async def test_acknowledges_an_open_incident(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """The transition is applied and the updated incident returned."""
    await _seed(uow_factory)

    response = await client.patch("/api/v1/incidents/inc-1", json={"status": "ACKNOWLEDGED"})

    assert response.status_code == 200
    assert response.json()["status"] == "ACKNOWLEDGED"


async def test_the_change_is_visible_to_a_later_read(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """The status is persisted, not merely echoed back.

    A handler that returned the requested status without writing it would pass
    the test above and lose the operator's decision.
    """
    await _seed(uow_factory)

    await client.patch("/api/v1/incidents/inc-1", json={"status": "ACKNOWLEDGED"})

    listed = await client.get("/api/v1/incidents?status=ACKNOWLEDGED")
    assert [incident["incident_id"] for incident in listed.json()] == ["inc-1"]


async def test_an_illegal_transition_is_a_conflict(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """A resolved incident cannot be reopened.

    This is the assertion that finally exercises `invalid_incident_transition`,
    which has been in the error catalog since the lifecycle was written and was
    unreachable over HTTP until this route existed. 409 rather than 422 because
    the request is well formed -- it is the incident's current state that
    refuses it, and a client that retried the same body later might succeed.
    """
    await _seed(uow_factory)
    await client.patch("/api/v1/incidents/inc-1", json={"status": "RESOLVED"})

    response = await client.patch("/api/v1/incidents/inc-1", json={"status": "ACKNOWLEDGED"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "invalid_incident_transition"


async def test_open_is_not_a_valid_target(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Nothing transitions back to OPEN, so asking for it is a conflict.

    Deliberately a 409 and not a 422, even though it reads like a validation
    error: `OPEN` is a real value of the enum and the body is well formed. The
    lifecycle simply has no edge leading back to it -- reopening is modelled as
    a new incident so the audit trail stays append-only.
    """
    await _seed(uow_factory)

    response = await client.patch("/api/v1/incidents/inc-1", json={"status": "OPEN"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "invalid_incident_transition"


async def test_repeating_a_transition_is_a_conflict(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Acknowledging twice conflicts rather than succeeding quietly.

    The transition table has no self-transitions. That is the right answer for
    a double-clicked button: the second request means someone else acted on the
    incident in between, and reporting success would hide it.
    """
    await _seed(uow_factory)
    await client.patch("/api/v1/incidents/inc-1", json={"status": "ACKNOWLEDGED"})

    response = await client.patch("/api/v1/incidents/inc-1", json={"status": "ACKNOWLEDGED"})

    assert response.status_code == 409


async def test_an_unknown_incident_is_not_found(client: AsyncClient) -> None:
    """An unknown identifier is a 404, distinguishable from an illegal move."""
    response = await client.patch("/api/v1/incidents/missing", json={"status": "ACKNOWLEDGED"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "incident_not_found"


async def test_an_unknown_status_is_rejected(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """A status outside the enum is a 422 from the framework, not a 409."""
    await _seed(uow_factory)

    response = await client.patch("/api/v1/incidents/inc-1", json={"status": "ESCALATED"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_failed"


async def test_the_fleet_open_count_follows_the_status(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Resolving an incident drops it from the fleet's open count.

    The number a dashboard actually shows, and the one place the new route
    meets the new bulk count query. `is_open` covers OPEN and ACKNOWLEDGED
    alike, so acknowledging must leave the count alone while dismissing must
    not -- and the two are asserted together, because a count that moved on
    both would pass either check on its own.
    """
    await _seed(uow_factory)

    async def open_count() -> int:
        fleet = await client.get("/api/v1/machines")
        [machine] = fleet.json()
        return int(machine["open_incident_count"])

    assert await open_count() == 1

    await client.patch("/api/v1/incidents/inc-1", json={"status": "ACKNOWLEDGED"})
    assert await open_count() == 1, "an acknowledged incident still needs attention"

    await client.patch("/api/v1/incidents/inc-1", json={"status": "DISMISSED"})
    assert await open_count() == 0
