"""Registering machines, and the telemetry the registry unlocks."""

from __future__ import annotations

from datetime import timedelta

from httpx import AsyncClient

from api.domain.value_objects.machine_id import MachineId
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.factories import DEFAULT_NOW

MACHINES_URL = "/api/v1/machines"


def a_sample(machine_id: str = "M004") -> dict[str, object]:
    return {
        "event_id": f"sim-bd-20260923-{machine_id}-00000000",
        "machine_id": machine_id,
        "recorded_at": (DEFAULT_NOW + timedelta(minutes=1)).isoformat(),
        "temperature": 61.5,
        "vibration": 1.42,
        "rpm": 1480.0,
        "current": 12.5,
        "load": 0.72,
        "voltage": 400.0,
    }


async def test_a_machine_is_registered(client: AsyncClient) -> None:
    response = await client.post(MACHINES_URL, json={"machine_id": "M004", "name": "Pump 4"})

    assert response.status_code == 201
    assert response.json()["machine_id"] == "M004"
    assert response.json()["name"] == "Pump 4"
    assert response.json()["registered_at"]


async def test_registering_twice_is_idempotent(client: AsyncClient) -> None:
    """Rerunning the pipeline must not fail on a machine it already declared.

    200 rather than 201 on the second call, because the status is the only
    place a caller learns which happened.
    """
    first = await client.post(MACHINES_URL, json={"machine_id": "M004", "name": "Pump 4"})
    second = await client.post(MACHINES_URL, json={"machine_id": "M004", "name": "Pump 4"})

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["registered_at"] == first.json()["registered_at"], (
        "re-registering must not restamp the machine"
    )


async def test_a_blank_name_is_refused(client: AsyncClient) -> None:
    """`Machine` validates its timestamp but not its name, so the schema does."""
    response = await client.post(MACHINES_URL, json={"machine_id": "M004", "name": ""})

    assert response.status_code == 422


async def test_an_over_long_machine_id_is_refused(client: AsyncClient) -> None:
    response = await client.post(MACHINES_URL, json={"machine_id": "M" * 17, "name": "Pump"})

    assert response.status_code == 422


async def test_registration_unlocks_telemetry(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Registration is what makes a machine's telemetry acceptable.

    The point of the registry: telemetry for an unregistered machine is a 404,
    and registering it is what makes the same reading acceptable. This is the
    order the pipeline has to run in, and the reason a fresh deployment that
    skips registration sees every message rejected.
    """
    refused = await client.post("/api/v1/telemetry", json={"records": [a_sample()]})
    assert refused.status_code == 404

    await client.post(MACHINES_URL, json={"machine_id": "M004", "name": "Pump 4"})
    accepted = await client.post("/api/v1/telemetry", json={"records": [a_sample()]})

    assert accepted.status_code == 201
    assert accepted.json()["accepted"] == 1

    async with uow_factory() as uow:
        assert await uow.telemetry.count_for(MachineId("M004")) == 1


async def test_a_registered_machine_appears_in_the_fleet(
    client: AsyncClient,
) -> None:
    await client.post(MACHINES_URL, json={"machine_id": "M004", "name": "Pump 4"})

    fleet = await client.get(MACHINES_URL)

    assert [machine["machine"]["machine_id"] for machine in fleet.json()] == ["M004"]
    assert fleet.json()[0]["is_reporting"] is False
