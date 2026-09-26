"""Telemetry ingestion, end to end through the API.

The payloads here are built in the exact shape the simulator publishes to MQTT,
so this is also where the wire agreement is pinned from the API's side. The
simulator's own half lives in `simulator.tests.integration.test_api_compatibility`.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from httpx import AsyncClient

from api.domain.entities.prediction import PREDICTION_WINDOW_READINGS
from api.domain.value_objects.machine_id import MachineId
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.factories import DEFAULT_NOW, make_machine

INGEST_URL = "/api/v1/telemetry"
SESSION_ID = "sim-bd-20260923"


def a_sample(index: int, machine_id: str = "M003") -> dict[str, object]:
    """One reading, in the flat shape `TelemetrySample.as_row()` produces."""
    return {
        "event_id": f"{SESSION_ID}-{machine_id}-{index:08d}",
        "machine_id": machine_id,
        "recorded_at": (DEFAULT_NOW + timedelta(minutes=index)).isoformat(),
        "session_id": SESSION_ID,
        "temperature": 61.5,
        "vibration": 1.42,
        "rpm": 1480.0,
        "current": 12.5,
        "load": 0.72,
        "voltage": 400.0,
    }


def a_batch(count: int, machine_id: str = "M003") -> list[dict[str, object]]:
    return [a_sample(index, machine_id) for index in range(count)]


async def registered(uow_factory: InMemoryUnitOfWorkFactory, machine_id: str = "M003") -> None:
    async with uow_factory() as uow:
        await uow.machines.add(make_machine(machine_id))


async def test_a_batch_is_stored_and_counted(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    await registered(uow_factory)

    response = await client.post(INGEST_URL, json={"records": a_batch(3)})

    assert response.status_code == 201
    assert response.json() == {"accepted": 3, "duplicates": 0, "ready": []}

    async with uow_factory() as uow:
        assert await uow.telemetry.count_for(MachineId("M003")) == 3


async def test_redelivering_a_batch_creates_no_duplicate_rows(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """The spec's central requirement, end to end.

    An at-least-once transport will deliver the same message twice. The second
    delivery is reported rather than rejected -- it is an expected event, not
    an error -- and no row is created for it.
    """
    await registered(uow_factory)
    batch = a_batch(PREDICTION_WINDOW_READINGS)

    first = await client.post(INGEST_URL, json={"records": batch})
    second = await client.post(INGEST_URL, json={"records": batch})

    assert first.json()["accepted"] == PREDICTION_WINDOW_READINGS
    assert second.status_code == 201
    assert second.json()["accepted"] == 0
    assert second.json()["duplicates"] == PREDICTION_WINDOW_READINGS

    async with uow_factory() as uow:
        assert await uow.telemetry.count_for(MachineId("M003")) == PREDICTION_WINDOW_READINGS


async def test_a_machine_becomes_ready_at_exactly_the_window_length(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """One reading short is not ready; the next one is.

    The boundary is the whole point of the field: the orchestrator scores what
    this names, and naming a machine one reading early would have the model
    refused by the very next call.
    """
    await registered(uow_factory)

    short = await client.post(INGEST_URL, json={"records": a_batch(PREDICTION_WINDOW_READINGS - 1)})
    assert short.json()["ready"] == []

    last = await client.post(
        INGEST_URL,
        json={"records": a_batch(PREDICTION_WINDOW_READINGS)[PREDICTION_WINDOW_READINGS - 1 :]},
    )

    assert last.json()["accepted"] == 1
    assert last.json()["ready"] == ["M003"]


async def test_a_redelivery_never_reports_readiness(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """A duplicate batch reports nothing ready, because nothing changed.

    Without the short circuit a redelivery would re-report a warm machine and
    the orchestrator would score it again for no new data.
    """
    await registered(uow_factory)
    batch = a_batch(PREDICTION_WINDOW_READINGS)
    await client.post(INGEST_URL, json={"records": batch})

    redelivered = await client.post(INGEST_URL, json={"records": batch})

    assert redelivered.json()["ready"] == []


async def test_readiness_is_sorted_rather_than_left_in_arrival_order(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Deterministic output, so a retry produces an identical response."""
    for machine_id in ("M003", "M001", "M002"):
        await registered(uow_factory, machine_id)

    records = []
    for machine_id in ("M003", "M001", "M002"):
        records.extend(a_batch(PREDICTION_WINDOW_READINGS, machine_id))

    response = await client.post(INGEST_URL, json={"records": records})

    assert response.json()["ready"] == ["M001", "M002", "M003"]


async def test_an_unregistered_machine_is_a_404_not_a_500(
    client: AsyncClient,
) -> None:
    """Named, so an operator can fix it.

    The foreign key would reject this too, but as an integrity error reported as
    an internal failure -- which says nothing about which machine, or that the
    fix is to register it.
    """
    response = await client.post(INGEST_URL, json={"records": a_batch(1, "M999")})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "machine_not_found"
    assert "M999" in response.json()["error"]["message"]


async def test_an_empty_batch_is_refused(client: AsyncClient) -> None:
    """A batch with nothing in it is a caller mistake, not a no-op."""
    response = await client.post(INGEST_URL, json={"records": []})

    assert response.status_code == 422


async def test_an_oversized_batch_is_refused_before_the_use_case(client: AsyncClient) -> None:
    """The schema bound fires first, so no list is allocated beyond it."""
    response = await client.post(INGEST_URL, json={"records": a_batch(501)})

    assert response.status_code == 422


async def test_an_over_long_event_id_is_refused(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Caught here, where SQLite would not catch it.

    The column is `String(64)` but SQLite ignores lengths, so an identifier this
    long would pass every test in the default tier and fail against PostgreSQL.
    """
    await registered(uow_factory)
    sample = a_sample(0)
    sample["event_id"] = "e" * 65

    response = await client.post(INGEST_URL, json={"records": [sample]})

    assert response.status_code == 422


@pytest.mark.parametrize(
    "missing",
    ["event_id", "machine_id", "recorded_at", "temperature", "voltage"],
)
async def test_a_malformed_reading_is_refused(client: AsyncClient, missing: str) -> None:
    """Every field the model depends on is required, not defaulted."""
    sample = a_sample(0)
    del sample[missing]

    response = await client.post(INGEST_URL, json={"records": [sample]})

    assert response.status_code == 422


async def test_a_session_id_is_optional(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Telemetry recorded outside a simulation has no session to belong to."""
    await registered(uow_factory)
    sample = a_sample(0)
    del sample["session_id"]

    response = await client.post(INGEST_URL, json={"records": [sample]})

    assert response.status_code == 201, response.text
    assert response.json()["accepted"] == 1
