"""Machine monitoring endpoints."""

from __future__ import annotations

from datetime import timedelta

from httpx import AsyncClient

from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.factories import (
    DEFAULT_NOW,
    make_incident,
    make_machine,
    make_prediction,
    make_reading,
    make_telemetry,
)


async def test_machine_list_is_empty_before_any_machines_exist(
    client: AsyncClient,
) -> None:
    """An empty fleet is a 200 with an empty list, not a 404.

    This is the expected state at the end of Phase 1: the simulator that
    registers machines arrives in Phase 2.
    """
    response = await client.get("/api/v1/machines")

    assert response.status_code == 200
    assert response.json() == []


async def test_machine_list_returns_current_state(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Each machine carries its latest reading and prediction."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003", name="Compressor motor"))
        await uow.telemetry.add_many_idempotent(
            [
                make_telemetry(
                    event_id="evt-1",
                    machine_id="M003",
                    recorded_at=DEFAULT_NOW,
                    reading=make_reading(vibration=2.3),
                )
            ]
        )
        await uow.predictions.add(make_prediction(machine_id="M003", probability=0.81))

    response = await client.get("/api/v1/machines")

    assert response.status_code == 200
    [machine] = response.json()
    assert machine["machine"]["machine_id"] == "M003"
    assert machine["machine"]["name"] == "Compressor motor"
    assert machine["latest_telemetry"]["reading"]["vibration"] == 2.3
    assert machine["latest_prediction"]["failure_probability"] == 0.81
    assert machine["risk_level"] == "CRITICAL"
    assert machine["is_reporting"] is True


async def test_machine_list_handles_a_machine_with_no_data(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """Missing data is null, not zero.

    Reporting 0.0 would be indistinguishable from a genuine measurement of
    zero, which for a vibration reading means something quite different.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M001"))

    [machine] = (await client.get("/api/v1/machines")).json()

    assert machine["latest_telemetry"] is None
    assert machine["latest_prediction"] is None
    assert machine["risk_level"] is None
    assert machine["is_reporting"] is False


async def test_machine_detail_returns_summary_and_incidents(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """The detail view consolidates state and history."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.incidents.add(make_incident(machine_id="M003", incident_id="inc-1"))

    response = await client.get("/api/v1/machines/M003")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["machine"]["machine_id"] == "M003"
    assert [incident["incident_id"] for incident in body["recent_incidents"]] == ["inc-1"]


async def test_unknown_machine_returns_a_structured_404(client: AsyncClient) -> None:
    """A missing machine produces the shared error envelope."""
    response = await client.get("/api/v1/machines/M999")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "machine_not_found"
    assert "M999" in error["message"]


async def test_malformed_machine_identifier_returns_422(client: AsyncClient) -> None:
    """An identifier failing the domain pattern is a validation failure."""
    response = await client.get("/api/v1/machines/not-a-machine-id")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_value"


async def test_telemetry_returns_a_raw_series_for_a_short_window(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """A one-hour window is unbucketed and labelled as such."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.telemetry.add_many_idempotent(
            [
                make_telemetry(
                    event_id=f"evt-{minutes}",
                    machine_id="M003",
                    recorded_at=DEFAULT_NOW - timedelta(minutes=minutes),
                )
                for minutes in (0, 10)
            ]
        )

    response = await client.get("/api/v1/machines/M003/telemetry?window=1h")

    assert response.status_code == 200
    body = response.json()
    assert body["window"] == "1h"
    assert body["resolution"]["is_raw"] is True
    assert body["resolution"]["bucket_seconds"] is None
    assert body["resolution"]["aggregation"] == "RAW"
    assert len(body["points"]) == 2


async def test_telemetry_states_its_resolution_for_a_long_window(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """An aggregated response says how it was reduced.

    A consumer must be able to tell a measurement from an aggregate, so the
    resolution is part of the contract rather than implicit.
    """
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.telemetry.add_many_idempotent(
            [make_telemetry(event_id="evt-1", machine_id="M003", recorded_at=DEFAULT_NOW)]
        )

    body = (await client.get("/api/v1/machines/M003/telemetry?window=30d")).json()

    assert body["resolution"]["is_raw"] is False
    assert body["resolution"]["bucket_seconds"] == 3600
    assert body["resolution"]["aggregation"] == "MEAN"
    assert body["points"][0]["sample_count"] == 1


async def test_telemetry_rejects_an_unsupported_window(client: AsyncClient) -> None:
    """Only the PRD's windows are selectable."""
    response = await client.get("/api/v1/machines/M003/telemetry?window=42h")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_failed"


async def test_prediction_history_is_returned(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """The prediction history endpoint reports stored predictions."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.predictions.add(
            make_prediction(machine_id="M003", prediction_id="pred-1", probability=0.81)
        )

    response = await client.get("/api/v1/machines/M003/predictions")

    assert response.status_code == 200
    [prediction] = response.json()
    assert prediction["prediction_id"] == "pred-1"
    assert prediction["failure_probability"] == 0.81
    assert prediction["risk_level"] == "CRITICAL"
    assert prediction["horizon_seconds"] == 3600
    assert prediction["model_version"] == "lstm-v1"


async def test_machine_incidents_are_returned(
    client: AsyncClient, uow_factory: InMemoryUnitOfWorkFactory
) -> None:
    """The per-machine incident endpoint is scoped to that machine."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.incidents.add(make_incident(machine_id="M003", incident_id="inc-1"))

    response = await client.get("/api/v1/machines/M003/incidents")

    assert response.status_code == 200
    [incident] = response.json()
    assert incident["incident_id"] == "inc-1"
    assert incident["status"] == "OPEN"
    assert incident["incident_type"] == "UNCLASSIFIED"
