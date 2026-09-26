"""Telemetry ingestion endpoint."""

from __future__ import annotations

from fastapi import APIRouter, status

from api.domain.entities.telemetry import TelemetryRecord
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.sensor_reading import SensorReading
from api.presentation.dependencies import IngestTelemetryDep, IngestTokenDep
from api.presentation.schemas.telemetry_ingest import (
    IngestResultSchema,
    TelemetryIngestItem,
    TelemetryIngestRequest,
)

router = APIRouter(prefix="/telemetry", tags=["telemetry"])


@router.post(
    "",
    response_model=IngestResultSchema,
    status_code=status.HTTP_201_CREATED,
    dependencies=[IngestTokenDep],
    summary="Ingest a batch of telemetry",
)
async def ingest_telemetry(
    payload: TelemetryIngestRequest,
    use_case: IngestTelemetryDep,
) -> IngestResultSchema:
    """Store readings, tolerating redelivery.

    A redelivered batch is not an error. It reports `accepted: 0` and a
    `duplicates` count equal to the batch size, because the transport is
    at-least-once and a retry is an expected event rather than a fault.
    """
    result = await use_case.execute([_to_record(item) for item in payload.records])
    return IngestResultSchema(
        accepted=result.accepted,
        duplicates=result.duplicates,
        ready=[machine_id.value for machine_id in result.ready],
    )


def _to_record(item: TelemetryIngestItem) -> TelemetryRecord:
    """Build a domain record from one wire item.

    The flat wire shape becomes the domain's nested one here, and `session_id`
    becomes `simulation_session_id`. Both renames happen at the boundary so
    neither side has to know the other's vocabulary.
    """
    return TelemetryRecord(
        event_id=item.event_id,
        machine_id=MachineId(item.machine_id),
        recorded_at=item.recorded_at,
        reading=SensorReading(
            temperature=item.temperature,
            vibration=item.vibration,
            rpm=item.rpm,
            current=item.current,
            load=item.load,
            voltage=item.voltage,
        ),
        simulation_session_id=item.session_id,
    )
