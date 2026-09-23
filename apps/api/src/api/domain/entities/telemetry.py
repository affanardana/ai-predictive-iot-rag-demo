"""Telemetry record entity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from api.domain.timestamps import ensure_aware
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.sensor_reading import SensorReading


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """One telemetry sample published by a machine.

    Immutable: a measurement, once taken, is a historical fact.

    `event_id` is the idempotency key. The transport layer may deliver the
    same MQTT message more than once, and the non-functional requirements
    forbid that from producing duplicate rows, so this identifier is carried
    end to end and enforced by a unique constraint in the database.
    """

    event_id: str
    machine_id: MachineId
    recorded_at: datetime
    reading: SensorReading
    simulation_session_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the event identifier and timestamp."""
        if not self.event_id.strip():
            raise ValueError("Telemetry event_id must not be blank.")
        ensure_aware(self.recorded_at, "recorded_at")
