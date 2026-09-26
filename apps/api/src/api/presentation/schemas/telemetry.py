"""Telemetry response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from api.domain.value_objects.time_window import Aggregation, TimeWindow


class SensorReadingSchema(BaseModel):
    """The six telemetry signals."""

    model_config = ConfigDict(frozen=True)

    temperature: float = Field(description="Winding or housing temperature.")
    vibration: float = Field(description="Velocity RMS.")
    rpm: float = Field(description="Shaft speed.")
    current: float = Field(description="Motor current draw.")
    load: float = Field(description="Mechanical load.")
    voltage: float = Field(description="Supply voltage.")


class TelemetryRecordSchema(BaseModel):
    """A single stored telemetry record."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(description="Idempotency key from the transport layer.")
    machine_id: str
    recorded_at: datetime
    reading: SensorReadingSchema


class TelemetryPointSchema(BaseModel):
    """One point on a telemetry series."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    reading: SensorReadingSchema
    sample_count: int = Field(
        description="1 for a raw measurement, otherwise the number of readings combined.",
    )


class SeriesResolutionSchema(BaseModel):
    """How a returned series was reduced.

    Exposed because a consumer must be able to tell a measurement from an
    aggregate. The PRD's evidence model treats those differently, and hiding
    the distinction here would make it impossible to honour later.
    """

    model_config = ConfigDict(frozen=True)

    is_raw: bool
    bucket_seconds: int | None = Field(
        default=None,
        description="Bucket width in seconds, or null when the series is raw.",
    )
    aggregation: Aggregation


class ResolvedWindowSchema(BaseModel):
    """The instants a series actually covers.

    `window` names a duration; this says where that duration was applied. The
    two can differ, and a client that assumed otherwise would misplace the
    whole series on a time axis: the end is the newest stored reading whenever
    that leads the wall clock, which is exactly what happens while a simulator
    is being played back faster than real time.
    """

    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime = Field(
        description=(
            "Where the series ends. Later than the wall clock during a "
            "fast-forwarded simulation, because it follows the data."
        ),
    )


class TelemetrySeriesSchema(BaseModel):
    """A window of telemetry at a stated resolution."""

    model_config = ConfigDict(frozen=True)

    machine_id: str
    window: TimeWindow
    resolution: SeriesResolutionSchema
    interval: ResolvedWindowSchema
    points: list[TelemetryPointSchema]
