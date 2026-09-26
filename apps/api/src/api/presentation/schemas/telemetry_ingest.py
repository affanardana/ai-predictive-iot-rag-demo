"""Telemetry ingestion schemas.

The wire format is not this API's to choose. It is the flat row the simulator
publishes to MQTT and n8n forwards, so the schema accepts it verbatim rather
than asking the transport to reshape it. Every *response* schema in
`schemas/telemetry.py` nests the six signals under `reading`; this one does not,
and the difference is deliberate.
"""

from __future__ import annotations

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

#: Column widths, restated rather than imported. The originals live in
#: `api.infrastructure.persistence.sql.models`, and the layer contract forbids
#: presentation from importing infrastructure -- for good reason, since a
#: schema's job is to describe the wire, not the table.
#:
#: Restating them matters because SQLite ignores `String(n)`: an over-long
#: `event_id` would pass every test in the default tier and fail against
#: PostgreSQL, where the column is real.
EVENT_ID_MAX_LENGTH = 64
SESSION_ID_MAX_LENGTH = 64
MACHINE_ID_MAX_LENGTH = 16

#: Matches `IngestTelemetry`'s own bound. Declared here as well so an oversized
#: payload is refused by the framework before a use case is constructed.
MAX_BATCH_RECORDS = 500


class TelemetryIngestItem(BaseModel):
    """One reading, as the simulator emits it.

    `session_id` carries the name the simulator uses; the domain and the column
    call the same thing `simulation_session_id`. Renaming happens here, at the
    boundary, rather than in either of them.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(
        min_length=1,
        max_length=EVENT_ID_MAX_LENGTH,
        description="Idempotency key. Redelivering one of these creates no second row.",
    )
    machine_id: str = Field(min_length=1, max_length=MACHINE_ID_MAX_LENGTH)
    recorded_at: AwareDatetime = Field(description="When the reading was taken, with a UTC offset.")
    session_id: str | None = Field(default=None, max_length=SESSION_ID_MAX_LENGTH)

    temperature: float
    vibration: float
    rpm: float
    current: float
    load: float
    voltage: float


class TelemetryIngestRequest(BaseModel):
    """A batch of readings from the transport."""

    records: list[TelemetryIngestItem] = Field(
        min_length=1,
        max_length=MAX_BATCH_RECORDS,
        description="Oldest first. A redelivered batch is accepted and reported as duplicates.",
    )


class IngestResultSchema(BaseModel):
    """What became of a batch.

    `ready` names the machines that now hold a full prediction window, which is
    what the orchestrator uses to decide when to ask for a prediction.
    """

    model_config = ConfigDict(frozen=True)

    accepted: int = Field(ge=0, description="Records stored by this call.")
    duplicates: int = Field(ge=0, description="Records dropped because their event_id existed.")
    ready: list[str] = Field(description="Machines holding a complete prediction window.")
