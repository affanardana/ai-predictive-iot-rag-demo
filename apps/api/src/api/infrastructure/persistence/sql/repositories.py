"""SQL repository implementations.

Every method wraps its work in `translating_persistence_errors()`, so a driver
exception becomes a domain `PersistenceError` before it leaves this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, func, select, type_coerce
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql import ColumnElement

from api.domain.entities.incident import Incident
from api.domain.entities.machine import Machine
from api.domain.entities.prediction import Prediction
from api.domain.entities.telemetry import TelemetryRecord
from api.domain.errors import IncidentNotFoundError
from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.risk_level import IncidentSeverity
from api.domain.value_objects.sensor_reading import SensorReading
from api.domain.value_objects.time_window import Aggregation
from api.infrastructure.error_translation import translating_persistence_errors
from api.infrastructure.persistence.sql.mappers import (
    as_aware,
    incident_from_model,
    incident_to_model,
    machine_from_model,
    machine_to_model,
    prediction_from_model,
    prediction_to_model,
    telemetry_from_model,
)
from api.infrastructure.persistence.sql.models import (
    IncidentModel,
    MachineModel,
    PredictionModel,
    TelemetryModel,
)

#: Dialects whose `INSERT ... ON CONFLICT ... RETURNING` and epoch bucketing are
#: implemented here. Both are exercised: PostgreSQL by production and the
#: postgres test tier, SQLite by the default offline test run.
SUPPORTED_DIALECTS = frozenset({"postgresql", "sqlite"})

_UNKNOWN_DIALECT_MESSAGE = (
    "Bucketed telemetry aggregation is implemented for PostgreSQL and SQLite "
    "only; got dialect '{dialect}'. Add an expression for the new dialect "
    "rather than aggregating in Python, which would transfer the whole window."
)


def _bucket_expression(
    dialect_name: str,
    column: InstrumentedAttribute[datetime],
    bucket_seconds: int,
) -> ColumnElement[datetime]:
    """Floor a timestamp onto a fixed bucket grid aligned to the Unix epoch.

    Expressed as integer arithmetic on the epoch rather than `date_trunc`,
    because the bucket sizes this system produces -- 2, 14, and 60 minutes for
    the 24h, 7d, and 30d windows -- are not all units `date_trunc` accepts.

    The two branches are written out rather than factored into a shared
    "seconds since epoch" helper, because PostgreSQL and SQLite have no common
    spelling for it and a shared helper would have to be typed as returning an
    opaque expression.
    """
    if bucket_seconds < 1:
        raise ValueError("Bucket size must be at least one second.")

    if dialect_name == "postgresql":
        postgres_epoch = func.extract("epoch", column)
        postgres_floored = func.floor(postgres_epoch / bucket_seconds) * bucket_seconds
        return type_coerce(func.to_timestamp(postgres_floored), DateTime(timezone=True))

    if dialect_name == "sqlite":
        # Coerced to Integer so the division truncates rather than producing a
        # float, which is what makes the floor exact. The result is a naive UTC
        # string that `as_aware` re-attaches UTC to on the way out.
        sqlite_epoch = type_coerce(func.strftime("%s", column), Integer)
        sqlite_floored = func.floor(sqlite_epoch / bucket_seconds) * bucket_seconds
        return type_coerce(func.datetime(sqlite_floored, "unixepoch"), DateTime)

    raise NotImplementedError(_UNKNOWN_DIALECT_MESSAGE.format(dialect=dialect_name))


def _aggregator(
    aggregation: Aggregation,
) -> Callable[..., Any]:
    """Return the SQL function implementing `aggregation`."""
    mapping: dict[Aggregation, Callable[..., Any]] = {
        Aggregation.MEAN: func.avg,
        Aggregation.MIN: func.min,
        Aggregation.MAX: func.max,
    }
    try:
        return mapping[aggregation]
    except KeyError:
        raise ValueError(f"Aggregation '{aggregation}' cannot be applied to a bucket.") from None


class SqlMachineRepository:
    """Machine registry backed by the operational database."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_all(self) -> Sequence[Machine]:
        """Return every registered machine, ordered by identifier."""
        with translating_persistence_errors():
            result = await self._session.execute(
                select(MachineModel).order_by(MachineModel.machine_id)
            )
            return [machine_from_model(model) for model in result.scalars().all()]

    async def get(self, machine_id: MachineId) -> Machine | None:
        """Return one machine, or None if it is not registered."""
        with translating_persistence_errors():
            model = await self._session.get(MachineModel, machine_id.value)
            return machine_from_model(model) if model is not None else None

    async def add(self, machine: Machine) -> None:
        """Register a machine."""
        with translating_persistence_errors():
            self._session.add(machine_to_model(machine))
            await self._session.flush()


class SqlTelemetryRepository:
    """Telemetry store backed by the operational database."""

    def __init__(self, session: AsyncSession, dialect_name: str) -> None:
        self._session = session
        self._dialect_name = dialect_name

    async def add_many_idempotent(self, records: Sequence[TelemetryRecord]) -> int:
        """Insert records, skipping any whose `event_id` already exists.

        Redelivery is an expected event rather than an error, so conflicting
        rows are dropped silently and the count of genuinely new records is
        returned. `RETURNING` is used instead of a row count because both
        supported dialects return rows only for inserts that actually happened.
        """
        if not records:
            return 0

        if self._dialect_name not in SUPPORTED_DIALECTS:
            raise NotImplementedError(_UNKNOWN_DIALECT_MESSAGE.format(dialect=self._dialect_name))

        insert_statement = (
            postgresql_insert if self._dialect_name == "postgresql" else sqlite_insert
        )
        values = [
            {
                "event_id": record.event_id,
                "machine_id": record.machine_id.value,
                "recorded_at": record.recorded_at,
                "temperature": record.reading.temperature,
                "vibration": record.reading.vibration,
                "rpm": record.reading.rpm,
                "current": record.reading.current,
                "load": record.reading.load,
                "voltage": record.reading.voltage,
                "simulation_session_id": record.simulation_session_id,
            }
            for record in records
        ]

        statement = (
            insert_statement(TelemetryModel)
            .values(values)
            .on_conflict_do_nothing(index_elements=["event_id"])
            .returning(TelemetryModel.event_id)
        )

        with translating_persistence_errors():
            result = await self._session.execute(statement)
            return len(result.scalars().all())

    async def latest_for(self, machine_id: MachineId) -> TelemetryRecord | None:
        """Return the most recent record for a machine."""
        with translating_persistence_errors():
            result = await self._session.execute(
                select(TelemetryModel)
                .where(TelemetryModel.machine_id == machine_id.value)
                .order_by(TelemetryModel.recorded_at.desc())
                .limit(1)
            )
            model = result.scalars().first()
            return telemetry_from_model(model) if model is not None else None

    async def latest_records(self, machine_id: MachineId, limit: int) -> Sequence[TelemetryRecord]:
        """Return the most recent `limit` measurements, oldest first."""
        with translating_persistence_errors():
            # Newest-first so the LIMIT keeps the most recent rows, then
            # reversed into the chronological order the port promises.
            #
            # `event_id` breaks ties so the result is deterministic. Two rows
            # sharing a `recorded_at` are possible, and without a tie-break
            # which one falls outside the limit would depend on the plan.
            result = await self._session.execute(
                select(TelemetryModel)
                .where(TelemetryModel.machine_id == machine_id.value)
                .order_by(TelemetryModel.recorded_at.desc(), TelemetryModel.event_id.desc())
                .limit(limit)
            )
            models = list(result.scalars().all())

        models.reverse()
        return [telemetry_from_model(model) for model in models]

    async def count_for(self, machine_id: MachineId) -> int:
        """Return how many measurements are stored for a machine."""
        with translating_persistence_errors():
            result = await self._session.execute(
                # A count over the existing `ix_telemetry_machine_id_recorded_at`
                # index, so it does not read the rows themselves.
                select(func.count())
                .select_from(TelemetryModel)
                .where(TelemetryModel.machine_id == machine_id.value)
            )
            return int(result.scalar_one())

    async def window_raw(
        self,
        machine_id: MachineId,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> Sequence[TelemetryRecord]:
        """Return measurements in `[start, end]`, oldest first."""
        with translating_persistence_errors():
            # Ordered newest-first so the LIMIT keeps the most recent points,
            # then reversed into the chronological order the port promises.
            result = await self._session.execute(
                select(TelemetryModel)
                .where(
                    TelemetryModel.machine_id == machine_id.value,
                    TelemetryModel.recorded_at >= start,
                    TelemetryModel.recorded_at <= end,
                )
                .order_by(TelemetryModel.recorded_at.desc())
                .limit(limit)
            )
            models = list(result.scalars().all())

        models.reverse()
        return [telemetry_from_model(model) for model in models]

    async def window_bucketed(
        self,
        machine_id: MachineId,
        start: datetime,
        end: datetime,
        bucket_seconds: int,
        aggregation: Aggregation,
    ) -> Sequence[tuple[datetime, tuple[float, ...], int]]:
        """Aggregate measurements into fixed buckets, oldest first."""
        bucket = _bucket_expression(
            self._dialect_name, TelemetryModel.recorded_at, bucket_seconds
        ).label("bucket_start")

        signal_names = SensorReading.signal_names()
        aggregate = _aggregator(aggregation)

        statement = (
            select(
                bucket,
                *[aggregate(getattr(TelemetryModel, name)).label(name) for name in signal_names],
                func.count().label("sample_count"),
            )
            .where(
                TelemetryModel.machine_id == machine_id.value,
                TelemetryModel.recorded_at >= start,
                TelemetryModel.recorded_at <= end,
            )
            .group_by(bucket)
            .order_by(bucket)
        )

        with translating_persistence_errors():
            result = await self._session.execute(statement)
            rows = result.all()

        return [
            (
                as_aware(row.bucket_start),
                tuple(float(getattr(row, name)) for name in signal_names),
                int(row.sample_count),
            )
            for row in rows
        ]


class SqlPredictionRepository:
    """Prediction store backed by the operational database."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, prediction: Prediction) -> None:
        """Persist a prediction."""
        with translating_persistence_errors():
            self._session.add(prediction_to_model(prediction))
            await self._session.flush()

    async def latest_for(self, machine_id: MachineId) -> Prediction | None:
        """Return the most recent prediction for a machine."""
        with translating_persistence_errors():
            result = await self._session.execute(
                select(PredictionModel)
                .where(PredictionModel.machine_id == machine_id.value)
                .order_by(PredictionModel.predicted_at.desc())
                .limit(1)
            )
            model = result.scalars().first()
            return prediction_from_model(model) if model is not None else None

    async def history_for(self, machine_id: MachineId, limit: int) -> Sequence[Prediction]:
        """Return recent predictions for a machine, most recent first."""
        with translating_persistence_errors():
            result = await self._session.execute(
                select(PredictionModel)
                .where(PredictionModel.machine_id == machine_id.value)
                .order_by(PredictionModel.predicted_at.desc())
                .limit(limit)
            )
            return [prediction_from_model(model) for model in result.scalars().all()]


class SqlIncidentRepository:
    """Incident store backed by the operational database."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, incident: Incident) -> None:
        """Persist a new incident."""
        with translating_persistence_errors():
            self._session.add(incident_to_model(incident))
            await self._session.flush()

    async def get(self, incident_id: str) -> Incident | None:
        """Return one incident, or None if unknown."""
        with translating_persistence_errors():
            model = await self._session.get(IncidentModel, incident_id)
            return incident_from_model(model) if model is not None else None

    async def update(self, incident: Incident) -> None:
        """Persist changes to an existing incident.

        The lifecycle currently only changes `status`, but the full set of
        mutable columns is written so a later rule that revises severity or type
        cannot silently fail to persist.

        Raises:
            IncidentNotFoundError: if no row carries this identifier.
        """
        with translating_persistence_errors():
            model = await self._session.get(IncidentModel, incident.incident_id)
            if model is None:
                raise IncidentNotFoundError(incident.incident_id)

            replacement = incident_to_model(incident)
            model.status = replacement.status
            model.severity = replacement.severity
            model.incident_type = replacement.incident_type
            model.failure_probability = replacement.failure_probability
            model.prediction_id = replacement.prediction_id
            await self._session.flush()

    async def list_for_machine(
        self,
        machine_id: MachineId,
        limit: int,
    ) -> Sequence[Incident]:
        """Return incidents for a machine, most recent first."""
        with translating_persistence_errors():
            result = await self._session.execute(
                select(IncidentModel)
                .where(IncidentModel.machine_id == machine_id.value)
                .order_by(IncidentModel.detected_at.desc())
                .limit(limit)
            )
            return [incident_from_model(model) for model in result.scalars().all()]

    async def list_filtered(
        self,
        status: IncidentStatus | None,
        severity: IncidentSeverity | None,
        limit: int,
    ) -> Sequence[Incident]:
        """Return incidents, optionally filtered, most recent first."""
        statement = select(IncidentModel)
        if status is not None:
            statement = statement.where(IncidentModel.status == status.value)
        if severity is not None:
            statement = statement.where(IncidentModel.severity == severity.value)
        statement = statement.order_by(IncidentModel.detected_at.desc()).limit(limit)

        with translating_persistence_errors():
            result = await self._session.execute(statement)
            return [incident_from_model(model) for model in result.scalars().all()]
