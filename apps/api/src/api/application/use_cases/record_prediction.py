"""Use case: ask the model about a machine, and record what it said."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from api.application.summaries import OPEN_INCIDENT_SCAN_LIMIT
from api.domain.entities.incident import Incident
from api.domain.entities.prediction import PREDICTION_WINDOW_READINGS, Prediction
from api.domain.errors import InsufficientTelemetryHistoryError, MachineNotFoundError
from api.domain.ports.clock import Clock
from api.domain.ports.events import EventKind, EventPublisher, MachineEvent
from api.domain.ports.predictor import ModelOutput, Predictor
from api.domain.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
from api.domain.services.incident_policy import IncidentPolicy
from api.domain.services.risk_level_classifier import RiskLevelClassifier
from api.domain.value_objects.failure_probability import FailureProbability
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.sensor_reading import SensorReading


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """A scoring outcome: what the model said, and the incident it caused.

    The incident is reported rather than left for the caller to look up,
    because the distinction that matters downstream is between "scored, nothing
    changed" and "scored, and this machine now has an incident" -- and only one
    place knows which. `IngestResult` and `RegistrationResult` already have
    this shape.
    """

    prediction: Prediction
    incident: Incident | None = None


@dataclass(frozen=True, slots=True)
class RecordPrediction:
    """Score a machine's stored telemetry and persist the outcome.

    The only place that knows both what the model produces and what the product
    makes of it. The model returns a probability; the risk bands are the
    application's, applied here through `RiskLevelClassifier`, and so is the
    rule about when that warrants an incident, applied through `IncidentPolicy`.
    """

    unit_of_work_factory: UnitOfWorkFactory
    predictor: Predictor
    clock: Clock
    classifier: RiskLevelClassifier
    incident_policy: IncidentPolicy
    events: EventPublisher

    async def execute(self, machine_id: MachineId) -> PredictionResult:
        """Score one machine, recording a prediction and any incident.

        Raises:
            MachineNotFoundError: if the machine is not registered.
            InsufficientTelemetryHistoryError: if too little history is stored.
            PredictionUnavailableError: if inference could not be performed.
        """
        readings = await self._read_window(machine_id)

        # Deliberately outside any transaction. Inference is a network call to
        # another service, and holding a database connection open across it
        # would tie up a pooled connection for the duration of someone else's
        # latency.
        output = await self.predictor.predict(readings)

        result = await self._persist(machine_id, output, readings[-1])

        # After the transaction has committed. A subscriber told about a
        # prediction that then rolled back would refetch and find nothing,
        # which reads as a bug in the dashboard rather than in the write.
        await self.events.publish(
            MachineEvent(kind=EventKind.PREDICTION_RECORDED, machine_id=machine_id)
        )
        if result.incident is not None:
            await self.events.publish(
                MachineEvent(kind=EventKind.INCIDENT_RAISED, machine_id=machine_id)
            )
        return result

    async def _read_window(self, machine_id: MachineId) -> Sequence[SensorReading]:
        """Return the readings to score, or report how short the history fell.

        The window comes from stored telemetry rather than from the caller.
        Readings arrive over MQTT and are persisted before anything scores
        them, so by the time a prediction is asked for the data is already
        here -- and reading order, which the model is sensitive to, becomes the
        API's responsibility instead of a caller's.
        """
        async with self.unit_of_work_factory() as uow:
            if await uow.machines.get(machine_id) is None:
                raise MachineNotFoundError(str(machine_id))
            records = await uow.telemetry.latest_records(machine_id, PREDICTION_WINDOW_READINGS)

        if len(records) < PREDICTION_WINDOW_READINGS:
            raise InsufficientTelemetryHistoryError(
                have=len(records), need=PREDICTION_WINDOW_READINGS
            )
        return [record.reading for record in records]

    async def _persist(
        self,
        machine_id: MachineId,
        output: ModelOutput,
        latest: SensorReading,
    ) -> PredictionResult:
        """Store the prediction, and any incident it raises, in one transaction."""
        # `FailureProbability` clamps and rejects NaN, so a model that returned
        # nonsense fails here rather than reaching the CHECK constraint.
        probability = FailureProbability(output.failure_probability)
        risk_level = self.classifier.classify(probability)
        prediction = Prediction.create(
            machine_id=machine_id,
            predicted_at=self.clock.now(),
            probability=probability,
            risk_level=risk_level,
            model_version=output.model_version,
        )

        async with self.unit_of_work_factory() as uow:
            # One transaction for both. The unit-of-work contract names this
            # case directly: a prediction and the incident it raises either
            # commit together or not at all.
            await uow.predictions.add(prediction)
            incident = await self._incident_for(uow, prediction, latest)
            if incident is not None:
                await uow.incidents.add(incident)

        return PredictionResult(prediction=prediction, incident=incident)

    async def _incident_for(
        self,
        uow: UnitOfWork,
        prediction: Prediction,
        latest: SensorReading,
    ) -> Incident | None:
        """Build an incident, unless one is already open for this machine."""
        risk_level = prediction.risk_level
        if not self.incident_policy.should_raise(risk_level):
            return None

        # Suppressed while an incident is already open. `should_raise` is true
        # for every HIGH or CRITICAL scoring, and a degrading machine *stays*
        # in those bands, so without this a single run would file an incident
        # per reading -- hundreds of them, and the incident list would stop
        # meaning anything.
        #
        # The scan is bounded, so an open incident buried past the limit would
        # not suppress. That is the same bound `summaries.py` already accepts
        # for the same query, and a machine with that many incidents has a
        # problem a duplicate would not make worse.
        existing = await uow.incidents.list_for_machine(
            prediction.machine_id, limit=OPEN_INCIDENT_SCAN_LIMIT
        )
        if any(incident.is_open for incident in existing):
            return None

        return Incident.create(
            machine_id=prediction.machine_id,
            incident_type=self.incident_policy.type_for(latest),
            severity=self.incident_policy.severity_for(risk_level),
            probability=prediction.probability,
            detected_at=self.clock.now(),
            prediction_id=prediction.prediction_id,
        )
