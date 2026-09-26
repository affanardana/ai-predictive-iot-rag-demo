"""Scoring a machine: the window, the prediction, and any incident it raises.

The rule under test here is the one Phase 6 introduces: a machine that stays in
a high risk band raises *one* incident, not one per reading.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from api.application.use_cases import RecordPrediction
from api.domain.entities.prediction import PREDICTION_WINDOW_READINGS
from api.domain.errors import InsufficientTelemetryHistoryError, MachineNotFoundError
from api.domain.ports.predictor import ModelOutput
from api.domain.services.incident_policy import DefaultIncidentPolicy
from api.domain.services.risk_level_classifier import RiskLevelClassifier
from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.incident_type import IncidentType
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.risk_level import IncidentSeverity, RiskLevel
from api.domain.value_objects.risk_thresholds import RiskThresholds
from api.infrastructure.persistence.memory.store import InMemoryStore
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.factories import DEFAULT_NOW, make_machine, make_reading, make_telemetry
from tests.support.fakes import FixedClock, RecordingEventPublisher


class StubPredictor:
    """Returns a fixed probability, so the band is the only variable."""

    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.calls = 0

    async def predict(self, readings):
        self.calls += 1
        return ModelOutput(failure_probability=self.probability, model_version="run-test")


async def seed(
    factory: InMemoryUnitOfWorkFactory,
    count: int = PREDICTION_WINDOW_READINGS,
    machine_id: str = "M003",
) -> None:
    async with factory() as uow:
        if await uow.machines.get(MachineId(machine_id)) is None:
            await uow.machines.add(make_machine(machine_id))
        await uow.telemetry.add_many_idempotent(
            [
                make_telemetry(
                    event_id=f"{machine_id}-evt-{index:04d}",
                    machine_id=machine_id,
                    recorded_at=DEFAULT_NOW + timedelta(minutes=index),
                    reading=make_reading(),
                )
                for index in range(count)
            ]
        )


def a_use_case(
    factory: InMemoryUnitOfWorkFactory,
    probability: float,
) -> tuple[RecordPrediction, StubPredictor, RecordingEventPublisher]:
    predictor = StubPredictor(probability)
    events = RecordingEventPublisher()
    use_case = RecordPrediction(
        unit_of_work_factory=factory,
        predictor=predictor,
        clock=FixedClock(),
        classifier=RiskLevelClassifier(thresholds=RiskThresholds()),
        incident_policy=DefaultIncidentPolicy(),
        events=events,
    )
    return use_case, predictor, events


@pytest.fixture
def factory() -> InMemoryUnitOfWorkFactory:
    return InMemoryUnitOfWorkFactory(store=InMemoryStore())


async def test_a_high_risk_prediction_raises_an_incident(
    factory: InMemoryUnitOfWorkFactory,
) -> None:
    await seed(factory)
    use_case, _, _ = a_use_case(factory, probability=0.75)

    result = await use_case.execute(MachineId("M003"))

    async with factory() as uow:
        incidents = await uow.incidents.list_for_machine(MachineId("M003"), limit=10)

    assert result.prediction.risk_level is RiskLevel.HIGH
    assert len(incidents) == 1
    assert incidents[0].severity is IncidentSeverity.HIGH
    assert incidents[0].probability.value == pytest.approx(0.75)


async def test_the_incident_is_linked_to_the_prediction_that_raised_it(
    factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The incident points back at the prediction that raised it.

    AC-005 requires the association, and the column is `ON DELETE SET NULL`, so
    the link is a reference rather than an ownership.
    """
    await seed(factory)
    use_case, _, _ = a_use_case(factory, probability=0.95)

    result = await use_case.execute(MachineId("M003"))

    async with factory() as uow:
        incidents = await uow.incidents.list_for_machine(MachineId("M003"), limit=10)

    assert incidents[0].prediction_id == result.prediction.prediction_id


async def test_the_incident_is_unclassified(factory: InMemoryUnitOfWorkFactory) -> None:
    """The stub, asserted rather than assumed.

    The model is a binary failure classifier and cannot say what is failing, so
    this stays UNCLASSIFIED until `incident_type.py`'s options are chosen. A
    test that failed here would mean someone had guessed.
    """
    await seed(factory)
    use_case, _, _ = a_use_case(factory, probability=0.95)

    await use_case.execute(MachineId("M003"))

    async with factory() as uow:
        incidents = await uow.incidents.list_for_machine(MachineId("M003"), limit=10)

    assert incidents[0].incident_type is IncidentType.UNCLASSIFIED


@pytest.mark.parametrize("probability", [0.0, 0.29, 0.30, 0.59])
async def test_a_band_below_high_raises_no_incident(
    factory: InMemoryUnitOfWorkFactory, probability: float
) -> None:
    """The boundary `DefaultIncidentPolicy` documents.

    A warning is the system saying "watch this". Raising an incident for one
    would train operators to ignore the list.
    """
    await seed(factory)
    use_case, _, _ = a_use_case(factory, probability=probability)

    await use_case.execute(MachineId("M003"))

    async with factory() as uow:
        incidents = await uow.incidents.list_for_machine(MachineId("M003"), limit=10)

    assert incidents == []


@pytest.mark.parametrize("probability", [0.60, 0.79, 0.80, 1.0])
async def test_every_band_at_or_above_high_raises_one(
    factory: InMemoryUnitOfWorkFactory, probability: float
) -> None:
    await seed(factory)
    use_case, _, _ = a_use_case(factory, probability=probability)

    await use_case.execute(MachineId("M003"))

    async with factory() as uow:
        incidents = await uow.incidents.list_for_machine(MachineId("M003"), limit=10)

    assert len(incidents) == 1


async def test_scoring_again_while_an_incident_is_open_raises_nothing(
    factory: InMemoryUnitOfWorkFactory,
) -> None:
    """The rule Phase 6 adds, and the reason it exists.

    A degrading machine stays in the high band for the rest of the run. Without
    suppression, every reading past that point files another incident -- at one
    reading a minute, hundreds per demo, and the incident list stops meaning
    anything.
    """
    await seed(factory)
    use_case, _, _ = a_use_case(factory, probability=0.95)

    first = await use_case.execute(MachineId("M003"))
    second = await use_case.execute(MachineId("M003"))
    third = await use_case.execute(MachineId("M003"))

    async with factory() as uow:
        incidents = await uow.incidents.list_for_machine(MachineId("M003"), limit=10)

    ids = [result.prediction.prediction_id for result in (first, second, third)]
    assert len(set(ids)) == 3, "each scoring is its own prediction"
    assert len(incidents) == 1, "three predictions, one incident"


async def test_an_acknowledged_incident_still_suppresses(
    factory: InMemoryUnitOfWorkFactory,
) -> None:
    """Acknowledging says "seen", not "resolved".

    If acknowledgement reopened the door, an operator working through the list
    would generate a fresh incident for each one they touched.
    """
    await seed(factory)
    use_case, _, _ = a_use_case(factory, probability=0.95)
    await use_case.execute(MachineId("M003"))

    async with factory() as uow:
        incidents = await uow.incidents.list_for_machine(MachineId("M003"), limit=10)
        incidents[0].acknowledge()
        await uow.incidents.update(incidents[0])
        assert incidents[0].status is IncidentStatus.ACKNOWLEDGED

    await use_case.execute(MachineId("M003"))

    async with factory() as uow:
        after = await uow.incidents.list_for_machine(MachineId("M003"), limit=10)

    assert len(after) == 1


@pytest.mark.parametrize("terminal", [IncidentStatus.RESOLVED, IncidentStatus.DISMISSED])
async def test_a_closed_incident_does_not_suppress(
    factory: InMemoryUnitOfWorkFactory, terminal: IncidentStatus
) -> None:
    """Once nothing is open, a high reading is news again.

    A machine that was repaired and then degrades a second time is a second
    thing that happened, and the record should say so.
    """
    await seed(factory)
    use_case, _, _ = a_use_case(factory, probability=0.95)
    await use_case.execute(MachineId("M003"))

    async with factory() as uow:
        incidents = await uow.incidents.list_for_machine(MachineId("M003"), limit=10)
        incident = incidents[0]
        if terminal is IncidentStatus.RESOLVED:
            incident.resolve()
        else:
            incident.dismiss()
        await uow.incidents.update(incident)

    await use_case.execute(MachineId("M003"))

    async with factory() as uow:
        after = await uow.incidents.list_for_machine(MachineId("M003"), limit=10)

    assert len(after) == 2


async def test_suppression_is_per_machine(factory: InMemoryUnitOfWorkFactory) -> None:
    """One machine's open incident must not silence another's."""
    await seed(factory, machine_id="M003")
    await seed(factory, machine_id="M004")
    use_case, _, _ = a_use_case(factory, probability=0.95)

    await use_case.execute(MachineId("M003"))
    await use_case.execute(MachineId("M004"))

    async with factory() as uow:
        for machine_id in ("M003", "M004"):
            incidents = await uow.incidents.list_for_machine(MachineId(machine_id), limit=10)
            assert len(incidents) == 1, machine_id


async def test_a_short_history_is_refused_with_both_counts(
    factory: InMemoryUnitOfWorkFactory,
) -> None:
    await seed(factory, count=PREDICTION_WINDOW_READINGS - 1)
    use_case, predictor, _ = a_use_case(factory, probability=0.95)

    with pytest.raises(InsufficientTelemetryHistoryError) as raised:
        await use_case.execute(MachineId("M003"))

    assert raised.value.have == PREDICTION_WINDOW_READINGS - 1
    assert raised.value.need == PREDICTION_WINDOW_READINGS
    assert predictor.calls == 0, "no model call should be made for a window that cannot exist"


async def test_an_unknown_machine_is_refused_before_inference(
    factory: InMemoryUnitOfWorkFactory,
) -> None:
    use_case, predictor, _ = a_use_case(factory, probability=0.95)

    with pytest.raises(MachineNotFoundError):
        await use_case.execute(MachineId("M999"))

    assert predictor.calls == 0
