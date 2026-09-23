"""Prediction history use case."""

from __future__ import annotations

from datetime import timedelta

import pytest

from api.application.use_cases import GetPredictionHistory
from api.domain.errors import MachineNotFoundError
from api.domain.value_objects.machine_id import MachineId
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from tests.support.factories import DEFAULT_NOW, make_machine, make_prediction


async def test_raises_for_an_unknown_machine(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """An unregistered machine is a 404."""
    with pytest.raises(MachineNotFoundError):
        await GetPredictionHistory(unit_of_work_factory=uow_factory).execute(MachineId("M999"))


async def test_returns_predictions_most_recent_first(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """Predictions are ordered newest first, so the UI charts a trend."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        for hours, probability in ((3, 0.20), (1, 0.55), (2, 0.35)):
            await uow.predictions.add(
                make_prediction(
                    machine_id="M003",
                    prediction_id=f"pred-{hours}",
                    predicted_at=DEFAULT_NOW - timedelta(hours=hours),
                    probability=probability,
                )
            )

    history = await GetPredictionHistory(unit_of_work_factory=uow_factory).execute(
        MachineId("M003")
    )

    assert [prediction.prediction_id for prediction in history] == [
        "pred-1",
        "pred-2",
        "pred-3",
    ]
    assert history[0].probability.value == pytest.approx(0.55)


async def test_respects_the_limit(uow_factory: InMemoryUnitOfWorkFactory) -> None:
    """Only the most recent predictions are returned."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        for hours in range(5):
            await uow.predictions.add(
                make_prediction(
                    machine_id="M003",
                    prediction_id=f"pred-{hours}",
                    predicted_at=DEFAULT_NOW - timedelta(hours=hours),
                )
            )

    history = await GetPredictionHistory(unit_of_work_factory=uow_factory, limit=2).execute(
        MachineId("M003")
    )

    assert [prediction.prediction_id for prediction in history] == ["pred-0", "pred-1"]


async def test_returns_empty_for_a_machine_with_no_predictions(
    uow_factory: InMemoryUnitOfWorkFactory,
) -> None:
    """A registered machine with no predictions yields an empty history."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M003"))
        await uow.predictions.add(make_prediction(machine_id="M004", prediction_id="other"))

    history = await GetPredictionHistory(unit_of_work_factory=uow_factory).execute(
        MachineId("M003")
    )

    assert history == []
