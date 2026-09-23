"""Failure probability value object."""

from __future__ import annotations

import pytest

from api.domain.errors import DomainValidationError
from api.domain.value_objects.failure_probability import FailureProbability


def test_accepts_values_within_the_unit_interval() -> None:
    """Ordinary probabilities are stored unchanged."""
    assert FailureProbability(0.0).value == 0.0
    assert FailureProbability(0.42).value == 0.42
    assert FailureProbability(1.0).value == 1.0


@pytest.mark.parametrize(
    ("given", "expected"),
    [(-0.5, 0.0), (1.5, 1.0), (1.0000000001, 1.0)],
)
def test_clamps_out_of_range_values(given: float, expected: float) -> None:
    """Values outside [0, 1] are clamped rather than rejected.

    A network's sigmoid output can drift a hair past the bounds through
    floating-point error, and a probability of 1.0000000001 is meaningless
    rather than dangerous.
    """
    assert FailureProbability(given).value == expected


@pytest.mark.parametrize("given", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_values(given: float) -> None:
    """NaN and infinity are rejected, not clamped.

    NaN is the important case: it survives every comparison silently, so it
    would flow through the risk classifier and land in whichever band the
    comparison order happened to select.
    """
    with pytest.raises(DomainValidationError):
        FailureProbability(given)


def test_is_a_value_error_so_pydantic_reports_it_as_validation() -> None:
    """The error doubles as a `ValueError` for framework boundaries."""
    assert issubclass(DomainValidationError, ValueError)


def test_exposes_percentage_and_float_conversions() -> None:
    """Convenience conversions agree with the stored value."""
    probability = FailureProbability(0.815)

    assert probability.as_percentage() == pytest.approx(81.5)
    assert float(probability) == pytest.approx(0.815)


def test_instances_with_equal_values_are_interchangeable() -> None:
    """Value objects compare by value, not identity."""
    assert FailureProbability(0.5) == FailureProbability(0.5)
    assert hash(FailureProbability(0.5)) == hash(FailureProbability(0.5))
