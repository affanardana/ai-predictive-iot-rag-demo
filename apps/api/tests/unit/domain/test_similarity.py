"""Cosine similarity in the domain."""

from __future__ import annotations

import pytest

from api.domain.errors import DomainValidationError
from api.domain.services.similarity import cosine_similarity


def test_identical_vectors_are_fully_similar() -> None:
    """The boundary the ranking relies on."""
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_orthogonal_vectors_are_unrelated() -> None:
    """A right angle is zero similarity, not an error."""
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_opposite_vectors_are_fully_dissimilar() -> None:
    """The other end of the range."""
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([0.0, 0.0], [1.0, 2.0]),
        ([1.0, 2.0], [0.0, 0.0]),
        ([0.0, 0.0], [0.0, 0.0]),
    ],
)
def test_a_zero_vector_scores_zero_rather_than_dividing_by_zero(
    left: list[float], right: list[float]
) -> None:
    """A vector with no magnitude has no direction to compare."""
    assert cosine_similarity(left, right) == 0.0


def test_mismatched_dimensions_are_refused() -> None:
    """Vectors from two different models are not comparable."""
    with pytest.raises(DomainValidationError):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
