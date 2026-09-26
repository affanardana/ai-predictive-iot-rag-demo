"""Cosine similarity, in the domain.

The in-memory adapter ranks with this, and a PostgreSQL test asserts pgvector's
`<=>` agrees with it — so the comparison the database performs and the one the
domain defines are held to each other rather than assumed equal. Written in
plain Python because the domain may not import numpy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from api.domain.errors import DomainValidationError


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine of the angle between two vectors.

    A zero vector has no direction, so it is defined here as similarity 0.0
    rather than raising: an embedding of all zeros is a degenerate result, not
    a reason to fail a search.
    """
    if len(left) != len(right):
        raise DomainValidationError(
            f"Cannot compare vectors of {len(left)} and {len(right)} dimensions."
        )

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    # Clamped because the arithmetic is not exact: a vector compared against
    # itself can land a hair outside the range cosine is defined on.
    return min(1.0, max(-1.0, dot / (left_norm * right_norm)))
