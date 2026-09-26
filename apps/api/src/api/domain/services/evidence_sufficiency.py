"""Deciding whether retrieved passages can support an answer.

PRD section 19: when the evidence is not there, the system must say so rather
than answer from the nearest neighbour. This is the rule that decides, and it
returns the reason so the caller has something to show.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Default minimum rerank score for a passage to count as evidence. Refined by
#: a measured run of `ml knowledge evaluate`; see the changelog.
DEFAULT_MINIMUM_SCORE = 0.0


@dataclass(frozen=True, slots=True)
class Sufficiency:
    """Whether the retrieved evidence can support an answer, and why not."""

    is_sufficient: bool
    reason: str


def assess_sufficiency(
    scores: Sequence[float],
    *,
    minimum_score: float = DEFAULT_MINIMUM_SCORE,
) -> Sufficiency:
    """Return whether these candidate scores can support an answer.

    `scores` are the reranker's, best first; only the best one decides.
    """
    if not scores:
        return Sufficiency(False, "No maintenance document in the corpus matched this question.")

    best = max(scores)
    if best < minimum_score:
        return Sufficiency(
            False,
            f"The closest maintenance document scored {best:.2f}, below the {minimum_score:.2f} "
            "needed to support an answer.",
        )
    return Sufficiency(True, "")
