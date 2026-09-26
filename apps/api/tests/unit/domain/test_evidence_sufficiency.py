"""Whether retrieved evidence can support an answer (PRD section 19)."""

from __future__ import annotations

from api.domain.services.evidence_sufficiency import assess_sufficiency


def test_no_candidates_is_insufficient_with_a_reason() -> None:
    """Nothing retrieved is the clearest case of "the evidence is not there"."""
    result = assess_sufficiency([])

    assert result.is_sufficient is False
    assert result.reason


def test_a_score_above_the_threshold_is_sufficient() -> None:
    """The threshold is the line between evidence and the nearest neighbour."""
    assert assess_sufficiency([0.8], minimum_score=0.5).is_sufficient is True


def test_a_score_below_the_threshold_is_insufficient_and_says_why() -> None:
    """A weak match is reported as weak rather than answered from."""
    result = assess_sufficiency([0.2], minimum_score=0.5)

    assert result.is_sufficient is False
    assert "0.20" in result.reason
    assert "0.50" in result.reason


def test_only_the_best_candidate_decides() -> None:
    """A long tail of weak matches does not make a strong one weaker."""
    assert assess_sufficiency([0.9, 0.1, 0.05], minimum_score=0.5).is_sufficient is True
