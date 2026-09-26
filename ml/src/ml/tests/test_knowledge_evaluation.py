"""The retrieval metrics, on hand-worked examples.

The measured run needs a database and two models; the arithmetic does not, and
it is what CI can hold. Each case below is small enough to check by hand from
the numbers in it.
"""

from __future__ import annotations

import pytest

from ml.knowledge.evaluation import (
    Answer,
    Question,
    measure,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    render_comparison,
    render_scores,
    summarise_scores,
)

RELEVANT = ("bearing-inspection-sop", "vibration-diagnosis-guide")


def a_question(**overrides: object) -> Question:
    """Build a labelled question."""
    fields: dict[str, object] = {
        "question_id": "q1",
        "question": "What should I inspect when vibration rises?",
        "relevant_document_keys": RELEVANT,
        "expected_phrase": "accelerometer",
        "answerable": True,
    }
    fields.update(overrides)
    return Question(**fields)  # type: ignore[arg-type]


def an_answer(ranked: tuple[str, ...], **overrides: object) -> Answer:
    """Build one question's outcome."""
    fields: dict[str, object] = {
        "question": a_question(),
        "ranked_document_keys": ranked,
        "phrase_rank": 1,
        "sufficient": True,
    }
    fields.update(overrides)
    return Answer(**fields)  # type: ignore[arg-type]


def test_recall_counts_the_relevant_documents_that_came_back() -> None:
    """One of two relevant documents inside the top three is 0.5."""
    ranked = ("bearing-inspection-sop", "electrical-motor-safety", "overheating-x")

    assert recall_at_k(ranked, RELEVANT, 3) == pytest.approx(0.5)


def test_recall_beyond_the_cut_is_not_counted() -> None:
    """The k in recall@k is the point of it."""
    ranked = ("electrical-motor-safety", "overheating-x", "lubrication", "bearing-inspection-sop")

    assert recall_at_k(ranked, RELEVANT, 3) == 0.0
    assert recall_at_k(ranked, RELEVANT, 4) == pytest.approx(0.5)


def test_recall_with_no_relevant_documents_is_zero() -> None:
    """An unanswerable question has nothing to recall, and must not divide by it."""
    assert recall_at_k(("anything",), (), 3) == 0.0


def test_precision_is_over_the_cut_not_over_what_was_returned() -> None:
    """One relevant in the top three is a third, even if only one came back."""
    assert precision_at_k(("bearing-inspection-sop",), RELEVANT, 3) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_one_over_the_first_hit() -> None:
    """Rank two is a half, rank one is one."""
    assert reciprocal_rank(("electrical-motor-safety", "bearing-inspection-sop"), RELEVANT) == 0.5
    assert reciprocal_rank(("bearing-inspection-sop",), RELEVANT) == 1.0


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_in_the_cut() -> None:
    """A miss is zero rather than an error."""
    assert reciprocal_rank(("electrical-motor-safety",), RELEVANT) == 0.0


def test_ndcg_rewards_putting_the_relevant_document_first() -> None:
    """The ideal ordering scores one, and a worse one scores less."""
    first = ndcg_at_k(("bearing-inspection-sop", "vibration-diagnosis-guide"), RELEVANT, 5)
    second = ndcg_at_k(("vibration-diagnosis-guide", "bearing-inspection-sop"), RELEVANT, 5)
    later = ndcg_at_k(("electrical-motor-safety", "bearing-inspection-sop"), RELEVANT, 5)

    assert first == pytest.approx(1.0)
    assert second == pytest.approx(1.0)
    assert later < second
    assert later > 0.0


def test_ndcg_ignores_what_falls_below_the_cut() -> None:
    """A relevant document at rank six is not in nDCG@5."""
    ranked = ("x", "y", "z", "w", "v", "bearing-inspection-sop")

    assert ndcg_at_k(ranked, RELEVANT, 5) == 0.0


def test_the_abstention_rate_counts_only_the_unanswerable_questions() -> None:
    """AC-009 as a number: how often the system refuses rather than guesses.

    Answerable questions that were refused do not count here -- they are a
    different failure, and averaging the two would hide both.
    """
    metrics = measure(
        [
            an_answer(("bearing-inspection-sop",), sufficient=True),
            an_answer(
                ("electrical-motor-safety",),
                question=a_question(question_id="q2", answerable=False, relevant_document_keys=()),
                sufficient=False,
            ),
            an_answer(
                ("electrical-motor-safety",),
                question=a_question(question_id="q3", answerable=False, relevant_document_keys=()),
                sufficient=True,
            ),
        ]
    )

    assert metrics.answerable_count == 1
    assert metrics.unanswerable_count == 2
    assert metrics.abstention_rate == pytest.approx(0.5)


def test_the_phrase_rate_catches_the_right_document_with_the_wrong_section() -> None:
    """Document-level labels call that a success; a phrase does not.

    Answering from the bearing SOP but from a passage that does not mention the
    accelerometer is exactly the failure a Copilot would surface, and it is the
    reason the label set carries a phrase at all.
    """
    metrics = measure(
        [
            an_answer(("bearing-inspection-sop",), phrase_rank=1),
            an_answer(("bearing-inspection-sop",), phrase_rank=3),
            an_answer(("bearing-inspection-sop",), phrase_rank=None),
        ]
    )

    assert metrics.recall[3] == pytest.approx(0.5)
    assert metrics.phrase_match_rate == pytest.approx(1 / 3)


@pytest.mark.parametrize("k", [3, 5, 10])
def test_ndcg_cannot_exceed_one(k: int) -> None:
    """The ceiling the first measured run broke.

    A document whose several passages fill the top of the ranking was counted
    once per passage, so document-level nDCG came out at 1.72 — a number that
    should have been impossible and was, which is how the mistake was found.
    """
    ranked = ("bearing-inspection-sop",) * 5

    assert ndcg_at_k(ranked, ("bearing-inspection-sop",), k) == pytest.approx(1.0)


def test_every_metric_stays_inside_its_range() -> None:
    """Recall, precision and nDCG are fractions, and MRR is at most one."""
    answers = [
        an_answer(("bearing-inspection-sop", "bearing-inspection-sop", "other")),
        an_answer(("other", "other", "other")),
        an_answer(
            (),
            question=a_question(question_id="q2", answerable=False, relevant_document_keys=()),
            sufficient=False,
        ),
    ]

    metrics = measure(answers)

    assert all(0.0 <= value <= 1.0 for value in metrics.recall.values())
    assert 0.0 <= metrics.precision_at_3 <= 1.0
    assert 0.0 <= metrics.mrr_at_10 <= 1.0
    assert 0.0 <= metrics.ndcg_at_5 <= 1.0


def test_the_two_stages_scores_are_reported_so_the_threshold_can_be_read() -> None:
    """The abstention rate is unreadable without knowing the scales.

    A cosine similarity sits in [-1, 1] and rarely goes below zero; a
    cross-encoder logit is centred near zero. Both are compared against one
    configured threshold, so the separation is what says whether a stage's
    scores actually distinguish answerable from unanswerable questions.
    """
    answers = [
        an_answer(("bearing-inspection-sop",), top_score=0.6),
        an_answer(
            ("other",),
            question=a_question(question_id="q2", answerable=False, relevant_document_keys=()),
            sufficient=False,
            top_score=-4.0,
        ),
    ]

    summary = summarise_scores(answers)

    assert summary.answerable_median == pytest.approx(0.6)
    assert summary.unanswerable_median == pytest.approx(-4.0)
    assert summary.separation == pytest.approx(4.6)
    assert "separation" in render_scores(summary, summary)


def test_a_stage_that_cannot_tell_the_difference_reports_no_separation() -> None:
    """Zero separation is the honest answer when the scores do not separate."""
    answers = [
        an_answer(("bearing-inspection-sop",), top_score=0.5),
        an_answer(
            ("other",),
            question=a_question(question_id="q2", answerable=False, relevant_document_keys=()),
            sufficient=False,
            top_score=0.5,
        ),
    ]

    assert summarise_scores(answers).separation == pytest.approx(0.0)


def test_the_comparison_prints_both_runs_side_by_side() -> None:
    """The pair is what makes "reranking is implemented" a measurement."""
    answers = [an_answer(("bearing-inspection-sop", "vibration-diagnosis-guide"))]
    both = measure(answers)

    rendered = render_comparison(both, both, documents=10)

    assert "recall@3" in rendered
    assert "abstained when unanswerable" in rendered
    assert "corpus: 10 documents" in rendered
