"""Measuring retrieval against a labelled question set.

Two halves, deliberately separable. The metric functions are pure and unit
tested, so the arithmetic is checked in the offline tier; the measured run needs
a database, both models and the corpus, and its numbers are *recorded* rather
than asserted -- see `CHANGELOG.md` and `ml/README.md`.

`MASTERPLAN.md` names no retrieval metric, so the set here is chosen to answer
two questions: does the right document come back (recall, MRR, nDCG), and does
the system refuse when the corpus does not cover the question (abstention).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ml.knowledge.errors import KnowledgeError

#: What recall is reported at. Annotated with the corpus size wherever it is
#: quoted, because recall@10 over a hundred passages measures very little.
RECALL_KS = (3, 5, 10)

QUESTIONS_VERSION = 1

DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class Question:
    """One labelled question."""

    question_id: str
    question: str
    relevant_document_keys: tuple[str, ...]
    expected_phrase: str
    answerable: bool


@dataclass(frozen=True, slots=True)
class Answer:
    """What the corpus returned for one question."""

    question: Question
    ranked_document_keys: tuple[str, ...]
    #: One-based rank of the first passage carrying the expected phrase, or None
    #: if no returned passage carried it.
    phrase_rank: int | None
    sufficient: bool


@dataclass(frozen=True, slots=True)
class Metrics:
    """The numbers one run produced."""

    recall: dict[int, float]
    precision_at_3: float
    mrr_at_10: float
    ndcg_at_5: float
    #: Share of answerable questions whose top-ranked passage carried the phrase
    #: that section is known to contain. Catches "right document, wrong section",
    #: which document-level labels alone cannot.
    phrase_match_rate: float
    #: Share of unanswerable questions the system refused. This is AC-009 as a
    #: number rather than as a claim.
    abstention_rate: float
    answerable_count: int
    unanswerable_count: int


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Return the share of relevant documents present in the top `k`."""
    wanted = set(relevant)
    if not wanted:
        return 0.0
    return len(set(ranked[:k]) & wanted) / len(wanted)


def precision_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Return the share of the top `k` that is relevant."""
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / k


def reciprocal_rank(ranked: Sequence[str], relevant: Sequence[str], k: int = 10) -> float:
    """Return 1/rank of the first relevant document, or 0 if none is in the top `k`."""
    wanted = set(relevant)
    for rank, key in enumerate(ranked[:k], start=1):
        if key in wanted:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int = 5) -> float:
    """Return binary-gain nDCG@k, where every relevant document counts once."""
    wanted = set(relevant)
    if not wanted:
        return 0.0
    gains = [1.0 if key in wanted else 0.0 for key in ranked[:k]]
    discounted = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(wanted), k) + 1))
    return discounted / ideal if ideal else 0.0


def as_payload(metrics: Metrics) -> dict[str, Any]:
    """Return one run's metrics as JSON-serialisable data.

    Keys are strings because JSON has no integer keys, which would otherwise
    turn a written report's `recall` into something a reader has to re-parse.
    """
    return {
        "recall": {str(k): v for k, v in metrics.recall.items()},
        "precision_at_3": metrics.precision_at_3,
        "mrr_at_10": metrics.mrr_at_10,
        "ndcg_at_5": metrics.ndcg_at_5,
        "phrase_match_rate": metrics.phrase_match_rate,
        "abstention_rate": metrics.abstention_rate,
        "answerable_count": metrics.answerable_count,
        "unanswerable_count": metrics.unanswerable_count,
    }


def measure(answers: Sequence[Answer]) -> Metrics:
    """Compute every metric over one run."""
    answerable = [answer for answer in answers if answer.question.answerable]
    unanswerable = [answer for answer in answers if not answer.question.answerable]

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return Metrics(
        recall={
            k: mean(
                [
                    recall_at_k(
                        answer.ranked_document_keys,
                        answer.question.relevant_document_keys,
                        k,
                    )
                    for answer in answerable
                ]
            )
            for k in RECALL_KS
        },
        precision_at_3=mean(
            [
                precision_at_k(
                    answer.ranked_document_keys,
                    answer.question.relevant_document_keys,
                    3,
                )
                for answer in answerable
            ]
        ),
        mrr_at_10=mean(
            [
                reciprocal_rank(answer.ranked_document_keys, answer.question.relevant_document_keys)
                for answer in answerable
            ]
        ),
        ndcg_at_5=mean(
            [
                ndcg_at_k(answer.ranked_document_keys, answer.question.relevant_document_keys)
                for answer in answerable
            ]
        ),
        phrase_match_rate=mean([1.0 if answer.phrase_rank == 1 else 0.0 for answer in answerable]),
        abstention_rate=mean([1.0 if not answer.sufficient else 0.0 for answer in unanswerable]),
        answerable_count=len(answerable),
        unanswerable_count=len(unanswerable),
    )


def render_comparison(without: Metrics, with_rerank: Metrics, *, documents: int) -> str:
    """Render both runs side by side.

    The pair is the honest way to report "reranking is implemented" on a corpus
    where recall is already saturated: if the two columns are equal, that is the
    finding, and it is recorded as one rather than presented as an improvement.
    """
    rows: list[tuple[str, str, str]] = [
        ("answerable questions", str(without.answerable_count), str(with_rerank.answerable_count)),
        (
            "unanswerable questions",
            str(without.unanswerable_count),
            str(with_rerank.unanswerable_count),
        ),
    ]
    for k in RECALL_KS:
        rows.append((f"recall@{k}", f"{without.recall[k]:.3f}", f"{with_rerank.recall[k]:.3f}"))
    rows.extend(
        [
            ("precision@3", f"{without.precision_at_3:.3f}", f"{with_rerank.precision_at_3:.3f}"),
            ("mrr@10", f"{without.mrr_at_10:.3f}", f"{with_rerank.mrr_at_10:.3f}"),
            ("ndcg@5", f"{without.ndcg_at_5:.3f}", f"{with_rerank.ndcg_at_5:.3f}"),
            (
                "expected phrase first",
                f"{without.phrase_match_rate:.3f}",
                f"{with_rerank.phrase_match_rate:.3f}",
            ),
            (
                "abstained when unanswerable",
                f"{without.abstention_rate:.3f}",
                f"{with_rerank.abstention_rate:.3f}",
            ),
        ]
    )

    width = max(len(label) for label, _, _ in rows)
    lines = [
        f"{'':<{width}}  {'vector only':>11}  {'reranked':>11}",
        f"{'':<{width}}  {'-' * 11}  {'-' * 11}",
    ]
    lines.extend(f"{label:<{width}}  {left:>11}  {right:>11}" for label, left, right in rows)
    # Recall is quoted with the corpus size, because recall@10 over a hundred
    # passages measures very little and a reader has to know which it is.
    lines.append("")
    lines.append(f"corpus: {documents} documents")
    return "\n".join(lines)


def read_questions(path: Path) -> tuple[Question, ...]:
    """Read the labelled question set.

    Raises:
        KnowledgeError: if the file is missing, malformed, or of an unknown
            version.
    """
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise KnowledgeError(f"'{path}' could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise KnowledgeError(f"'{path}' is not valid JSON: {exc}") from exc

    found = payload.get("questions_version")
    if found != QUESTIONS_VERSION:
        raise KnowledgeError(
            f"'{path}' is question set version {found}, and this build understands "
            f"version {QUESTIONS_VERSION}."
        )

    questions = tuple(_question(raw, path) for raw in payload.get("questions", ()))
    if not questions:
        raise KnowledgeError(f"'{path}' lists no questions.")
    return questions


def _question(raw: object, path: Path) -> Question:
    """Build one labelled question."""
    if not isinstance(raw, dict) or not raw.get("id") or not raw.get("question"):
        raise KnowledgeError(f"'{path}' has a question without an id or text: {raw!r}")
    return Question(
        question_id=str(raw["id"]),
        question=str(raw["question"]),
        relevant_document_keys=tuple(str(key) for key in raw.get("relevant_document_keys", ())),
        expected_phrase=str(raw.get("expected_phrase", "")),
        answerable=bool(raw.get("answerable", False)),
    )


def ask(
    questions: Sequence[Question],
    *,
    client: httpx.Client,
    api_url: str,
    rerank: bool,
    limit: int = 5,
) -> tuple[Answer, ...]:
    """Ask the API every question, and record what came back.

    The reranked and un-reranked runs differ in one field of one request, which
    is what makes the difference between them attributable to the reranker.

    Raises:
        KnowledgeError: if the API cannot be reached or refuses a search.
    """
    answers: list[Answer] = []
    for question in questions:
        body = _search(client, api_url, question, rerank=rerank, limit=limit)
        matches = body.get("matches", [])
        answers.append(
            Answer(
                question=question,
                ranked_document_keys=tuple(
                    str(match["citation"]["document_key"]) for match in matches
                ),
                phrase_rank=_phrase_rank(matches, question.expected_phrase),
                sufficient=bool(body.get("sufficient", False)),
            )
        )
    return tuple(answers)


def _search(
    client: httpx.Client,
    api_url: str,
    question: Question,
    *,
    rerank: bool,
    limit: int,
) -> dict[str, Any]:
    """Run one search, returning the API's own response body."""
    try:
        response = client.post(
            f"{api_url.rstrip('/')}/api/v1/knowledge/search",
            json={"query": question.question, "limit": limit, "rerank": rerank},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        raise KnowledgeError(f"Could not reach the API at {api_url}: {exc}") from exc

    if response.status_code >= 400:
        raise KnowledgeError(
            f"Search for '{question.question_id}' failed ({response.status_code}): {response.text}"
        )
    body: dict[str, Any] = response.json()
    return body


def _phrase_rank(matches: Sequence[dict[str, Any]], phrase: str) -> int | None:
    """Return the one-based rank of the first passage carrying `phrase`.

    Document-level labels alone would call "the right document, the wrong
    section" a success. A phrase the relevant section is known to contain does
    not, which is the failure a Copilot would actually surface.
    """
    if not phrase:
        return None
    wanted = phrase.casefold()
    for rank, match in enumerate(matches, start=1):
        if wanted in str(match.get("content", "")).casefold():
            return rank
    return None
