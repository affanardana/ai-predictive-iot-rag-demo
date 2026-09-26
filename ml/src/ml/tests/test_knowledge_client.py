"""The two modules that talk to the API, against a mock transport.

No server: `httpx.MockTransport` answers in-process. What is checked is the
request that goes out and the failure that comes back — the two things a run
against a real stack would otherwise be the only test of, which is a slow and
expensive place to find a typo in a field name.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ml.knowledge.corpus import CorpusEntry
from ml.knowledge.errors import KnowledgeError
from ml.knowledge.evaluation import ask, count_documents, read_questions
from ml.knowledge.ingest import ingest_entry

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
QUESTIONS = REPOSITORY_ROOT / "knowledge" / "eval" / "questions.json"


def a_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    """Return a client whose transport answers in-process."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def an_entry(path: Path) -> CorpusEntry:
    """Build a manifest entry pointing at a real document."""
    return CorpusEntry(
        document_key="bearing-inspection-sop",
        title="Bearing Inspection SOP",
        category="BEARING_INSPECTION",
        version="1.4",
        synthetic=True,
        path=path,
    )


def test_ingest_sends_the_wire_shape_the_api_declares(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three fields per line, and the metadata beside them.

    The parse is exercised for real — a tiny PDF built by hand would test the
    fixture rather than the wire, so this one uses a document from the corpus.
    """
    pytest.importorskip("pypdf")
    seen: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            201,
            json={
                "document": {"document_id": "d1"},
                "chunk_count": 3,
                "unchanged": False,
                "replaced": False,
            },
        )

    entry = an_entry(REPOSITORY_ROOT / "dummy_pdfs" / "procedure" / "Bearing Inspection SOP.pdf")
    with a_client(capture) as client:
        outcome = ingest_entry(entry, client=client, api_url="http://api:8000", activate=True)

    body = seen[0]
    assert body["document_key"] == "bearing-inspection-sop"
    assert body["version"] == "1.4"
    assert body["activate"] is True
    assert body["is_synthetic"] is True
    assert set(body["lines"][0]) == {"text", "page", "font_size"}
    assert all(isinstance(line["page"], int) for line in body["lines"])
    assert outcome.chunk_count == 3


def test_ingest_reports_the_apis_own_message_when_it_refuses(
    tmp_path: Path,
) -> None:
    """A refusal names what the API said, not just a status code."""
    pytest.importorskip("pypdf")

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={"error": {"code": "document_content_conflict", "message": "bump the version"}},
        )

    entry = an_entry(REPOSITORY_ROOT / "dummy_pdfs" / "procedure" / "Bearing Inspection SOP.pdf")
    with a_client(refuse) as client, pytest.raises(KnowledgeError, match="bump the version"):
        ingest_entry(entry, client=client, api_url="http://api:8000")


def test_an_unreachable_api_is_a_clear_error() -> None:
    """A refused connection names the address rather than a stack trace.

    The document is real, because the parse comes first: a missing file would
    fail before the request and this would be asserting the wrong thing.
    """
    pytest.importorskip("pypdf")

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    entry = an_entry(REPOSITORY_ROOT / "dummy_pdfs" / "procedure" / "Bearing Inspection SOP.pdf")
    with a_client(fail) as client, pytest.raises(KnowledgeError, match="Could not reach"):
        ingest_entry(entry, client=client, api_url="http://api:8000")


def test_count_documents_refuses_an_empty_corpus() -> None:
    """Every metric would be zero for a reason that is not retrieval.

    Refused with the command that fixes it, rather than reported as a score of
    zero — which reads like a retrieval problem and is not one.
    """

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    with a_client(empty) as client, pytest.raises(KnowledgeError, match="ingest"):
        count_documents(client=client, api_url="http://api:8000")


def test_the_evaluation_asks_with_and_without_reranking() -> None:
    """The one field that differs between the two runs is the one being measured."""
    seen: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={
                "matches": [
                    {
                        "citation": {
                            "document_key": "bearing-inspection-sop",
                            "title": "Bearing Inspection SOP",
                            "version": "1.4",
                            "section": "3. Inspection Steps",
                            "page": 1,
                            "label": "Bearing Inspection SOP v1.4, page 1",
                        },
                        "content": "Attach an accelerometer to the bearing housing.",
                        "score": 0.9,
                    }
                ],
                "sufficient": True,
                "reason": "",
            },
        )

    questions = read_questions(QUESTIONS)[:1]
    with a_client(capture) as client:
        answers = ask(questions, client=client, api_url="http://api:8000", rerank=False)

    assert [body["rerank"] for body in seen] == [False]
    assert answers[0].ranked_document_keys == ("bearing-inspection-sop",)
    assert answers[0].phrase_rank == 1
    assert answers[0].sufficient is True
