"""The corpus manifest, held against the corpus.

These run without `pypdf`: reading the manifest is JSON and paths. The one test
that opens a PDF says so and skips without the extra.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.knowledge.corpus import read_corpus
from ml.knowledge.errors import KnowledgeError

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
MANIFEST = REPOSITORY_ROOT / "knowledge" / "corpus.json"

#: PRD section 17's six categories. Asserted here as strings rather than
#: imported from the API, because `ml` may not import `api` -- and the check
#: that matters is that the corpus covers them at all.
PRD_CATEGORIES = frozenset(
    {
        "MOTOR_MAINTENANCE_MANUAL",
        "BEARING_INSPECTION",
        "OVERHEATING_TROUBLESHOOTING",
        "VIBRATION_DIAGNOSIS",
        "ELECTRICAL_SAFETY",
        "PREVENTIVE_MAINTENANCE_SCHEDULE",
    }
)


def test_the_manifest_and_the_corpus_are_a_bijection() -> None:
    """Every PDF is listed, and every entry resolves to a PDF.

    This is what stops the manifest rotting when an eleventh document appears:
    an unlisted file would be silently absent from retrieval, and nothing else
    would notice.
    """
    entries = read_corpus(MANIFEST)
    listed = {entry.path.resolve() for entry in entries}
    on_disk = {path.resolve() for path in (REPOSITORY_ROOT / "dummy_pdfs").rglob("*.pdf")}

    assert listed == on_disk
    assert len(entries) == 10


def test_an_unlisted_pdf_is_refused(tmp_path: Path) -> None:
    """Adding a file without a manifest entry fails loudly."""
    manifest = _manifest(
        tmp_path, entries=[_entry("knowledge-doc", "dummy_pdfs/guide/Missing.pdf")]
    )

    with pytest.raises(KnowledgeError, match="not there"):
        read_corpus(manifest)


def test_an_empty_manifest_is_refused(tmp_path: Path) -> None:
    """A manifest with no documents would ingest nothing and report success."""
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({"manifest_version": 1, "documents": []}), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="no documents"):
        read_corpus(manifest)


def test_an_unknown_manifest_version_is_refused(tmp_path: Path) -> None:
    """A newer manifest read by older code would be silently misread."""
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({"manifest_version": 99, "documents": []}), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="version 99"):
        read_corpus(manifest)


def test_every_entry_is_synthetic_and_categorised() -> None:
    """PRD section 17: demonstration material, and classified."""
    entries = read_corpus(MANIFEST)

    assert all(entry.synthetic for entry in entries)
    assert {entry.category for entry in entries} >= PRD_CATEGORIES
    assert all(entry.document_key and entry.title and entry.version for entry in entries)


def test_the_category_is_not_the_directory() -> None:
    """The trap the manifest exists to avoid.

    `procedure/` holds a bearing SOP and a lubrication procedure; `guide/` holds
    two named categories plus one PRD section 17 does not name. Taking the path
    as the category would misfile them and look correct doing it.
    """
    entries = {entry.document_key: entry for entry in read_corpus(MANIFEST)}

    lubrication = entries["lubrication-procedure"]
    assert lubrication.path.parent.name == "procedure"
    assert lubrication.category == "LUBRICATION"

    stator = entries["stator-winding-test-guide"]
    assert stator.path.parent.name == "guide"
    assert stator.category == "STATOR_WINDING_TEST"


def test_the_version_is_the_documents_own_revision_when_it_states_one() -> None:
    """One document of ten declares a revision, and the manifest carries it.

    The other nine declare none, and carry the corpus's own revision rather
    than an invented one -- a fabricated version would be the one field in a
    citation that cannot be checked against the page it points at.
    """
    pytest.importorskip("pypdf")
    from ml.knowledge.extraction import extract_lines

    entries = {entry.document_key: entry for entry in read_corpus(MANIFEST)}
    sop = entries["bearing-inspection-sop"]

    stated = [line.text for line in extract_lines(sop.path) if "Revision:" in line.text]
    assert stated, "the SOP no longer states a revision, so the manifest needs review."
    assert f"Revision: {sop.version}" in stated[0]


def test_every_labelled_phrase_is_where_the_label_says_it_is() -> None:
    """The label set is checked against the corpus it describes.

    A phrase that is not there would score zero for a reason that has nothing to
    do with retrieval, and the evaluation would blame the ranker for it. This is
    the cheapest possible guard against a label set that rots: it needs no
    models, no database and no server.
    """
    pytest.importorskip("pypdf")
    from ml.knowledge.evaluation import read_questions
    from ml.knowledge.extraction import extract_lines

    entries = {entry.document_key: entry for entry in read_corpus(MANIFEST)}
    questions = read_questions(REPOSITORY_ROOT / "knowledge" / "eval" / "questions.json")
    texts = {
        key: "\n".join(line.text for line in extract_lines(entry.path)).casefold()
        for key, entry in entries.items()
    }

    for question in questions:
        if not question.answerable:
            continue
        assert question.relevant_document_keys, question.question_id
        assert any(
            question.expected_phrase.casefold() in texts[key]
            for key in question.relevant_document_keys
        ), (
            f"'{question.expected_phrase}' is not in any document "
            f"labelled for {question.question_id}."
        )


def test_the_question_set_has_unanswerable_questions() -> None:
    """AC-009 is only measurable if some questions have no answer in the corpus.

    Without them the evaluation reports how well retrieval ranks and says
    nothing about whether the system refuses when it should -- which is the half
    that stops a Copilot inventing a procedure.
    """
    from ml.knowledge.evaluation import read_questions

    questions = read_questions(REPOSITORY_ROOT / "knowledge" / "eval" / "questions.json")

    assert sum(1 for question in questions if not question.answerable) >= 3
    assert sum(1 for question in questions if question.answerable) >= 8


def _manifest(tmp_path: Path, *, entries: list[dict]) -> Path:
    """Write a manifest under a fake repository root."""
    root = tmp_path
    (root / "knowledge").mkdir(parents=True, exist_ok=True)
    manifest = root / "knowledge" / "corpus.json"
    manifest.write_text(json.dumps({"manifest_version": 1, "documents": entries}), encoding="utf-8")
    return manifest


def _entry(key: str, path: str) -> dict:
    """Build one manifest entry."""
    return {
        "document_key": key,
        "title": key,
        "category": "BEARING_INSPECTION",
        "version": "1.0",
        "synthetic": True,
        "path": path,
    }
