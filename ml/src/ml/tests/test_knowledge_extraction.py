"""Extracting lines from the maintenance corpus.

Marked by skip rather than by a marker: `pypdf` is the `knowledge` extra, and CI
installs `--all-packages` without extras, so these skip there. Locally they run,
which is where the corpus is.

The module-level `importorskip` is not redundant with anything -- there is no
marker to filter on, and the imports below it need the extra at collection time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pypdf")

from ml.knowledge.extraction import PdfLine, extract_lines

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CORPUS = REPOSITORY_ROOT / "dummy_pdfs"


def test_the_corpus_is_where_the_tests_think_it_is() -> None:
    """A wrong path would turn every test below into a skip-shaped pass."""
    assert CORPUS.is_dir(), f"{CORPUS} is not a directory."


def test_extraction_recovers_the_documents_own_words() -> None:
    """The text comes back legible, which is the whole of what extraction does.

    Asserted against a phrase a reader can find on the page, so a change in
    `pypdf`'s fragment stream shows up here rather than as a retrieval score
    that quietly moved.
    """
    lines = extract_lines(CORPUS / "procedure" / "Bearing Inspection SOP.pdf")

    text = "\n".join(line.text for line in lines)
    assert "Standard Operating Procedure: Bearing Inspection" in text
    assert "Revision: 1.4" in text
    assert "Attach an accelerometer to the bearing housing" in text


def test_reading_order_is_top_to_bottom() -> None:
    """The document's own order, not the order the content stream emits.

    Measured: the text matrix's y decreases up the page and `cm` flips it, so
    reading the raw matrix back to front is the natural mistake -- and it is
    invisible in the text, which is what makes it worth a test.
    """
    lines = extract_lines(CORPUS / "procedure" / "Bearing Inspection SOP.pdf")

    headings = [line.text for line in lines if "Purpose" in line.text or "Safety" in line.text]
    assert headings == ["1. Purpose", "2. Safety Prerequisites"]

    title_index = next(i for i, line in enumerate(lines) if line.text.startswith("Standard"))
    assert title_index == 0


def test_the_font_size_survives_extraction() -> None:
    """Every line carries the size it was set at, which is the only structure.

    These documents have no outline and no tagged structure -- measured -- so
    the size is what a heading is recognised by, and the API's chunker is where
    that decision is made.
    """
    lines = extract_lines(CORPUS / "procedure" / "Bearing Inspection SOP.pdf")

    body = min(line.font_size for line in lines)
    headings = [
        line.font_size for line in lines if line.text in {"1. Purpose", "2. Safety Prerequisites"}
    ]
    assert len(set(headings)) == 1
    assert headings[0] > body * 1.05


def test_a_multi_page_document_reports_each_line_s_page() -> None:
    """`Citation.page` is exact only if the page survives extraction."""
    lines = extract_lines(CORPUS / "guide" / "Overheating Troubleshooting Guide.pdf")

    pages = sorted({line.page for line in lines})
    assert pages == [1, 2]
    assert all(isinstance(line, PdfLine) for line in lines)


def test_every_document_in_the_corpus_parses() -> None:
    """A file that parses to nothing would ingest as an empty document.

    Ten documents, none of them empty, and every line placed on a real page --
    which is the assertion that would catch a corpus file being replaced by
    something unreadable.
    """
    pdfs = sorted(CORPUS.rglob("*.pdf"))

    assert len(pdfs) == 10
    for path in pdfs:
        lines = extract_lines(path)
        assert lines, f"{path.name} produced no text."
        assert all(line.page >= 1 for line in lines)
        assert all(line.font_size > 0 for line in lines)
