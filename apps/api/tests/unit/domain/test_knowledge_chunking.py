"""Document chunking: page, heading and size boundaries."""

from __future__ import annotations

import pytest

from api.domain.errors import ChunkTooLongError
from api.domain.services.knowledge_chunking import (
    CHUNK_MAX_CHARACTERS,
    CHUNK_OVERLAP_CHARACTERS,
    TextChunk,
    chunk_document,
    content_hash,
    ensure_within_limit,
)
from api.domain.value_objects.text_line import TextLine

BODY = 13.33
TITLE = 26.66
SECTION = 17.33
SUBHEADING = BODY + 1.5


def line(text: str, *, page: int = 1, size: float = BODY) -> TextLine:
    """Build one extracted line."""
    return TextLine(text=text, page=page, font_size=size)


def paragraph(prefix: str, *, words: int = 300) -> str:
    """Build a single line long enough to need splitting on its own."""
    return " ".join(f"{prefix}{index:04d}" for index in range(words))


def test_empty_input_produces_no_chunks() -> None:
    """Nothing extracted means nothing to store."""
    assert chunk_document([]) == ()


def test_a_document_without_headings_still_chunks() -> None:
    """A heading-less document is chunked by page and size, with no section.

    `Citation.section` is empty rather than invented: the corpus is not
    guaranteed to have headings, and a made-up section would be a fabrication in
    the one field whose purpose is to be checkable.
    """
    chunks = chunk_document([line("First line."), line("Second line.")])

    assert [chunk.content for chunk in chunks] == ["First line.\nSecond line."]
    assert chunks[0].section == ""


def test_no_chunk_spans_a_page() -> None:
    """Page is an exact field, which requires a chunk to sit on one page."""
    chunks = chunk_document(
        [
            line("Page one text.", page=1),
            line("Page two text.", page=2),
        ]
    )

    assert [(chunk.page, chunk.content) for chunk in chunks] == [
        (1, "Page one text."),
        (2, "Page two text."),
    ]


def test_no_chunk_spans_a_heading() -> None:
    """A section boundary splits chunks even when the size allows more."""
    chunks = chunk_document(
        [
            line("1. Purpose", size=SECTION),
            line("Inspect the bearing."),
            line("2. Safety", size=SECTION),
            line("Lock out the machine."),
        ]
    )

    assert [chunk.content for chunk in chunks] == [
        "1. Purpose\nInspect the bearing.",
        "2. Safety\nLock out the machine.",
    ]


def test_section_names_nest() -> None:
    """A chunk under a subheading cites both headings, outermost first."""
    chunks = chunk_document(
        [
            line("Bearing Inspection", size=TITLE),
            line("3. Inspection Steps", size=SECTION),
            line("3.2 Vibration Analysis", size=SUBHEADING),
            line("Attach an accelerometer."),
        ]
    )

    assert chunks[-1].section == "Bearing Inspection > 3. Inspection Steps > 3.2 Vibration Analysis"


def test_sections_are_found_relative_to_the_documents_own_body_size() -> None:
    """Heading detection is relative, so a 16pt body does not hide its headings.

    Absolute thresholds break on the next document set in a different size: the
    corpus mixes 13.33pt and 16pt bodies.
    """
    chunks = chunk_document(
        [
            line("Sensor Calibration", size=21.33),
            line("1. Zero Adjustment", size=17.0),
            line("Adjust the zero offset.", size=16.0),
        ]
    )

    assert chunks[-1].section == "Sensor Calibration > 1. Zero Adjustment"


def test_the_character_cap_holds() -> None:
    """No chunk exceeds the embedding model's window.

    MiniLM truncates silently at 256 word-pieces, so an over-long chunk is
    embedded without its tail and nothing says so.
    """
    chunks = chunk_document([line(paragraph("word"))])

    assert len(chunks) > 1
    assert max(len(chunk.content) for chunk in chunks) <= CHUNK_MAX_CHARACTERS


def test_consecutive_pieces_of_one_paragraph_overlap() -> None:
    """A split inside a paragraph repeats its tail, so a fact on the seam survives."""
    chunks = chunk_document([line(paragraph("word"))])

    tail = chunks[0].content[-30:]
    assert tail in chunks[1].content[: CHUNK_OVERLAP_CHARACTERS + 30]


def test_overlap_does_not_cross_a_section_boundary() -> None:
    """A chunk carrying two sections would have no single honest section."""
    chunks = chunk_document(
        [
            line("1. First", size=SECTION),
            line(paragraph("alpha")),
            line("2. Second", size=SECTION),
            line(paragraph("beta")),
        ]
    )

    first_section = [chunk.content for chunk in chunks if chunk.section == "1. First"]
    second_section = [chunk.content for chunk in chunks if chunk.section == "2. Second"]

    assert len(first_section) > 1
    assert first_section[-1][-30:] not in second_section[0]


def test_chunking_is_deterministic() -> None:
    """The same document always produces the same chunks and the same hash."""
    lines = [
        line("1. Purpose", size=SECTION),
        line(paragraph("word")),
        line("Page two.", page=2),
    ]

    assert chunk_document(lines) == chunk_document(lines)
    assert content_hash(chunk_document(lines)) == content_hash(chunk_document(lines))


def test_the_hash_changes_when_a_page_changes() -> None:
    """A citation records the page, so the two are different content."""
    text = "Inspect the bearing."
    first = chunk_document([line(text, page=1)])
    second = chunk_document([line(text, page=2)])

    assert content_hash(first) != content_hash(second)


def test_an_oversized_chunk_is_refused_before_it_is_embedded() -> None:
    """The cap is defended twice, because the failure it guards is silent."""
    oversized = TextChunk(section="", page=1, content="x" * (CHUNK_MAX_CHARACTERS + 1))

    with pytest.raises(ChunkTooLongError):
        ensure_within_limit([oversized])
