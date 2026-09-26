"""Turning extracted document lines into retrievable chunks.

The rules live here rather than in the parser because chunking is a domain
rule: the parser only reports what a page says and how large it was set.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from api.domain.errors import ChunkTooLongError
from api.domain.value_objects.text_line import TextLine

#: Longest chunk that will be embedded, in characters.
#:
#: `all-MiniLM-L6-v2` stops at 256 word-pieces and says nothing when it does, so
#: a longer chunk would be embedded without its tail. About 1000 characters is
#: roughly 250 word-pieces of English prose.
CHUNK_MAX_CHARACTERS = 1000

#: How much of a chunk's tail is repeated at the start of the next one.
CHUNK_OVERLAP_CHARACTERS = 150

#: A line is a heading when its font size exceeds the document's body size by
#: this much. Relative, because body size varies between documents: 17.33pt is a
#: heading in one and body text in another.
HEADING_SIZE_RATIO = 1.05


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A piece of a document, before it is embedded."""

    section: str
    page: int
    content: str


def chunk_document(lines: Sequence[TextLine]) -> tuple[TextChunk, ...]:
    """Split one document's lines into chunks.

    Boundaries, coarsest first: page, then heading, then size. A chunk never
    spans a page or a section, which is what lets a citation name one page and
    one section rather than the place a chunk mostly sits.
    """
    if not lines:
        return ()
    levels = _heading_levels(lines)
    chunks: list[TextChunk] = []
    for section, page, text in _segments(lines, levels):
        chunks.extend(
            TextChunk(section=section, page=page, content=piece) for piece in _split(text)
        )
    return tuple(chunks)


def content_hash(chunks: Sequence[TextChunk]) -> str:
    """Return a stable hash of a document's chunked content.

    Covers section, page and text, so the ingest endpoint can tell a re-ingest
    of unchanged content (a no-op) from one that would silently change what an
    existing citation resolves to.
    """
    payload = [[chunk.section, chunk.page, chunk.content] for chunk in chunks]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_within_limit(chunks: Sequence[TextChunk]) -> None:
    """Raise if any chunk is longer than the embedding model's window.

    A second check, after the chunker's own cap: the failure it guards against
    is a silent one, so it is worth asserting at the point where the text is
    about to be embedded rather than trusting the earlier one.
    """
    for chunk in chunks:
        if len(chunk.content) > CHUNK_MAX_CHARACTERS:
            raise ChunkTooLongError(len(chunk.content), CHUNK_MAX_CHARACTERS)


def _body_size(lines: Sequence[TextLine]) -> float:
    """Return the document's body size: the modal font size by characters set."""
    weights: Counter[float] = Counter()
    for line in lines:
        weights[_size(line)] += len(line.text)
    return weights.most_common(1)[0][0]


def _heading_levels(lines: Sequence[TextLine]) -> dict[float, int]:
    """Map each heading font size to its level, largest size first."""
    body_size = _body_size(lines)
    sizes = sorted(
        {_size(line) for line in lines if _size(line) > body_size * HEADING_SIZE_RATIO},
        reverse=True,
    )
    return {size: level for level, size in enumerate(sizes, start=1)}


def _size(line: TextLine) -> float:
    """Return the line's font size, rounded so equal sizes compare equal."""
    return round(line.font_size, 2)


def _section(headings: dict[int, str]) -> str:
    """Join the headings in force, outermost first."""
    return " > ".join(headings[level] for level in sorted(headings))


def _segments(
    lines: Sequence[TextLine],
    levels: dict[float, int],
) -> Iterator[tuple[str, int, str]]:
    """Yield `(section, page, text)` for each run of lines between boundaries.

    A heading closes the run before it and opens the one after, and a page
    change closes the run too. The section carries across a page change, since
    a section that continues onto the next page is still that section.
    """
    headings: dict[int, str] = {}
    page = lines[0].page
    current: list[str] = []

    def flush() -> tuple[str, int, str] | None:
        if not current:
            return None
        return _section(headings), page, "\n".join(current)

    for line in lines:
        if line.page != page:
            finished = flush()
            if finished is not None:
                yield finished
            current = []
            page = line.page

        level = levels.get(_size(line))
        if level is None:
            current.append(line.text.strip())
            continue

        finished = flush()
        if finished is not None:
            yield finished
        current = [line.text.strip()]
        for deeper in [depth for depth in headings if depth >= level]:
            del headings[deeper]
        headings[level] = line.text.strip()

    finished = flush()
    if finished is not None:
        yield finished


def _split(text: str) -> list[str]:
    """Split one segment into pieces of at most `CHUNK_MAX_CHARACTERS`.

    Overlap is applied only between pieces of the same segment, never across a
    heading or a page: a piece carrying text from two sections has no single
    honest section to cite.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_MAX_CHARACTERS:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_MAX_CHARACTERS, len(text))
        if end < len(text):
            end = _break_at(text, start, end)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP_CHARACTERS, start + 1)
    return pieces


def _break_at(text: str, start: int, end: int) -> int:
    """Move `end` back to the nearest line or word break, if there is one."""
    floor = start + CHUNK_MAX_CHARACTERS // 2
    for separator in ("\n", " "):
        cut = text.rfind(separator, floor, end)
        if cut != -1:
            return cut
    return end
