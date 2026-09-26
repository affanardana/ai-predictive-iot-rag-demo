"""Reading a maintenance PDF as lines, with the page and size they were set at.

The corpus is browser prints. Measured: no `/Outlines`, no `/StructTreeRoot`,
and no space characters -- words are positioned, not separated. So what survives
extraction is the text, the page it sits on, and the size it was set at, and the
font size is the only heading signal any of these documents carry.

This module reports that measurement and stops there. Whether 17.33pt is a
heading depends on what the rest of the document uses for body text, which is a
rule, and rules live in the API's domain behind the port boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ml.knowledge.errors import KnowledgeError

#: How far apart two fragments' baselines may sit and still be one line.
LINE_TOLERANCE = 0.5

#: A horizontal gap wider than this fraction of the font size is a space.
SPACE_GAP_RATIO = 0.25

#: Estimated advance width per character, as a fraction of the font size. The
#: visitor reports where a fragment starts and never how wide it was, so the end
#: of one has to be estimated to decide whether the next began a new word.
GLYPH_WIDTH_RATIO = 0.5


@dataclass(frozen=True, slots=True)
class PdfLine:
    """One line of a document, and the two facts that place it."""

    text: str
    page: int
    font_size: float


@dataclass(frozen=True, slots=True)
class _Fragment:
    """One run of text as the PDF reports it, before lines are assembled."""

    x: float
    y: float
    text: str
    font_size: float

    @property
    def end(self) -> float:
        """Where this fragment is estimated to stop, horizontally."""
        return self.x + len(self.text) * self.font_size * GLYPH_WIDTH_RATIO


def extract_lines(path: Path) -> tuple[PdfLine, ...]:
    """Return every line of `path`, in reading order.

    Raises:
        KnowledgeError: if the file cannot be read as a PDF.
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
    except Exception as exc:
        # pypdf raises several unrelated types for a malformed file, and which
        # one is an implementation detail callers should not have to know.
        raise KnowledgeError(f"'{path.name}' could not be read as a PDF: {exc}") from exc

    lines: list[PdfLine] = []
    for page_number in range(1, page_count + 1):
        fragments: list[_Fragment] = []
        reader.pages[page_number - 1].extract_text(visitor_text=_collector(fragments))
        lines.extend(_to_lines(fragments, page_number))
    return tuple(lines)


def _collector(into: list[_Fragment]) -> Callable[..., None]:
    """Return a visitor that appends each text run to `into`.

    Positions are composed through both matrices rather than read off the text
    matrix alone, and the size is scaled by the same factors. Measured on the
    corpus: `tm` is `[1, 0, 0, -1, x, y]` with y increasing *downward*, and `cm`
    scales by 0.75 -- so raw y sorts a page back to front, and a raw `Tf` size
    is a third larger than the points it is actually set in. Composing both puts
    positions and sizes in one space, where larger y is higher up the page.
    """

    def visit(
        text: str,
        cm: Sequence[float],
        tm: Sequence[float],
        _font: object,
        font_size: float,
    ) -> None:
        # The visitor is also called with "\n" to mark a line break, which is
        # not text and carries no size of its own.
        if not text.strip():
            return
        x = cm[0] * tm[4] + cm[2] * tm[5] + cm[4]
        y = cm[1] * tm[4] + cm[3] * tm[5] + cm[5]
        scale = math.sqrt(tm[0] ** 2 + tm[1] ** 2) * math.sqrt(cm[0] ** 2 + cm[1] ** 2)
        into.append(
            _Fragment(x=x, y=y, text=text, font_size=font_size * scale if scale else font_size)
        )

    return visit


def _to_lines(fragments: Sequence[_Fragment], page: int) -> list[PdfLine]:
    """Group fragments into lines, top to bottom and left to right."""
    ordered = sorted(fragments, key=lambda fragment: (-fragment.y, fragment.x))
    lines: list[PdfLine] = []
    group: list[_Fragment] = []
    for fragment in ordered:
        if group and abs(group[-1].y - fragment.y) > LINE_TOLERANCE:
            lines.append(_line(group, page))
            group = []
        group.append(fragment)
    if group:
        lines.append(_line(group, page))
    return lines


def _line(group: Sequence[_Fragment], page: int) -> PdfLine:
    """Join one line's fragments, inserting the spaces the PDF did not write.

    The size recorded is the largest in the line. These documents set a line at
    one size, and where a line mixes sizes the larger is what a reader would
    call it.
    """
    parts: list[str] = []
    previous: _Fragment | None = None
    for fragment in group:
        if (
            previous is not None
            and fragment.x - previous.end > SPACE_GAP_RATIO * fragment.font_size
        ):
            parts.append(" ")
        parts.append(fragment.text)
        previous = fragment
    return PdfLine(
        text="".join(parts).strip(),
        page=page,
        font_size=max(fragment.font_size for fragment in group),
    )
