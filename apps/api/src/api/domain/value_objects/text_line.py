"""One line of text recovered from a document, with where it came from."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextLine:
    """A line of parsed text and the two facts that place it.

    This is the boundary between parsing and chunking. The parser runs in the
    `ingest` container, which owns `pypdf`; the chunker runs here, because
    chunking is a *rule* and rules live behind the port boundary. So this type
    crosses the wire, and its three fields are the whole contract.

    `font_size` is carried because it is the only structure these documents
    have. They are browser prints with no PDF outline and no tagged structure --
    measured, not assumed -- and the heading hierarchy survives solely as a
    difference in type size. Dropping the field here would leave the chunker
    unable to tell a heading from a paragraph, and the section metadata PRD
    section 18 requires would have nothing to derive from.

    It is deliberately a raw measurement rather than a classification. Whether
    17.33 is a heading depends on what the *rest of that document* uses for body
    text, and no single line can answer that.
    """

    text: str
    page: int
    font_size: float

    def __post_init__(self) -> None:
        """Validate that the line carries text and a plausible position."""
        if not self.text.strip():
            raise ValueError("TextLine text must not be blank.")
        if self.page < 1:
            raise ValueError(f"TextLine page must be at least 1, got {self.page}.")
        if self.font_size <= 0:
            raise ValueError(f"TextLine font_size must be positive, got {self.font_size}.")
