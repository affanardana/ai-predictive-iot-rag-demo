"""Where a piece of documented evidence came from."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Citation:
    """A reference to one place in one document.

    PRD section 18 requires the dashboard to show the document name, its
    version, the section, and the page where available. Those four are the
    fields here, plus `document_key` for identity -- a title is a display
    string and two documents could share one, whereas the key is what
    `(document_key, version)` names in the schema.

    `section` is empty when a document has no headings at all. The corpus
    contains such documents, and inventing a section name for them would be a
    fabrication in the one field whose entire purpose is to be checkable.

    `page` is not optional. Every chunk is confined to a single page, which is
    what lets this be exact rather than "the page the chunk mostly sits on".
    """

    document_key: str
    title: str
    version: str
    section: str
    page: int

    def __post_init__(self) -> None:
        """Validate that the citation can actually be resolved and shown."""
        if not self.document_key.strip():
            raise ValueError("Citation document_key must not be blank.")
        if not self.title.strip():
            raise ValueError("Citation title must not be blank.")
        if not self.version.strip():
            raise ValueError("Citation version must not be blank.")
        if self.page < 1:
            raise ValueError(f"Citation page must be at least 1, got {self.page}.")

    @property
    def label(self) -> str:
        """Render the citation for display and for `Evidence.source`.

        `Evidence` requires documented evidence to name its source as a string,
        so this is what that string is. It is a property rather than a method
        because it is a rendering of the value, and the format is pinned by a
        test -- a citation is only useful if a reader can find the passage it
        points at.
        """
        parts = [f"{self.title} v{self.version}"]
        if self.section:
            parts.append(f"section {self.section}")
        parts.append(f"page {self.page}")
        return ", ".join(parts)
