"""Parsing the corpus and sending it to the API.

The parse happens here and the rules do not. Chunking, hashing, versioning,
activation and the model-identity check all live behind the API's port
boundary, because a second caller holding a copy of those rules is the failure
ADR 0004 exists to prevent -- the same arrangement n8n has for telemetry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from ml.knowledge.corpus import CorpusEntry
from ml.knowledge.errors import KnowledgeError
from ml.knowledge.extraction import extract_lines

#: Parsing a ten-document corpus takes a couple of seconds; embedding it on a
#: one-core box takes considerably longer, and the first call after a restart
#: loads two models.
DEFAULT_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class IngestOutcome:
    """What the API did with one document."""

    document_key: str
    version: str
    chunk_count: int
    unchanged: bool
    replaced: bool

    def render(self) -> str:
        """Describe the outcome in one line."""
        if self.unchanged:
            return f"{self.document_key} v{self.version}: unchanged"
        verb = "replaced" if self.replaced else "ingested"
        return f"{self.document_key} v{self.version}: {verb}, {self.chunk_count} passages"


def ingest_entry(
    entry: CorpusEntry,
    *,
    client: httpx.Client,
    api_url: str,
    activate: bool = False,
    allow_replace: bool = False,
) -> IngestOutcome:
    """Parse one document and send it to the API.

    Raises:
        KnowledgeError: if the document cannot be parsed, or the API refuses it.
    """
    lines = extract_lines(entry.path)
    if not lines:
        raise KnowledgeError(f"'{entry.path.name}' produced no text.")

    payload = {
        "document_key": entry.document_key,
        "title": entry.title,
        "category": entry.category,
        "version": entry.version,
        "source_path": _relative(entry.path),
        "is_synthetic": entry.synthetic,
        "activate": activate,
        "allow_replace": allow_replace,
        # The wire type the API's schema declares: page, text, font size. The
        # chunker decides what a heading is, so the size travels raw.
        "lines": [
            {"text": line.text, "page": line.page, "font_size": line.font_size} for line in lines
        ],
    }

    try:
        response = client.post(
            f"{api_url.rstrip('/')}/api/v1/knowledge/documents",
            json=payload,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        raise KnowledgeError(f"Could not reach the API at {api_url}: {exc}") from exc

    if response.status_code >= 400:
        raise KnowledgeError(
            f"The API refused '{entry.document_key}' v{entry.version} "
            f"({response.status_code}): {_message(response)}"
        )

    body = response.json()
    return IngestOutcome(
        document_key=entry.document_key,
        version=entry.version,
        chunk_count=int(body["chunk_count"]),
        unchanged=bool(body["unchanged"]),
        replaced=bool(body["replaced"]),
    )


def ingest_corpus(
    entries: Sequence[CorpusEntry],
    *,
    client: httpx.Client,
    api_url: str,
    activate: bool = False,
    allow_replace: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[IngestOutcome, ...]:
    """Send every document in the manifest, in order.

    Raises:
        KnowledgeError: on the first document the API or the parser refuses. A
            partial ingest is visible in the outcomes already reported, and
            re-running it is free for everything that did not change.
    """
    outcomes: list[IngestOutcome] = []
    for entry in entries:
        if progress is not None:
            progress(f"parsing {entry.path.name}")
        outcome = ingest_entry(
            entry,
            client=client,
            api_url=api_url,
            activate=activate,
            allow_replace=allow_replace,
        )
        if progress is not None:
            progress(outcome.render())
        outcomes.append(outcome)
    return tuple(outcomes)


def _relative(path: Path) -> str:
    """Return `path` relative to the working directory when it can be.

    Stored for provenance rather than resolved: the column exists so a reader
    can find the file a document came from, and an absolute path from one
    machine is worth less than a relative one that reads the same everywhere.
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _message(response: httpx.Response) -> str:
    """Return the API's own error message, when it sent one."""
    try:
        error = response.json().get("error", {})
        return str(error.get("message", response.text))
    except (ValueError, AttributeError):
        return response.text
