"""The corpus manifest: which file is which document.

Four facts only a human knows -- key, title, category, version -- plus the path
to the file. Everything derivable (page count, chunk count, content hash) is
computed at ingest and never written back, following the principle
`ml.dataset.plan` states: the manifest is the generator's input, not a
transcription of its output.

The directory is deliberately not consulted for the category. Measured on this
corpus: `procedure/` holds a bearing SOP and a lubrication procedure, and
`guide/` holds two categories PRD section 17 names plus one it does not. Taking
the path as the category would misfile documents and look correct doing it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ml.knowledge.errors import KnowledgeError

MANIFEST_VERSION = 1

#: `knowledge/corpus.json` sits two levels below the repository root, and the
#: paths inside it are written relative to that root.
MANIFEST_DEPTH = 2


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """One document, as the manifest declares it."""

    document_key: str
    title: str
    category: str
    version: str
    synthetic: bool
    path: Path


def read_corpus(manifest: Path) -> tuple[CorpusEntry, ...]:
    """Read the manifest, and check it against the files it describes.

    Raises:
        KnowledgeError: if the manifest is malformed, is of an unknown version,
            names a file that is not there, or leaves a PDF in the corpus
            directory unlisted.
    """
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except OSError as exc:
        raise KnowledgeError(f"'{manifest}' could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise KnowledgeError(f"'{manifest}' is not valid JSON: {exc}") from exc

    found = payload.get("manifest_version")
    if found != MANIFEST_VERSION:
        raise KnowledgeError(
            f"'{manifest}' is manifest version {found}, and this build understands "
            f"version {MANIFEST_VERSION}."
        )

    root = manifest.resolve().parents[MANIFEST_DEPTH - 1]
    entries = tuple(_entry(raw, root, manifest) for raw in payload.get("documents", ()))
    if not entries:
        raise KnowledgeError(f"'{manifest}' lists no documents.")
    _check_nothing_is_unlisted(entries, manifest)
    return entries


def _entry(raw: object, root: Path, manifest: Path) -> CorpusEntry:
    """Build one entry, refusing anything the API would refuse later."""
    if not isinstance(raw, dict):
        raise KnowledgeError(f"'{manifest}' has an entry that is not an object.")
    missing = [
        field
        for field in ("document_key", "title", "category", "version", "path")
        if not raw.get(field)
    ]
    if missing:
        raise KnowledgeError(f"'{manifest}' has an entry missing {', '.join(missing)}: {raw!r}")

    path = root / str(raw["path"])
    if not path.exists():
        raise KnowledgeError(f"'{manifest}' lists '{raw['path']}', which is not there.")
    return CorpusEntry(
        document_key=str(raw["document_key"]),
        title=str(raw["title"]),
        category=str(raw["category"]),
        version=str(raw["version"]),
        synthetic=bool(raw.get("synthetic", False)),
        path=path,
    )


def _check_nothing_is_unlisted(
    entries: tuple[CorpusEntry, ...],
    manifest: Path,
) -> None:
    """Fail if a PDF sits in the corpus directory without a manifest entry.

    A document that exists but is not listed would be silently absent from
    retrieval, and the person who added it would have no way to tell. The
    directory comes from the entries themselves, so the manifest stays the only
    place the corpus is described.
    """
    roots = {entry.path.parent for entry in entries}
    for directory in sorted(roots):
        for path in sorted(directory.rglob("*.pdf")):
            if not any(entry.path == path for entry in entries):
                raise KnowledgeError(
                    f"'{path}' is in the corpus but not in '{manifest.name}'. "
                    "Add it, or remove the file."
                )
