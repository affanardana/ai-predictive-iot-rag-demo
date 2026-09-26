"""The maintenance corpus: reading it, sending it, and measuring retrieval.

This package parses PDFs into lines and hands them to the API, which owns every
rule about what a document is. Chunking, versioning and activation all happen
behind that boundary, so nothing here decides them.

It needs the `knowledge` extra for `pypdf` and `httpx`, which is why the CLI
imports it lazily: `ml` rides into the inference service as `ml[train]`, and a
PDF parser has no business in the model-serving image.
"""

from __future__ import annotations

__all__: list[str] = []
