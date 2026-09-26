"""The error this package raises, in its own import-free module."""

from __future__ import annotations


class KnowledgeError(RuntimeError):
    """Raised when the corpus cannot be read, sent, or evaluated."""
