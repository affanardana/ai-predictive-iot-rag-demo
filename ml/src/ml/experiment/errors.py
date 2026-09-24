"""The error this package raises, in its own import-free module."""

from __future__ import annotations


class ExperimentError(RuntimeError):
    """Raised when a run cannot be configured, rebuilt, or reproduced."""
