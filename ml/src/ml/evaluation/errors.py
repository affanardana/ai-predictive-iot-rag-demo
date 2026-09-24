"""The error this package raises.

Its own module with no imports, for the same reason `ml.dataset.errors` has
one: the modules that detect a bad evaluation are spread across the package, and
none of them should have to import another to raise the same type.
"""

from __future__ import annotations


class EvaluationError(RuntimeError):
    """Raised when an evaluation cannot be computed, or must not be reported."""
