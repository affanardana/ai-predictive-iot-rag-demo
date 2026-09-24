"""The one error this package raises for a dataset that does not add up.

Its own module, with no imports, for a specific reason: the checks that detect a
malformed dataset are split between the pure modules (which read the plan) and
the numpy ones (which read the arrays). Both need to raise the same type, and
`ml.dataset.artifacts` cannot be imported from `ml.dataset.plan` without
dragging numpy into a module that is deliberately free of it.
"""

from __future__ import annotations


class DatasetError(RuntimeError):
    """Raised when the generated files do not describe the plan that made them."""
