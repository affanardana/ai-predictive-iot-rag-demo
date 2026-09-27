"""What can go wrong while measuring the Copilot.

Its own error rather than a shared one, matching `ml.knowledge.errors`: the
caller is a command line that prints the message and exits non-zero, and an
error type that says which part of the pipeline failed is what makes that
message worth reading.
"""

from __future__ import annotations


class CopilotEvaluationError(Exception):
    """The scorecard could not be produced."""
