"""Evidence model.

PRD section 16 requires the Copilot to distinguish what was measured, what the
model predicted, what the documentation says, and what was reasoned from those
things -- and specifically forbids presenting an inference as an observation.

This lives in the domain rather than in the response schema because it is a
product rule about what the system may claim, not a serialization detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from api.domain.errors import DomainValidationError


class EvidenceKind(StrEnum):
    """Provenance of a statement the system makes."""

    OBSERVED = "OBSERVED"
    PREDICTED = "PREDICTED"
    DOCUMENTED = "DOCUMENTED"
    INFERRED = "INFERRED"


@dataclass(frozen=True, slots=True)
class Evidence[T]:
    """A value together with the provenance that justifies it."""

    kind: EvidenceKind
    value: T
    source: str | None = None

    def __post_init__(self) -> None:
        """Require documented evidence to name its source.

        PRD section 18 requires the UI to expose which document contributed to
        an answer. Enforcing it here means an unsourced documentation claim
        cannot be constructed at all, rather than being caught in review.
        """
        if self.kind is EvidenceKind.DOCUMENTED and not self.source:
            raise DomainValidationError("Documented evidence must name its source document.")

    @classmethod
    def observed(cls, value: T) -> Evidence[T]:
        """Evidence measured directly by sensors."""
        return cls(kind=EvidenceKind.OBSERVED, value=value)

    @classmethod
    def predicted(cls, value: T) -> Evidence[T]:
        """Evidence produced by the ML model."""
        return cls(kind=EvidenceKind.PREDICTED, value=value)

    @classmethod
    def documented(cls, value: T, source: str) -> Evidence[T]:
        """Evidence retrieved from maintenance documentation."""
        return cls(kind=EvidenceKind.DOCUMENTED, value=value, source=source)

    @classmethod
    def inferred(cls, value: T) -> Evidence[T]:
        """A conclusion reasoned from other evidence."""
        return cls(kind=EvidenceKind.INFERRED, value=value)
