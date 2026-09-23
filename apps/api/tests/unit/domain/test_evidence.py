"""Evidence model."""

from __future__ import annotations

import pytest

from api.domain.errors import DomainValidationError
from api.domain.value_objects.evidence import Evidence, EvidenceKind


def test_observed_and_predicted_need_no_source() -> None:
    """Measured and modelled values carry no document reference."""
    assert Evidence.observed(2.3).source is None
    assert Evidence.predicted(0.81).source is None
    assert Evidence.inferred("bearing wear").source is None


def test_factories_set_the_matching_kind() -> None:
    """Each factory produces its own evidence kind."""
    assert Evidence.observed(1.0).kind is EvidenceKind.OBSERVED
    assert Evidence.predicted(1.0).kind is EvidenceKind.PREDICTED
    assert Evidence.inferred(1.0).kind is EvidenceKind.INFERRED
    assert Evidence.documented(1.0, source="SOP-BRG-02").kind is EvidenceKind.DOCUMENTED


def test_documented_evidence_records_its_source() -> None:
    """A documentation claim names the document it came from."""
    evidence = Evidence.documented(
        "Check bearing housing temperature.", source="SOP-BRG-02 rev 1.4"
    )

    assert evidence.source == "SOP-BRG-02 rev 1.4"


@pytest.mark.parametrize("source", [None, ""])
def test_documented_evidence_requires_a_source(source: str | None) -> None:
    """Constructing an unsourced documentation claim is impossible.

    PRD section 18 requires the UI to expose which document supported an
    answer. Enforcing it in the constructor means an unsourced claim cannot
    exist at all, rather than being caught in review.
    """
    with pytest.raises(DomainValidationError):
        Evidence(kind=EvidenceKind.DOCUMENTED, value="Check the bearing.", source=source)


def test_preserves_the_wrapped_value() -> None:
    """Evidence is a transparent wrapper."""
    readings = [1.4, 1.9, 2.3]

    assert Evidence.observed(readings).value == readings
