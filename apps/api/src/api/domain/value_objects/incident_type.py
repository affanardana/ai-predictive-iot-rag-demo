"""Incident type classification.

OPEN QUESTION (deliberately unresolved -- see the Phase 1 plan):

The predictive model is a binary 60-minute failure classifier. It answers
*"will this machine fail soon?"*, not *"what kind of failure is it?"*. Yet the
PRD requires an `incident_type` on every incident.

Three ways forward, none of them a Phase 1 decision:

1. Have the simulator publish its scenario label and inherit it. Cheap, but
   that is ground truth leaking into the product layer, and it sits close to
   the "engineering over demo tricks" line the masterplan draws.
2. Infer the type from telemetry signatures with explicit rules. Honest and
   self-contained, but real work that belongs with the model.
3. Leave incidents unclassified until one of the above is chosen.

Phase 1 takes option 3 via `DefaultIncidentPolicy.type_for()`, which always
returns `UNCLASSIFIED`. The stub is intentionally visible rather than hidden
behind a plausible-looking guess.
"""

from __future__ import annotations

from enum import StrEnum


class IncidentType(StrEnum):
    """Category of a detected failure condition."""

    BEARING_DEGRADATION = "BEARING_DEGRADATION"
    OVERHEATING = "OVERHEATING"
    OVERLOAD = "OVERLOAD"
    UNCLASSIFIED = "UNCLASSIFIED"
