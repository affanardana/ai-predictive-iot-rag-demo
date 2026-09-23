"""Risk level and incident severity.

`RiskLevel` and `IncidentSeverity` are modelled separately on purpose. A
machine's *current risk* and an incident's *severity* are different concepts,
and they diverge the moment an incident is acknowledged or resolved -- a
resolved incident keeps its original severity while the machine's risk may have
returned to normal. Collapsing the two would make that impossible to express.
"""

from __future__ import annotations

from enum import StrEnum


class RiskLevel(StrEnum):
    """Application-level risk derived from a model's failure probability."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentSeverity(StrEnum):
    """How severe an individual incident is judged to be."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
