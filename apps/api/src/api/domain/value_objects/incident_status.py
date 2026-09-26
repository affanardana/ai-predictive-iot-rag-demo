"""Incident lifecycle status and its permitted transitions."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from api.domain.errors import InvalidIncidentTransitionError


class IncidentStatus(StrEnum):
    """Lifecycle state of a maintenance incident."""

    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


#: `RESOLVED` and `DISMISSED` are terminal: reopening is modelled as a new
#: incident rather than a resurrection, so the audit trail stays append-only.
#: `OPEN` may go straight to `RESOLVED` because an incident can legitimately
#: clear before anyone acknowledges it.
ALLOWED_TRANSITIONS: Mapping[IncidentStatus, frozenset[IncidentStatus]] = {
    IncidentStatus.OPEN: frozenset(
        {IncidentStatus.ACKNOWLEDGED, IncidentStatus.RESOLVED, IncidentStatus.DISMISSED}
    ),
    IncidentStatus.ACKNOWLEDGED: frozenset({IncidentStatus.RESOLVED, IncidentStatus.DISMISSED}),
    IncidentStatus.RESOLVED: frozenset(),
    IncidentStatus.DISMISSED: frozenset(),
}


#: The statuses that still require someone's attention. These are what the
#: fleet view counts and what `RecordPrediction` suppresses against.
#:
#: Declared here rather than left to `Incident.is_open` alone because the count
#: is also computed in SQL -- `IncidentRepository.open_counts_by_machine`
#: filters on these values. A second copy of the rule inside a query would be
#: free to drift from the domain's, and the drift would show up as a fleet
#: summary that disagrees with the incident list beside it.
OPEN_INCIDENT_STATUSES: frozenset[IncidentStatus] = frozenset(
    {IncidentStatus.OPEN, IncidentStatus.ACKNOWLEDGED}
)


def ensure_transition_allowed(current: IncidentStatus, requested: IncidentStatus) -> None:
    """Raise if moving from `current` to `requested` is not permitted."""
    if requested not in ALLOWED_TRANSITIONS[current]:
        raise InvalidIncidentTransitionError(current=current.value, requested=requested.value)
