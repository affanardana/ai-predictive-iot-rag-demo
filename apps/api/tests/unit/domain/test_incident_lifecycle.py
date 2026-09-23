"""Incident lifecycle transitions."""

from __future__ import annotations

import pytest

from api.domain.entities.incident import Incident
from api.domain.errors import InvalidIncidentTransitionError
from api.domain.value_objects.incident_status import ALLOWED_TRANSITIONS, IncidentStatus
from tests.support.factories import make_incident


def test_new_incidents_start_open() -> None:
    """An incident is created awaiting attention."""
    incident = make_incident()

    assert incident.status is IncidentStatus.OPEN
    assert incident.is_open


def test_open_can_be_acknowledged() -> None:
    """The ordinary first move."""
    incident = make_incident()

    incident.acknowledge()

    assert incident.status is IncidentStatus.ACKNOWLEDGED
    assert incident.is_open


def test_open_can_resolve_directly() -> None:
    """An incident may clear before anyone acknowledges it."""
    incident = make_incident()

    incident.resolve()

    assert incident.status is IncidentStatus.RESOLVED


def test_acknowledged_can_resolve() -> None:
    """The ordinary closing move."""
    incident = make_incident()
    incident.acknowledge()

    incident.resolve()

    assert incident.status is IncidentStatus.RESOLVED


def test_dismiss_marks_a_false_positive() -> None:
    """Dismissal closes without action."""
    incident = make_incident()

    incident.dismiss()

    assert incident.status is IncidentStatus.DISMISSED


@pytest.mark.parametrize("terminal", [IncidentStatus.RESOLVED, IncidentStatus.DISMISSED])
@pytest.mark.parametrize("action", ["acknowledge", "resolve", "dismiss"])
def test_terminal_states_accept_no_further_transition(
    terminal: IncidentStatus, action: str
) -> None:
    """Resolved and dismissed are terminal.

    Reopening is modelled as a new incident rather than a resurrection, so the
    audit trail stays append-only and an incident never loses the status it was
    closed with.
    """
    incident = make_incident()
    if terminal is IncidentStatus.RESOLVED:
        incident.resolve()
    else:
        incident.dismiss()

    with pytest.raises(InvalidIncidentTransitionError):
        getattr(incident, action)()


def test_acknowledged_cannot_be_acknowledged_again() -> None:
    """Re-acknowledging is not a transition."""
    incident = make_incident()
    incident.acknowledge()

    with pytest.raises(InvalidIncidentTransitionError):
        incident.acknowledge()


def test_error_names_both_states() -> None:
    """The failure explains what was attempted."""
    incident = make_incident()
    incident.resolve()

    with pytest.raises(InvalidIncidentTransitionError) as caught:
        incident.dismiss()

    assert caught.value.current == IncidentStatus.RESOLVED.value
    assert caught.value.requested == IncidentStatus.DISMISSED.value
    assert "RESOLVED" in str(caught.value)
    assert "DISMISSED" in str(caught.value)


def test_terminal_states_are_not_open() -> None:
    """Closed incidents are excluded from open counts."""
    resolved = make_incident(incident_id="inc-a")
    resolved.resolve()
    dismissed = make_incident(incident_id="inc-b")
    dismissed.dismiss()

    assert not resolved.is_open
    assert not dismissed.is_open


def test_transition_table_covers_every_status() -> None:
    """Each status declares its successors, and non-terminal ones have some.

    Guards against a status being added to the enum without a transition entry,
    which would raise `KeyError` at runtime instead of failing here.
    """
    assert set(ALLOWED_TRANSITIONS) == set(IncidentStatus)

    for status, successors in ALLOWED_TRANSITIONS.items():
        if status in (IncidentStatus.RESOLVED, IncidentStatus.DISMISSED):
            assert not successors
        else:
            assert successors


def test_create_factory_assigns_an_identifier() -> None:
    """The factory produces a distinct, non-blank identifier."""
    template = make_incident()

    def _new() -> Incident:
        return Incident.create(
            machine_id=template.machine_id,
            incident_type=template.incident_type,
            severity=template.severity,
            probability=template.probability,
            detected_at=template.detected_at,
        )

    first, second = _new(), _new()

    assert first.incident_id
    assert first.incident_id != second.incident_id
    assert first.status is IncidentStatus.OPEN
