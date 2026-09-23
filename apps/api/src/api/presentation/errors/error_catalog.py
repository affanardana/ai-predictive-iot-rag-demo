"""The single table mapping domain failures onto HTTP statuses.

Keeping this in one place is what stops infrastructure vocabulary and driver
messages from reaching a client: every failure that crosses the boundary has to
be classified here, and anything unclassified becomes a generic 500.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from api.domain.errors import (
    DomainError,
    DomainValidationError,
    IncidentNotFoundError,
    InvalidIncidentTransitionError,
    InvalidTelemetryError,
    MachineNotFoundError,
    PersistenceError,
    RepositoryUnavailableError,
)


@dataclass(frozen=True, slots=True)
class ErrorMapping:
    """HTTP status and stable error code for one kind of failure."""

    status_code: int
    code: str


ERROR_CATALOG: Mapping[type[DomainError], ErrorMapping] = {
    MachineNotFoundError: ErrorMapping(404, "machine_not_found"),
    IncidentNotFoundError: ErrorMapping(404, "incident_not_found"),
    # A conflict rather than a validation error: the request was well-formed,
    # the incident's current state simply does not permit the move.
    InvalidIncidentTransitionError: ErrorMapping(409, "invalid_incident_transition"),
    InvalidTelemetryError: ErrorMapping(422, "invalid_telemetry"),
    DomainValidationError: ErrorMapping(422, "invalid_value"),
    RepositoryUnavailableError: ErrorMapping(503, "repository_unavailable"),
    PersistenceError: ErrorMapping(500, "persistence_error"),
    DomainError: ErrorMapping(400, "domain_error"),
}


def resolve(error: DomainError) -> ErrorMapping:
    """Find the mapping for `error`, walking its class hierarchy.

    Walking the MRO means a new subclass inherits its parent's status unless it
    registers its own, so adding a domain error cannot accidentally produce an
    unhandled exception.
    """
    for klass in type(error).__mro__:
        mapping = ERROR_CATALOG.get(klass)
        if mapping is not None:
            return mapping
    return ErrorMapping(500, "internal_error")
