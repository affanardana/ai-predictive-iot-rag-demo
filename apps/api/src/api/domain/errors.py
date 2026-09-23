"""Domain error hierarchy.

Two distinct kinds of failure are modelled here, because callers handle them
differently:

* `DomainValidationError` -- a value is malformed at construction time. It also
  inherits `ValueError`, so Pydantic surfaces it as a validation error at the
  presentation boundary with no extra wiring.
* Everything else under `DomainError` -- a business rule or state conflict. The
  presentation layer maps these onto specific HTTP statuses.

Infrastructure exceptions must be translated into one of these before they
reach an upper layer; see `api.infrastructure.error_translation`.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for every error raised by the domain layer."""


class DomainValidationError(DomainError, ValueError):
    """A value object was constructed with a value that violates its invariant."""


class MachineNotFoundError(DomainError):
    """Raised when a machine identifier does not resolve to a known machine."""

    def __init__(self, machine_id: str) -> None:
        self.machine_id = machine_id
        super().__init__(f"Machine '{machine_id}' was not found.")


class IncidentNotFoundError(DomainError):
    """Raised when an incident identifier does not resolve to a known incident."""

    def __init__(self, incident_id: str) -> None:
        self.incident_id = incident_id
        super().__init__(f"Incident '{incident_id}' was not found.")


class InvalidTelemetryError(DomainError):
    """Raised when a telemetry record violates a domain invariant."""


class InvalidIncidentTransitionError(DomainError):
    """Raised when an incident status transition is not permitted."""

    def __init__(self, current: str, requested: str) -> None:
        self.current = current
        self.requested = requested
        super().__init__(f"An incident in status '{current}' cannot transition to '{requested}'.")


# ---------------------------------------------------------------------------
# Storage contract failures
#
# These live in the domain, not in application or infrastructure, because the
# domain owns the repository ports. If the domain defines the storage contract,
# it also defines how that contract fails -- which keeps infrastructure able to
# raise these without importing a sibling layer (the layer contract forbids
# infrastructure and application from depending on each other).
# ---------------------------------------------------------------------------


class PersistenceError(DomainError):
    """A storage operation failed.

    Carries a human-readable summary only. The original driver exception is
    deliberately not chained onto this message, because driver errors routinely
    embed connection strings and credentials; adapters log the original
    separately, through the redacting logger.
    """


class RepositoryUnavailableError(PersistenceError):
    """The backing store could not be reached, as opposed to rejecting a query."""
